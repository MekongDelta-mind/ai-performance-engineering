// baseline_cublaslt_gemm_fp4.cu -- Naive FP4 GEMM baseline for tensor core comparison
//
// This baseline uses block-scaled FP4 (NVFP4) format to provide an apples-to-apples
// comparison with the optimized FP4 cuBLASLt version that uses tensor cores.
//
// BOOK REFERENCE (Ch9/Ch19): FP4 on Blackwell provides 3-5x throughput over FP16
// due to 4-bit precision with per-block scaling for numerical stability.
// This baseline shows naive FP4 GEMM without tensor core acceleration.
//
// NOTE: Block-scaled FP4 uses groups of 16 elements sharing a single scale factor.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " "           \
                << cudaGetErrorString(status) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

// Block scaling parameters for FP4
constexpr int BLOCK_SIZE_SCALE = 16;  // Elements per scaling block

// FP4 representation: packed 2 values per byte (nibble format)
// Value range: [-6, 6] with 4-bit representation (signed)
struct PackedFP4 {
    uint8_t data;  // Contains 2 x 4-bit values

    __host__ __device__ void pack(float v0, float v1, float scale) {
        // Quantize to NVFP4 (E2M1) and pack (low nibble = v0, high nibble = v1).
        __nv_fp4_storage_t fp4_0 = __nv_cvt_float_to_fp4(v0 / scale, __NV_E2M1, cudaRoundNearest);
        __nv_fp4_storage_t fp4_1 = __nv_cvt_float_to_fp4(v1 / scale, __NV_E2M1, cudaRoundNearest);
        data = ((fp4_1 & 0x0F) << 4) | (fp4_0 & 0x0F);
    }

    __host__ __device__ void unpack(float& v0, float& v1, float scale) const {
        __nv_fp4_e2m1 fp4_0;
        __nv_fp4_e2m1 fp4_1;
        fp4_0.__x = static_cast<__nv_fp4_storage_t>(data & 0x0F);
        fp4_1.__x = static_cast<__nv_fp4_storage_t>((data >> 4) & 0x0F);
        v0 = static_cast<float>(fp4_0) * scale;
        v1 = static_cast<float>(fp4_1) * scale;
    }
};

// Naive tiled FP4 GEMM kernel (no tensor cores)
// Uses block-scaled FP4 with FP32 accumulation
template<int TILE_SIZE = 32>
__global__ void tiled_fp4_gemm_kernel(const PackedFP4* __restrict__ A,
                                       const __nv_fp8_e4m3* __restrict__ A_scales,
                                       const PackedFP4* __restrict__ B,
                                       const __nv_fp8_e4m3* __restrict__ B_scales,
                                       __half* __restrict__ C,
                                       int M, int N, int K,
                                       float alpha, float beta) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        const int k_idx = t * TILE_SIZE + threadIdx.x;
        const int k_row = t * TILE_SIZE + threadIdx.y;
        
        // Load tile of A into shared memory (unpack FP4 to FP32)
        if (row < M && k_idx < K) {
            // Determine which packed byte and position within byte
            const int a_idx = row * (K / 2) + k_idx / 2;
            const int scale_idx = row * (K / BLOCK_SIZE_SCALE) + k_idx / BLOCK_SIZE_SCALE;
            const float scale_a = static_cast<float>(A_scales[scale_idx]);
            
            float v0, v1;
            A[a_idx].unpack(v0, v1, scale_a);
            As[threadIdx.y][threadIdx.x] = (k_idx % 2 == 0) ? v0 : v1;
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile of B into shared memory (unpack FP4 to FP32)
        if (k_row < K && col < N) {
            const int b_idx = k_row * (N / 2) + col / 2;
            const int scale_idx = k_row * (N / BLOCK_SIZE_SCALE) + col / BLOCK_SIZE_SCALE;
            const float scale_b = static_cast<float>(B_scales[scale_idx]);
            
            float v0, v1;
            B[b_idx].unpack(v0, v1, scale_b);
            Bs[threadIdx.y][threadIdx.x] = (col % 2 == 0) ? v0 : v1;
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        float c_val = (beta != 0.0f) ? __half2float(C[row * N + col]) : 0.0f;
        float result = alpha * sum + beta * c_val;
        C[row * N + col] = __float2half(result);
    }
}

// Helper function to quantize FP32 to block-scaled FP4
void quantize_to_fp4(const float* input, PackedFP4* output, __nv_fp8_e4m3* scales,
                     int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        NVTX_RANGE("iteration");
        for (int block_start = 0; block_start < cols; block_start += BLOCK_SIZE_SCALE) {
            NVTX_RANGE("iteration");
            // Find max absolute value in block
            float max_abs = 0.0f;
            for (int i = 0; i < BLOCK_SIZE_SCALE && block_start + i < cols; ++i) {
                NVTX_RANGE("iteration");
                max_abs = std::max(max_abs, std::abs(input[r * cols + block_start + i]));
            }
            
            // Scale factor to map max_abs to FP4 range [-6, 6]
            float scale = (max_abs > 0.0f) ? max_abs / 6.0f : 1.0f;
            scales[r * (cols / BLOCK_SIZE_SCALE) + block_start / BLOCK_SIZE_SCALE] = __nv_fp8_e4m3(scale);
            
            // Quantize pairs of values
            for (int i = 0; i < BLOCK_SIZE_SCALE; i += 2) {
                NVTX_RANGE("iteration");
                if (block_start + i + 1 < cols) {
                    float v0 = input[r * cols + block_start + i];
                    float v1 = input[r * cols + block_start + i + 1];
                    output[r * (cols / 2) + (block_start + i) / 2].pack(v0, v1, scale);
                }
            }
        }
    }
}

int main() {
    NVTX_RANGE("main");
    // Matrix dimensions - must be divisible by BLOCK_SIZE_SCALE
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;
    constexpr int kIterations = 10;
    constexpr int kBatchCount = 8;

    // Sizes for packed FP4 (2 values per byte)
    const size_t packed_A_size = (M * K / 2) * kBatchCount;
    const size_t packed_B_size = (K * N / 2) * kBatchCount;
    const size_t elements_C = static_cast<size_t>(M) * N;
    
    // Scale factors per block
    const size_t scales_A_size = (M * K / BLOCK_SIZE_SCALE) * kBatchCount;
    const size_t scales_B_size = (K * N / BLOCK_SIZE_SCALE) * kBatchCount;

    // Host allocation
    float* h_A_fp32 = new float[M * K * kBatchCount];
    float* h_B_fp32 = new float[K * N * kBatchCount];
    PackedFP4* h_A = new PackedFP4[packed_A_size];
    PackedFP4* h_B = new PackedFP4[packed_B_size];
    __nv_fp8_e4m3* h_A_scales = new __nv_fp8_e4m3[scales_A_size];
    __nv_fp8_e4m3* h_B_scales = new __nv_fp8_e4m3[scales_B_size];
    __half* h_C = new __half[elements_C * kBatchCount];

    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < M * K * kBatchCount; ++i) {
        NVTX_RANGE("setup");
        h_A_fp32[i] = dis(gen);
    }
    for (size_t i = 0; i < K * N * kBatchCount; ++i) {
        NVTX_RANGE("setup");
        h_B_fp32[i] = dis(gen);
    }
    
    // Quantize to block-scaled FP4
    for (int batch = 0; batch < kBatchCount; ++batch) {
        NVTX_RANGE("batch");
        quantize_to_fp4(h_A_fp32 + batch * M * K, 
                        h_A + batch * (M * K / 2),
                        h_A_scales + batch * (M * K / BLOCK_SIZE_SCALE),
                        M, K);
        quantize_to_fp4(h_B_fp32 + batch * K * N,
                        h_B + batch * (K * N / 2),
                        h_B_scales + batch * (K * N / BLOCK_SIZE_SCALE),
                        K, N);
    }
    
    for (size_t i = 0; i < elements_C * kBatchCount; ++i) {
        NVTX_RANGE("setup");
        h_C[i] = __float2half(0.0f);
    }

    // Device allocation
    PackedFP4 *d_A = nullptr, *d_B = nullptr;
    __nv_fp8_e4m3 *d_A_scales = nullptr, *d_B_scales = nullptr;
    __half *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, packed_A_size * sizeof(PackedFP4)));
    CUDA_CHECK(cudaMalloc(&d_B, packed_B_size * sizeof(PackedFP4)));
    CUDA_CHECK(cudaMalloc(&d_A_scales, scales_A_size * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_B_scales, scales_B_size * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_C, elements_C * kBatchCount * sizeof(__half)));

    // Pre-load all data before timing
    CUDA_CHECK(cudaMemcpy(d_A, h_A, packed_A_size * sizeof(PackedFP4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, packed_B_size * sizeof(PackedFP4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_scales, h_A_scales, scales_A_size * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_scales, h_B_scales, scales_B_size * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, elements_C * kBatchCount * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    constexpr int TILE_SIZE = 32;
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE,
                   (M + TILE_SIZE - 1) / TILE_SIZE);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Warmup
    for (int batch = 0; batch < kBatchCount; ++batch) {
        NVTX_RANGE("compute_kernel");
        const size_t offset_A = batch * (M * K / 2);
        const size_t offset_B = batch * (K * N / 2);
        const size_t offset_C = batch * elements_C;
        const size_t offset_A_scales = batch * (M * K / BLOCK_SIZE_SCALE);
        const size_t offset_B_scales = batch * (K * N / BLOCK_SIZE_SCALE);
        
        tiled_fp4_gemm_kernel<TILE_SIZE><<<grid_size, block_size, 0, stream>>>(
            d_A + offset_A, d_A_scales + offset_A_scales,
            d_B + offset_B, d_B_scales + offset_B_scales,
            d_C + offset_C, M, N, K, alpha, beta);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Timed section: Kernel execution only
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iter = 0; iter < kIterations; ++iter) {
        NVTX_RANGE("batch");
        for (int batch = 0; batch < kBatchCount; ++batch) {
            NVTX_RANGE("compute_kernel");
            const size_t offset_A = batch * (M * K / 2);
            const size_t offset_B = batch * (K * N / 2);
            const size_t offset_C = batch * elements_C;
            const size_t offset_A_scales = batch * (M * K / BLOCK_SIZE_SCALE);
            const size_t offset_B_scales = batch * (K * N / BLOCK_SIZE_SCALE);
            
            tiled_fp4_gemm_kernel<TILE_SIZE><<<grid_size, block_size, 0, stream>>>(
                d_A + offset_A, d_A_scales + offset_A_scales,
                d_B + offset_B, d_B_scales + offset_B_scales,
                d_C + offset_C, M, N, K, alpha, beta);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(kIterations * kBatchCount);
    
    // Calculate TFLOPS (effective computation)
    const double flops = 2.0 * M * N * K * kBatchCount * kIterations;
    const double tflops = flops / (total_ms * 1e9);
    
    std::cout << "Naive Tiled FP4 GEMM (baseline): " << avg_ms << " ms" << std::endl;
    std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;

#ifdef VERIFY
    CUDA_CHECK(cudaMemcpy(h_C, d_C, elements_C * kBatchCount * sizeof(__half), cudaMemcpyDeviceToHost));
    const size_t elements = elements_C * kBatchCount;
    double checksum = 0.0;
    for (size_t i = 0; i < elements; ++i) {
        checksum += std::abs(__half2float(h_C[i]));
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A_scales));
    CUDA_CHECK(cudaFree(d_B_scales));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A_fp32;
    delete[] h_B_fp32;
    delete[] h_A;
    delete[] h_B;
    delete[] h_A_scales;
    delete[] h_B_scales;
    delete[] h_C;

    return 0;
}
