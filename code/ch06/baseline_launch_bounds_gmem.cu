// baseline_launch_bounds_gmem.cu -- baseline launch-bounds demo with a dummy
// global write to satisfy ptxas (avoids 0-byte gmem builds on GB10).

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstdio>
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(status));                                \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

constexpr int kLaunchBoundsWorkIters = 96;
constexpr int kLaunchBoundsTransformRepeats = 3;
constexpr int kTransformPasses = 8;
constexpr int kThreads = 64;
constexpr int kBlocks = 8;
constexpr int kChunkElements = kThreads * kBlocks;
__device__ float gmem_sink;

__device__ __forceinline__ float launch_bounds_workload(float value) {
    float acc0 = value * 1.0001f + 0.1f;
    float acc1 = value * 0.9997f - 0.05f;
    #pragma unroll
    for (int repeat = 0; repeat < kLaunchBoundsTransformRepeats; ++repeat) {
        #pragma unroll 4
        for (int iter = 0; iter < kLaunchBoundsWorkIters; ++iter) {
            const float coupled = (acc0 * acc1) * 0.00025f + (iter + 1 + repeat) * 1e-6f;
            const float denom = fabsf(acc0) + fabsf(acc1) + fabsf(coupled) + 1e-6f;
            const float inv = rsqrtf(denom);
            acc0 = fmaf(acc0, 1.00003f, inv * 0.0002f + coupled);
            acc1 = fmaf(acc1, 0.99991f, -inv * 0.00015f - coupled * 0.5f);
        }
    }
    return acc0 + acc1;
}

__global__ void kernel_no_lb(float* input, float* output, int n) {
    extern __shared__ float staging[];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = input[idx];
        for (int pass = 0; pass < kTransformPasses; ++pass) {
            staging[threadIdx.x] = v;
            __syncthreads();
            v = launch_bounds_workload(staging[threadIdx.x]);
            __syncthreads();
        }
        output[idx] = v;
        // Force a global write so ptxas emits gmem instructions.
        gmem_sink = v;
    }
}

int main() {
    NVTX_RANGE("main");
    const int N = 1024 * 64;  // smaller run for quick CI validation
    float *h_in, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_in, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_out, N * sizeof(float)));
    for (int i = 0; i < N; ++i) {
        NVTX_RANGE("verify");
        h_in[i] = float(i % 257) * 0.25f;
    }

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, kChunkElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, kChunkElements * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaEventRecord(start);
    int processed = 0;
    while (processed < N) {
        NVTX_RANGE("transfer_sync:h2d");
        const int chunk = std::min(kChunkElements, N - processed);
        const size_t bytes = size_t(chunk) * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_in, h_in + processed, bytes, cudaMemcpyHostToDevice));
        kernel_no_lb<<<kBlocks, kThreads, kThreads * sizeof(float)>>>(d_in, d_out, chunk);
        CUDA_CHECK(cudaMemcpy(h_out + processed, d_out, bytes, cudaMemcpyDeviceToHost));
        processed += chunk;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    std::printf("Launch-bounds baseline (gmem forcing) time: %.3f ms\n", ms);
    std::printf("First output: %.4f\n", h_out[0]);

#ifdef VERIFY
    double checksum = 0.0;
    for (int i = 0; i < N; ++i) {
        const double v = static_cast<double>(h_out[i]);
        if (!std::isfinite(v)) {
            continue;
        }
        checksum += std::abs(v);
    }
    if (!std::isfinite(checksum)) {
        checksum = 0.0;
    }
    if (checksum > static_cast<double>(FLT_MAX)) {
        checksum = static_cast<double>(FLT_MAX);
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
