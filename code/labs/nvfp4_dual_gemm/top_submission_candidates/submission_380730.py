#!POPCORN leaderboard nvfp4_dual_gemm
#!POPCORN gpu NVIDIA

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


CUDA_SOURCE = r"""
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <cstdlib>

constexpr int WARP_SIZE = 32;
constexpr int MMA_K = 64;

// L2 Cache Hints
constexpr uint64_t EVICT_FIRST  = 0x12F0000000000000ULL;
constexpr uint64_t EVICT_LAST   = 0x14F0000000000000ULL;
constexpr uint64_t EVICT_NORMAL = 0x1000000000000000ULL;

__device__ __forceinline__
constexpr uint64_t desc_encode(uint64_t x) {
    return (x & 0x3'FFFFULL) >> 4ULL;
}

// Old SiLU helper (baseline): FP32 expf + FP32 divide, then multiply by y, returns packed half2.
__device__ __forceinline__
half2 silu_mul_h2(float x0, float x1, float y0, float y1) {
    const float2 x = make_float2(x0, x1);
    const float2 y = make_float2(y0, y1);
    const float2 s = make_float2(
        __fdividef(x.x, 1.0f + __expf(-x.x)),
        __fdividef(x.y, 1.0f + __expf(-x.y))
    );
    const float2 p = __fmul2_rn(s, y);
    return __float22half2_rn(p);
}

__device__ __forceinline__
void stg_32b(const void* dst, unsigned long long v0, unsigned long long v1,
            unsigned long long v2, unsigned long long v3) {
    asm volatile(
        "st.global.v4.b64 [%0], {%1, %2, %3, %4};"
        :: "l"(dst), "l"(v0), "l"(v1), "l"(v2), "l"(v3)
        : "memory"
    );
}

__device__ __forceinline__
uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred)
        : "r"(0xFFFFFFFF)
    );
    return pred;
}

__device__ __forceinline__
void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ __forceinline__
void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks)
    );
}

__device__ __forceinline__
void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAX:\n\t"
        "mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_RELAX;\n\t"
        "bra.uni LAB_WAIT_RELAX;\n\t"
        "DONE_RELAX:\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks)
    );
}

template <int CTA_GROUP>
__device__ __forceinline__
void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t cache_policy) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%6.L2::cache_hint "
        "[%0], [%1, {%2, %3, %4}], [%5], %7;"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr), "n"(CTA_GROUP), "l"(cache_policy)
        : "memory"
    );
}

template <int CTA_GROUP>
__device__ __forceinline__
void tma_3d_gmem2smem_mcast(int dst, const void *tmap_ptr, int x, int y, int z,
                           int mbar_addr, uint16_t cta_mask, uint64_t cache_policy) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::%6.L2::cache_hint "
        "[%0], [%1, {%2, %3, %4}], [%5], %7, %8;"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr), "n"(CTA_GROUP), "h"(cta_mask), "l"(cache_policy)
        : "memory"
    );
}

__device__ __forceinline__
void tcgen05_cp_cta2(int taddr, uint64_t s_desc) {
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
}

// Regular MMA without collector
__device__ __forceinline__
void tcgen05_mma_cta2(
    int d_tmem,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t i_desc,
    int scale_A_tmem,
    int scale_B_tmem,
    int enable_input_d
) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
           "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d)
    );
}

// MMA with collector::a::fill - fills the A collector buffer
__device__ __forceinline__
void tcgen05_mma_cta2_collector_fill(
    int d_tmem,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t i_desc,
    int scale_A_tmem,
    int scale_B_tmem,
    int enable_input_d
) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::fill "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
           "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d)
    );
}

// MMA with collector::a::lastuse - reuses A from collector buffer
__device__ __forceinline__
void tcgen05_mma_cta2_collector_lastuse(
    int d_tmem,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t i_desc,
    int scale_A_tmem,
    int scale_B_tmem,
    int enable_input_d
) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
           "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d)
    );
}

__device__ __forceinline__
void tcgen05_ld_32x32bx8(float *tmp, int addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
          "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
        : "r"(addr)
    );
}

__device__ __forceinline__
void tcgen05_ld_32x32bx64_addr(float *tmp, int addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
        "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        "  %8,  %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
        : "=f"(tmp[0]),  "=f"(tmp[1]),  "=f"(tmp[2]),  "=f"(tmp[3]),
          "=f"(tmp[4]),  "=f"(tmp[5]),  "=f"(tmp[6]),  "=f"(tmp[7]),
          "=f"(tmp[8]),  "=f"(tmp[9]),  "=f"(tmp[10]), "=f"(tmp[11]),
          "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
          "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]),
          "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
          "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]),
          "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31]),
          "=f"(tmp[32]), "=f"(tmp[33]), "=f"(tmp[34]), "=f"(tmp[35]),
          "=f"(tmp[36]), "=f"(tmp[37]), "=f"(tmp[38]), "=f"(tmp[39]),
          "=f"(tmp[40]), "=f"(tmp[41]), "=f"(tmp[42]), "=f"(tmp[43]),
          "=f"(tmp[44]), "=f"(tmp[45]), "=f"(tmp[46]), "=f"(tmp[47]),
          "=f"(tmp[48]), "=f"(tmp[49]), "=f"(tmp[50]), "=f"(tmp[51]),
          "=f"(tmp[52]), "=f"(tmp[53]), "=f"(tmp[54]), "=f"(tmp[55]),
          "=f"(tmp[56]), "=f"(tmp[57]), "=f"(tmp[58]), "=f"(tmp[59]),
          "=f"(tmp[60]), "=f"(tmp[61]), "=f"(tmp[62]), "=f"(tmp[63])
        : "r"(addr)
    );
}

// ---------------- TensorMap creation ----------------

void check_cu(CUresult err) {
    if (err == CUDA_SUCCESS) return;
    const char *error_msg_ptr;
    if (cuGetErrorString(err, &error_msg_ptr) != CUDA_SUCCESS)
        error_msg_ptr = "unable to get error string";
    TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", error_msg_ptr);
}

void init_AB_tmap(
    CUtensorMap *tmap,
    const char *ptr,
    uint64_t global_height,
    uint64_t global_width,
    uint32_t shared_height,
    uint32_t shared_width
) {
    constexpr uint32_t rank = 3;
    uint64_t globalDim[rank]       = {256, global_height, global_width / 256};
    uint64_t globalStrides[rank-1] = {global_width / 2, 128};
    uint32_t boxDim[rank]          = {256, shared_height, shared_width / 256};
    uint32_t elementStrides[rank]  = {1, 1, 1};
    auto err = cuTensorMapEncodeTiled(
        tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        rank,
        (void *)ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    check_cu(err);
}

void init_SF_tmap(
    CUtensorMap *tmap,
    const char *ptr,
    uint64_t mn,
    uint64_t K,
    uint32_t block_k
) {
    constexpr uint32_t rank = 3;
    const uint64_t k_blocks = K / 64;
    const uint64_t mn_blocks = mn / 128;
    const uint32_t tile_k_blocks = block_k / 64;
    constexpr uint64_t SF_BLOCK_BYTES = 512;
    constexpr uint64_t X_ELEMS = SF_BLOCK_BYTES / sizeof(uint16_t);
    uint64_t globalDim[rank]       = {X_ELEMS, mn_blocks, k_blocks};
    uint64_t globalStrides[rank-1] = {k_blocks * SF_BLOCK_BYTES, SF_BLOCK_BYTES};
    uint32_t boxDim[rank]          = {(uint32_t)X_ELEMS, 1, tile_k_blocks};
    uint32_t elementStrides[rank]  = {1, 1, 1};
    auto err = cuTensorMapEncodeTiled(
        tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT16,
        rank,
        (void *)ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    check_cu(err);
}

// ============================================================================
// N=64 kernel WITH collector (for M=256)
// ============================================================================
template <int BLOCK_M, int BLOCK_K, int NUM_STAGES>
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(BLOCK_M + 2 * WARP_SIZE)
void dual_gemm_cta2_collector_n64_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B1_tmap,
    const __grid_constant__ CUtensorMap B2_tmap,
    const __grid_constant__ CUtensorMap SFA_tmap,
    const __grid_constant__ CUtensorMap SFB1_tmap,
    const __grid_constant__ CUtensorMap SFB2_tmap,
    half *C_ptr,
    int M, int N, int K
) {
    constexpr int CTA_GROUP = 2;
    constexpr int BLOCK_N = 64;
    constexpr int HALF_BLOCK_N = BLOCK_N / CTA_GROUP;
    constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    const int cluster_pid = bid / CTA_GROUP;
    const int grid_n_clusters = N / BLOCK_N;
    const int cluster_m = cluster_pid / grid_n_clusters;
    const int cluster_n = cluster_pid % grid_n_clusters;
    const int off_m = cluster_m * (BLOCK_M * CTA_GROUP) + cta_rank * BLOCK_M;
    const int off_n = cluster_n * BLOCK_N;
    const int sf_y_A = off_m / 128;
    const int sf_y_B = off_n / 128;
    const int B_col_offset = off_n + cta_rank * HALF_BLOCK_N;

    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

    constexpr int A_size    = BLOCK_M * BLOCK_K / 2;
    constexpr int B1_size   = HALF_BLOCK_N * BLOCK_K / 2;
    constexpr int B2_size   = HALF_BLOCK_N * BLOCK_K / 2;
    constexpr int SFA_size  = 128 * BLOCK_K / 16;
    constexpr int SFB1_size = 128 * BLOCK_K / 16;
    constexpr int SFB2_size = 128 * BLOCK_K / 16;
    constexpr int STAGE_SIZE = A_size + B1_size + B2_size + SFA_size + SFB1_size + SFB2_size;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbars[NUM_STAGES * 2 + 1];
    __shared__ int tmem_addr[1];
    const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
    const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
    const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

    constexpr int ACC_BASE = 0;
    constexpr int ACC1_OFF = 0;
    constexpr int ACC2_OFF = BLOCK_N;
    constexpr int SFA_COLS_PER_K = 8;
    constexpr int SFB_COLS_PER_K = 4;
    constexpr int SFA_tmem  = 2 * BLOCK_N;
    constexpr int SFB1_tmem = SFA_tmem + SFA_COLS_PER_K * (BLOCK_K / MMA_K);
    constexpr int SFB2_tmem = SFB1_tmem + SFB_COLS_PER_K * (BLOCK_K / MMA_K);
    constexpr int TOTAL_TMEM_COLS = 256;

    if (warp_id == 0 && elect_sync()) {
        for (int i = 0; i < NUM_STAGES; i++) {
            mbarrier_init(tma_mbar_addr + i * 8, CTA_GROUP);
            mbarrier_init(mma_mbar_addr + i * 8, 1);
        }
        mbarrier_init(mainloop_mbar_addr, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                    :: "r"(addr), "r"(TOTAL_TMEM_COLS));
    }

    asm volatile("bar.sync 1, %0;" :: "r"(64) : "memory");

//    __syncthreads();
    const int taddr = tmem_addr[0];

    constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) | ((uint32_t)BLOCK_N >> 3U << 17U) | (2U << 27U);
    constexpr int SBO_AB = 8 * 128;
    constexpr int SBO_SF = 8 * 16;
    constexpr uint64_t AB_desc_base = (desc_encode(SBO_AB) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
    constexpr uint64_t SF_desc_base = (desc_encode(SBO_SF) << 32ULL) | (1ULL << 46ULL);

    const int num_iters = K / BLOCK_K;
    const uint64_t cache_A = EVICT_FIRST;
    const uint64_t cache_B = EVICT_FIRST;

    // TMA warp
    if (warp_id == NUM_WARPS - 2 && elect_sync()) {
        int tma_stage = 0;
        int mma_phase = 1;
        int it = 0;
        for (int iter_k = 0; iter_k < num_iters; iter_k++, it++) {
            if (it >= NUM_STAGES)
                mbarrier_wait_relaxed(mma_mbar_addr + tma_stage * 8, mma_phase);

            const int mbar_addr = (tma_mbar_addr + tma_stage * 8) & 0xFEFFFFFF;
            const int base_smem = smem + tma_stage * STAGE_SIZE;
            const int A_smem   = base_smem;
            const int B1_smem  = base_smem + A_size;
            const int B2_smem  = B1_smem + B1_size;
            const int SFA_smem = base_smem + A_size + B1_size + B2_size;
            const int SFB1_smem = SFA_smem + SFA_size;
            const int SFB2_smem = SFB1_smem + SFB1_size;

            constexpr int TENSOR_TMA_SIZE = A_size + B1_size + B2_size;
            const int SF_TMA_SIZE = SFA_size + ((cta_rank == 0) ? (CTA_GROUP * (SFB1_size + SFB2_size)) : 0);
            const int TOTAL_TMA_SIZE = TENSOR_TMA_SIZE + SF_TMA_SIZE;
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"(mbar_addr), "r"(TOTAL_TMA_SIZE) : "memory");

            const int z_ab = iter_k * (BLOCK_K / 256);
            const int z_sf = iter_k * (BLOCK_K / 64);

            tma_3d_gmem2smem<CTA_GROUP>(A_smem, &A_tmap, 0, off_m, z_ab, mbar_addr, cache_A);
            tma_3d_gmem2smem<CTA_GROUP>(B1_smem, &B1_tmap, 0, B_col_offset, z_ab, mbar_addr, cache_B);
            tma_3d_gmem2smem<CTA_GROUP>(B2_smem, &B2_tmap, 0, B_col_offset, z_ab, mbar_addr, cache_B);
            tma_3d_gmem2smem<CTA_GROUP>(SFA_smem, &SFA_tmap, 0, sf_y_A, z_sf, mbar_addr, cache_A);
            if (cta_rank == 0) {
                constexpr uint16_t cta_mask = (1u << CTA_GROUP) - 1u;
                tma_3d_gmem2smem_mcast<CTA_GROUP>(SFB1_smem, &SFB1_tmap, 0, sf_y_B, z_sf, mbar_addr, cta_mask, cache_B);
                tma_3d_gmem2smem_mcast<CTA_GROUP>(SFB2_smem, &SFB2_tmap, 0, sf_y_B, z_sf, mbar_addr, cta_mask, cache_B);
            }

            tma_stage = (tma_stage + 1) % NUM_STAGES;
            if (tma_stage == 0) mma_phase ^= 1;
        }
    } 
    // MMA warp with collector A reuse
    else if (cta_rank == 0 && warp_id == NUM_WARPS - 1 && elect_sync()) {
        int tma_stage = 0;
        int tma_phase = 0;
        constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
        const int scale_B_base_off = (cluster_n & 1) * (BLOCK_N / 32);
        
        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
            mbarrier_wait(tma_mbar_addr + tma_stage * 8, tma_phase);

            const int base_smem = smem + tma_stage * STAGE_SIZE;
            const int A_smem   = base_smem;
            const int B1_smem  = base_smem + A_size;
            const int B2_smem  = base_smem + A_size + B1_size;
            const int SFA_smem = base_smem + A_size + B1_size + B2_size;
            const int SFB1_smem = SFA_smem + SFA_size;
            const int SFB2_smem = SFB1_smem + SFB1_size;

            const uint64_t SFA_desc  = SF_desc_base + ((uint64_t)SFA_smem >> 4ULL);
            const uint64_t SFB1_desc = SF_desc_base + ((uint64_t)SFB1_smem >> 4ULL);
            const uint64_t SFB2_desc = SF_desc_base + ((uint64_t)SFB2_smem >> 4ULL);

            constexpr int SF_ITERS = BLOCK_K / MMA_K;
            constexpr int MMA_ITERS = 256 / MMA_K;
            
            uint64_t a_descs[MMA_ITERS];
            uint64_t b1_descs[MMA_ITERS];
            uint64_t b2_descs[MMA_ITERS];
            #pragma unroll
            for (int k2 = 0; k2 < MMA_ITERS; k2++) {
                const int off = k2 * 32;
                a_descs[k2] = AB_desc_base + desc_encode(A_smem + off);
                b1_descs[k2] = AB_desc_base + desc_encode(B1_smem + off);
                b2_descs[k2] = AB_desc_base + desc_encode(B2_smem + off);
            }
            const int scale_A_base = SFA_tmem;
            const int scale_B1_base = SFB1_tmem + scale_B_base_off;
            const int scale_B2_base = SFB2_tmem + scale_B_base_off;

            // Copy all scale factors first
            #pragma unroll
            for (int k = 0; k < SF_ITERS; k++) {
                tcgen05_cp_cta2(SFA_tmem + k * SFA_COLS_PER_K,  SFA_desc  + (uint64_t)k * 32ULL);
                tcgen05_cp_cta2(SFB1_tmem + k * SFB_COLS_PER_K, SFB1_desc + (uint64_t)k * 32ULL);
                tcgen05_cp_cta2(SFB2_tmem + k * SFB_COLS_PER_K, SFB2_desc + (uint64_t)k * 32ULL);
            }

            // Collector buffer for A-matrix reuse:
            // MMA1 with collector::a::fill (A @ B1) - reads A from SMEM, fills collector
            // MMA2 with collector::a::lastuse (A @ B2) - reuses A from collector
            #pragma unroll
            for (int k2 = 0; k2 < MMA_ITERS; k2++) {
                const uint64_t a_desc  = a_descs[k2];
                const uint64_t b1_desc = b1_descs[k2];
                const uint64_t b2_desc = b2_descs[k2];
                const int k_sf = k2;
                const int scale_A  = scale_A_base + k_sf * SFA_COLS_PER_K;
                const int scale_B1 = scale_B1_base + k_sf * SFB_COLS_PER_K;
                const int scale_B2 = scale_B2_base + k_sf * SFB_COLS_PER_K;
                const int enable_d = (k2 == 0) ? iter_k : 1;
                
                tcgen05_mma_cta2_collector_fill(ACC_BASE + ACC1_OFF, a_desc, b1_desc, i_desc, scale_A, scale_B1, enable_d);
                tcgen05_mma_cta2_collector_lastuse(ACC_BASE + ACC2_OFF, a_desc, b2_desc, i_desc, scale_A, scale_B2, enable_d);
            }

            asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                        :: "r"(mma_mbar_addr + tma_stage * 8), "h"(cta_mask) : "memory");

            tma_stage = (tma_stage + 1) % NUM_STAGES;
            if (tma_stage == 0) tma_phase ^= 1;
        }
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                    :: "r"(mainloop_mbar_addr), "h"(cta_mask) : "memory");
    }
    // Epilogue warps
    else if (warp_id < 4) {
        mbarrier_wait(mainloop_mbar_addr, 0);
        asm volatile("tcgen05.fence::after_thread_sync;");

        if (tid < BLOCK_M) {
            constexpr int WIDTH = 64;
            const int tmem_row = cta_rank * 128 + warp_id * 32;
            float acc1[WIDTH], acc2[WIDTH];
            const int addr1 = taddr + (tmem_row << 16) + (ACC_BASE + ACC1_OFF);
            const int addr2 = taddr + (tmem_row << 16) + (ACC_BASE + ACC2_OFF);
            tcgen05_ld_32x32bx64_addr(acc1, addr1);
            tcgen05_ld_32x32bx64_addr(acc2, addr2);
            asm volatile("tcgen05.wait::ld.sync.aligned;");

            half* row_ptr = C_ptr + (off_m + tid) * N + off_n;
            #pragma unroll
            for (int i = 0; i < WIDTH; i += 16) {
                float e[16];
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    e[j] = __expf(-acc1[i + j]);
                }

                half2 h0 = silu_mul_h2(acc1[i + 0],  acc1[i + 1],  acc2[i + 0],  acc2[i + 1]);
                half2 h1 = silu_mul_h2(acc1[i + 2],  acc1[i + 3],  acc2[i + 2],  acc2[i + 3]);
                half2 h2 = silu_mul_h2(acc1[i + 4],  acc1[i + 5],  acc2[i + 4],  acc2[i + 5]);
                half2 h3 = silu_mul_h2(acc1[i + 6],  acc1[i + 7],  acc2[i + 6],  acc2[i + 7]);
                half2 h4 = silu_mul_h2(acc1[i + 8],  acc1[i + 9],  acc2[i + 8],  acc2[i + 9]);
                half2 h5 = silu_mul_h2(acc1[i + 10], acc1[i + 11], acc2[i + 10], acc2[i + 11]);
                half2 h6 = silu_mul_h2(acc1[i + 12], acc1[i + 13], acc2[i + 12], acc2[i + 13]);
                half2 h7 = silu_mul_h2(acc1[i + 14], acc1[i + 15], acc2[i + 14], acc2[i + 15]);

                const uint32_t u0 = *reinterpret_cast<uint32_t*>(&h0);
                const uint32_t u1 = *reinterpret_cast<uint32_t*>(&h1);
                const uint32_t u2 = *reinterpret_cast<uint32_t*>(&h2);
                const uint32_t u3 = *reinterpret_cast<uint32_t*>(&h3);
                const uint32_t u4 = *reinterpret_cast<uint32_t*>(&h4);
                const uint32_t u5 = *reinterpret_cast<uint32_t*>(&h5);
                const uint32_t u6 = *reinterpret_cast<uint32_t*>(&h6);
                const uint32_t u7 = *reinterpret_cast<uint32_t*>(&h7);

                const unsigned long long q0 = (unsigned long long)u0 | ((unsigned long long)u1 << 32);
                const unsigned long long q1 = (unsigned long long)u2 | ((unsigned long long)u3 << 32);
                const unsigned long long q2 = (unsigned long long)u4 | ((unsigned long long)u5 << 32);
                const unsigned long long q3 = (unsigned long long)u6 | ((unsigned long long)u7 << 32);
                stg_32b((const void*)(row_ptr + i), q0, q1, q2, q3);
            }
        }
    }

    if (warp_id < 4) {
        asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");
        if (warp_id == 0)
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(TOTAL_TMEM_COLS));
    }
}

// ============================================================================
// N=128 kernel WITHOUT collector (baseline for M=512)
// ============================================================================
template <int BLOCK_M, int BLOCK_K, int NUM_STAGES>
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(BLOCK_M + 2 * WARP_SIZE)
void dual_gemm_cta2_baseline_n128_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B1_tmap,
    const __grid_constant__ CUtensorMap B2_tmap,
    const __grid_constant__ CUtensorMap SFA_tmap,
    const __grid_constant__ CUtensorMap SFB1_tmap,
    const __grid_constant__ CUtensorMap SFB2_tmap,
    half *C_ptr,
    int M, int N, int K
) {
    constexpr int CTA_GROUP = 2;
    constexpr int BLOCK_N = 128;
    constexpr int HALF_BLOCK_N = BLOCK_N / CTA_GROUP;
    constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    const int cluster_pid = bid / CTA_GROUP;
    const int grid_n_clusters = N / BLOCK_N;
    const int cluster_m = cluster_pid / grid_n_clusters;
    const int cluster_n = cluster_pid % grid_n_clusters;
    const int off_m = cluster_m * (BLOCK_M * CTA_GROUP) + cta_rank * BLOCK_M;
    const int off_n = cluster_n * BLOCK_N;
    const int sf_y_A = off_m / 128;
    const int sf_y_B = off_n / 128;
    const int B_col_offset = off_n + cta_rank * HALF_BLOCK_N;

    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

    constexpr int A_size    = BLOCK_M * BLOCK_K / 2;
    constexpr int B1_size   = HALF_BLOCK_N * BLOCK_K / 2;
    constexpr int B2_size   = HALF_BLOCK_N * BLOCK_K / 2;
    constexpr int SFA_size  = 128 * BLOCK_K / 16;
    constexpr int SFB1_size = 128 * BLOCK_K / 16;
    constexpr int SFB2_size = 128 * BLOCK_K / 16;
    // Add padding to reduce bank conflicts (128 bytes = 1 cache line)
    constexpr int PAD = 128;
    constexpr int STAGE_SIZE = A_size + PAD + B1_size + B2_size + PAD + SFA_size + SFB1_size + SFB2_size;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbars[NUM_STAGES * 2 + 1];
    __shared__ int tmem_addr[1];
    const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
    const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
    const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

    constexpr int ACC_BASE = 0;
    constexpr int ACC1_OFF = 0;
    constexpr int ACC2_OFF = BLOCK_N;
    constexpr int SFA_COLS_PER_K = 8;
    constexpr int SFB_COLS_PER_K = 4;
    constexpr int SFA_tmem  = 2 * BLOCK_N;
    constexpr int SFB1_tmem = SFA_tmem + SFA_COLS_PER_K * (BLOCK_K / MMA_K);
    constexpr int SFB2_tmem = SFB1_tmem + SFB_COLS_PER_K * (BLOCK_K / MMA_K);
    constexpr int TOTAL_TMEM_COLS = 512;

    if (warp_id == 0 && elect_sync()) {
        for (int i = 0; i < NUM_STAGES; i++) {
            mbarrier_init(tma_mbar_addr + i * 8, CTA_GROUP);
            mbarrier_init(mma_mbar_addr + i * 8, 1);
        }
        mbarrier_init(mainloop_mbar_addr, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                    :: "r"(addr), "r"(TOTAL_TMEM_COLS));
    }

//    asm volatile("bar.sync 1, %0;" :: "r"(64) : "memory");

    __syncthreads();
    const int taddr = tmem_addr[0];

    constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) | ((uint32_t)BLOCK_N >> 3U << 17U) | (2U << 27U);
    constexpr int SBO_AB = 8 * 128;
    constexpr int SBO_SF = 8 * 16;
    constexpr uint64_t AB_desc_base = (desc_encode(SBO_AB) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
    constexpr uint64_t SF_desc_base = (desc_encode(SBO_SF) << 32ULL) | (1ULL << 46ULL);

    const int num_iters = K / BLOCK_K;
    const uint64_t cache_A = EVICT_FIRST;
    const uint64_t cache_B = EVICT_FIRST;

    if (warp_id == NUM_WARPS - 2 && elect_sync()) {
        int tma_stage = 0;
        int mma_phase = 1;
        int it = 0;
        for (int iter_k = 0; iter_k < num_iters; iter_k++, it++) {
            if (it >= NUM_STAGES)
                mbarrier_wait_relaxed(mma_mbar_addr + tma_stage * 8, mma_phase);

            const int mbar_addr = (tma_mbar_addr + tma_stage * 8) & 0xFEFFFFFF;
            const int base_smem = smem + tma_stage * STAGE_SIZE;
            const int A_smem   = base_smem;
            const int B1_smem  = base_smem + A_size + PAD;  // padding after A
            const int B2_smem  = B1_smem + B1_size;
            const int SFA_smem = B2_smem + B2_size + PAD;   // padding after B2
            const int SFB1_smem = SFA_smem + SFA_size;
            const int SFB2_smem = SFB1_smem + SFB1_size;

            constexpr int TENSOR_TMA_SIZE = A_size + B1_size + B2_size;
            const int SF_TMA_SIZE = SFA_size + ((cta_rank == 0) ? (CTA_GROUP * (SFB1_size + SFB2_size)) : 0);
            const int TOTAL_TMA_SIZE = TENSOR_TMA_SIZE + SF_TMA_SIZE;
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"(mbar_addr), "r"(TOTAL_TMA_SIZE) : "memory");

            const int z_ab = iter_k * (BLOCK_K / 256);
            const int z_sf = iter_k * (BLOCK_K / 64);

            tma_3d_gmem2smem<CTA_GROUP>(B1_smem, &B1_tmap, 0, B_col_offset, z_ab, mbar_addr, cache_B);
            tma_3d_gmem2smem<CTA_GROUP>(B2_smem, &B2_tmap, 0, B_col_offset, z_ab, mbar_addr, cache_B);
            if (cta_rank == 0) {
                constexpr uint16_t cta_mask = (1u << CTA_GROUP) - 1u;
                tma_3d_gmem2smem_mcast<CTA_GROUP>(SFB1_smem, &SFB1_tmap, 0, sf_y_B, z_sf, mbar_addr, cta_mask, cache_B);
                tma_3d_gmem2smem_mcast<CTA_GROUP>(SFB2_smem, &SFB2_tmap, 0, sf_y_B, z_sf, mbar_addr, cta_mask, cache_B);
            }
            tma_3d_gmem2smem<CTA_GROUP>(A_smem, &A_tmap, 0, off_m, z_ab, mbar_addr, cache_A);
            tma_3d_gmem2smem<CTA_GROUP>(SFA_smem, &SFA_tmap, 0, sf_y_A, z_sf, mbar_addr, cache_A);

            tma_stage = (tma_stage + 1) % NUM_STAGES;
            if (tma_stage == 0) mma_phase ^= 1;
        }
    } else if (cta_rank == 0 && warp_id == NUM_WARPS - 1 && elect_sync()) {
        int tma_stage = 0;
        int tma_phase = 0;
        constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
        constexpr int scale_B_base_off = 0;
        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
            mbarrier_wait(tma_mbar_addr + tma_stage * 8, tma_phase);

            const int base_smem = smem + tma_stage * STAGE_SIZE;
            const int A_smem   = base_smem;
            const int B1_smem  = base_smem + A_size + PAD;  // padding after A
            const int B2_smem  = B1_smem + B1_size;
            const int SFA_smem = B2_smem + B2_size + PAD;   // padding after B2
            const int SFB1_smem = SFA_smem + SFA_size;
            const int SFB2_smem = SFB1_smem + SFB1_size;

            const uint64_t SFA_desc  = SF_desc_base + ((uint64_t)SFA_smem >> 4ULL);
            const uint64_t SFB1_desc = SF_desc_base + ((uint64_t)SFB1_smem >> 4ULL);
            const uint64_t SFB2_desc = SF_desc_base + ((uint64_t)SFB2_smem >> 4ULL);

            constexpr int SF_ITERS = BLOCK_K / MMA_K;
            constexpr int MMA_ITERS = 256 / MMA_K;
            constexpr int HALF = (SF_ITERS > 1) ? (SF_ITERS / 2) : 1;
            uint64_t a_descs[MMA_ITERS];
            uint64_t b1_descs[MMA_ITERS];
            uint64_t b2_descs[MMA_ITERS];
            #pragma unroll
            for (int k2 = 0; k2 < MMA_ITERS; k2++) {
                const int off = k2 * 32;
                a_descs[k2] = AB_desc_base + desc_encode(A_smem + off);
                b1_descs[k2] = AB_desc_base + desc_encode(B1_smem + off);
                b2_descs[k2] = AB_desc_base + desc_encode(B2_smem + off);
            }
            const int scale_A_base = SFA_tmem;
            const int scale_B1_base = SFB1_tmem + scale_B_base_off;
            const int scale_B2_base = SFB2_tmem + scale_B_base_off;

            #pragma unroll
            for (int k = 0; k < HALF; k++) {
                tcgen05_cp_cta2(SFA_tmem + k * SFA_COLS_PER_K,  SFA_desc  + (uint64_t)k * 32ULL);
                tcgen05_cp_cta2(SFB1_tmem + k * SFB_COLS_PER_K, SFB1_desc + (uint64_t)k * 32ULL);
                tcgen05_cp_cta2(SFB2_tmem + k * SFB_COLS_PER_K, SFB2_desc + (uint64_t)k * 32ULL);
            }

            #pragma unroll
            for (int k2 = 0; k2 < HALF; k2++) {
                const uint64_t a_desc  = a_descs[k2];
                const uint64_t b1_desc = b1_descs[k2];
                const uint64_t b2_desc = b2_descs[k2];
                const int k_sf = k2;
                const int scale_A  = scale_A_base + k_sf * SFA_COLS_PER_K;
                const int scale_B1 = scale_B1_base + k_sf * SFB_COLS_PER_K;
                const int scale_B2 = scale_B2_base + k_sf * SFB_COLS_PER_K;
                const int enable_d = (k2 == 0) ? iter_k : 1;
                tcgen05_mma_cta2(ACC_BASE + ACC1_OFF, a_desc, b1_desc, i_desc, scale_A, scale_B1, enable_d);
                tcgen05_mma_cta2(ACC_BASE + ACC2_OFF, a_desc, b2_desc, i_desc, scale_A, scale_B2, enable_d);
            }

            #pragma unroll
            for (int k = HALF; k < SF_ITERS; k++) {
                tcgen05_cp_cta2(SFA_tmem + k * SFA_COLS_PER_K,  SFA_desc  + (uint64_t)k * 32ULL);
                tcgen05_cp_cta2(SFB1_tmem + k * SFB_COLS_PER_K, SFB1_desc + (uint64_t)k * 32ULL);
                tcgen05_cp_cta2(SFB2_tmem + k * SFB_COLS_PER_K, SFB2_desc + (uint64_t)k * 32ULL);
            }

            #pragma unroll
            for (int k2 = HALF; k2 < MMA_ITERS; k2++) {
                const uint64_t a_desc  = a_descs[k2];
                const uint64_t b1_desc = b1_descs[k2];
                const uint64_t b2_desc = b2_descs[k2];
                const int k_sf = k2;
                const int scale_A  = scale_A_base + k_sf * SFA_COLS_PER_K;
                const int scale_B1 = scale_B1_base + k_sf * SFB_COLS_PER_K;
                const int scale_B2 = scale_B2_base + k_sf * SFB_COLS_PER_K;
                const int enable_d = 1;
                tcgen05_mma_cta2(ACC_BASE + ACC1_OFF, a_desc, b1_desc, i_desc, scale_A, scale_B1, enable_d);
                tcgen05_mma_cta2(ACC_BASE + ACC2_OFF, a_desc, b2_desc, i_desc, scale_A, scale_B2, enable_d);
            }

            asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                        :: "r"(mma_mbar_addr + tma_stage * 8), "h"(cta_mask) : "memory");

            tma_stage = (tma_stage + 1) % NUM_STAGES;
            if (tma_stage == 0) tma_phase ^= 1;
        }
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                    :: "r"(mainloop_mbar_addr), "h"(cta_mask) : "memory");
    } else if (warp_id < 4) {
        mbarrier_wait(mainloop_mbar_addr, 0);
        asm volatile("tcgen05.fence::after_thread_sync;");

        if (tid < BLOCK_M) {
            constexpr int CHUNK = 16;
            const int tmem_row = cta_rank * 128 + warp_id * 32;
            half* row_ptr = C_ptr + (off_m + tid) * N + off_n;

            #pragma unroll 1
            for (int seg = 0; seg < 128; seg += 64) {
                #pragma unroll
                for (int base = 0; base < 64; base += CHUNK) {
                    float acc1[CHUNK], acc2[CHUNK];
                    const int addr1 = taddr + (tmem_row << 16) + (ACC_BASE + ACC1_OFF + seg + base);
                    const int addr2 = taddr + (tmem_row << 16) + (ACC_BASE + ACC2_OFF + seg + base);
                    tcgen05_ld_32x32bx8(acc1 + 0, addr1 + 0);
                    tcgen05_ld_32x32bx8(acc1 + 8, addr1 + 8);
                    tcgen05_ld_32x32bx8(acc2 + 0, addr2 + 0);
                    tcgen05_ld_32x32bx8(acc2 + 8, addr2 + 8);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");

                    float e[16];
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        e[j] = __expf(-acc1[j]);
                    }

                    half2 h0 = silu_mul_h2(acc1[0],  acc1[1],  acc2[0],  acc2[1]);
                    half2 h1 = silu_mul_h2(acc1[2],  acc1[3],  acc2[2],  acc2[3]);
                    half2 h2 = silu_mul_h2(acc1[4],  acc1[5],  acc2[4],  acc2[5]);
                    half2 h3 = silu_mul_h2(acc1[6],  acc1[7],  acc2[6],  acc2[7]);
                    half2 h4 = silu_mul_h2(acc1[8],  acc1[9],  acc2[8],  acc2[9]);
                    half2 h5 = silu_mul_h2(acc1[10], acc1[11], acc2[10], acc2[11]);
                    half2 h6 = silu_mul_h2(acc1[12], acc1[13], acc2[12], acc2[13]);
                    half2 h7 = silu_mul_h2(acc1[14], acc1[15], acc2[14], acc2[15]);

                    const uint32_t u0 = *reinterpret_cast<uint32_t*>(&h0);
                    const uint32_t u1 = *reinterpret_cast<uint32_t*>(&h1);
                    const uint32_t u2 = *reinterpret_cast<uint32_t*>(&h2);
                    const uint32_t u3 = *reinterpret_cast<uint32_t*>(&h3);
                    const uint32_t u4 = *reinterpret_cast<uint32_t*>(&h4);
                    const uint32_t u5 = *reinterpret_cast<uint32_t*>(&h5);
                    const uint32_t u6 = *reinterpret_cast<uint32_t*>(&h6);
                    const uint32_t u7 = *reinterpret_cast<uint32_t*>(&h7);

                    const unsigned long long q0 = (unsigned long long)u0 | ((unsigned long long)u1 << 32);
                    const unsigned long long q1 = (unsigned long long)u2 | ((unsigned long long)u3 << 32);
                    const unsigned long long q2 = (unsigned long long)u4 | ((unsigned long long)u5 << 32);
                    const unsigned long long q3 = (unsigned long long)u6 | ((unsigned long long)u7 << 32);
                    stg_32b((const void*)(row_ptr + seg + base), q0, q1, q2, q3);
                }
            }
        }
    }

    if (warp_id < 4) {
        asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");
        if (warp_id == 0)
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(TOTAL_TMEM_COLS));
    }
}

// ============================================================================
// Launch wrappers
// ============================================================================

// M=256 launcher with collector kernel
template <int BLOCK_N, int BLOCK_M, int BLOCK_K, int NUM_STAGES>
at::Tensor dual_gemm_launch_collector(
    const at::Tensor& A,
    const at::Tensor& B1,
    const at::Tensor& B2,
    const at::Tensor& SFA,
    const at::Tensor& SFB1,
    const at::Tensor& SFB2,
    at::Tensor& C
) {
    const int M = (int)A.size(0);
    const int N = (int)B1.size(0);
    const int K = (int)A.size(1) * 2;

    auto A_ptr    = reinterpret_cast<const char *>(A.data_ptr());
    auto B1_ptr   = reinterpret_cast<const char *>(B1.data_ptr());
    auto B2_ptr   = reinterpret_cast<const char *>(B2.data_ptr());
    auto SFA_ptr  = reinterpret_cast<const char *>(SFA.data_ptr());
    auto SFB1_ptr = reinterpret_cast<const char *>(SFB1.data_ptr());
    auto SFB2_ptr = reinterpret_cast<const char *>(SFB2.data_ptr());
    auto C_ptr    = reinterpret_cast<half *>(C.data_ptr());

    CUtensorMap A_tmap, B1_tmap, B2_tmap;
    init_AB_tmap(&A_tmap,  A_ptr,  M, K, BLOCK_M,     BLOCK_K);
    init_AB_tmap(&B1_tmap, B1_ptr, N, K, BLOCK_N / 2, BLOCK_K);
    init_AB_tmap(&B2_tmap, B2_ptr, N, K, BLOCK_N / 2, BLOCK_K);

    CUtensorMap SFA_tmap, SFB1_tmap, SFB2_tmap;
    init_SF_tmap(&SFA_tmap,  SFA_ptr,  M, K, BLOCK_K);
    init_SF_tmap(&SFB1_tmap, SFB1_ptr, N, K, BLOCK_K);
    init_SF_tmap(&SFB2_tmap, SFB2_ptr, N, K, BLOCK_K);

    constexpr int tb_size = BLOCK_M + 2 * WARP_SIZE;
    constexpr int A_size_c    = BLOCK_M * BLOCK_K / 2;
    constexpr int B_size_c    = (BLOCK_N / 2) * BLOCK_K / 2;
    constexpr int SFA_size_c  = 128 * BLOCK_K / 16;
    constexpr int SFB_size_c  = 128 * BLOCK_K / 16;
    const int smem_size = (A_size_c + B_size_c + B_size_c + SFA_size_c + SFB_size_c + SFB_size_c) * NUM_STAGES;

    const int grid_m_clusters = M / (BLOCK_M * 2);
    const int grid_n_clusters = N / BLOCK_N;
    const int num_tiles = grid_m_clusters * grid_n_clusters;
    int clusters = num_tiles;
    if (clusters < 1) clusters = 1;
    dim3 grid(clusters * 2, 1, 1);

    auto kernel_fn = dual_gemm_cta2_collector_n64_kernel<BLOCK_M, BLOCK_K, NUM_STAGES>;
    if (smem_size > 48000) cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel_fn<<<grid, tb_size, smem_size>>>(A_tmap, B1_tmap, B2_tmap, SFA_tmap, SFB1_tmap, SFB2_tmap, C_ptr, M, N, K);

    return C;
}

// Non-M=256 launcher with baseline kernel
template <int BLOCK_N, int BLOCK_M, int BLOCK_K, int NUM_STAGES>
at::Tensor dual_gemm_launch_baseline(
    const at::Tensor& A,
    const at::Tensor& B1,
    const at::Tensor& B2,
    const at::Tensor& SFA,
    const at::Tensor& SFB1,
    const at::Tensor& SFB2,
    at::Tensor& C
) {
    const int M = (int)A.size(0);
    const int N = (int)B1.size(0);
    const int K = (int)A.size(1) * 2;

    auto A_ptr    = reinterpret_cast<const char *>(A.data_ptr());
    auto B1_ptr   = reinterpret_cast<const char *>(B1.data_ptr());
    auto B2_ptr   = reinterpret_cast<const char *>(B2.data_ptr());
    auto SFA_ptr  = reinterpret_cast<const char *>(SFA.data_ptr());
    auto SFB1_ptr = reinterpret_cast<const char *>(SFB1.data_ptr());
    auto SFB2_ptr = reinterpret_cast<const char *>(SFB2.data_ptr());
    auto C_ptr    = reinterpret_cast<half *>(C.data_ptr());

    CUtensorMap A_tmap, B1_tmap, B2_tmap;
    init_AB_tmap(&A_tmap,  A_ptr,  M, K, BLOCK_M,     BLOCK_K);
    init_AB_tmap(&B1_tmap, B1_ptr, N, K, BLOCK_N / 2, BLOCK_K);
    init_AB_tmap(&B2_tmap, B2_ptr, N, K, BLOCK_N / 2, BLOCK_K);

    CUtensorMap SFA_tmap, SFB1_tmap, SFB2_tmap;
    init_SF_tmap(&SFA_tmap,  SFA_ptr,  M, K, BLOCK_K);
    init_SF_tmap(&SFB1_tmap, SFB1_ptr, N, K, BLOCK_K);
    init_SF_tmap(&SFB2_tmap, SFB2_ptr, N, K, BLOCK_K);

    constexpr int tb_size = BLOCK_M + 2 * WARP_SIZE;
    constexpr int A_size_c    = BLOCK_M * BLOCK_K / 2;
    constexpr int B_size_c    = (BLOCK_N / 2) * BLOCK_K / 2;
    constexpr int SFA_size_c  = 128 * BLOCK_K / 16;
    constexpr int SFB_size_c  = 128 * BLOCK_K / 16;
    // Add padding for bank conflict mitigation (2 x 128 bytes per stage)
    constexpr int PAD_c = 256;
    const int smem_size = (A_size_c + B_size_c + B_size_c + SFA_size_c + SFB_size_c + SFB_size_c + PAD_c) * NUM_STAGES;

    const int grid_m_clusters = M / (BLOCK_M * 2);
    const int grid_n_clusters = N / BLOCK_N;
    const int num_tiles = grid_m_clusters * grid_n_clusters;
    int clusters = num_tiles;
    if (clusters < 1) clusters = 1;
    dim3 grid(clusters * 2, 1, 1);

    auto kernel_fn = dual_gemm_cta2_baseline_n128_kernel<BLOCK_M, BLOCK_K, NUM_STAGES>;
    if (smem_size > 48000) cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel_fn<<<grid, tb_size, smem_size>>>(A_tmap, B1_tmap, B2_tmap, SFA_tmap, SFB1_tmap, SFB2_tmap, C_ptr, M, N, K);

    return C;
}

at::Tensor dual_gemm(
    const at::Tensor& A,
    const at::Tensor& B1,
    const at::Tensor& B2,
    const at::Tensor& SFA,
    const at::Tensor& SFB1,
    const at::Tensor& SFB2,
    at::Tensor& C
) {
    const int K = (int)A.size(1) * 2;
    const int M = (int)A.size(0);
    const int N = (int)B1.size(0);
    TORCH_CHECK((K % 256) == 0, "Unsupported K: ", K);
    TORCH_CHECK((M % 256) == 0, "Unsupported M: ", M);
    TORCH_CHECK((N % 64) == 0, "Unsupported N: ", N);

    if (M == 256) {
        // Use collector kernel for M=256 (shows small improvement)
        if ((N == 3072 && K == 4096) || (N == 4096 && K == 7168)) {
            return dual_gemm_launch_collector<64, 128, 256, 7>(A, B1, B2, SFA, SFB1, SFB2, C);
        }
        const int num_iters = K / 256;
        const int stages = (num_iters < 7) ? num_iters : 7;
        switch (stages) {
            case 1: return dual_gemm_launch_collector<64, 128, 256, 1>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 2: return dual_gemm_launch_collector<64, 128, 256, 2>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 3: return dual_gemm_launch_collector<64, 128, 256, 3>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 4: return dual_gemm_launch_collector<64, 128, 256, 4>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 5: return dual_gemm_launch_collector<64, 128, 256, 5>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 6: return dual_gemm_launch_collector<64, 128, 256, 6>(A, B1, B2, SFA, SFB1, SFB2, C);
            default: return dual_gemm_launch_collector<64, 128, 256, 7>(A, B1, B2, SFA, SFB1, SFB2, C);
        }
    } else {
        // Use baseline kernel for M=512 and other cases
        TORCH_CHECK((N % 128) == 0, "Unsupported N for N=128 path: ", N);
        if (M == 512 && K == 7168 && (N == 3072 || N == 4096)) {
            return dual_gemm_launch_baseline<128, 128, 256, 5>(A, B1, B2, SFA, SFB1, SFB2, C);
        }
        const int num_iters = K / 256;
        const int stages = (num_iters < 5) ? num_iters : 5;
        switch (stages) {
            case 1: return dual_gemm_launch_baseline<128, 128, 256, 1>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 2: return dual_gemm_launch_baseline<128, 128, 256, 2>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 3: return dual_gemm_launch_baseline<128, 128, 256, 3>(A, B1, B2, SFA, SFB1, SFB2, C);
            case 4: return dual_gemm_launch_baseline<128, 128, 256, 4>(A, B1, B2, SFA, SFB1, SFB2, C);
            default: return dual_gemm_launch_baseline<128, 128, 256, 5>(A, B1, B2, SFA, SFB1, SFB2, C);
        }
    }
}

TORCH_LIBRARY(dual_gemm_final_combined_module, m) {
    m.def("dual_gemm(Tensor A, Tensor B1, Tensor B2, Tensor SFA, Tensor SFB1, Tensor SFB2, Tensor(a!) C) -> Tensor");
    m.impl("dual_gemm", &dual_gemm);
}
"""


_compiled_module = None


def _get_module():
    global _compiled_module
    if _compiled_module is None:
        _compiled_module = load_inline(
            name="dual_gemm_final_combined_cuda",
            cpp_sources="",
            cuda_sources=CUDA_SOURCE,
            functions=None,
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_100a,code=sm_100a",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--relocatable-device-code=false",
            ],
            extra_ldflags=["-lcuda"],
            with_cuda=True,
            verbose=False,
            is_python_module=False,
        )
    return _compiled_module


def custom_kernel(data: input_t) -> output_t:
    a, b1, b2, _, _, _, sfa_permuted, sfb1_permuted, sfb2_permuted, c = data
    _get_module()
    return torch.ops.dual_gemm_final_combined_module.dual_gemm(
        a, b1, b2, sfa_permuted, sfb1_permuted, sfb2_permuted, c
    )
