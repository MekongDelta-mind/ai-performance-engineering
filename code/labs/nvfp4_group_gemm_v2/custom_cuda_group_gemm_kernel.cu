#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/atomic>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/__ptx/instructions/cp_async_bulk_tensor.h>

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

// Tunable compile-time knobs (set via nvcc -D... flags).
#ifndef NVFP4_GROUP_GEMM_V2_PIPELINE_STAGES
#define NVFP4_GROUP_GEMM_V2_PIPELINE_STAGES 2
#endif

#ifndef NVFP4_GROUP_GEMM_V2_TMEM_COLUMNS
#define NVFP4_GROUP_GEMM_V2_TMEM_COLUMNS 512
#endif

namespace {

namespace cg = cooperative_groups;
namespace cptx = cuda::ptx;

__device__ __forceinline__ int8_t fp4_e2m1_to_intq(uint8_t nibble) {
  // Signed symmetric lookup in quantized integer space for E2M1 values.
  // The final FP value is 0.25 * intq (matches the Python LUT of {0,0.5,1,1.5,2,3,4,6,...}).
  constexpr int8_t kLut[16] = {
      0,  1,  2,  3,  //
      4,  6,  8,  12, //
      0,  -1, -2, -3, //
      -4, -6, -8, -12 //
  };
  return kLut[nibble & 0x0F];
}

__host__ __device__ __forceinline__ int ceil_div_int(int a, int b) {
  return (a + b - 1) / b;
}

__host__ __device__ __forceinline__ size_t shared_bytes_for_tile(int block_m, int block_n, int kpack_tile_bytes, int scale_tile) {
  // A/B are packed bytes.
  // Scales are fp16: (block_m + block_n) * scale_tile halfs.
  // Add padding for alignment.
  return static_cast<size_t>(block_m) * static_cast<size_t>(kpack_tile_bytes) +
         static_cast<size_t>(block_n) * static_cast<size_t>(kpack_tile_bytes) +
         static_cast<size_t>(block_m + block_n) * static_cast<size_t>(scale_tile) * sizeof(half) +
         32;
}

__global__ void nvfp4_group_gemm_v2_scalar_kernel(
    const uint64_t* __restrict__ a_ptrs,
    const uint64_t* __restrict__ b_ptrs,
    const uint64_t* __restrict__ sfa_ptrs,
    const uint64_t* __restrict__ sfb_ptrs,
    const uint64_t* __restrict__ c_ptrs,
    const int32_t* __restrict__ m_sizes,
    const int32_t* __restrict__ n_sizes,
    const int32_t* __restrict__ k_halves,
    const int32_t* __restrict__ k_scales,
    int block_m,
    int block_n,
    int kpack_tile_bytes,
    int scale_tile) {
  const int group_idx = static_cast<int>(blockIdx.z);

  const uint8_t* const a_packed = reinterpret_cast<const uint8_t*>(a_ptrs[group_idx]);
  const uint8_t* const b_packed = reinterpret_cast<const uint8_t*>(b_ptrs[group_idx]);
  const half* const sfa_half = reinterpret_cast<const half*>(sfa_ptrs[group_idx]);
  const half* const sfb_half = reinterpret_cast<const half*>(sfb_ptrs[group_idx]);
  half* const c_out = reinterpret_cast<half*>(c_ptrs[group_idx]);

  const int m_size = m_sizes[group_idx];
  const int n_size = n_sizes[group_idx];
  const int k_half = k_halves[group_idx];   // packed bytes == K/2
  const int k_scale = k_scales[group_idx];  // K/16

  const int local_n = static_cast<int>(threadIdx.x);
  const int local_m = static_cast<int>(threadIdx.y);
  const int global_m = static_cast<int>(blockIdx.y) * block_m + local_m;
  const int global_n = static_cast<int>(blockIdx.x) * block_n + local_n;

  if (static_cast<int>(blockIdx.y) * block_m >= m_size || static_cast<int>(blockIdx.x) * block_n >= n_size) {
    return;
  }

  const int linear_tid = local_m * block_n + local_n;
  const int threads = block_m * block_n;

  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;

  uint8_t* const sh_a = reinterpret_cast<uint8_t*>(smem_ptr);
  smem_ptr += static_cast<size_t>(block_m) * static_cast<size_t>(kpack_tile_bytes);

  uint8_t* const sh_b = reinterpret_cast<uint8_t*>(smem_ptr);
  smem_ptr += static_cast<size_t>(block_n) * static_cast<size_t>(kpack_tile_bytes);

  // Align for half.
  uintptr_t aligned_ptr = reinterpret_cast<uintptr_t>(smem_ptr);
  aligned_ptr = (aligned_ptr + alignof(half) - 1U) & ~(static_cast<uintptr_t>(alignof(half) - 1U));
  half* const sh_sa = reinterpret_cast<half*>(aligned_ptr);
  aligned_ptr += static_cast<size_t>(block_m) * static_cast<size_t>(scale_tile) * sizeof(half);
  half* const sh_sb = reinterpret_cast<half*>(aligned_ptr);

  float acc = 0.0f;

  for (int kp_base = 0; kp_base < k_half; kp_base += kpack_tile_bytes) {
    const int valid_kpack = ((k_half - kp_base) < kpack_tile_bytes) ? (k_half - kp_base) : kpack_tile_bytes;
    const int valid_scale = (valid_kpack + 7) >> 3;  // 8 packed bytes == 16 FP4 values

    // Load A packed bytes.
    for (int idx = linear_tid; idx < block_m * kpack_tile_bytes; idx += threads) {
      const int row = idx / kpack_tile_bytes;
      const int kk = idx - row * kpack_tile_bytes;
      const int gm = static_cast<int>(blockIdx.y) * block_m + row;
      const int gk = kp_base + kk;
      uint8_t value = 0;
      if (gm < m_size && kk < valid_kpack) {
        value = a_packed[static_cast<size_t>(gm) * static_cast<size_t>(k_half) + static_cast<size_t>(gk)];
      }
      sh_a[idx] = value;
    }

    // Load B packed bytes.
    for (int idx = linear_tid; idx < block_n * kpack_tile_bytes; idx += threads) {
      const int row = idx / kpack_tile_bytes;
      const int kk = idx - row * kpack_tile_bytes;
      const int gn = static_cast<int>(blockIdx.x) * block_n + row;
      const int gk = kp_base + kk;
      uint8_t value = 0;
      if (gn < n_size && kk < valid_kpack) {
        value = b_packed[static_cast<size_t>(gn) * static_cast<size_t>(k_half) + static_cast<size_t>(gk)];
      }
      sh_b[idx] = value;
    }

    // Load A/B scale factors (FP16, K/16).
    for (int idx = linear_tid; idx < block_m * scale_tile; idx += threads) {
      const int row = idx / scale_tile;
      const int s = idx - row * scale_tile;
      const int gm = static_cast<int>(blockIdx.y) * block_m + row;
      const int gs = (kp_base >> 3) + s;  // kp_base bytes -> (kp_base*2)/16 == kp_base/8
      half value = __float2half_rn(0.0f);
      if (gm < m_size && s < valid_scale && gs < k_scale) {
        value = sfa_half[static_cast<size_t>(gm) * static_cast<size_t>(k_scale) + static_cast<size_t>(gs)];
      }
      sh_sa[idx] = value;
    }

    for (int idx = linear_tid; idx < block_n * scale_tile; idx += threads) {
      const int row = idx / scale_tile;
      const int s = idx - row * scale_tile;
      const int gn = static_cast<int>(blockIdx.x) * block_n + row;
      const int gs = (kp_base >> 3) + s;
      half value = __float2half_rn(0.0f);
      if (gn < n_size && s < valid_scale && gs < k_scale) {
        value = sfb_half[static_cast<size_t>(gn) * static_cast<size_t>(k_scale) + static_cast<size_t>(gs)];
      }
      sh_sb[idx] = value;
    }

    __syncthreads();

    if (global_m < m_size && global_n < n_size) {
      const uint8_t* const a_row = sh_a + static_cast<size_t>(local_m) * static_cast<size_t>(kpack_tile_bytes);
      const uint8_t* const b_row = sh_b + static_cast<size_t>(local_n) * static_cast<size_t>(kpack_tile_bytes);
      const half* const sa_row = sh_sa + static_cast<size_t>(local_m) * static_cast<size_t>(scale_tile);
      const half* const sb_row = sh_sb + static_cast<size_t>(local_n) * static_cast<size_t>(scale_tile);

#pragma unroll
      for (int s = 0; s < 16; ++s) {
        if (s >= valid_scale) {
          break;
        }
        const float scale = __half2float(sa_row[s]) * __half2float(sb_row[s]);
        const int kk_base = s << 3;  // 8 packed bytes -> 16 values
        const int remaining = valid_kpack - kk_base;
        const int active = remaining > 8 ? 8 : remaining;

#pragma unroll
        for (int t = 0; t < 8; ++t) {
          if (t >= active) {
            break;
          }
          const uint8_t pa = a_row[kk_base + t];
          const uint8_t pb = b_row[kk_base + t];

          const int a0 = static_cast<int>(fp4_e2m1_to_intq(static_cast<uint8_t>(pa & 0x0F)));
          const int a1 = static_cast<int>(fp4_e2m1_to_intq(static_cast<uint8_t>((pa >> 4) & 0x0F)));
          const int b0 = static_cast<int>(fp4_e2m1_to_intq(static_cast<uint8_t>(pb & 0x0F)));
          const int b1 = static_cast<int>(fp4_e2m1_to_intq(static_cast<uint8_t>((pb >> 4) & 0x0F)));

          acc += scale * (0.25f * static_cast<float>((a0 * b0) + (a1 * b1)));
        }
      }
    }

    __syncthreads();
  }

  if (global_m < m_size && global_n < n_size) {
    c_out[static_cast<size_t>(global_m) * static_cast<size_t>(n_size) + static_cast<size_t>(global_n)] =
        __float2half_rn(acc);
  }
}

}  // namespace

void nvfp4_group_gemm_v2_forward_grouped_cuda(
    torch::Tensor a_ptrs,
    torch::Tensor b_ptrs,
    torch::Tensor sfa_ptrs,
    torch::Tensor sfb_ptrs,
    torch::Tensor c_ptrs,
    torch::Tensor m_sizes,
    torch::Tensor n_sizes,
    torch::Tensor k_halves,
    torch::Tensor k_scales,
    int max_m_size,
    int max_n_size,
    int block_m,
    int block_n,
    int kpack_tile_bytes) {
  TORCH_CHECK(a_ptrs.is_cuda(), "a_ptrs must be CUDA tensor");
  TORCH_CHECK(b_ptrs.is_cuda(), "b_ptrs must be CUDA tensor");
  TORCH_CHECK(sfa_ptrs.is_cuda(), "sfa_ptrs must be CUDA tensor");
  TORCH_CHECK(sfb_ptrs.is_cuda(), "sfb_ptrs must be CUDA tensor");
  TORCH_CHECK(c_ptrs.is_cuda(), "c_ptrs must be CUDA tensor");
  TORCH_CHECK(m_sizes.is_cuda(), "m_sizes must be CUDA tensor");
  TORCH_CHECK(n_sizes.is_cuda(), "n_sizes must be CUDA tensor");
  TORCH_CHECK(k_halves.is_cuda(), "k_halves must be CUDA tensor");
  TORCH_CHECK(k_scales.is_cuda(), "k_scales must be CUDA tensor");

  TORCH_CHECK(a_ptrs.scalar_type() == torch::kInt64, "a_ptrs must be torch.int64");
  TORCH_CHECK(b_ptrs.scalar_type() == torch::kInt64, "b_ptrs must be torch.int64");
  TORCH_CHECK(sfa_ptrs.scalar_type() == torch::kInt64, "sfa_ptrs must be torch.int64");
  TORCH_CHECK(sfb_ptrs.scalar_type() == torch::kInt64, "sfb_ptrs must be torch.int64");
  TORCH_CHECK(c_ptrs.scalar_type() == torch::kInt64, "c_ptrs must be torch.int64");
  TORCH_CHECK(m_sizes.scalar_type() == torch::kInt, "m_sizes must be torch.int32");
  TORCH_CHECK(n_sizes.scalar_type() == torch::kInt, "n_sizes must be torch.int32");
  TORCH_CHECK(k_halves.scalar_type() == torch::kInt, "k_halves must be torch.int32");
  TORCH_CHECK(k_scales.scalar_type() == torch::kInt, "k_scales must be torch.int32");

  TORCH_CHECK(a_ptrs.dim() == 1, "a_ptrs must be 1D");
  TORCH_CHECK(b_ptrs.dim() == 1, "b_ptrs must be 1D");
  TORCH_CHECK(sfa_ptrs.dim() == 1, "sfa_ptrs must be 1D");
  TORCH_CHECK(sfb_ptrs.dim() == 1, "sfb_ptrs must be 1D");
  TORCH_CHECK(c_ptrs.dim() == 1, "c_ptrs must be 1D");
  TORCH_CHECK(m_sizes.dim() == 1, "m_sizes must be 1D");
  TORCH_CHECK(n_sizes.dim() == 1, "n_sizes must be 1D");
  TORCH_CHECK(k_halves.dim() == 1, "k_halves must be 1D");
  TORCH_CHECK(k_scales.dim() == 1, "k_scales must be 1D");

  TORCH_CHECK(a_ptrs.is_contiguous(), "a_ptrs must be contiguous");
  TORCH_CHECK(b_ptrs.is_contiguous(), "b_ptrs must be contiguous");
  TORCH_CHECK(sfa_ptrs.is_contiguous(), "sfa_ptrs must be contiguous");
  TORCH_CHECK(sfb_ptrs.is_contiguous(), "sfb_ptrs must be contiguous");
  TORCH_CHECK(c_ptrs.is_contiguous(), "c_ptrs must be contiguous");
  TORCH_CHECK(m_sizes.is_contiguous(), "m_sizes must be contiguous");
  TORCH_CHECK(n_sizes.is_contiguous(), "n_sizes must be contiguous");
  TORCH_CHECK(k_halves.is_contiguous(), "k_halves must be contiguous");
  TORCH_CHECK(k_scales.is_contiguous(), "k_scales must be contiguous");

  const int groups = static_cast<int>(m_sizes.numel());
  TORCH_CHECK(groups > 0, "grouped call requires at least one group");
  TORCH_CHECK(b_ptrs.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
  TORCH_CHECK(sfa_ptrs.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
  TORCH_CHECK(sfb_ptrs.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
  TORCH_CHECK(c_ptrs.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
  TORCH_CHECK(n_sizes.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
  TORCH_CHECK(k_halves.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
  TORCH_CHECK(k_scales.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");

  TORCH_CHECK(max_m_size > 0, "max_m_size must be > 0");
  TORCH_CHECK(max_n_size > 0, "max_n_size must be > 0");
  TORCH_CHECK(block_m > 0 && block_n > 0, "block sizes must be > 0");
  TORCH_CHECK(block_m * block_n <= 1024, "block_m * block_n must be <= 1024");
  TORCH_CHECK(kpack_tile_bytes > 0, "kpack_tile_bytes must be > 0");
  TORCH_CHECK((kpack_tile_bytes % 8) == 0, "kpack_tile_bytes must be divisible by 8");

  const int scale_tile = (kpack_tile_bytes + 7) >> 3;

  const dim3 block(static_cast<unsigned int>(block_n), static_cast<unsigned int>(block_m), 1);
  const dim3 grid(static_cast<unsigned int>(ceil_div_int(max_n_size, block_n)),
                  static_cast<unsigned int>(ceil_div_int(max_m_size, block_m)),
                  static_cast<unsigned int>(groups));

  const size_t shared_bytes = shared_bytes_for_tile(block_m, block_n, kpack_tile_bytes, scale_tile);

  nvfp4_group_gemm_v2_scalar_kernel<<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint64_t*>(a_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<const uint64_t*>(b_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<const uint64_t*>(sfa_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<const uint64_t*>(sfb_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<const uint64_t*>(c_ptrs.data_ptr<int64_t>()),
      m_sizes.data_ptr<int32_t>(),
      n_sizes.data_ptr<int32_t>(),
      k_halves.data_ptr<int32_t>(),
      k_scales.data_ptr<int32_t>(),
      block_m,
      block_n,
      kpack_tile_bytes,
      scale_tile);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

// -----------------------------------------------------------------------------
// Minimal UMMA descriptor structs (transliterated from CUTLASS/CuTe headers).
// -----------------------------------------------------------------------------

namespace umma {

enum class Major : uint8_t {
  K = 0,
  MN = 1,
};

enum class LayoutType : uint8_t {
  SWIZZLE_NONE = 0,
  SWIZZLE_128B_BASE32B = 1,
  SWIZZLE_128B = 2,
  SWIZZLE_64B = 4,
  SWIZZLE_32B = 6,
};

union SmemDescriptor {
  uint64_t desc_ = 0;
  struct {
    uint16_t start_address_ : 14, : 2;
    uint16_t leading_byte_offset_ : 14, : 2;
    uint16_t stride_byte_offset_ : 14, version_ : 2;
    uint8_t : 1, base_offset_ : 3, lbo_mode_ : 1, : 3;
    uint8_t : 5, layout_type_ : 3;
  };
  struct {
    uint32_t lo;
    uint32_t hi;
  };
  __host__ __device__ constexpr operator uint64_t() const noexcept { return desc_; }
};

union InstrDescriptorBlockScaled {
  uint32_t desc_ = 0;
  struct {
    uint16_t sparse_id2_ : 2,
             sparse_flag_ : 1,
             : 1,
             b_sf_id_ : 2,
             : 1,
             a_format_ : 3,
             b_format_ : 3,
             a_negate_ : 1,
             b_negate_ : 1,
             a_major_ : 1;
    uint16_t b_major_ : 1,
             n_dim_ : 6,
             scale_format_ : 1,
             m_dim_ : 5,
             a_sf_id_ : 2,
             k_size_ : 1;
  };
  __host__ __device__ constexpr operator uint32_t() const noexcept { return desc_; }
};

__device__ __forceinline__ uint64_t make_runtime_instr_desc_block_scaled(
    InstrDescriptorBlockScaled desc_i, uint32_t tmem_sfa_addr, uint32_t tmem_sfb_addr) {
  // The first 2-bits of TMEM address includes byte address.
  desc_i.a_sf_id_ = (tmem_sfa_addr & 0xC0000000u) >> 30;
  desc_i.b_sf_id_ = (tmem_sfb_addr & 0xC0000000u) >> 30;
  // Upper 32b contains the instruction descriptor. Lower 32b unused for dense MMA.
  return (static_cast<uint64_t>(static_cast<uint32_t>(desc_i)) << 32);
}

}  // namespace umma

// -----------------------------------------------------------------------------
// tcgen05 PTX helpers (transliterated from CUTLASS/CuTe headers).
// -----------------------------------------------------------------------------

namespace tcgen05 {

__device__ __forceinline__ uint32_t cast_smem_ptr_to_uint(const void* ptr) {
  // Prefer NVCC builtin to avoid PTX inline-asm operand width issues on SM100.
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ bool elect_one_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const unsigned mask = __activemask();
  const int leader = __ffs(static_cast<int>(mask)) - 1;
  return ((static_cast<int>(threadIdx.x) & 31) == leader);
#else
  return (threadIdx.x == 0);
#endif
}

__device__ __forceinline__ uint32_t block_rank_in_cluster() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t rank = 0;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank) :);
  return rank;
#else
  return 0u;
#endif
}

__device__ __forceinline__ void tmem_alloc_cta1(uint32_t* dst_ptr_smem, int num_columns) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t dst_intptr = cast_smem_ptr_to_uint(dst_ptr_smem);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               :
               : "r"(dst_intptr), "r"(num_columns));
#else
  (void)dst_ptr_smem;
  (void)num_columns;
#endif
}

__device__ __forceinline__ void tmem_alloc_cta2(uint32_t* dst_ptr_smem, int num_columns) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t dst_intptr = cast_smem_ptr_to_uint(dst_ptr_smem);
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
               :
               : "r"(dst_intptr), "r"(num_columns));
#else
  (void)dst_ptr_smem;
  (void)num_columns;
#endif
}

__device__ __forceinline__ void tmem_dealloc_cta1(uint32_t tmem_ptr, int num_columns) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("{\n\t"
               "tcgen05.dealloc.cta_group::1.sync.aligned.b32  %0, %1; \n\t"
               "}"
               :
               : "r"(tmem_ptr), "r"(num_columns));
#else
  (void)tmem_ptr;
  (void)num_columns;
#endif
}

__device__ __forceinline__ void tmem_dealloc_cta2(uint32_t tmem_ptr, int num_columns) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("{\n\t"
               "tcgen05.dealloc.cta_group::2.sync.aligned.b32  %0, %1; \n\t"
               "}"
               :
               : "r"(tmem_ptr), "r"(num_columns));
#else
  (void)tmem_ptr;
  (void)num_columns;
#endif
}

__device__ __forceinline__ void tmem_relinquish_alloc_permit_cta1() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::);
#endif
}

__device__ __forceinline__ void tmem_relinquish_alloc_permit_cta2() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::);
#endif
}

__device__ __forceinline__ void utccp_cp_cta1_128x128b(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc));
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void utccp_cp_cta1_32x128b_warpx4(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc));
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void utccp_cp_cta1_64x128b_warpx2_02_13(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc) : "memory");
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void utccp_cp_cta1_64x128b_warpx2_01_23(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::1.64x128b.warpx2::01_23 [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc) : "memory");
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void utccp_cp_cta2_32x128b_warpx4(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc));
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void utccp_cp_cta2_64x128b_warpx2_02_13(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc));
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void utccp_cp_cta2_64x128b_warpx2_01_23(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc));
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void mma_cta1_mxf4nvf4_block16(
    uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c, uint32_t accumulate, uint32_t idesc_hi, uint32_t tsfa_addr,
    uint32_t tsfb_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "setp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
               "}\n"
               :
               : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc_hi), "r"(accumulate), "r"(tsfa_addr),
                 "r"(tsfb_addr));
#else
  (void)desc_a;
  (void)desc_b;
  (void)tmem_c;
  (void)accumulate;
  (void)idesc_hi;
  (void)tsfa_addr;
  (void)tsfb_addr;
#endif
}

__device__ __forceinline__ void mma_cta2_mxf4nvf4_block16(
    uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c, uint32_t accumulate, uint32_t idesc_hi, uint32_t tsfa_addr,
    uint32_t tsfb_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "setp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
               "}\n"
               :
               : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc_hi), "r"(accumulate), "r"(tsfa_addr),
                 "r"(tsfb_addr));
#else
  (void)desc_a;
  (void)desc_b;
  (void)tmem_c;
  (void)accumulate;
  (void)idesc_hi;
  (void)tsfa_addr;
  (void)tsfb_addr;
#endif
}

template <int CtaGroup>
__device__ __forceinline__ void tmem_alloc(uint32_t* dst_ptr_smem, int num_columns) {
  if constexpr (CtaGroup == 2) {
    tmem_alloc_cta2(dst_ptr_smem, num_columns);
  } else {
    tmem_alloc_cta1(dst_ptr_smem, num_columns);
  }
}

template <int CtaGroup>
__device__ __forceinline__ void tmem_dealloc(uint32_t tmem_ptr, int num_columns) {
  if constexpr (CtaGroup == 2) {
    tmem_dealloc_cta2(tmem_ptr, num_columns);
  } else {
    tmem_dealloc_cta1(tmem_ptr, num_columns);
  }
}

template <int CtaGroup>
__device__ __forceinline__ void tmem_relinquish_alloc_permit() {
  if constexpr (CtaGroup == 2) {
    tmem_relinquish_alloc_permit_cta2();
  } else {
    tmem_relinquish_alloc_permit_cta1();
  }
}

template <int CtaGroup>
__device__ __forceinline__ void utccp_cp_32x128b_warpx4(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
  if constexpr (CtaGroup == 2) {
    utccp_cp_cta2_32x128b_warpx4(src_smem_desc, dst_tmem_addr);
  } else {
    utccp_cp_cta1_32x128b_warpx4(src_smem_desc, dst_tmem_addr);
  }
}

template <int CtaGroup>
__device__ __forceinline__ void utccp_cp_64x128b_warpx2_02_13(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
  if constexpr (CtaGroup == 2) {
    utccp_cp_cta2_64x128b_warpx2_02_13(src_smem_desc, dst_tmem_addr);
  } else {
    utccp_cp_cta1_64x128b_warpx2_02_13(src_smem_desc, dst_tmem_addr);
  }
}

template <int CtaGroup>
__device__ __forceinline__ void utccp_cp_64x128b_warpx2_01_23(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
  if constexpr (CtaGroup == 2) {
    utccp_cp_cta2_64x128b_warpx2_01_23(src_smem_desc, dst_tmem_addr);
  } else {
    utccp_cp_cta1_64x128b_warpx2_01_23(src_smem_desc, dst_tmem_addr);
  }
}

template <int CtaGroup>
__device__ __forceinline__ void mma_mxf4nvf4_block16(
    uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c, uint32_t accumulate, uint32_t idesc_hi, uint32_t tsfa_addr,
    uint32_t tsfb_addr) {
  if constexpr (CtaGroup == 2) {
    mma_cta2_mxf4nvf4_block16(desc_a, desc_b, tmem_c, accumulate, idesc_hi, tsfa_addr, tsfb_addr);
  } else {
    mma_cta1_mxf4nvf4_block16(desc_a, desc_b, tmem_c, accumulate, idesc_hi, tsfa_addr, tsfb_addr);
  }
}

__device__ __forceinline__ uint32_t tmem_ld_32dp32b_x1_pack16b(uint32_t src_addr) {
  uint32_t dst0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32"
               "{%0},"
               "[%1];\n"
               : "=r"(dst0)
               : "r"(src_addr));
#else
  dst0 = 0;
  (void)src_addr;
#endif
  return dst0;
}

__device__ __forceinline__ void tmem_ld_32dp32b_x8(
    uint32_t src_addr, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3, uint32_t& dst4, uint32_t& dst5,
    uint32_t& dst6, uint32_t& dst7) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32"
               "{%0, %1, %2, %3, %4, %5, %6, %7},"
               "[%8];\n"
               : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3), "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
               : "r"(src_addr)
               : "memory");
#else
  dst0 = 0;
  dst1 = 0;
  dst2 = 0;
  dst3 = 0;
  dst4 = 0;
  dst5 = 0;
  dst6 = 0;
  dst7 = 0;
  (void)src_addr;
#endif
}

__device__ __forceinline__ void tmem_ld_32dp32b_x8_pack16b(
    uint32_t src_addr, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3, uint32_t& dst4, uint32_t& dst5,
    uint32_t& dst6, uint32_t& dst7) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32"
               "{%0, %1, %2, %3, %4, %5, %6, %7},"
               "[%8];\n"
               : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3), "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
               : "r"(src_addr)
               : "memory");
#else
  dst0 = 0;
  dst1 = 0;
  dst2 = 0;
  dst3 = 0;
  dst4 = 0;
  dst5 = 0;
  dst6 = 0;
  dst7 = 0;
  (void)src_addr;
#endif
}

__device__ __forceinline__ uint32_t tmem_ld_32dp32b_x1(uint32_t src_addr) {
  uint32_t dst0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32"
               "{%0},"
               "[%1];\n"
               : "=r"(dst0)
               : "r"(src_addr));
#else
  dst0 = 0;
  (void)src_addr;
#endif
  return dst0;
}

__device__ __forceinline__ void tmem_wait_ld_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.wait::ld.sync.aligned;" : : : "memory");
#endif
}

__device__ __forceinline__ void tmem_wait_st_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.wait::st.sync.aligned;" : : : "memory");
#endif
}

// TMEM pointers are encoded as {col:16, dp:8, idx:8}. When forming derived addresses,
// preserve the high idx bits from the allocated base pointer rather than assuming idx=0.
__device__ __forceinline__ uint32_t tmem_addr_add(uint32_t base, uint32_t dp_add, uint32_t col_add) {
  const uint32_t base_col = base & 0x0000FFFFu;
  const uint32_t base_dp = (base >> 16) & 0x000000FFu;
  const uint32_t base_idx = base & 0xFF000000u;
  const uint32_t col = base_col + col_add;
  const uint32_t dp = base_dp + dp_add;
  return base_idx | (dp << 16) | (col & 0x0000FFFFu);
}

}  // namespace tcgen05

// -----------------------------------------------------------------------------
// TMA descriptor encoding (host-side). Stored as 16x int64 per descriptor.
// -----------------------------------------------------------------------------

using EncodeFn = PFN_cuTensorMapEncodeTiled_v12000;

static EncodeFn load_cuTensorMapEncodeTiled() {
  void* func_ptr = nullptr;
  cudaDriverEntryPointQueryResult query_result{};

  cudaError_t err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &func_ptr, 13000, cudaEnableDefault,
                                                     &query_result);
  if (err != cudaSuccess || query_result != cudaDriverEntryPointSuccess) {
    err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &func_ptr, 12000, cudaEnableDefault, &query_result);
  }
  if (err != cudaSuccess || query_result != cudaDriverEntryPointSuccess || func_ptr == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<EncodeFn>(func_ptr);
}

static void encode_2d_u8_tensor_map_or_throw(
    CUtensorMap* out_desc, EncodeFn encode, void* base, uint64_t width, uint64_t height, uint64_t ld_bytes,
    uint32_t box_width, uint32_t box_height, CUtensorMapSwizzle swizzle_mode) {
  constexpr uint32_t rank = 2;
  uint64_t dims[rank] = {width, height};
  uint64_t stride[rank - 1] = {ld_bytes};
  uint32_t box[rank] = {box_width, box_height};
  uint32_t elem_stride[rank] = {1, 1};

  constexpr auto interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  constexpr auto promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
  constexpr auto oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  auto fn = encode ? encode : cuTensorMapEncodeTiled;
  CUresult res = fn(out_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, base, dims, stride, box, elem_stride, interleave,
                    swizzle_mode, promotion, oob_fill);
  if (res != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    const char* err_name = nullptr;
    cuGetErrorString(res, &err_str);
    cuGetErrorName(res, &err_name);
    TORCH_CHECK(false, "cuTensorMapEncodeTiled failed: ", (err_str ? err_str : "unknown"), " (",
                (err_name ? err_name : "unknown"), ", ", static_cast<int>(res), ")");
  }
}

static std::pair<torch::Tensor, torch::Tensor> build_ab_tma_descs_cuda(
    torch::Tensor a_ptrs_cpu, torch::Tensor b_ptrs_cpu, torch::Tensor m_sizes_cpu, torch::Tensor n_sizes_cpu,
    torch::Tensor k_halves_cpu, int64_t b_box_height_rows) {
  TORCH_CHECK(!a_ptrs_cpu.is_cuda(), "a_ptrs_cpu must be CPU tensor");
  TORCH_CHECK(!b_ptrs_cpu.is_cuda(), "b_ptrs_cpu must be CPU tensor");
  TORCH_CHECK(!m_sizes_cpu.is_cuda(), "m_sizes_cpu must be CPU tensor");
  TORCH_CHECK(!n_sizes_cpu.is_cuda(), "n_sizes_cpu must be CPU tensor");
  TORCH_CHECK(!k_halves_cpu.is_cuda(), "k_halves_cpu must be CPU tensor");

  TORCH_CHECK(a_ptrs_cpu.scalar_type() == torch::kInt64, "a_ptrs_cpu must be torch.int64");
  TORCH_CHECK(b_ptrs_cpu.scalar_type() == torch::kInt64, "b_ptrs_cpu must be torch.int64");
  TORCH_CHECK(m_sizes_cpu.scalar_type() == torch::kInt, "m_sizes_cpu must be torch.int32");
  TORCH_CHECK(n_sizes_cpu.scalar_type() == torch::kInt, "n_sizes_cpu must be torch.int32");
  TORCH_CHECK(k_halves_cpu.scalar_type() == torch::kInt, "k_halves_cpu must be torch.int32");

  TORCH_CHECK(b_box_height_rows == 64 || b_box_height_rows == 128 || b_box_height_rows == 256,
              "b_box_height_rows must be 64, 128, or 256. Got b_box_height_rows=", b_box_height_rows);
  const uint32_t b_box_height = static_cast<uint32_t>(b_box_height_rows);

  const int64_t groups = m_sizes_cpu.numel();
  TORCH_CHECK(groups > 0, "groups must be > 0");
  TORCH_CHECK(a_ptrs_cpu.numel() == groups, "a_ptrs_cpu length mismatch");
  TORCH_CHECK(b_ptrs_cpu.numel() == groups, "b_ptrs_cpu length mismatch");
  TORCH_CHECK(n_sizes_cpu.numel() == groups, "n_sizes_cpu length mismatch");
  TORCH_CHECK(k_halves_cpu.numel() == groups, "k_halves_cpu length mismatch");

  auto encode = load_cuTensorMapEncodeTiled();
  TORCH_CHECK(encode != nullptr, "cuTensorMapEncodeTiled unavailable on this runtime");

  static_assert(sizeof(CUtensorMap) == 128, "Unexpected CUtensorMap size");

  std::vector<CUtensorMap> a_descs_host(static_cast<size_t>(groups));
  std::vector<CUtensorMap> b_descs_host(static_cast<size_t>(groups));

  auto* a_ptrs = a_ptrs_cpu.data_ptr<int64_t>();
  auto* b_ptrs = b_ptrs_cpu.data_ptr<int64_t>();
  auto* ms = m_sizes_cpu.data_ptr<int32_t>();
  auto* ns = n_sizes_cpu.data_ptr<int32_t>();
  auto* ks = k_halves_cpu.data_ptr<int32_t>();

  for (int64_t i = 0; i < groups; ++i) {
    const int32_t m = ms[i];
    const int32_t n = ns[i];
    const int32_t k_bytes = ks[i];
    TORCH_CHECK(m > 0 && n > 0 && k_bytes > 0, "Invalid sizes for group ", i);
    // The caller passes explicit padded heights for TMA encoding (setup-time). Do not re-pad
    // here, otherwise the tensormap height can exceed the backing allocation (and TMA reads
    // become out-of-bounds, which is undefined).
    const int32_t m_padded = m;
    const int32_t n_padded = n;

    encode_2d_u8_tensor_map_or_throw(&a_descs_host[static_cast<size_t>(i)], encode,
                                     reinterpret_cast<void*>(a_ptrs[i]), static_cast<uint64_t>(k_bytes),
                                     static_cast<uint64_t>(m_padded), static_cast<uint64_t>(k_bytes),
                                     /*box_width=*/128, /*box_height=*/128, CU_TENSOR_MAP_SWIZZLE_128B);

    encode_2d_u8_tensor_map_or_throw(&b_descs_host[static_cast<size_t>(i)], encode,
                                     reinterpret_cast<void*>(b_ptrs[i]), static_cast<uint64_t>(k_bytes),
                                     static_cast<uint64_t>(n_padded), static_cast<uint64_t>(k_bytes),
                                     /*box_width=*/128, /*box_height=*/b_box_height, CU_TENSOR_MAP_SWIZZLE_128B);
  }

  auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  torch::Tensor a_descs = torch::empty({groups, 16}, opts);
  torch::Tensor b_descs = torch::empty({groups, 16}, opts);

  cudaError_t err_a =
      cudaMemcpy(a_descs.data_ptr(), a_descs_host.data(), static_cast<size_t>(groups) * sizeof(CUtensorMap),
                 cudaMemcpyHostToDevice);
  TORCH_CHECK(err_a == cudaSuccess, "cudaMemcpy a_descs failed: ", cudaGetErrorString(err_a));
  cudaError_t err_b =
      cudaMemcpy(b_descs.data_ptr(), b_descs_host.data(), static_cast<size_t>(groups) * sizeof(CUtensorMap),
                 cudaMemcpyHostToDevice);
  TORCH_CHECK(err_b == cudaSuccess, "cudaMemcpy b_descs failed: ", cudaGetErrorString(err_b));

  return {a_descs, b_descs};
}

static std::pair<torch::Tensor, torch::Tensor> build_scale_tma_descs_cuda(
    torch::Tensor sfa_ptrs_cpu, torch::Tensor sfb_ptrs_cpu, torch::Tensor m_sizes_cpu, torch::Tensor n_sizes_cpu,
    torch::Tensor k_halves_cpu, int64_t sfb_box_height_rows) {
  TORCH_CHECK(!sfa_ptrs_cpu.is_cuda(), "sfa_ptrs_cpu must be CPU tensor");
  TORCH_CHECK(!sfb_ptrs_cpu.is_cuda(), "sfb_ptrs_cpu must be CPU tensor");
  TORCH_CHECK(!m_sizes_cpu.is_cuda(), "m_sizes_cpu must be CPU tensor");
  TORCH_CHECK(!n_sizes_cpu.is_cuda(), "n_sizes_cpu must be CPU tensor");
  TORCH_CHECK(!k_halves_cpu.is_cuda(), "k_halves_cpu must be CPU tensor");

  TORCH_CHECK(sfa_ptrs_cpu.scalar_type() == torch::kInt64, "sfa_ptrs_cpu must be torch.int64");
  TORCH_CHECK(sfb_ptrs_cpu.scalar_type() == torch::kInt64, "sfb_ptrs_cpu must be torch.int64");
  TORCH_CHECK(m_sizes_cpu.scalar_type() == torch::kInt, "m_sizes_cpu must be torch.int32");
  TORCH_CHECK(n_sizes_cpu.scalar_type() == torch::kInt, "n_sizes_cpu must be torch.int32");
  TORCH_CHECK(k_halves_cpu.scalar_type() == torch::kInt, "k_halves_cpu must be torch.int32");

  TORCH_CHECK(sfb_box_height_rows == 128 || sfb_box_height_rows == 256,
              "sfb_box_height_rows must be 128 or 256. Got sfb_box_height_rows=", sfb_box_height_rows);
  const uint32_t sfb_box_height = static_cast<uint32_t>(sfb_box_height_rows);

  const int64_t groups = m_sizes_cpu.numel();
  TORCH_CHECK(groups > 0, "groups must be > 0");
  TORCH_CHECK(sfa_ptrs_cpu.numel() == groups, "sfa_ptrs_cpu length mismatch");
  TORCH_CHECK(sfb_ptrs_cpu.numel() == groups, "sfb_ptrs_cpu length mismatch");
  TORCH_CHECK(n_sizes_cpu.numel() == groups, "n_sizes_cpu length mismatch");
  TORCH_CHECK(k_halves_cpu.numel() == groups, "k_halves_cpu length mismatch");

  auto encode = load_cuTensorMapEncodeTiled();
  TORCH_CHECK(encode != nullptr, "cuTensorMapEncodeTiled unavailable on this runtime");

  static_assert(sizeof(CUtensorMap) == 128, "Unexpected CUtensorMap size");

  std::vector<CUtensorMap> sfa_descs_host(static_cast<size_t>(groups));
  std::vector<CUtensorMap> sfb_descs_host(static_cast<size_t>(groups));

  auto* sfa_ptrs = sfa_ptrs_cpu.data_ptr<int64_t>();
  auto* sfb_ptrs = sfb_ptrs_cpu.data_ptr<int64_t>();
  auto* ms = m_sizes_cpu.data_ptr<int32_t>();
  auto* ns = n_sizes_cpu.data_ptr<int32_t>();
  auto* ks = k_halves_cpu.data_ptr<int32_t>();

  for (int64_t i = 0; i < groups; ++i) {
    const int32_t m = ms[i];
    const int32_t n = ns[i];
    const int32_t k_bytes = ks[i];
    TORCH_CHECK(m > 0 && n > 0 && k_bytes > 0, "Invalid sizes for group ", i);

    constexpr int kTileBytes = 128;
    constexpr int sfaRowsPerTile = 128;
    constexpr int sfbRowsPerTile = 128;
    constexpr int sfRowBytes = 16;

    const int32_t k_tiles = ceil_div_int(k_bytes, kTileBytes);
    const int32_t m_tiles = ceil_div_int(m, 128);
    const int32_t n_tiles = ceil_div_int(n, 128);

    const int64_t sfa_height = static_cast<int64_t>(m_tiles) * static_cast<int64_t>(k_tiles) * sfaRowsPerTile;
    const int64_t sfb_height = static_cast<int64_t>(n_tiles) * static_cast<int64_t>(k_tiles) * sfbRowsPerTile;

    encode_2d_u8_tensor_map_or_throw(&sfa_descs_host[static_cast<size_t>(i)], encode,
                                     reinterpret_cast<void*>(sfa_ptrs[i]),
                                     static_cast<uint64_t>(sfRowBytes),
                                     static_cast<uint64_t>(sfa_height),
                                     static_cast<uint64_t>(sfRowBytes),
                                     /*box_width=*/sfRowBytes,
                                     /*box_height=*/sfaRowsPerTile,
                                     CU_TENSOR_MAP_SWIZZLE_NONE);

    encode_2d_u8_tensor_map_or_throw(&sfb_descs_host[static_cast<size_t>(i)], encode,
                                     reinterpret_cast<void*>(sfb_ptrs[i]),
                                     static_cast<uint64_t>(sfRowBytes),
                                     static_cast<uint64_t>(sfb_height),
                                     static_cast<uint64_t>(sfRowBytes),
                                     /*box_width=*/sfRowBytes,
                                     /*box_height=*/sfb_box_height,
                                     CU_TENSOR_MAP_SWIZZLE_NONE);
  }

  auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  torch::Tensor sfa_descs = torch::empty({groups, 16}, opts);
  torch::Tensor sfb_descs = torch::empty({groups, 16}, opts);

  cudaError_t err_sfa =
      cudaMemcpy(sfa_descs.data_ptr(), sfa_descs_host.data(), static_cast<size_t>(groups) * sizeof(CUtensorMap),
                 cudaMemcpyHostToDevice);
  TORCH_CHECK(err_sfa == cudaSuccess, "cudaMemcpy sfa_descs failed: ", cudaGetErrorString(err_sfa));
  cudaError_t err_sfb =
      cudaMemcpy(sfb_descs.data_ptr(), sfb_descs_host.data(), static_cast<size_t>(groups) * sizeof(CUtensorMap),
                 cudaMemcpyHostToDevice);
  TORCH_CHECK(err_sfb == cudaSuccess, "cudaMemcpy sfb_descs failed: ", cudaGetErrorString(err_sfb));

  return {sfa_descs, sfb_descs};
}

// -----------------------------------------------------------------------------
// tcgen05 kernel (correctness-first; single-stage). CtaGroup is 1 or 2.
// -----------------------------------------------------------------------------

namespace cuda_device = cuda::device::experimental;

template <int CtaGroup, int UnrollN>
__global__ void nvfp4_group_gemm_v2_tcgen05_kernel(
    const uint64_t* __restrict__ a_ptrs,
    const uint64_t* __restrict__ b_ptrs,
    const uint64_t* __restrict__ sfa_ptrs,  // points to packed [m_tiles,k_tiles,128,16] uint8 scales
    const uint64_t* __restrict__ sfb_ptrs,  // points to packed [n_tiles,k_tiles,128,16] uint8 scales
    const uint64_t* __restrict__ c_ptrs,
    const int32_t* __restrict__ m_sizes,
    const int32_t* __restrict__ n_sizes,
    const int32_t* __restrict__ k_halves,
    const int32_t* __restrict__ k_scales,
    const uint64_t* __restrict__ a_descs_u64,  // [groups,16]
    const uint64_t* __restrict__ b_descs_u64,  // [groups,16]
    const uint64_t* __restrict__ sfa_descs_u64,  // [groups,16]
    const uint64_t* __restrict__ sfb_descs_u64,  // [groups,16]
    // Optional packed-CTA mapping (GPU MODE-style): grid=(1,1,total_ctas), blockIdx.z selects a CTA.
    // When these are null, we use the legacy max-based grid mapping: grid=(n_tiles, m_tiles, groups).
    const int32_t* __restrict__ cta_group_idx_map,  // [total_ctas] or nullptr
    const int32_t* __restrict__ cta_tile_m_map,     // [total_ctas] or nullptr
    const int32_t* __restrict__ cta_tile_n_map,     // [total_ctas] or nullptr
    int cta2_desc_a_row_offset_rows,
    int cta2_desc_b_row_offset_rows,
    int cta2_desc_sfa_row_offset_rows,
    int cta2_epilogue_row_base_rows,
    int cta2_epilogue_addr_mode,
    int cta2_sfb_slot_mode,
    int cta2_tmem_c_word_offset,
    int cta2_tmem_sf_word_offset,
    int cta2_tmem_sf_rank_word_offset,
    int cta2_tsfa_word_offset,
    int cta2_tsfb_word_offset,
    int debug_tmem_dump,
    int debug_tmem_only_rank,
    int debug_tmem_idx_add,
    int cta2_partition_b,
    int debug_print_ptrs,
    int cta2_idesc_m_dim_override,
    int cta2_idesc_n_dim_override,
    int cluster_dim_x,
    int enable_tma_multicast
) {
  // cta_group::2 note:
  // We intentionally operate on a 256x128 *cluster* tile so each participating CTA owns a full
  // 128-row fragment (M_MMA=128) and we can reuse the known-correct 1SM TMEM accumulator layout.
  // This matches CUTLASS's SM100 blockscaled 2SM builder constraints (TileShape_M == 256) and avoids
  // the much trickier M=128 (M_MMA=64) 2SM accumulator/scale-factor layouts.
  constexpr int CTA_TILE_M = 128;
  constexpr int CLUSTER_TILE_M = (CtaGroup == 2) ? 256 : 128;
  constexpr int TILE_M = CTA_TILE_M;
  constexpr int TILE_N = 128;
  constexpr int K_TILE_BYTES = 128;   // 256 FP4 elems
  constexpr int K_SEG_BYTES = 32;     // 64 FP4 elems
  // TMEM allocation: use the full 512-column TMEM slice, matching the GPU MODE reference.
  // This avoids allocator interleaving behavior for smaller slices and keeps the layout consistent
  // across bring-up and tuned variants.
  constexpr int TMEM_COLUMNS = NVFP4_GROUP_GEMM_V2_TMEM_COLUMNS;
  constexpr int SF_BYTES_PER_ROW = 16;  // (mm4, kk4) packed as 4x4 bytes
  constexpr int SFA_ROWS = 128;         // 4 chunks * 32 rows
  constexpr int SFB_ROWS = 128;         // 4 chunks * 32 rows
  constexpr int SFA_TILE_BYTES = SFA_ROWS * SF_BYTES_PER_ROW;
  constexpr int SFB_TILE_BYTES = SFB_ROWS * SF_BYTES_PER_ROW;
  // CUTLASS `tmem_sf_frg` for SFVecSize=16 (ScaleFactorDuplicated4by1) uses:
  // - 16 TMEM columns per K64 segment
  // - 4 K64 segments per K256 tile
  // => 64 columns per operand tile (SFA or SFB) per CTA.
  //
  // Verified via `UMMA::tmem_sf_frg<uint8_t,16,1,...>::make(((128,(16,4)),1,4))`:
  // `k_tile` (segment) stride = +16 columns and total col extent = 64.
  constexpr uint32_t SF_COLS_PER_SEG = 16u;
  constexpr uint32_t SF_COLS_PER_TILE = 64u;
  // Bring-up knob:
  //   0 = full (TMA + scales + UTCCP + MMA + epilogue)
  //   2 = TMA-only sanity (alloc TMEM, load A/B once, then dealloc + return)
  //   3 = TMA + scales + UTCCP (no MMA/epilogue; one K-tile, then dealloc + return)
  //   4 = TMA + scales + UTCCP + MMA (no epilogue; one K-tile, then dealloc + return)
  constexpr int DEBUG_STAGE = 0;
  // Shared-memory stages: 2-stage double-buffered pipeline across K tiles.
  constexpr int PIPELINE_STAGES = NVFP4_GROUP_GEMM_V2_PIPELINE_STAGES;
  static_assert(PIPELINE_STAGES == 1 || PIPELINE_STAGES == 2, "PIPELINE_STAGES must be 1 or 2");

  static_assert(CtaGroup == 1 || CtaGroup == 2, "CtaGroup must be 1 or 2");
  static_assert(UnrollN == 1 || UnrollN == 2, "UnrollN must be 1 or 2");

  // Optional cluster-mode optimization: multicast A + SFA across CTAs that share (tile_m, k_tile).
  // This reduces redundant L2 traffic for A/SFA across the N tiles of the same M tile.
  // NOTE: Requires cluster_dim_x > 1 and that the grid's N-CTA dimension is an exact multiple
  // of cluster_dim_x (otherwise partial clusters could deadlock on cluster sync).
  const bool use_tma_multicast = (CtaGroup == 1) && (enable_tma_multicast != 0) && (cluster_dim_x > 1);
  const uint16_t tma_multicast_mask =
      use_tma_multicast ? static_cast<uint16_t>((1u << static_cast<unsigned>(cluster_dim_x)) - 1u) : uint16_t{0};

  int group_idx = 0;
  const int cluster_rank =
      (CtaGroup == 2 || use_tma_multicast) ? static_cast<int>(tcgen05::block_rank_in_cluster()) : 0;
  const int cluster_rank_b = cluster_rank;
  cg::cluster_group cluster = cg::this_cluster();
  int tile_n = 0;
  int tile_m = 0;
  if (cta_group_idx_map != nullptr && cta_tile_m_map != nullptr && cta_tile_n_map != nullptr) {
    // Packed-CTA mode: the host launches exactly the required CTAs per group (no early-return CTAs).
    const int cta_linear = static_cast<int>(blockIdx.z);
    group_idx = cta_group_idx_map[cta_linear];
    tile_m = cta_tile_m_map[cta_linear];
    tile_n = cta_tile_n_map[cta_linear];
  } else {
    // Legacy max-based grid: (tile_n, tile_m, group_idx).
    group_idx = static_cast<int>(blockIdx.z);
    // For cluster-mode launches, `blockIdx.x` indexes CTAs along N. Each CTA covers `UnrollN`
    // adjacent N tiles, so convert CTA index -> tile_n start.
    tile_n = (CtaGroup == 2) ? ((static_cast<int>(blockIdx.x) >> 1) * UnrollN)
                             : (static_cast<int>(blockIdx.x) * UnrollN);
    tile_m = static_cast<int>(blockIdx.y);
  }
  // cta_group::2 bring-up: legacy SFB-slot mapping knob (kept for future UTCCP schedule experiments).
  (void)cta2_sfb_slot_mode;

  const int m_size = m_sizes[group_idx];
  const int n_size = n_sizes[group_idx];
  const int k_bytes_total = k_halves[group_idx];
  const int k_tiles_total = ceil_div_int(k_bytes_total, K_TILE_BYTES);
  const int n_tiles_group = ceil_div_int(n_size, TILE_N);
  const int n_tiles_tma = (UnrollN == 2) ? ((n_tiles_group + 1) & ~1) : n_tiles_group;
  const int cta2_partition_b_mode =
      (CtaGroup == 2 && UnrollN == 2) ? 2 : cta2_partition_b;

  const int m_offset_cluster = tile_m * CLUSTER_TILE_M;
  const int n_offset = tile_n * TILE_N;

  // Cluster-wide bounds check: must be uniform across CTAs in the cluster to avoid divergence
  // across required cta_group::2 synchronization points.
  if (m_offset_cluster >= m_size || n_offset >= n_size) {
    return;
  }

  const uint8_t* const a_gmem = reinterpret_cast<const uint8_t*>(a_ptrs[group_idx]);
  const uint8_t* const b_gmem = reinterpret_cast<const uint8_t*>(b_ptrs[group_idx]);
  half* const c_out = reinterpret_cast<half*>(c_ptrs[group_idx]);
  // Legacy bring-up parameters (kept for host-side experimentation). The current 256x128 2SM path
  // intentionally ignores the descriptor-row/epilogue mapping knobs in favor of a simpler bring-up.
  (void)cta2_desc_a_row_offset_rows;
  (void)cta2_desc_b_row_offset_rows;
  (void)cta2_desc_sfa_row_offset_rows;
  (void)cta2_epilogue_row_base_rows;
  (void)cta2_epilogue_addr_mode;
  (void)debug_tmem_dump;
  (void)debug_tmem_only_rank;
  (void)debug_tmem_idx_add;
  (void)debug_print_ptrs;
  (void)cta2_idesc_m_dim_override;
  (void)cta2_idesc_n_dim_override;

  // Shared storage (double-buffered across K tiles).
  __shared__ alignas(128) uint8_t sA[PIPELINE_STAGES][CTA_TILE_M][K_TILE_BYTES];
  __shared__ alignas(128) uint8_t sB[PIPELINE_STAGES][UnrollN][TILE_N][K_TILE_BYTES];
  __shared__ alignas(16) uint8_t sSFA[PIPELINE_STAGES][SFA_ROWS][SF_BYTES_PER_ROW];
  __shared__ alignas(16) uint8_t sSFB[PIPELINE_STAGES][UnrollN][SFB_ROWS][SF_BYTES_PER_ROW];
  // Warp-specialized CTA1 pipeline state (used only in the cta_group::1 fast path).
  // `stage_sf_ready_iter[stage]` holds the k_tile_idx whose SFA/SFB have been copied into TMEM
  // for that stage.
  __shared__ volatile int stage_sf_ready_iter[PIPELINE_STAGES];
  // `consumer_done_iter` holds the last k_tile_idx for which MMA has been issued (the stage can be reused).
  __shared__ volatile int consumer_done_iter;
  // UMMA completion barrier (cta_group::2 only). tcgen05.mma is asynchronous; tcgen05.commit +
  // mbarrier wait ensures the accumulator tile is fully materialized in TMEM before epilogue loads.
  __shared__ alignas(8) uint64_t umma_done_barrier;

  // Barrier for TMA loads (per-stage). We only use the barrier from thread0 and then
  // synchronize the full CTA with __syncthreads() when data is needed.
  using block_barrier = cuda::barrier<cuda::thread_scope_block>;
  struct alignas(alignof(block_barrier)) BarrierStorage {
    unsigned char bytes[sizeof(block_barrier)];
  };
  __shared__ BarrierStorage bar_storage[PIPELINE_STAGES];
  block_barrier* bars[PIPELINE_STAGES];
#pragma unroll
  for (int stage = 0; stage < PIPELINE_STAGES; ++stage) {
    bars[stage] = reinterpret_cast<block_barrier*>(bar_storage[stage].bytes);
  }

  if (threadIdx.x == 0) {
  #pragma unroll
    for (int stage = 0; stage < PIPELINE_STAGES; ++stage) {
      init(bars[stage], /*expected_count=*/1);
      stage_sf_ready_iter[stage] = -1;
    }
    consumer_done_iter = -1;
    cuda_device::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  if constexpr (CtaGroup == 2) {
    // Initialize the UMMA completion barrier once per CTA. We always wait for parity=1 since the
    // barrier is freshly initialized for this kernel invocation.
    if (threadIdx.x == 0) {
      const uint32_t bar_addr = tcgen05::cast_smem_ptr_to_uint(&umma_done_barrier);
      // cta_group::2 MMAs logically involve 2 CTAs; require 2 arrivals before the barrier flips.
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(bar_addr), "r"(2) : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;\n" : : : "memory");
    }
    __syncthreads();
    cg::this_cluster().sync();
  }

  // TMA expects the tensor map descriptor in global/const memory (not shared). Our descriptors are
  // stored in CUDA global memory as a packed [groups,16] int64 tensor (16 * 8 bytes = 128 bytes).
  const CUtensorMap* const a_descs = reinterpret_cast<const CUtensorMap*>(a_descs_u64);
  const CUtensorMap* const b_descs = reinterpret_cast<const CUtensorMap*>(b_descs_u64);
  const CUtensorMap* const sfa_descs = reinterpret_cast<const CUtensorMap*>(sfa_descs_u64);
  const CUtensorMap* const sfb_descs = reinterpret_cast<const CUtensorMap*>(sfb_descs_u64);
  const CUtensorMap* const a_desc = a_descs + group_idx;
  const CUtensorMap* const b_desc = b_descs + group_idx;
  const CUtensorMap* const sfa_desc = sfa_descs + group_idx;
  const CUtensorMap* const sfb_desc = sfb_descs + group_idx;

  // For cta_group::2, each CTA owns a full 128-row fragment within the 256-row cluster tile.
  // Load A/SFA at a per-rank M offset. B/SFB are currently duplicated (both CTAs load the same N tile).
  const int m_offset = m_offset_cluster + ((CtaGroup == 2) ? (cluster_rank * CTA_TILE_M) : 0);
  const int sfa_tile_m = (CtaGroup == 2) ? (tile_m * 2 + cluster_rank) : tile_m;

  // Legacy bring-up knobs (kept for debugging). For the 256x128 cluster tile path these offsets
  // should be 0; we compute the correct per-rank partitioning via the TMA load offsets above.
  const uint64_t cta2_desc_a_row_offset = 0ull;
  const uint64_t cta2_desc_b_row_offset = 0ull;
  const uint64_t cta2_desc_sfa_row_offset = 0ull;

  // Prepare smem descriptors for A/B tiles (per pipeline stage).
  umma::SmemDescriptor desc_a_base[PIPELINE_STAGES]{};
  umma::SmemDescriptor desc_b_base[PIPELINE_STAGES][UnrollN]{};
  // Prepare smem descriptors for scale tiles (Layout_K_INTER_Atom<uint8_t> tiled to 128x16).
  umma::SmemDescriptor desc_sfa[PIPELINE_STAGES]{};
  umma::SmemDescriptor desc_sfb[PIPELINE_STAGES][UnrollN]{};

  // For SWIZZLE_128B major-K UMMA descriptors, advancing by one logical MN row is
  // not a simple linear +row*row_bytes adjustment due to the position-dependent
  // swizzle. CUTLASS's canonical UMMA_K layout (see make_umma_desc<Major::K>)
  // implies the following u128-addressing strides for SW128:
  //   - within an 8-row swizzle atom: +8 u128 per row
  //   - across 8-row blocks:        +stride_u128 per block (SBO)
  // For our use case we only need the base pointer shift for MN offsets at K=0.
  auto sw128_major_k_row_offset_bytes = [](int row_offset_rows, uint32_t stride_u128) -> uint32_t {
    // row_offset_rows is in units of uint8 rows (MN index). The address arithmetic is in units of
    // uint128 (4 LSB stripped). Apply the same Swizzle<3,4,3> transform as CUTLASS uses for SW128.
    const uint32_t within = static_cast<uint32_t>(row_offset_rows & 7);
    const uint32_t block = static_cast<uint32_t>(row_offset_rows >> 3);
    const uint32_t logical_u128 = within * 8u + block * stride_u128;
    // Swizzle<3,4,3> is defined on *byte* offsets. We operate in u128 (16-byte) units here,
    // so the equivalent transform is Swizzle<3,0,3>: XOR bits[0:2] with bits[3:5] shifted down.
    const uint32_t yyy = (logical_u128 & 0x00000038u) >> 3;
    const uint32_t swizzled_u128 = logical_u128 ^ yyy;
    return swizzled_u128 * 16u;
  };

#pragma unroll
  for (int stage = 0; stage < PIPELINE_STAGES; ++stage) {
    const uint32_t sA_addr = tcgen05::cast_smem_ptr_to_uint(&sA[stage][0][0]);
    const uint32_t sSFA_addr = tcgen05::cast_smem_ptr_to_uint(&sSFA[stage][0][0]);

    desc_a_base[stage].version_ = 1;
    desc_a_base[stage].lbo_mode_ = 0;
    desc_a_base[stage].base_offset_ = 0;
    desc_a_base[stage].layout_type_ = static_cast<uint8_t>(umma::LayoutType::SWIZZLE_128B);
    desc_a_base[stage].start_address_ = static_cast<uint16_t>(sA_addr >> 4);
    desc_a_base[stage].leading_byte_offset_ = 1;
    // Canonical stride for a 128-row K-major SW128 operand tile in units of uint128 (4LSB removed).
    desc_a_base[stage].stride_byte_offset_ = 64;

#pragma unroll
    for (int u = 0; u < UnrollN; ++u) {
      uint32_t sB_addr = tcgen05::cast_smem_ptr_to_uint(&sB[stage][u][0][0]);

      desc_b_base[stage][u].version_ = 1;
      desc_b_base[stage][u].lbo_mode_ = 0;
      desc_b_base[stage][u].base_offset_ = 0;
      desc_b_base[stage][u].layout_type_ = static_cast<uint8_t>(umma::LayoutType::SWIZZLE_128B);
      if constexpr (CtaGroup == 2) {
        // Partition B along N/2 across the two CTAs by shifting the *SMEM descriptor base* (not the
        // global coordinate). This keeps the shared-memory tile fully initialized and avoids UB
        // from reading uninitialized rows.
        //
        // Mode:
        //   cta2_partition_b=2 (default): descriptor shift by N/2 within the 128-row shared tile
        //   cta2_partition_b=1: global N shift by N/2 (experimental; requires special padding)
        if (cta2_partition_b_mode == 2) {
          sB_addr += static_cast<uint32_t>(cluster_rank) *
                     sw128_major_k_row_offset_bytes(static_cast<int>(TILE_N / 2), /*stride_u128=*/64u);
        }
      }
      desc_b_base[stage][u].start_address_ = static_cast<uint16_t>(sB_addr >> 4);
      desc_b_base[stage][u].leading_byte_offset_ = 1;
      // Keep the canonical SW128 major-K stride used by the known-correct 1SM path. (We will
      // transliterate the exact CUTLASS descriptor builder once CTA2 semantics are validated.)
      desc_b_base[stage][u].stride_byte_offset_ = 64;

      const uint32_t sSFB_addr = tcgen05::cast_smem_ptr_to_uint(&sSFB[stage][u][0][0]);
      desc_sfb[stage][u].version_ = 1;
      desc_sfb[stage][u].lbo_mode_ = 0;
      desc_sfb[stage][u].base_offset_ = 0;
      desc_sfb[stage][u].layout_type_ = static_cast<uint8_t>(umma::LayoutType::SWIZZLE_NONE);
      if constexpr (UnrollN == 2) {
        // UnrollN=2 packs two adjacent N tiles consecutively in shared memory.
        // For u==1, offset descriptors by 128 rows into the same [256,128] buffer.
        constexpr uint32_t kSFBUnroll2RowOffsetRows = SFB_ROWS;
        constexpr uint32_t kSFBUnroll2BRowOffsetRows = TILE_N;
        const bool use_unroll2_tma_pack = (CtaGroup == 1) || (CtaGroup == 2);
        if (use_unroll2_tma_pack) {
          if (u == 0) {
            desc_b_base[stage][u].start_address_ = static_cast<uint16_t>(sB_addr >> 4);
            desc_sfb[stage][u].start_address_ = static_cast<uint16_t>(sSFB_addr >> 4);
          } else {
            const uint32_t sB_unroll2_base = tcgen05::cast_smem_ptr_to_uint(&sB[stage][0][0][0]);
            const uint32_t sSFB_unroll2_base = tcgen05::cast_smem_ptr_to_uint(&sSFB[stage][0][0][0]);
            const uint32_t b_row_offset_bytes =
                sw128_major_k_row_offset_bytes(static_cast<int>(kSFBUnroll2BRowOffsetRows), /*stride_u128=*/64u);
            desc_b_base[stage][u].start_address_ = static_cast<uint16_t>((sB_unroll2_base + b_row_offset_bytes) >> 4u);
            desc_sfb[stage][u].start_address_ = static_cast<uint16_t>((sSFB_unroll2_base >> 4u) + kSFBUnroll2RowOffsetRows);
          }
        } else {
          desc_b_base[stage][u].start_address_ = static_cast<uint16_t>(sB_addr >> 4);
          desc_sfb[stage][u].start_address_ = static_cast<uint16_t>(sSFB_addr >> 4);
        }
      } else {
        desc_b_base[stage][u].start_address_ = static_cast<uint16_t>(sB_addr >> 4);
        desc_sfb[stage][u].start_address_ = static_cast<uint16_t>(sSFB_addr >> 4);
      }
      desc_sfb[stage][u].leading_byte_offset_ = 1;
      desc_sfb[stage][u].stride_byte_offset_ = 8;
    }

    desc_sfa[stage].version_ = 1;
    desc_sfa[stage].lbo_mode_ = 0;
    desc_sfa[stage].base_offset_ = 0;
    desc_sfa[stage].layout_type_ = static_cast<uint8_t>(umma::LayoutType::SWIZZLE_NONE);
    desc_sfa[stage].start_address_ = static_cast<uint16_t>(sSFA_addr >> 4);
    desc_sfa[stage].leading_byte_offset_ = 1;
    desc_sfa[stage].stride_byte_offset_ = 8;
  }

  // Allocate TMEM once per CTA.
  __shared__ uint32_t tmem_base;
  // NOTE: tcgen05.alloc/dealloc are warp-synchronous: issue from a single fully-active warp.
  if constexpr (CtaGroup == 2) {
    // Ensure the 2 participating CTAs reach alloc together (requirement for cta_group::2).
    cluster.sync();
  }
  if (threadIdx.x < 32) {
    // Columns must be power-of-2, 32..512.
    tcgen05::tmem_alloc<CtaGroup>(&tmem_base, /*num_columns=*/TMEM_COLUMNS);
  }
  __syncthreads();
  if constexpr (CtaGroup == 2) {
    cluster.sync();
  }

  // TMEM address plan (word addressing).
  //
  // Important: `tcgen05.alloc.cta_group::2` returns a TMEM pointer that is shared across both CTAs
  // in the group (see CUTLASS `cute/arch/tmem_allocator_sm100.hpp`). We must partition the
  // allocation so each CTA writes/reads disjoint regions for its accumulators and (at least) SFA.
  const uint32_t tmem_addr_base = tmem_base;
  const uint32_t tmem_c_rank =
      (CtaGroup == 2)
          ? tcgen05::tmem_addr_add(
                tmem_addr_base,
                /*dp_add=*/0u,
                static_cast<uint32_t>(cluster_rank) * 128u + static_cast<uint32_t>(cta2_tmem_c_word_offset))
          : tmem_addr_base;

  uint32_t tmem_c_tiles[UnrollN];
  tmem_c_tiles[0] = tmem_c_rank;
  if constexpr (UnrollN == 2) {
    // Second 128-column accumulator tile lives at +128 columns.
    tmem_c_tiles[1] = tcgen05::tmem_addr_add(tmem_c_rank, /*dp_add=*/0u, /*col_add=*/128u);
  }

  uint32_t tmem_sfa_ptrs[4];
  uint32_t tmem_sfb_ptrs[4 * UnrollN];
  // Scale-factor layout in TMEM (CUTLASS `tmem_sf_frg`, ScaleFactorDuplicated4by1):
  // - Each operand tile (SFA or SFB) consumes 64 TMEM columns (4 K64 segments * 16 cols/seg).
  // - cta_group::1:
  //     UnrollN=1: accum uses 128 cols, so place SFA at +128, SFB at +192 (64 cols each).
  //     UnrollN=2: accum uses 256 cols (2x 128-col tiles), so place SFA at +256, then SFB at +320
  //               (reserve 64 cols per unrolled N tile => +320/+384).
  // - cta_group::2:
  //     We allocate disjoint accumulator windows per rank (rank0 at +0, rank1 at +128), so the
  //     group consumes 256 cols. Place scales at +256, then reserve 128 cols per rank (64 SFA + 64 SFB).
  const uint32_t tmem_sf_base =
      (CtaGroup == 2)
          ? tcgen05::tmem_addr_add(
                tmem_addr_base,
                /*dp_add=*/0u,
                256u + static_cast<uint32_t>(cta2_tmem_sf_word_offset))
          : tcgen05::tmem_addr_add(
                tmem_addr_base,
                /*dp_add=*/0u,
                (UnrollN == 2) ? 256u : static_cast<uint32_t>(cta2_tmem_sf_word_offset));
  const uint32_t tmem_sf_rank_base =
      (CtaGroup == 2)
          ? (tmem_sf_base + static_cast<uint32_t>(cluster_rank) * static_cast<uint32_t>(cta2_tmem_sf_rank_word_offset))
          : tmem_sf_base;

  const uint32_t tmem_sfa_base = tmem_sf_rank_base;
  const uint32_t tmem_sfb_base = tcgen05::tmem_addr_add(tmem_sfa_base, /*dp_add=*/0u, /*col_add=*/SF_COLS_PER_TILE);
#pragma unroll
  for (int seg = 0; seg < 4; ++seg) {
    tmem_sfa_ptrs[seg] = tcgen05::tmem_addr_add(
        tmem_sfa_base, /*dp_add=*/0u, /*col_add=*/static_cast<uint32_t>(seg) * SF_COLS_PER_SEG);
  }
#pragma unroll
  for (int u = 0; u < UnrollN; ++u) {
    const uint32_t tmem_sfb_base_u =
        tcgen05::tmem_addr_add(tmem_sfb_base, /*dp_add=*/0u, /*col_add=*/static_cast<uint32_t>(u) * SF_COLS_PER_TILE);
#pragma unroll
    for (int seg = 0; seg < 4; ++seg) {
      tmem_sfb_ptrs[u * 4 + seg] = tcgen05::tmem_addr_add(
          tmem_sfb_base_u, /*dp_add=*/0u, /*col_add=*/static_cast<uint32_t>(seg) * SF_COLS_PER_SEG);
    }
  }

  if constexpr (CtaGroup == 2) {
    if (debug_print_ptrs != 0 && group_idx == 0 && tile_m == 0 && tile_n == 0 && threadIdx.x == 0) {
      const uint32_t tsfa0 = static_cast<uint32_t>(tmem_sfa_ptrs[0] + static_cast<uint32_t>(cta2_tsfa_word_offset));
      const uint32_t tsfb0 = static_cast<uint32_t>(tmem_sfb_ptrs[0] + static_cast<uint32_t>(cta2_tsfb_word_offset));
      const uint32_t sfa_id = (tsfa0 & 0xC0000000u) >> 30;
      const uint32_t sfb_id = (tsfb0 & 0xC0000000u) >> 30;
      printf("cta2 rank=%d tmem_base=0x%08x tmem_c=0x%08x tsfa0=0x%08x(tsfa_id=%u) tsfb0=0x%08x(tsfb_id=%u)\\n",
             cluster_rank, tmem_base, tmem_c_rank, tsfa0, sfa_id, tsfb0, sfb_id);
    }
  }

  // Instruction descriptor (block scaled, MXF4 E2M1, scale UE4M3, K-major operands).
  umma::InstrDescriptorBlockScaled idesc{};
  idesc.sparse_id2_ = 0;
  idesc.sparse_flag_ = 0;
  idesc.b_sf_id_ = 0;
  idesc.a_format_ = 1;  // MXF4Format::E2M1
  idesc.b_format_ = 1;  // MXF4Format::E2M1
  idesc.a_negate_ = 0;
  idesc.b_negate_ = 0;
  idesc.a_major_ = static_cast<uint8_t>(umma::Major::K);
  idesc.b_major_ = static_cast<uint8_t>(umma::Major::K);
  idesc.n_dim_ = (TILE_N >> 3);
  idesc.scale_format_ = 0;  // ScaleFormat::UE4M3
  // For cta_group::2, we use the M=256 2-CTA cluster MMA variant (each CTA contributes 128 rows).
  // The instruction descriptor must encode the *full* M dimension for the 2-CTA group instruction.
  // For cta_group::1, keep the canonical full tile M dimension. CUTLASS encodes `m_dim_ = (M >> 4)`
  // with compile-time M (128 here). Some runtime-masked M encodings can trigger illegal instruction
  // faults on SM100 when using block-scaled UMMA.
  idesc.m_dim_ = (CtaGroup == 2) ? (CLUSTER_TILE_M >> 4) : (TILE_M >> 4);
  idesc.a_sf_id_ = 0;
  idesc.k_size_ = 0;  // MXF4 dense K64
  if constexpr (CtaGroup == 2) {
    // Bring-up debug knobs: allow overriding the descriptor M/N dims to validate 2CTA semantics.
    if (cta2_idesc_m_dim_override > 0) {
      idesc.m_dim_ = static_cast<uint16_t>(cta2_idesc_m_dim_override);
    }
    if (cta2_idesc_n_dim_override > 0) {
      idesc.n_dim_ = static_cast<uint16_t>(cta2_idesc_n_dim_override);
    }
  }

  // Precompute per-segment TMEM scale pointers and runtime instruction descriptor hi bits.
  // These are invariant across K tiles: each K tile overwrites the same TMEM scale slots.
  uint32_t tmem_sfa_seg[4];
  uint32_t tmem_sfb_seg[UnrollN][4];
  uint32_t idesc_hi_seg[UnrollN][4];
  if (threadIdx.x == 0) {
#pragma unroll
    for (int seg = 0; seg < 4; ++seg) {
      const uint32_t tmem_sfa =
          (CtaGroup == 2) ? static_cast<uint32_t>(tmem_sfa_ptrs[seg] + static_cast<uint32_t>(cta2_tsfa_word_offset))
                          : tmem_sfa_ptrs[seg];
      tmem_sfa_seg[seg] = tmem_sfa;
#pragma unroll
      for (int u = 0; u < UnrollN; ++u) {
        const uint32_t tmem_sfb =
            (CtaGroup == 2) ? static_cast<uint32_t>(tmem_sfb_ptrs[u * 4 + seg] + static_cast<uint32_t>(cta2_tsfb_word_offset))
                            : tmem_sfb_ptrs[u * 4 + seg];
        tmem_sfb_seg[u][seg] = tmem_sfb;
        const uint64_t idesc_runtime = umma::make_runtime_instr_desc_block_scaled(idesc, tmem_sfa, tmem_sfb);
        idesc_hi_seg[u][seg] = static_cast<uint32_t>(idesc_runtime >> 32);
      }
    }
  }

  // Shared-memory -> TMEM scale copy helper.
  // UnrollN=1 and all SFA copies use 32x128b.warpx4.
  // UnrollN=2 correctness-first: copy SFB per-unrolled-tile with the same 32x128b.warpx4 primitive.
  // (We can revisit 64x128b warpx2 once the CUTLASS layout is fully exploited.)
  auto copy_scale_fragments = [&](uint64_t src_desc_base, const uint32_t* tmem_ptrs, bool is_sfb) -> void {
    constexpr int SF_COPY_32x128B_DESC_STEP = 32;
    (void)is_sfb;
    for (int seg = 0; seg < 4; ++seg) {
      const uint64_t src_desc = src_desc_base + static_cast<uint64_t>(seg * SF_COPY_32x128B_DESC_STEP);
      tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(src_desc, tmem_ptrs[seg]);
    }
  };


  // TMA helper: issue a single-stage load for the given K tile into the selected pipeline stage.
  // Only thread0 participates in the barrier (expected_count=1) and owns the returned token.
  auto issue_tma_tile = [&](int stage, int k_byte, int sfa_row_offset, int sfb_row_offset) -> block_barrier::arrival_token {
    block_barrier* bar = bars[stage];
    block_barrier::arrival_token tok;
    // Stage all async TMA work under a single barrier generation.
    // When enabled, multicast A + SFA from the cluster leader (rank0) to all CTAs in the cluster.
    if (threadIdx.x == 0) {
      constexpr size_t kBytesA = sizeof(sA[0]);
      int unroll_n_valid = 0;
#pragma unroll
      for (int u = 0; u < UnrollN; ++u) {
        if ((tile_n + u) < n_tiles_group) {
          ++unroll_n_valid;
        }
      }
      int unroll_n_tx = unroll_n_valid;
      if constexpr (UnrollN == 2) {
        const bool use_unroll2_tma_pack = (CtaGroup == 1) || (CtaGroup == 2);
        // UnrollN=2 always issues a single 256-row TMA load for B and SFB (2 adjacent N tiles).
        // The host pads N to a multiple of 256 so this stays in-bounds even for odd tile counts.
        if (use_unroll2_tma_pack) {
          unroll_n_tx = 2;
        }
      }
      size_t kBytesB = static_cast<size_t>(unroll_n_tx) * static_cast<size_t>(TILE_N) * static_cast<size_t>(K_TILE_BYTES);
      if constexpr (CtaGroup == 2 && UnrollN == 1) {
        // For legacy cta2_partition_b=1, each CTA loads only N/2 rows.
        if (cta2_partition_b_mode == 1) {
          kBytesB =
              static_cast<size_t>(unroll_n_tx) * static_cast<size_t>(TILE_N / 2) * static_cast<size_t>(K_TILE_BYTES);
        }
      }
      const size_t kBytesSFB = static_cast<size_t>(unroll_n_tx) * static_cast<size_t>(SFB_TILE_BYTES);
      const size_t kBytesSF = static_cast<size_t>(SFA_TILE_BYTES) + kBytesSFB;
      tok = cuda::device::barrier_arrive_tx(*bar, 1, kBytesA + kBytesB + kBytesSF);

      // For cta_group::2, CUTLASS partitions B across the two CTAs along N. In mode 1 we shift
      // the global N coordinate by N/2 so each CTA loads only one half. In other modes we keep the
      // global N coordinate and shift the shared-memory descriptor base per rank.
      //
      // Debug knob: allow disabling this shift to validate whether B is truly partitioned along N/2.
      const bool partition_b_global_shift = (CtaGroup == 2) && (cta2_partition_b_mode == 1);
      const int b_n_offset_base = partition_b_global_shift ? (n_offset + cluster_rank_b * (TILE_N / 2)) : n_offset;
      // B/SFB vary with tile_n, so always load them per-CTA.
      if constexpr (UnrollN == 2) {
        const bool use_unroll2_tma_pack = (CtaGroup == 1) || (CtaGroup == 2);
        if (use_unroll2_tma_pack) {
          // Load 2 adjacent N tiles in a single TMA transaction (box_height=256).
          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sB[stage][0][0][0], b_desc, k_byte, b_n_offset_base, *bar);
          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sSFB[stage][0][0][0], sfb_desc, /*x=*/0, sfb_row_offset, *bar);
        } else {
#pragma unroll
          for (int u = 0; u < UnrollN; ++u) {
            if ((tile_n + u) >= n_tiles_group) {
              continue;
            }
            const int k_tile_idx = k_byte / K_TILE_BYTES;
            const int b_n_offset = b_n_offset_base + u * TILE_N;
            const int sfb_row_offset_u =
                (UnrollN == 2) ? ((k_tile_idx * n_tiles_tma + tile_n + u) * SFB_ROWS)
                               : (sfb_row_offset + u * k_tiles_total * SFB_ROWS);
            cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sB[stage][u][0][0], b_desc, k_byte, b_n_offset, *bar);
            cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sSFB[stage][u][0][0], sfb_desc, /*x=*/0, sfb_row_offset_u, *bar);
          }
        }
      } else {
#pragma unroll
        for (int u = 0; u < UnrollN; ++u) {
          if ((tile_n + u) >= n_tiles_group) {
            continue;
          }
          const int k_tile_idx = k_byte / K_TILE_BYTES;
          const int b_n_offset = b_n_offset_base + u * TILE_N;
          const int sfb_row_offset_u =
              (UnrollN == 2) ? ((k_tile_idx * n_tiles_tma + tile_n + u) * SFB_ROWS)
                             : (sfb_row_offset + u * k_tiles_total * SFB_ROWS);
          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sB[stage][u][0][0], b_desc, k_byte, b_n_offset, *bar);
          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sSFB[stage][u][0][0], sfb_desc, /*x=*/0, sfb_row_offset_u, *bar);
        }
      }

      if (!use_tma_multicast) {
        cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sA[stage][0][0], a_desc, k_byte, m_offset, *bar);
        cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sSFA[stage][0][0], sfa_desc, /*x=*/0, sfa_row_offset, *bar);
      }

      cuda_device::cp_async_bulk_commit_group();
    }

    // TMA multicast requires all CTAs to have joined the barrier generation before the
    // leader issues the multicast op, otherwise some CTAs can miss completions.
    // See `ch10/tma_multicast_cluster.cu` for the required sync pattern.
    if (use_tma_multicast) {
      __syncthreads();
      cg::this_cluster().sync();
    }

    if (use_tma_multicast && cluster_rank == 0 && threadIdx.x == 0) {
      // Coordinates are in the descriptor's dimension order. For A: (x=k_byte, y=m_offset).
      const int coords_a[2] = {k_byte, m_offset};
      cptx::cp_async_bulk_tensor(cptx::space_cluster,
                                 cptx::space_global,
                                 &sA[stage][0][0],
                                 a_desc,
                                 coords_a,
                                 cuda::device::barrier_native_handle(*bar),
                                 tma_multicast_mask);

      // For packed SFA: (x=0, y=row_offset).
      const int coords_sfa[2] = {0, sfa_row_offset};
      cptx::cp_async_bulk_tensor(cptx::space_cluster,
                                 cptx::space_global,
                                 &sSFA[stage][0][0],
                                 sfa_desc,
                                 coords_sfa,
                                 cuda::device::barrier_native_handle(*bar),
                                 tma_multicast_mask);
    }
    return tok;
  };

  auto wait_tma_tile = [&](int stage, block_barrier::arrival_token& tok) {
    block_barrier* bar = bars[stage];
    if (threadIdx.x == 0) {
      // Wait for the outstanding TMA transactions associated with this barrier generation.
      bar->wait(std::move(tok));
      cuda_device::cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();
    if constexpr (CtaGroup == 2) {
      cg::this_cluster().sync();
    } else if (use_tma_multicast) {
      cg::this_cluster().sync();
    }
  };

  // Warp-specialized TMA wait for the common case (cta_group::1, no cluster multicast):
  // thread0 performs the barrier wait, then we warp-sync to keep warp0 from racing ahead.
  auto wait_tma_tile_warp0 = [&](int stage, block_barrier::arrival_token& tok) {
    block_barrier* bar = bars[stage];
    if (threadIdx.x == 0) {
      bar->wait(std::move(tok));
      cuda_device::cp_async_bulk_wait_group_read<0>();
    }
    __syncwarp();
  };

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;

  // Full-CTA bring-up path (used for cta_group::2 and for optional TMA multicast mode).
  auto run_full_cta_mainloop = [&]() {
    if constexpr (PIPELINE_STAGES == 1) {
      // Sequential K-tile loop (1 stage): load -> scales -> MMA.
      for (int k_tile_idx = 0; k_tile_idx < k_tiles_total; ++k_tile_idx) {
        const int k_byte = k_tile_idx * K_TILE_BYTES;
        const int sfa_row_offset = (sfa_tile_m * k_tiles_total + k_tile_idx) * SFA_ROWS;
        const int sfb_row_offset =
            (UnrollN == 2) ? ((k_tile_idx * n_tiles_tma + tile_n) * SFB_ROWS) : ((tile_n * k_tiles_total + k_tile_idx) * SFB_ROWS);
        auto tok0 = issue_tma_tile(/*stage=*/0, k_byte, sfa_row_offset, sfb_row_offset);
        wait_tma_tile(/*stage=*/0, tok0);

        if constexpr (DEBUG_STAGE == 2) {
          break;
        }

        if (threadIdx.x == 0) {
          const uint64_t desc_sfa_base = static_cast<uint64_t>(desc_sfa[0]) + cta2_desc_sfa_row_offset;
          copy_scale_fragments(desc_sfa_base, tmem_sfa_ptrs, /*is_sfb=*/false);
          for (int u = 0; u < UnrollN; ++u) {
            if ((tile_n + u) >= n_tiles_group) {
              continue;
            }
            const uint64_t desc_sfb_base = static_cast<uint64_t>(desc_sfb[0][u]);
            const uint32_t* tmem_sfb_ptrs_u = tmem_sfb_ptrs + u * 4;
            copy_scale_fragments(desc_sfb_base, tmem_sfb_ptrs_u, /*is_sfb=*/true);
          }
        }
        __syncthreads();
        if constexpr (CtaGroup == 2) {
          cg::this_cluster().sync();
        }
        // CUTLASS does not issue an explicit `tcgen05.wait::st` fence after UTCCP scale copies.
        // The subsequent UMMA uses the TMEM scale addresses and will naturally stall if needed.
        // Waiting here over-serializes the pipeline and costs latency (~16us -> 13us target).

        if constexpr (DEBUG_STAGE == 3) {
          break;
        }

        if (threadIdx.x == 0) {
#pragma unroll
	        for (int u = 0; u < UnrollN; ++u) {
	          if ((tile_n + u) >= n_tiles_group) {
	            continue;
	          }
#pragma unroll
	          for (int seg = 0; seg < 4; ++seg) {
	            const uint64_t desc_a =
	                static_cast<uint64_t>(desc_a_base[0]) + cta2_desc_a_row_offset +
	                static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
	            const uint64_t desc_b =
	                static_cast<uint64_t>(desc_b_base[0][u]) + cta2_desc_b_row_offset +
	                static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
	            const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
	            tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
	                desc_a, desc_b, tmem_c_tiles[u], accumulate, idesc_hi_seg[u][seg], tmem_sfa_seg[seg], tmem_sfb_seg[u][seg]);
	          }
	        }
	        }

        if constexpr (DEBUG_STAGE == 4) {
          break;
        }
      }
      return;
    }

    // Prologue: preload the first K tile into stage0.
    if (k_tiles_total > 0) {
      const int k_tile_idx = 0;
      const int k_byte = 0;
      const int sfa_row_offset = (sfa_tile_m * k_tiles_total + k_tile_idx) * SFA_ROWS;
      const int sfb_row_offset =
          (UnrollN == 2) ? ((k_tile_idx * n_tiles_tma + tile_n) * SFB_ROWS) : ((tile_n * k_tiles_total + k_tile_idx) * SFB_ROWS);
      auto tok0 = issue_tma_tile(/*stage=*/0, k_byte, sfa_row_offset, sfb_row_offset);
      wait_tma_tile(/*stage=*/0, tok0);
    }

    // Mainloop: iterate over K tiles with a 2-stage shared-memory pipeline.
    for (int k_tile_idx = 0; k_tile_idx < k_tiles_total; ++k_tile_idx) {
      const int stage_cur = k_tile_idx & 1;
      const int stage_next = stage_cur ^ 1;
      const int k_byte = k_tile_idx * K_TILE_BYTES;

      if constexpr (DEBUG_STAGE == 2) {
        break;
      }

      // Prefetch the next K tile into the alternate stage (overlaps with UTCCP+MMA of the current tile).
      const bool has_next = (k_tile_idx + 1) < k_tiles_total;
      block_barrier::arrival_token tok_next;
	      if (has_next) {
	        const int next_tile = k_tile_idx + 1;
	        const int next_k_byte = next_tile * K_TILE_BYTES;
	        const int sfa_row_offset_next = (sfa_tile_m * k_tiles_total + next_tile) * SFA_ROWS;
	        const int sfb_row_offset_next =
	            (UnrollN == 2) ? ((next_tile * n_tiles_tma + tile_n) * SFB_ROWS) : ((tile_n * k_tiles_total + next_tile) * SFB_ROWS);
	        tok_next = issue_tma_tile(stage_next, next_k_byte, sfa_row_offset_next, sfb_row_offset_next);
	      }

			      if (threadIdx.x == 0) {
			        const uint64_t desc_sfa_base = static_cast<uint64_t>(desc_sfa[stage_cur]) + cta2_desc_sfa_row_offset;
                copy_scale_fragments(desc_sfa_base, tmem_sfa_ptrs, /*is_sfb=*/false);
                for (int u = 0; u < UnrollN; ++u) {
                  if ((tile_n + u) >= n_tiles_group) {
                    continue;
                  }
                  const uint64_t desc_sfb_base = static_cast<uint64_t>(desc_sfb[stage_cur][u]);
                  const uint32_t* tmem_sfb_ptrs_u = tmem_sfb_ptrs + u * 4;
                  copy_scale_fragments(desc_sfb_base, tmem_sfb_ptrs_u, /*is_sfb=*/true);
                }
			      }

      __syncthreads();
      if constexpr (CtaGroup == 2) {
        cg::this_cluster().sync();
      }

      // UTCCP copies write TMEM asynchronously. Ensure scale-factor tiles are resident
      // before the subsequent MMA reads them (correctness-first; we can pipeline later).
      // See note above: avoid globally fencing TMEM stores after UTCCP scale copies.

      if constexpr (DEBUG_STAGE == 3) {
        break;
      }

      if (threadIdx.x == 0) {
#pragma unroll
	        for (int u = 0; u < UnrollN; ++u) {
	          if ((tile_n + u) >= n_tiles_group) {
	            continue;
	          }
#pragma unroll
	          for (int seg = 0; seg < 4; ++seg) {
	            const uint64_t desc_a =
	                static_cast<uint64_t>(desc_a_base[stage_cur]) + cta2_desc_a_row_offset +
	                static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
	            const uint64_t desc_b =
	                static_cast<uint64_t>(desc_b_base[stage_cur][u]) + cta2_desc_b_row_offset +
	                static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));

	            // accumulate=0 for first segment of first tile, else accumulate=1.
	            const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
	            tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
	                desc_a, desc_b, tmem_c_tiles[u], accumulate, idesc_hi_seg[u][seg], tmem_sfa_seg[seg], tmem_sfb_seg[u][seg]);
	          }
	        }
	      }

      if constexpr (DEBUG_STAGE == 4) {
        break;
      }

      // Ensure the next stage is resident in shared memory before the loop advances.
      if (has_next) {
        wait_tma_tile(stage_next, tok_next);
      } else {
        __syncthreads();
        if constexpr (CtaGroup == 2) {
          cg::this_cluster().sync();
        }
      }
    }
  };

  if constexpr (CtaGroup == 1) {
    if (!use_tma_multicast) {
      // Fast path: warp0 runs the mainloop; other warps stay idle until epilogue.
      // This avoids per-K-tile CTA-wide synchronization overhead in the common (non-multicast) mode.
    if constexpr (PIPELINE_STAGES == 1) {
      // Single-stage bring-up: keep the simple warp0-only mainloop.
      if (warp == 0) {
        for (int k_tile_idx = 0; k_tile_idx < k_tiles_total; ++k_tile_idx) {
          const int k_byte = k_tile_idx * K_TILE_BYTES;
          const int sfa_row_offset = (sfa_tile_m * k_tiles_total + k_tile_idx) * SFA_ROWS;
          const int sfb_row_offset = (UnrollN == 2) ? ((k_tile_idx * n_tiles_tma + tile_n) * SFB_ROWS)
                                                    : ((tile_n * k_tiles_total + k_tile_idx) * SFB_ROWS);
          auto tok0 = issue_tma_tile(/*stage=*/0, k_byte, sfa_row_offset, sfb_row_offset);
          wait_tma_tile_warp0(/*stage=*/0, tok0);

          if constexpr (DEBUG_STAGE == 2) {
            break;
          }

          if (lane == 0) {
            const uint64_t desc_sfa_base = static_cast<uint64_t>(desc_sfa[0]) + cta2_desc_sfa_row_offset;
            copy_scale_fragments(desc_sfa_base, tmem_sfa_ptrs, /*is_sfb=*/false);
            for (int u = 0; u < UnrollN; ++u) {
              if ((tile_n + u) >= n_tiles_group) {
                continue;
              }
              const uint64_t desc_sfb_base = static_cast<uint64_t>(desc_sfb[0][u]);
              const uint32_t* tmem_sfb_ptrs_u = tmem_sfb_ptrs + u * 4;
              copy_scale_fragments(desc_sfb_base, tmem_sfb_ptrs_u, /*is_sfb=*/true);
            }
          }
          // See note above: avoid globally fencing TMEM stores after UTCCP scale copies.

          if constexpr (DEBUG_STAGE == 3) {
            break;
          }

          if (lane == 0) {
#pragma unroll
            for (int seg = 0; seg < 4; ++seg) {
              const uint64_t desc_a =
                  static_cast<uint64_t>(desc_a_base[0]) + cta2_desc_a_row_offset +
                  static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
              const uint32_t accumulate = (k_byte == 0 && seg == 0) ? 0u : 1u;
              const uint64_t desc_b_base_u0 =
                  static_cast<uint64_t>(desc_b_base[0][0]) + cta2_desc_b_row_offset +
                  static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
              tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
                  desc_a, desc_b_base_u0, tmem_c_tiles[0], accumulate, idesc_hi_seg[0][seg], tmem_sfa_seg[seg],
                  tmem_sfb_seg[0][seg]);
              if constexpr (UnrollN == 2 && CtaGroup == 1) {
                if ((tile_n + 1) < n_tiles_group) {
                  const uint64_t desc_b_base_u1 =
                      static_cast<uint64_t>(desc_b_base[0][1]) + cta2_desc_b_row_offset +
                      static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
                  tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
                      desc_a, desc_b_base_u1, tmem_c_tiles[1], accumulate, idesc_hi_seg[1][seg], tmem_sfa_seg[seg],
                      tmem_sfb_seg[1][seg]);
                }
              }
            }
          }

          if constexpr (DEBUG_STAGE == 4) {
            break;
          }
        }
      }
    } else {
      if (warp == 0) {
        // Prologue: preload the first K tile into stage0.
        if (k_tiles_total > 0) {
          const int k_tile_idx = 0;
          const int k_byte = 0;
          const int sfa_row_offset = (sfa_tile_m * k_tiles_total + k_tile_idx) * SFA_ROWS;
          const int sfb_row_offset = (UnrollN == 2) ? ((k_tile_idx * n_tiles_tma + tile_n) * SFB_ROWS)
                                                    : ((tile_n * k_tiles_total + k_tile_idx) * SFB_ROWS);
          auto tok0 = issue_tma_tile(/*stage=*/0, k_byte, sfa_row_offset, sfb_row_offset);
          wait_tma_tile_warp0(/*stage=*/0, tok0);
        }

        // Mainloop: iterate over K tiles with a 2-stage shared-memory pipeline.
        for (int k_tile_idx = 0; k_tile_idx < k_tiles_total; ++k_tile_idx) {
          const int stage_cur = k_tile_idx & 1;
          const int stage_next = stage_cur ^ 1;
          const int k_byte = k_tile_idx * K_TILE_BYTES;

          if constexpr (DEBUG_STAGE == 2) {
            break;
          }

          // Prefetch the next K tile into the alternate stage.
          const bool has_next = (k_tile_idx + 1) < k_tiles_total;
          block_barrier::arrival_token tok_next;
          if (has_next) {
            const int next_tile = k_tile_idx + 1;
            const int next_k_byte = next_tile * K_TILE_BYTES;
            const int sfa_row_offset_next = (sfa_tile_m * k_tiles_total + next_tile) * SFA_ROWS;
            const int sfb_row_offset_next =
                (UnrollN == 2) ? ((next_tile * n_tiles_tma + tile_n) * SFB_ROWS) : ((tile_n * k_tiles_total + next_tile) * SFB_ROWS);
            tok_next = issue_tma_tile(stage_next, next_k_byte, sfa_row_offset_next, sfb_row_offset_next);
          }

          // Copy scale factors from shared memory -> TMEM.
          if (lane == 0) {
            const uint64_t desc_sfa_base = static_cast<uint64_t>(desc_sfa[stage_cur]) + cta2_desc_sfa_row_offset;
            copy_scale_fragments(desc_sfa_base, tmem_sfa_ptrs, /*is_sfb=*/false);
            for (int u = 0; u < UnrollN; ++u) {
              if ((tile_n + u) >= n_tiles_group) {
                continue;
              }
              const uint64_t desc_sfb_base = static_cast<uint64_t>(desc_sfb[stage_cur][u]);
              const uint32_t* tmem_sfb_ptrs_u = tmem_sfb_ptrs + u * 4;
              copy_scale_fragments(desc_sfb_base, tmem_sfb_ptrs_u, /*is_sfb=*/true);
            }
          }
          // See note above: avoid globally fencing TMEM stores after UTCCP scale copies.

          if constexpr (DEBUG_STAGE == 3) {
            break;
          }

          // Issue the 4 segment MMAs (K=256 => 4 x K=64 segments).
          if (lane == 0) {
#pragma unroll
            for (int seg = 0; seg < 4; ++seg) {
              const uint64_t desc_a =
                  static_cast<uint64_t>(desc_a_base[stage_cur]) + cta2_desc_a_row_offset +
                  static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
              const uint32_t accumulate = (k_byte == 0 && seg == 0) ? 0u : 1u;
              const uint64_t desc_b_base_u0 =
                  static_cast<uint64_t>(desc_b_base[stage_cur][0]) + cta2_desc_b_row_offset +
                  static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
              tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
                  desc_a, desc_b_base_u0, tmem_c_tiles[0], accumulate, idesc_hi_seg[0][seg], tmem_sfa_seg[seg],
                  tmem_sfb_seg[0][seg]);
              if constexpr (UnrollN == 2 && CtaGroup == 1) {
                if ((tile_n + 1) < n_tiles_group) {
                  const uint64_t desc_b_base_u1 =
                      static_cast<uint64_t>(desc_b_base[stage_cur][1]) + cta2_desc_b_row_offset +
                      static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
                  tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
                      desc_a, desc_b_base_u1, tmem_c_tiles[1], accumulate, idesc_hi_seg[1][seg], tmem_sfa_seg[seg],
                      tmem_sfb_seg[1][seg]);
                }
              }
            }
          }

          if constexpr (DEBUG_STAGE == 4) {
            break;
          }

          // Ensure the next stage is resident in shared memory before the loop advances.
          if (has_next) {
            wait_tma_tile_warp0(stage_next, tok_next);
          } else {
            __syncwarp();
          }
        }
      }
    }

    // Join all warps before any later TMEM loads in debug/epilogue.
    __syncthreads();
    } else {
      run_full_cta_mainloop();
    }
  } else {
    run_full_cta_mainloop();
  }

  if constexpr (CtaGroup == 2) {
    const uint32_t bar_addr = tcgen05::cast_smem_ptr_to_uint(&umma_done_barrier);
    // One lane per CTA issues the commit. The barrier flips only once both CTAs have arrived,
    // which prevents missing a short-lived intermediate parity.
    if (threadIdx.x == 0) {
      asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
                   :
                   : "r"(bar_addr)
                   : "memory");
    }
    // Wait for commit completion before any TMEM loads. (Correctness first; later we can
    // pipeline and use non-blocking checks.)
    if (threadIdx.x == 0) {
      uint32_t done = 0;
      constexpr uint32_t kPhase = 1;
      while (done == 0) {
        asm volatile("{\n\t"
                     ".reg .pred P1;\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
                     "selp.b32 %0, 1, 0, P1;\n\t"
                     "}\n"
                     : "=r"(done)
                     : "r"(bar_addr), "r"(kPhase)
                     : "memory");
      }
    }
    __syncthreads();
    cg::this_cluster().sync();
  }

  // Ensure UMMA writes (TMEM stores) are visible to TMEM loads before any debug/epilogue reads.
  // For cta_group::2, UMMA is explicitly asynchronous and requires a commit+mbarrier wait above.
  // For cta_group::1, `tcgen05.ld.sync` will naturally stall on outstanding writes, so avoid an
  // extra global wait here (latency sensitive).
  if constexpr (CtaGroup == 2) {
    tcgen05::tmem_wait_st_sync();
    tcgen05::tmem_wait_ld_sync();
  }

  // Optional debug: dump a 128x128 slice of TMEM (dp x col) into the output tile.
  // This is for offline mapping analysis only; it bypasses the normal epilogue.
  if constexpr (CtaGroup == 2) {
    if (debug_tmem_dump != 0) {
      // Only dump from the first tile/group to avoid races/overwrites.
      if (group_idx == 0 && tile_m == 0 && tile_n == 0 &&
          (debug_tmem_only_rank < 0 || cluster_rank == debug_tmem_only_rank)) {
        // TMEM loads operate on 32 DP lanes at a single column. Keep `col` uniform within the warp
        // and use the lane id to select DP lanes to avoid misaligned/invalid addressing.
        constexpr int DUMP_DP = 128;
        constexpr int DUMP_COL = 128;
        // Allow probing multiple TMEM subpartitions by adjusting the high idx bits.
        // TMEM pointers are encoded as {col:16, dp:8, idx:8}.
        const uint32_t idx_add_u = (debug_tmem_idx_add > 0) ? static_cast<uint32_t>(debug_tmem_idx_add) : 0u;
        const uint32_t tmem_dump_base = static_cast<uint32_t>(tmem_c_rank + (idx_add_u << 24));
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int warps_per_cta = static_cast<int>(blockDim.x) >> 5;
        for (int dp_base = warp * 32; dp_base < DUMP_DP; dp_base += warps_per_cta * 32) {
          const int dp = dp_base + lane;
          if (dp >= DUMP_DP) {
            continue;
          }
          for (int col = 0; col < DUMP_COL; ++col) {
            const uint32_t addr =
                tcgen05::tmem_addr_add(tmem_dump_base, static_cast<uint32_t>(dp), static_cast<uint32_t>(col));
            const uint32_t bits = tcgen05::tmem_ld_32dp32b_x1(addr);
            const float f = __uint_as_float(bits);
            const half h = __float2half_rn(f);
            const int gm = m_offset + dp;
            const int gn = n_offset + col;
            if (gm < m_size && gn < n_size) {
              c_out[static_cast<size_t>(gm) * static_cast<size_t>(n_size) + static_cast<size_t>(gn)] = h;
            }
          }
        }
      }

      __syncthreads();
      cluster.sync();
      if (threadIdx.x < 32) {
        tcgen05::tmem_dealloc<CtaGroup>(tmem_base, /*num_columns=*/TMEM_COLUMNS);
        tcgen05::tmem_relinquish_alloc_permit<CtaGroup>();
      }
      return;
    }
  }

  if constexpr (DEBUG_STAGE == 2) {
    __syncthreads();
    if (threadIdx.x < 32) {
      tcgen05::tmem_dealloc<CtaGroup>(tmem_base, /*num_columns=*/TMEM_COLUMNS);
      tcgen05::tmem_relinquish_alloc_permit<CtaGroup>();
    }
    return;
  }

  if constexpr (DEBUG_STAGE == 3 || DEBUG_STAGE == 4) {
    __syncthreads();
    if (threadIdx.x < 32) {
      tcgen05::tmem_dealloc<CtaGroup>(tmem_base, /*num_columns=*/TMEM_COLUMNS);
      tcgen05::tmem_relinquish_alloc_permit<CtaGroup>();
    }
    return;
  }

  // Epilogue: load accumulator tile from TMEM and store to global memory.
  // UMMA accumulators are FP32 in TMEM; convert to FP16 on store.
  const int warps_per_block = (static_cast<int>(blockDim.x) >> 5) > 0 ? (static_cast<int>(blockDim.x) >> 5) : 1;
  // For our 256x128 cluster tile (M_MMA=128 per CTA), the UMMA_2SM accumulator layout is
  // the simple 4x1 datapath atom: (m_local, n) -> dp=m_local, col=n.
  const int cta_rows = CTA_TILE_M;
  const int cta_row_base = 0;
  for (int row_chunk = warp; row_chunk * 32 < cta_rows; row_chunk += warps_per_block) {
    const int row_start = m_offset + cta_row_base + row_chunk * 32;
    // `tcgen05.ld.32dp...` is a warp-level operation: all lanes must participate.
    // If the entire warp maps to out-of-bounds rows, we can skip the loads safely.
    if (row_start >= m_size) {
      continue;
    }
    const int m_local = row_chunk * 32 + lane;
    const int gm = row_start + lane;
    const bool row_in_bounds = (gm < m_size);
    const size_t base = row_in_bounds ? (static_cast<size_t>(gm) * static_cast<size_t>(n_size)) : 0u;
    const uint32_t dp_lane = static_cast<uint32_t>(m_local);
    // Vectorized TMEM loads of FP32 accumulators. Use x8.b32 and explicitly convert FP32->FP16
    // to avoid interpreting FP32 bits as packed FP16 (which would corrupt results).
#pragma unroll
    for (int u = 0; u < UnrollN; ++u) {
      if ((tile_n + u) >= n_tiles_group) {
        continue;
      }
      const int n_offset_u = n_offset + u * TILE_N;
      for (int n_base = 0; n_base < TILE_N; n_base += 8) {
        const uint32_t col_lane = static_cast<uint32_t>(n_base);
        const uint32_t addr = tcgen05::tmem_addr_add(tmem_c_tiles[u], dp_lane, col_lane);
        uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
        tcgen05::tmem_ld_32dp32b_x8(addr, v0, v1, v2, v3, v4, v5, v6, v7);

        if (!row_in_bounds) {
          continue;
        }

        const int gn_base = n_offset_u + n_base;
        // Fast path: full 8-column block in-bounds (common for competition shapes).
        if (gn_base + 7 < n_size) {
          // Store as half2 for 32-bit wide stores. Alignment is guaranteed because N tiles are 128-wide.
          __half2* out_h2 = reinterpret_cast<__half2*>(c_out + base + static_cast<size_t>(gn_base));
          const float f0 = __uint_as_float(v0);
          const float f1 = __uint_as_float(v1);
          const float f2 = __uint_as_float(v2);
          const float f3 = __uint_as_float(v3);
          const float f4 = __uint_as_float(v4);
          const float f5 = __uint_as_float(v5);
          const float f6 = __uint_as_float(v6);
          const float f7 = __uint_as_float(v7);
          out_h2[0] = __floats2half2_rn(f0, f1);
          out_h2[1] = __floats2half2_rn(f2, f3);
          out_h2[2] = __floats2half2_rn(f4, f5);
          out_h2[3] = __floats2half2_rn(f6, f7);
        } else {
          // Tail path: store individual half elements.
          uint16_t* out_u16 = reinterpret_cast<uint16_t*>(c_out + base);
          const uint32_t vs[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int gn = gn_base + i;
            if (gn < n_size) {
              const float f = __uint_as_float(vs[i]);
              const half h = __float2half_rn(f);
              out_u16[static_cast<size_t>(gn)] = reinterpret_cast<const uint16_t&>(h);
            }
          }
        }
      }
    }
  }

  __syncthreads();
  if constexpr (CtaGroup == 2) {
    cluster.sync();
  }
  if (threadIdx.x < 32) {
    tcgen05::tmem_dealloc<CtaGroup>(tmem_base, /*num_columns=*/TMEM_COLUMNS);
    tcgen05::tmem_relinquish_alloc_permit<CtaGroup>();
  }
}

}  // namespace

void nvfp4_group_gemm_v2_forward_grouped_tcgen05_cuda(
    torch::Tensor a_ptrs,
    torch::Tensor b_ptrs,
    torch::Tensor sfa_ptrs,
    torch::Tensor sfb_ptrs,
    torch::Tensor c_ptrs,
    torch::Tensor m_sizes,
    torch::Tensor n_sizes,
    torch::Tensor k_halves,
    torch::Tensor k_scales,
    torch::Tensor a_descs,
    torch::Tensor b_descs,
    torch::Tensor sfa_descs,
    torch::Tensor sfb_descs,
    torch::Tensor cta_group_idx_map,
    torch::Tensor cta_tile_m_map,
    torch::Tensor cta_tile_n_map,
    int max_m_size,
    int max_n_size) {
  TORCH_CHECK(a_ptrs.is_cuda(), "a_ptrs must be CUDA tensor");
  TORCH_CHECK(b_ptrs.is_cuda(), "b_ptrs must be CUDA tensor");
  TORCH_CHECK(sfa_ptrs.is_cuda(), "sfa_ptrs must be CUDA tensor");
  TORCH_CHECK(sfb_ptrs.is_cuda(), "sfb_ptrs must be CUDA tensor");
  TORCH_CHECK(c_ptrs.is_cuda(), "c_ptrs must be CUDA tensor");
  TORCH_CHECK(m_sizes.is_cuda(), "m_sizes must be CUDA tensor");
  TORCH_CHECK(n_sizes.is_cuda(), "n_sizes must be CUDA tensor");
  TORCH_CHECK(k_halves.is_cuda(), "k_halves must be CUDA tensor");
  TORCH_CHECK(k_scales.is_cuda(), "k_scales must be CUDA tensor");
  TORCH_CHECK(a_descs.is_cuda(), "a_descs must be CUDA tensor");
  TORCH_CHECK(b_descs.is_cuda(), "b_descs must be CUDA tensor");
  TORCH_CHECK(sfa_descs.is_cuda(), "sfa_descs must be CUDA tensor");
  TORCH_CHECK(sfb_descs.is_cuda(), "sfb_descs must be CUDA tensor");
  TORCH_CHECK(cta_group_idx_map.is_cuda(), "cta_group_idx_map must be CUDA tensor");
  TORCH_CHECK(cta_tile_m_map.is_cuda(), "cta_tile_m_map must be CUDA tensor");
  TORCH_CHECK(cta_tile_n_map.is_cuda(), "cta_tile_n_map must be CUDA tensor");

  TORCH_CHECK(a_ptrs.scalar_type() == torch::kInt64, "a_ptrs must be torch.int64");
  TORCH_CHECK(b_ptrs.scalar_type() == torch::kInt64, "b_ptrs must be torch.int64");
  TORCH_CHECK(sfa_ptrs.scalar_type() == torch::kInt64, "sfa_ptrs must be torch.int64");
  TORCH_CHECK(sfb_ptrs.scalar_type() == torch::kInt64, "sfb_ptrs must be torch.int64");
  TORCH_CHECK(c_ptrs.scalar_type() == torch::kInt64, "c_ptrs must be torch.int64");
  TORCH_CHECK(m_sizes.scalar_type() == torch::kInt, "m_sizes must be torch.int32");
  TORCH_CHECK(n_sizes.scalar_type() == torch::kInt, "n_sizes must be torch.int32");
  TORCH_CHECK(k_halves.scalar_type() == torch::kInt, "k_halves must be torch.int32");
  TORCH_CHECK(k_scales.scalar_type() == torch::kInt, "k_scales must be torch.int32");
  TORCH_CHECK(a_descs.scalar_type() == torch::kInt64, "a_descs must be torch.int64");
  TORCH_CHECK(b_descs.scalar_type() == torch::kInt64, "b_descs must be torch.int64");
  TORCH_CHECK(sfa_descs.scalar_type() == torch::kInt64, "sfa_descs must be torch.int64");
  TORCH_CHECK(sfb_descs.scalar_type() == torch::kInt64, "sfb_descs must be torch.int64");
  TORCH_CHECK(cta_group_idx_map.scalar_type() == torch::kInt, "cta_group_idx_map must be torch.int32");
  TORCH_CHECK(cta_tile_m_map.scalar_type() == torch::kInt, "cta_tile_m_map must be torch.int32");
  TORCH_CHECK(cta_tile_n_map.scalar_type() == torch::kInt, "cta_tile_n_map must be torch.int32");

  TORCH_CHECK(a_descs.dim() == 2 && a_descs.size(1) == 16, "a_descs must be [groups,16] int64");
  TORCH_CHECK(b_descs.dim() == 2 && b_descs.size(1) == 16, "b_descs must be [groups,16] int64");
  TORCH_CHECK(sfa_descs.dim() == 2 && sfa_descs.size(1) == 16, "sfa_descs must be [groups,16] int64");
  TORCH_CHECK(sfb_descs.dim() == 2 && sfb_descs.size(1) == 16, "sfb_descs must be [groups,16] int64");

  const int groups = static_cast<int>(m_sizes.numel());
  TORCH_CHECK(groups > 0, "grouped call requires at least one group");
  TORCH_CHECK(a_ptrs.numel() == groups, "a_ptrs length mismatch");
  TORCH_CHECK(b_ptrs.numel() == groups, "b_ptrs length mismatch");
  TORCH_CHECK(sfa_ptrs.numel() == groups, "sfa_ptrs length mismatch");
  TORCH_CHECK(sfb_ptrs.numel() == groups, "sfb_ptrs length mismatch");
  TORCH_CHECK(c_ptrs.numel() == groups, "c_ptrs length mismatch");
  TORCH_CHECK(n_sizes.numel() == groups, "n_sizes length mismatch");
  TORCH_CHECK(k_halves.numel() == groups, "k_halves length mismatch");
  TORCH_CHECK(k_scales.numel() == groups, "k_scales length mismatch");
  TORCH_CHECK(a_descs.size(0) == groups, "a_descs groups mismatch");
  TORCH_CHECK(b_descs.size(0) == groups, "b_descs groups mismatch");
  TORCH_CHECK(sfa_descs.size(0) == groups, "sfa_descs groups mismatch");
  TORCH_CHECK(sfb_descs.size(0) == groups, "sfb_descs groups mismatch");

  TORCH_CHECK(max_m_size > 0, "max_m_size must be > 0");
  TORCH_CHECK(max_n_size > 0, "max_n_size must be > 0");

  const dim3 block(128, 1, 1);
  const int n_tiles = ceil_div_int(max_n_size, 128);
  const int m_tiles_1sm = ceil_div_int(max_m_size, 128);
  // The experimental cta_group::2 kernel operates on 256-row cluster tiles (2x 128-row CTAs).
  const int m_tiles_2sm = ceil_div_int(max_m_size, 256);
  auto parse_env_int = [](const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
      return default_value;
    }
    return std::atoi(value);
  };
  auto parse_env_int_optional = [](const char* name, int* out) -> bool {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
      return false;
    }
    *out = std::atoi(value);
    return true;
  };
  const int cta2_desc_a_row_offset_rows =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_CTA2_A_ROW_OFFSET", 64);
  const int cta2_desc_b_row_offset_rows =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_CTA2_B_ROW_OFFSET", 64);
  const int cta2_desc_sfa_row_offset_rows =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_CTA2_SFA_ROW_OFFSET", 0);
  const int cta2_epilogue_row_base_rows =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_CTA2_EPILOGUE_ROW_BASE", 64);
  const int cta2_epilogue_addr_mode =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_CTA2_EPILOGUE_ADDR_MODE", 0);
  const int cta2_sfb_slot_mode =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_CTA2_SFB_SLOT_MODE", 0);
  const int cta2_tmem_c_word_offset =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_CTA2_TMEM_C_WORD_OFFSET", 0);
  int cta2_tmem_sf_word_offset = 128;
  const bool cta2_tmem_sf_word_offset_set =
      parse_env_int_optional("AISP_NVFP4_GROUP_GEMM_V2_CTA2_TMEM_SF_WORD_OFFSET", &cta2_tmem_sf_word_offset);
  int cta2_tmem_sf_rank_word_offset = 32;
  const bool cta2_tmem_sf_rank_word_offset_set =
      // For the CUTLASS-aligned scale layout, cta_group::2 reserves 128 columns per rank:
      //   64 cols SFA + 64 cols SFB.
      parse_env_int_optional("AISP_NVFP4_GROUP_GEMM_V2_CTA2_TMEM_SF_RANK_WORD_OFFSET", &cta2_tmem_sf_rank_word_offset);
  const int cta2_tsfa_word_offset =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_CTA2_TSFA_WORD_OFFSET", 0);
  const int cta2_tsfb_word_offset =
      // Bring-up knob for adjusting the SFB TMEM pointer passed to UMMA (in 32-bit word columns).
      // With the current per-rank TMEM windowing, the correct default is 0.
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_CTA2_TSFB_WORD_OFFSET", 0);
  const int debug_tmem_dump =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_DEBUG_TMEM_DUMP", 0);
  const int debug_tmem_only_rank =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_DEBUG_TMEM_ONLY_RANK", -1);
  const int debug_tmem_idx_add =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_DEBUG_TMEM_IDX_ADD", 0);
  const int cta2_partition_b =
      // For cta_group::2, default to partitioning B via a global N shift by N/2 (mode=1).
      // Each CTA loads an N/2 slice of B into shared.
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_CTA2_PARTITION_B", 1);
  const int debug_print_ptrs =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_DEBUG_PRINT_PTRS", 0);
  const int cta2_idesc_m_dim_override =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_CTA2_IDESC_M_DIM_OVERRIDE", 0);
  const int cta2_idesc_n_dim_override =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_CTA2_IDESC_N_DIM_OVERRIDE", 0);
  const int enable_tma_multicast_env =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_ENABLE_TMA_MULTICAST", 0);
  const int unroll_n = parse_env_int("AISP_NVFP4_GROUP_GEMM_V2_UNROLL_N", 1);
  TORCH_CHECK(unroll_n == 1 || unroll_n == 2,
              "AISP_NVFP4_GROUP_GEMM_V2_UNROLL_N must be 1 or 2, got ",
              unroll_n);

  #if NVFP4_GROUP_GEMM_V2_TMEM_COLUMNS != 512
  TORCH_CHECK(unroll_n == 1,
              "AISP_NVFP4_GROUP_GEMM_V2_UNROLL_N=2 requires NVFP4_GROUP_GEMM_V2_TMEM_COLUMNS=512. "
              "Rebuild the extension with AISP_NVFP4_GROUP_GEMM_V2_TMEM_COLUMNS=512 under a new "
              "AISP_NVFP4_GROUP_GEMM_V2_EXT_NAME.");
  #endif

  const char* cluster_env = std::getenv("AISP_NVFP4_GROUP_GEMM_V2_CLUSTER_DIM_X");
  int cluster_dim_x = 1;
  if (cluster_env != nullptr && cluster_env[0] != '\0') {
    const int parsed = std::atoi(cluster_env);
    if (parsed > 1) {
      cluster_dim_x = parsed;
    }
  }

  const uint64_t* a_ptrs_dev = reinterpret_cast<const uint64_t*>(a_ptrs.data_ptr<int64_t>());
  const uint64_t* b_ptrs_dev = reinterpret_cast<const uint64_t*>(b_ptrs.data_ptr<int64_t>());
  const uint64_t* sfa_ptrs_dev = reinterpret_cast<const uint64_t*>(sfa_ptrs.data_ptr<int64_t>());
  const uint64_t* sfb_ptrs_dev = reinterpret_cast<const uint64_t*>(sfb_ptrs.data_ptr<int64_t>());
  const uint64_t* c_ptrs_dev = reinterpret_cast<const uint64_t*>(c_ptrs.data_ptr<int64_t>());
  const uint64_t* a_descs_dev = reinterpret_cast<const uint64_t*>(a_descs.data_ptr<int64_t>());
  const uint64_t* b_descs_dev = reinterpret_cast<const uint64_t*>(b_descs.data_ptr<int64_t>());
  const uint64_t* sfa_descs_dev = reinterpret_cast<const uint64_t*>(sfa_descs.data_ptr<int64_t>());
  const uint64_t* sfb_descs_dev = reinterpret_cast<const uint64_t*>(sfb_descs.data_ptr<int64_t>());
  const int32_t* cta_group_idx_dev =
      (cta_group_idx_map.numel() > 0) ? cta_group_idx_map.data_ptr<int32_t>() : nullptr;
  const int32_t* cta_tile_m_dev =
      (cta_tile_m_map.numel() > 0) ? cta_tile_m_map.data_ptr<int32_t>() : nullptr;
  const int32_t* cta_tile_n_dev =
      (cta_tile_n_map.numel() > 0) ? cta_tile_n_map.data_ptr<int32_t>() : nullptr;

  // Opt-in to large shared memory per block for the double-buffered pipeline.
  int max_shared_optin = 0;
  AT_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, at::cuda::current_device()));

  if (cluster_dim_x > 1) {
    int cluster_launch_supported = 0;
    AT_CUDA_CHECK(cudaDeviceGetAttribute(
        &cluster_launch_supported, cudaDevAttrClusterLaunch, at::cuda::current_device()));
    TORCH_CHECK(cluster_launch_supported == 1,
                "Cluster launch requested via AISP_NVFP4_GROUP_GEMM_V2_CLUSTER_DIM_X=",
                cluster_dim_x,
                ", but cudaDevAttrClusterLaunch is not supported on this device.");

    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = cluster_dim_x;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;

    const char* cta2_env = std::getenv("AISP_NVFP4_GROUP_GEMM_V2_ENABLE_EXPERIMENTAL_CTA2");
    const bool enable_experimental_cta2 = (cta2_env != nullptr && cta2_env[0] != '\0' && std::atoi(cta2_env) != 0);
    const bool use_cta_group2 = (cluster_dim_x == 2) && enable_experimental_cta2;
    const int cta2_tmem_sf_word_offset_eff = cta2_tmem_sf_word_offset_set ? cta2_tmem_sf_word_offset : 0;
    const int cta2_tmem_sf_rank_word_offset_eff =
        cta2_tmem_sf_rank_word_offset_set ? cta2_tmem_sf_rank_word_offset : 128;
    int enable_tma_multicast = (enable_tma_multicast_env != 0) ? 1 : 0;
    if (use_cta_group2) {
      // cta_group::2 bring-up currently does not support TMA multicast.
      enable_tma_multicast = 0;
    }
    if (enable_tma_multicast != 0) {
      TORCH_CHECK(cluster_dim_x <= 16,
                  "AISP_NVFP4_GROUP_GEMM_V2_ENABLE_TMA_MULTICAST=1 requires cluster_dim_x <= 16 "
                  "(mask is 16-bit). Got cluster_dim_x=",
                  cluster_dim_x);
      const int n_ctas_n = ceil_div_int(n_tiles, unroll_n);
      TORCH_CHECK((n_ctas_n % cluster_dim_x) == 0,
                  "AISP_NVFP4_GROUP_GEMM_V2_ENABLE_TMA_MULTICAST=1 requires the N-CTA dimension divisible by "
                  "cluster_dim_x to avoid partial clusters (would deadlock cluster sync). "
                  "n_ctas_n=",
                  n_ctas_n,
                  " n_tiles=",
                  n_tiles,
                  " unroll_n=",
                  unroll_n,
                  " cluster_dim_x=",
                  cluster_dim_x);
    }
    const int n_ctas_n = ceil_div_int(n_tiles, unroll_n);
    const int clustered_grid_x =
        use_cta_group2 ? (n_ctas_n * cluster_dim_x) : (ceil_div_int(n_ctas_n, cluster_dim_x) * cluster_dim_x);
    const int grid_y = use_cta_group2 ? m_tiles_2sm : m_tiles_1sm;
    cfg.gridDim = dim3(static_cast<unsigned int>(clustered_grid_x),
                       static_cast<unsigned int>(grid_y),
                       static_cast<unsigned int>(groups));
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    if (use_cta_group2) {
      if (unroll_n == 2) {
        cudaFuncAttributes func_attr{};
        AT_CUDA_CHECK(
            cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_v2_tcgen05_kernel<2, 2>));
        const int max_dynamic_smem =
            (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
                ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
                : 0;
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_v2_tcgen05_kernel<2, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem));
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_v2_tcgen05_kernel<2, 2>, cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared));
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_v2_tcgen05_kernel<2, 2>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

        AT_CUDA_CHECK(cudaLaunchKernelEx(
            &cfg,
            nvfp4_group_gemm_v2_tcgen05_kernel<2, 2>,
            a_ptrs_dev,
            b_ptrs_dev,
            sfa_ptrs_dev,
            sfb_ptrs_dev,
            c_ptrs_dev,
            m_sizes.data_ptr<int32_t>(),
            n_sizes.data_ptr<int32_t>(),
            k_halves.data_ptr<int32_t>(),
            k_scales.data_ptr<int32_t>(),
            a_descs_dev,
            b_descs_dev,
            sfa_descs_dev,
            sfb_descs_dev,
            /*cta_group_idx_map=*/nullptr,
            /*cta_tile_m_map=*/nullptr,
            /*cta_tile_n_map=*/nullptr,
            cta2_desc_a_row_offset_rows,
            cta2_desc_b_row_offset_rows,
            cta2_desc_sfa_row_offset_rows,
            cta2_epilogue_row_base_rows,
            cta2_epilogue_addr_mode,
            cta2_sfb_slot_mode,
            cta2_tmem_c_word_offset,
            cta2_tmem_sf_word_offset_eff,
            cta2_tmem_sf_rank_word_offset_eff,
            cta2_tsfa_word_offset,
            cta2_tsfb_word_offset,
            debug_tmem_dump,
            debug_tmem_only_rank,
            debug_tmem_idx_add,
            cta2_partition_b,
            debug_print_ptrs,
            cta2_idesc_m_dim_override,
            cta2_idesc_n_dim_override,
            cluster_dim_x,
            enable_tma_multicast));
      } else {
        cudaFuncAttributes func_attr{};
        AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_v2_tcgen05_kernel<2, 1>));
        const int max_dynamic_smem =
            (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
                ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
                : 0;
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_v2_tcgen05_kernel<2, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem));
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_v2_tcgen05_kernel<2, 1>, cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared));
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_v2_tcgen05_kernel<2, 1>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

        AT_CUDA_CHECK(cudaLaunchKernelEx(
            &cfg,
            nvfp4_group_gemm_v2_tcgen05_kernel<2, 1>,
            a_ptrs_dev,
            b_ptrs_dev,
            sfa_ptrs_dev,
            sfb_ptrs_dev,
            c_ptrs_dev,
            m_sizes.data_ptr<int32_t>(),
            n_sizes.data_ptr<int32_t>(),
            k_halves.data_ptr<int32_t>(),
            k_scales.data_ptr<int32_t>(),
            a_descs_dev,
            b_descs_dev,
            sfa_descs_dev,
            sfb_descs_dev,
            /*cta_group_idx_map=*/nullptr,
            /*cta_tile_m_map=*/nullptr,
            /*cta_tile_n_map=*/nullptr,
            cta2_desc_a_row_offset_rows,
            cta2_desc_b_row_offset_rows,
            cta2_desc_sfa_row_offset_rows,
            cta2_epilogue_row_base_rows,
            cta2_epilogue_addr_mode,
            cta2_sfb_slot_mode,
            cta2_tmem_c_word_offset,
            cta2_tmem_sf_word_offset_eff,
            cta2_tmem_sf_rank_word_offset_eff,
            cta2_tsfa_word_offset,
            cta2_tsfb_word_offset,
            debug_tmem_dump,
            debug_tmem_only_rank,
            debug_tmem_idx_add,
            cta2_partition_b,
            debug_print_ptrs,
            cta2_idesc_m_dim_override,
            cta2_idesc_n_dim_override,
            cluster_dim_x,
            enable_tma_multicast));
      }
    } else {
      if (unroll_n == 2) {
#if NVFP4_GROUP_GEMM_V2_TMEM_COLUMNS == 512
        cudaFuncAttributes func_attr{};
        AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_v2_tcgen05_kernel<1, 2>));
        const int max_dynamic_smem =
            (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
                ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
                : 0;
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_v2_tcgen05_kernel<1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem));
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_v2_tcgen05_kernel<1, 2>, cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared));
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_v2_tcgen05_kernel<1, 2>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

        AT_CUDA_CHECK(cudaLaunchKernelEx(
            &cfg,
            nvfp4_group_gemm_v2_tcgen05_kernel<1, 2>,
            a_ptrs_dev,
            b_ptrs_dev,
            sfa_ptrs_dev,
            sfb_ptrs_dev,
            c_ptrs_dev,
            m_sizes.data_ptr<int32_t>(),
            n_sizes.data_ptr<int32_t>(),
            k_halves.data_ptr<int32_t>(),
            k_scales.data_ptr<int32_t>(),
            a_descs_dev,
            b_descs_dev,
            sfa_descs_dev,
            sfb_descs_dev,
            /*cta_group_idx_map=*/nullptr,
            /*cta_tile_m_map=*/nullptr,
            /*cta_tile_n_map=*/nullptr,
            cta2_desc_a_row_offset_rows,
            cta2_desc_b_row_offset_rows,
            cta2_desc_sfa_row_offset_rows,
            cta2_epilogue_row_base_rows,
            cta2_epilogue_addr_mode,
            cta2_sfb_slot_mode,
            cta2_tmem_c_word_offset,
            cta2_tmem_sf_word_offset,
            cta2_tmem_sf_rank_word_offset,
            cta2_tsfa_word_offset,
            cta2_tsfb_word_offset,
            debug_tmem_dump,
            debug_tmem_only_rank,
            debug_tmem_idx_add,
            cta2_partition_b,
            debug_print_ptrs,
            cta2_idesc_m_dim_override,
            cta2_idesc_n_dim_override,
            cluster_dim_x,
            enable_tma_multicast));
#else
        TORCH_CHECK(false, "Cluster-mode UnrollN=2 requires NVFP4_GROUP_GEMM_V2_TMEM_COLUMNS=512.");
#endif
      } else {
        cudaFuncAttributes func_attr{};
        AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_v2_tcgen05_kernel<1, 1>));
        const int max_dynamic_smem =
            (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
                ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
                : 0;
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_v2_tcgen05_kernel<1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem));
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_v2_tcgen05_kernel<1, 1>, cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared));
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_v2_tcgen05_kernel<1, 1>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

        AT_CUDA_CHECK(cudaLaunchKernelEx(
            &cfg,
            nvfp4_group_gemm_v2_tcgen05_kernel<1, 1>,
            a_ptrs_dev,
            b_ptrs_dev,
            sfa_ptrs_dev,
            sfb_ptrs_dev,
            c_ptrs_dev,
            m_sizes.data_ptr<int32_t>(),
            n_sizes.data_ptr<int32_t>(),
            k_halves.data_ptr<int32_t>(),
            k_scales.data_ptr<int32_t>(),
            a_descs_dev,
            b_descs_dev,
            sfa_descs_dev,
            sfb_descs_dev,
            /*cta_group_idx_map=*/nullptr,
            /*cta_tile_m_map=*/nullptr,
            /*cta_tile_n_map=*/nullptr,
            cta2_desc_a_row_offset_rows,
            cta2_desc_b_row_offset_rows,
            cta2_desc_sfa_row_offset_rows,
            cta2_epilogue_row_base_rows,
            cta2_epilogue_addr_mode,
            cta2_sfb_slot_mode,
            cta2_tmem_c_word_offset,
            cta2_tmem_sf_word_offset,
            cta2_tmem_sf_rank_word_offset,
            cta2_tsfa_word_offset,
            cta2_tsfb_word_offset,
            debug_tmem_dump,
            debug_tmem_only_rank,
            debug_tmem_idx_add,
            cta2_partition_b,
            debug_print_ptrs,
            cta2_idesc_m_dim_override,
            cta2_idesc_n_dim_override,
            cluster_dim_x,
            enable_tma_multicast));
      }
    }
    AT_CUDA_CHECK(cudaGetLastError());
    return;
  }

  // Default (no cluster): launch exactly the CTAs required by each group, matching GPU MODE's
  // packed-CTA mapping to avoid extra early-return blocks for small M/N groups.
  TORCH_CHECK(cta_group_idx_map.numel() > 0, "cta_group_idx_map must be non-empty for non-cluster launch");
  TORCH_CHECK(cta_tile_m_map.numel() == cta_group_idx_map.numel(), "cta_tile_m_map length mismatch");
  TORCH_CHECK(cta_tile_n_map.numel() == cta_group_idx_map.numel(), "cta_tile_n_map length mismatch");
  const int total_ctas = static_cast<int>(cta_group_idx_map.numel());
  const dim3 grid(1u, 1u, static_cast<unsigned int>(total_ctas));

#if NVFP4_GROUP_GEMM_V2_TMEM_COLUMNS == 512
  if (unroll_n == 2) {
    {
      cudaFuncAttributes func_attr{};
      AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_v2_tcgen05_kernel<1, 2>));
      const int max_dynamic_smem =
          (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
              ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
              : 0;
      AT_CUDA_CHECK(cudaFuncSetAttribute(
          nvfp4_group_gemm_v2_tcgen05_kernel<1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem));
    }
    AT_CUDA_CHECK(cudaFuncSetAttribute(
        nvfp4_group_gemm_v2_tcgen05_kernel<1, 2>, cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
    nvfp4_group_gemm_v2_tcgen05_kernel<1, 2><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        a_ptrs_dev,
        b_ptrs_dev,
        sfa_ptrs_dev,
        sfb_ptrs_dev,
        c_ptrs_dev,
        m_sizes.data_ptr<int32_t>(),
        n_sizes.data_ptr<int32_t>(),
        k_halves.data_ptr<int32_t>(),
        k_scales.data_ptr<int32_t>(),
        a_descs_dev,
        b_descs_dev,
        sfa_descs_dev,
        sfb_descs_dev,
        cta_group_idx_dev,
        cta_tile_m_dev,
        cta_tile_n_dev,
        cta2_desc_a_row_offset_rows,
        cta2_desc_b_row_offset_rows,
        cta2_desc_sfa_row_offset_rows,
        cta2_epilogue_row_base_rows,
        cta2_epilogue_addr_mode,
        cta2_sfb_slot_mode,
        cta2_tmem_c_word_offset,
        cta2_tmem_sf_word_offset,
        cta2_tmem_sf_rank_word_offset,
        cta2_tsfa_word_offset,
        cta2_tsfb_word_offset,
        debug_tmem_dump,
        debug_tmem_only_rank,
        debug_tmem_idx_add,
        cta2_partition_b,
        debug_print_ptrs,
        cta2_idesc_m_dim_override,
        cta2_idesc_n_dim_override,
        /*cluster_dim_x=*/1,
        /*enable_tma_multicast=*/0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#endif

  {
    cudaFuncAttributes func_attr{};
    AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_v2_tcgen05_kernel<1, 1>));
    const int max_dynamic_smem =
        (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
            ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
            : 0;
    AT_CUDA_CHECK(cudaFuncSetAttribute(
        nvfp4_group_gemm_v2_tcgen05_kernel<1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem));
  }
  AT_CUDA_CHECK(cudaFuncSetAttribute(
      nvfp4_group_gemm_v2_tcgen05_kernel<1, 1>, cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
  nvfp4_group_gemm_v2_tcgen05_kernel<1, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      a_ptrs_dev,
      b_ptrs_dev,
      sfa_ptrs_dev,
      sfb_ptrs_dev,
      c_ptrs_dev,
      m_sizes.data_ptr<int32_t>(),
      n_sizes.data_ptr<int32_t>(),
      k_halves.data_ptr<int32_t>(),
      k_scales.data_ptr<int32_t>(),
      a_descs_dev,
      b_descs_dev,
      sfa_descs_dev,
      sfb_descs_dev,
      cta_group_idx_dev,
      cta_tile_m_dev,
      cta_tile_n_dev,
      cta2_desc_a_row_offset_rows,
      cta2_desc_b_row_offset_rows,
      cta2_desc_sfa_row_offset_rows,
      cta2_epilogue_row_base_rows,
      cta2_epilogue_addr_mode,
      cta2_sfb_slot_mode,
      cta2_tmem_c_word_offset,
      cta2_tmem_sf_word_offset,
      cta2_tmem_sf_rank_word_offset,
      cta2_tsfa_word_offset,
      cta2_tsfb_word_offset,
      debug_tmem_dump,
      debug_tmem_only_rank,
      debug_tmem_idx_add,
      cta2_partition_b,
      debug_print_ptrs,
      cta2_idesc_m_dim_override,
      cta2_idesc_n_dim_override,
      /*cluster_dim_x=*/1,
      /*enable_tma_multicast=*/0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvfp4_group_gemm_v2_forward_grouped_cuda", &nvfp4_group_gemm_v2_forward_grouped_cuda);
  m.def("nvfp4_group_gemm_v2_build_ab_tma_descs_cuda", &build_ab_tma_descs_cuda);
  m.def("nvfp4_group_gemm_v2_build_scale_tma_descs_cuda", &build_scale_tma_descs_cuda);
  m.def("nvfp4_group_gemm_v2_forward_grouped_tcgen05_cuda", &nvfp4_group_gemm_v2_forward_grouped_tcgen05_cuda);
}
