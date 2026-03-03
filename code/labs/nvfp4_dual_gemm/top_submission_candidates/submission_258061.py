import argparse
from typing import Type, Tuple, Union

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import from_dlpack

from functools import partial
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

#### COMPETITION SPECIFIC IMPORTS & SETTINGS
from task import input_t, output_t
from cutlass.cute.runtime import make_ptr

# FP4 data type for A and B
ab_dtype = cutlass.Float4E2M1FN
# FP8 data type for scale factors
sf_dtype = cutlass.Float8E4M3FN
# FP16 output type
c_dtype = cutlass.Float16
# Scale factor block size (16 elements share one scale)
sf_vec_size = 16
####
#### https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-eviction-priority-hints
_TMA_CACHE_EVICT_NORMAL = 0x1000000000000000
_TMA_CACHE_EVICT_FIRST = 0x12F0000000000000
_TMA_CACHE_EVICT_LAST = 0x14F0000000000000


@dsl_user_op
def tanh(a: float | cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


fadd2 = partial(cute.arch.add_packed_f32x2, ftz=False, rnd=nvvm.FPRoundingMode.RN)
fmul2 = partial(cute.arch.mul_packed_f32x2, ftz=False, rnd=nvvm.FPRoundingMode.RN)
ffma2 = partial(cute.arch.fma_packed_f32x2, ftz=False, rnd=nvvm.FPRoundingMode.RN)


class Sm100BlockScaledPersistentDualDenseGemmKernel:
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        prefetch_dist: Union[int, None] = None,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel with TMA prefetch support.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator, always set to Float32
            - sf_vec_size: Scalefactor A/B vector size.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        3. TMA Prefetch:
            - prefetch_dist: Prefetch distance for TMA operations.
              None = use num_ab_stage (default), 0 = disable prefetch, >0 = explicit distance.

        :param sf_vec_size: Scalefactor vector size.
        :type sf_vec_size: int
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param prefetch_dist: Prefetch distance for TMA operations (None=auto, 0=disable, >0=explicit).
        :type prefetch_dist: Union[int, None]
        """

        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        # Prefetch configuration: None=auto (num_ab_stage), 0=disable, >0=explicit distance
        self.prefetch_dist_param = prefetch_dist

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        # Set barrier id for epilogue sync and tmem ptr sync
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        """
        # Compute mma instruction shapes
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        # Compute number of TMEM columns for SFA/SFB/Accumulator
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (
            self.cta_tile_shape_mnk[0] // sf_atom_mn
        ) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (
            self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn
        ) * mma_inst_tile_k

        # Set prefetch distance for both initial and rolling prefetch (unified control)
        # None = use num_ab_stage (default), 0 = disable prefetch, >0 = explicit distance
        if self.prefetch_dist_param is None:
            self.prefetch_dist = self.num_ab_stage
        else:
            self.prefetch_dist = self.prefetch_dist_param

        # Check if prefetch is enabled (prefetch_dist > 0)
        self.prefetch_enabled = self.prefetch_dist > 0

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b1_ptr: cute.Pointer,
        b2_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb1_ptr: cute.Pointer,
        sfb2_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        epilogue_op: cutlass.Constexpr = lambda x: x
        * (1.0 / (1.0 + cute.math.exp(-x, fastmath=True))),  # Silu default
    ):
        m, n, k, l = problem_size  # noqa: E741
        self.m, self.n, self.k, self.l = m, n, k, l
        # Tensors
        a_tensor = cute.make_tensor(
            a_ptr, cute.make_layout((m, k, l), stride=(k, 1, m * k))
        )
        b_tensor1 = cute.make_tensor(
            b1_ptr, cute.make_layout((n, k, l), stride=(k, 1, n * k))
        )
        b_tensor2 = cute.make_tensor(
            b2_ptr, cute.make_layout((n, k, l), stride=(k, 1, n * k))
        )
        c_tensor = cute.make_tensor(
            c_ptr, cute.make_layout((m, n, l), stride=(n, 1, m * n))
        )

        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor1.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sf_dtype
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor1).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor1.shape, self.sf_vec_size
        )
        sfb_tensor1 = cute.make_tensor(sfb1_ptr, sfb_layout)
        sfb_tensor2 = cute.make_tensor(sfb2_ptr, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b1, tma_tensor_b1 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor1,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_b2, tma_tensor_b2 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor2,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb1, tma_tensor_sfb1 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor1,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb2, tma_tensor_sfb2 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor2,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb1.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb1.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb1.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb1.shape[1],
                tma_tensor_sfb1.shape[2],
            )
            # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb1.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb1.stride[1],
                tma_tensor_sfb1.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb1 = cute.make_tensor(
                tma_tensor_sfb1.iterator, tma_tensor_sfb_new_layout
            )
            tma_tensor_sfb2 = cute.make_tensor(
                tma_tensor_sfb2.iterator, tma_tensor_sfb_new_layout
            )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size * 2 + sfa_copy_size + sfb_copy_size * 2
        ) * atom_thr_size

        # Setup TMA store for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB1: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB2: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB1: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB2: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b1,
            tma_tensor_b1,
            tma_atom_b2,
            tma_tensor_b2,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb1,
            tma_tensor_sfb1,
            tma_atom_sfb2,
            tma_tensor_sfb2,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b1: cute.CopyAtom,
        mB_nkl1: cute.Tensor,
        tma_atom_b2: cute.CopyAtom,
        mB_nkl2: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb1: cute.CopyAtom,
        mSFB_nkl1: cute.Tensor,
        tma_atom_sfb2: cute.CopyAtom,
        mSFB_nkl2: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b1)
            cpasync.prefetch_descriptor(tma_atom_b2)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb1)
            cpasync.prefetch_descriptor(tma_atom_sfb2)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        # mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        mma_tile_coord_v = bidx & 1  # NOTE: Assume 2 CTA
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # num_acc_consumer_threads = len(self.epilog_warp_id) * (
        #    2 if use_2cta_instrs else 1
        # )
        num_acc_consumer_threads = 8
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(
            cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True
        )  # NOTE: No issues reported by compute sanitizer when outcomment

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB1 = storage.sB1.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sB2 = storage.sB2.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB1 = storage.sSFB1.get_tensor(sfb_smem_layout_staged)
        sSFB2 = storage.sSFB2.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl1 = cute.local_tile(
            mB_nkl1, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gB_nkl2 = cute.local_tile(
            mB_nkl2, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl1 = cute.local_tile(
            mSFB_nkl1,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        gSFB_nkl2 = cute.local_tile(
            mSFB_nkl2,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB1 = thr_mma.partition_B(gB_nkl1)
        tCgB2 = thr_mma.partition_B(gB_nkl2)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB1 = thr_mma_sfb.partition_B(gSFB_nkl1)
        tCgSFB2 = thr_mma_sfb.partition_B(gSFB_nkl2)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB1, tBgB1 = cpasync.tma_partition(
            tma_atom_b1,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB1, 0, 3),
            cute.group_modes(tCgB1, 0, 3),
        )
        tBsB2, tBgB2 = cpasync.tma_partition(
            tma_atom_b2,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB2, 0, 3),
            cute.group_modes(tCgB2, 0, 3),
        )

        #  TMA load SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB1, tBgSFB1 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb1,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB1, 0, 3),
            cute.group_modes(tCgSFB1, 0, 3),
        )
        tBsSFB1 = cute.filter_zeros(tBsSFB1)
        tBgSFB1 = cute.filter_zeros(tBgSFB1)
        tBsSFB2, tBgSFB2 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb2,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB2, 0, 3),
            cute.group_modes(tCgSFB2, 0, 3),
        )
        tBsSFB2 = cute.filter_zeros(tBsSFB2)
        tBgSFB2 = cute.filter_zeros(tBgSFB2)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB1 = tiled_mma.make_fragment_B(sB1)
        tCrB2 = tiled_mma.make_fragment_B(sB2)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE) # NOTE: STAGE == 1 always for dual gemm
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(
            cluster_shape_mn=self.cluster_shape_mn
        )  # NOTE: No issues reported by compute sanitizer when outcomment

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    # cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[0] >> 1,
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice1 = tBgB1[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]
                tBgB_slice2 = tBgB2[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]

                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                # ((atom_v, rest_v), RestK)
                tBgSFB_slice1 = tBgSFB1[(None, slice_n, None, mma_tile_coord_mnl[2])]
                tBgSFB_slice2 = tBgSFB2[(None, slice_n, None, mma_tile_coord_mnl[2])]

                #
                # Prefetch: Initial batch of prefetches to prime the pipeline
                #
                if self.prefetch_enabled:
                    for pf_k_tile in cutlass.range(
                        0, min(self.prefetch_dist, k_tile_cnt)
                    ):
                        cute.prefetch(
                            tma_atom_a,
                            tAgA_slice[(None, pf_k_tile)],
                        )
                        cute.prefetch(
                            tma_atom_b1,
                            tBgB_slice1[(None, pf_k_tile)],
                        )
                        cute.prefetch(
                            tma_atom_b2,
                            tBgB_slice2[(None, pf_k_tile)],
                        )
                        cute.prefetch(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, pf_k_tile)],
                        )
                        cute.prefetch(
                            tma_atom_sfb1,
                            tBgSFB_slice1[(None, pf_k_tile)],
                        )
                        cute.prefetch(
                            tma_atom_sfb2,
                            tBgSFB_slice2[(None, pf_k_tile)],
                        )

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )
                #
                # Tma load loop
                #
                UNROLL_TMA = 3 if cutlass.const_expr(self.m == 256) else 1
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=UNROLL_TMA):
                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # TMA load A/B/SFA/SFB
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                        cache_policy=cutlass.Int64(
                            cutlass.Int64(_TMA_CACHE_EVICT_FIRST).ir_value()
                        ),
                    )
                    cute.copy(
                        tma_atom_b1,
                        tBgB_slice1[(None, ab_producer_state.count)],
                        tBsB1[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                        cache_policy=cutlass.Int64(
                            cutlass.Int64(_TMA_CACHE_EVICT_FIRST).ir_value()
                        ),
                    )
                    cute.copy(
                        tma_atom_b2,
                        tBgB_slice2[(None, ab_producer_state.count)],
                        tBsB2[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                        cache_policy=cutlass.Int64(
                            cutlass.Int64(_TMA_CACHE_EVICT_FIRST).ir_value()
                        ),
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                        cache_policy=cutlass.Int64(
                            cutlass.Int64(_TMA_CACHE_EVICT_FIRST).ir_value()
                        ),
                    )
                    cute.copy(
                        tma_atom_sfb1,
                        tBgSFB_slice1[(None, ab_producer_state.count)],
                        tBsSFB1[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                        cache_policy=cutlass.Int64(
                            cutlass.Int64(_TMA_CACHE_EVICT_FIRST).ir_value()
                        ),
                    )
                    cute.copy(
                        tma_atom_sfb2,
                        tBgSFB_slice2[(None, ab_producer_state.count)],
                        tBsSFB2[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                        cache_policy=cutlass.Int64(
                            cutlass.Int64(_TMA_CACHE_EVICT_FIRST).ir_value()
                        ),
                    )

                    # Prefetch: Rolling prefetch for next tiles
                    if self.prefetch_enabled:
                        if k_tile < k_tile_cnt - self.prefetch_dist:
                            future_k_tile = ab_producer_state.count + self.prefetch_dist
                            cute.prefetch(
                                tma_atom_a,
                                tAgA_slice[(None, future_k_tile)],
                            )
                            cute.prefetch(
                                tma_atom_b1,
                                tBgB_slice1[(None, future_k_tile)],
                            )
                            cute.prefetch(
                                tma_atom_b2,
                                tBgB_slice2[(None, future_k_tile)],
                            )
                            cute.prefetch(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, future_k_tile)],
                            )
                            cute.prefetch(
                                tma_atom_sfb1,
                                tBgSFB_slice1[(None, future_k_tile)],
                            )
                            cute.prefetch(
                                tma_atom_sfb2,
                                tBgSFB_slice2[(None, future_k_tile)],
                            )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # Make accumulator tmem tensor
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base1 = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            acc_offset = tcgen05.find_tmem_tensor_col_offset(tCtAcc_base1)
            acc_tmem_ptr1 = cute.recast_ptr(
                acc_tmem_ptr + acc_offset,
                dtype=cutlass.Float32,
            )
            tCtAcc_base2 = cute.make_tensor(acc_tmem_ptr1, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + 2 * acc_offset,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr1 = cute.recast_ptr(
                acc_tmem_ptr + 2 * acc_offset + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout1 = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB1 = cute.make_tensor(sfb_tmem_ptr1, tCtSFB_layout1)
            sfb_tmem_ptr2 = cute.recast_ptr(
                acc_tmem_ptr
                + 2 * acc_offset
                + self.num_sfa_tmem_cols
                + self.num_sfb_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB2 = cute.make_tensor(sfb_tmem_ptr2, tCtSFB_layout1)
            #
            # Partition for S2T copy of SFA/SFB
            #
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t1,
                tCtSFB_compact_s2t1,
            ) = self.mainloop_s2t_copy_and_partition(sSFB1, tCtSFB1)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t2,
                tCtSFB_compact_s2t2,
            ) = self.mainloop_s2t_copy_and_partition(sSFB2, tCtSFB2)
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                acc_stage_index = acc_producer_state.index

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc1 = tCtAcc_base1[(None, None, None, acc_stage_index)]
                tCtAcc2 = tCtAcc_base2[(None, None, None, acc_stage_index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tCtSFB_mma1 = tCtSFB1
                tCtSFB_mma2 = tCtSFB2
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    offset = (
                        cutlass.Int32(2)
                        if mma_tile_coord_mnl[1] % 2 == 1
                        else cutlass.Int32(0)
                    )
                    shifted_ptr1 = cute.recast_ptr(
                        acc_tmem_ptr + 2 * acc_offset + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma1 = cute.make_tensor(shifted_ptr1, tCtSFB_layout1)
                    shifted_ptr2 = cute.recast_ptr(
                        acc_tmem_ptr
                        + 2 * acc_offset
                        + self.num_sfa_tmem_cols
                        + self.num_sfb_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma2 = cute.make_tensor(shifted_ptr2, tCtSFB_layout1)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr1 = cute.recast_ptr(
                        acc_tmem_ptr + 2 * acc_offset + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma1 = cute.make_tensor(shifted_ptr1, tCtSFB_layout1)
                    shifted_ptr2 = cute.recast_ptr(
                        acc_tmem_ptr
                        + 2 * acc_offset
                        + self.num_sfa_tmem_cols
                        + self.num_sfb_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma2 = cute.make_tensor(shifted_ptr2, tCtSFB_layout1)

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged1 = tCsSFB_compact_s2t1[
                            s2t_stage_coord
                        ]
                        tCsSFB_compact_s2t_staged2 = tCsSFB_compact_s2t2[
                            s2t_stage_coord
                        ]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged1,
                            tCtSFB_compact_s2t1,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged2,
                            tCtSFB_compact_s2t2,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma1[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc1,
                                tCrA[kblock_coord],
                                tCrB1[kblock_coord],
                                tCtAcc1,
                            )

                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma2[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc2,
                                tCrA[kblock_coord],
                                tCrB2[kblock_coord],
                                tCtAcc2,
                            )

                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)
        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base1 = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            acc_offset = tcgen05.find_tmem_tensor_col_offset(tCtAcc_base1)
            acc_tmem_ptr1 = cute.recast_ptr(
                acc_tmem_ptr + acc_offset,
                dtype=cutlass.Float32,
            )
            tCtAcc_base2 = cute.make_tensor(acc_tmem_ptr1, tCtAcc_fake.layout)
            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            (
                tiled_copy_t2r,
                tTR_tAcc_base1,
                tTR_rAcc1,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base1, tCgC, epi_tile, use_2cta_instrs
            )
            (
                tiled_copy_t2r,
                tTR_tAcc_base2,
                tTR_rAcc2,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base2, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc1.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC, epi_tile, sC
            )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]

                # Get accumulator stage index
                acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc1 = tTR_tAcc_base1[
                    (None, None, None, None, None, acc_stage_index)
                ]
                tTR_tAcc2 = tTR_tAcc_base2[
                    (None, None, None, None, None, acc_stage_index)
                ]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc1 = cute.group_modes(tTR_tAcc1, 3, cute.rank(tTR_tAcc1))
                tTR_tAcc2 = cute.group_modes(tTR_tAcc2, 3, cute.rank(tTR_tAcc2))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc1.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                UNROLL_FULL = (
                    True
                    if cutlass.const_expr(
                        (self.m == 256 and self.n == 4096)
                        or (self.m == 512 and self.n == 3072)
                    )
                    else False
                )
                for subtile_idx in cutlass.range(subtile_cnt, unroll_full=UNROLL_FULL):
                    real_subtile_idx = subtile_idx
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn1 = tTR_tAcc1[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn1, tTR_rAcc1)
                    tTR_tAcc_mn2 = tTR_tAcc2[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn2, tTR_rAcc2)

                    #
                    # Convert to C type
                    #
                    # FUSION: Silu(Acc1) * Acc2
                    x = tiled_copy_r2s.retile(tTR_rAcc1).load()
                    y = tiled_copy_r2s.retile(tTR_rAcc2).load()

                    NUM_ELEMS_PER_THREAD = 32  # NOTE: Could adjust epi tiler to do less/more but it does not help performance.
                    acc_res = cute.make_rmem_tensor(
                        cute.make_layout(NUM_ELEMS_PER_THREAD), dtype=cutlass.Float32
                    )
                    half_x_0, half_x_1 = fmul2((x[0], x[1]), (0.5, 0.5))
                    half_x_2, half_x_3 = fmul2((x[2], x[3]), (0.5, 0.5))
                    half_x_4, half_x_5 = fmul2((x[4], x[5]), (0.5, 0.5))
                    half_x_6, half_x_7 = fmul2((x[6], x[7]), (0.5, 0.5))
                    half_x_8, half_x_9 = fmul2((x[8], x[9]), (0.5, 0.5))
                    half_x_10, half_x_11 = fmul2((x[10], x[11]), (0.5, 0.5))
                    half_x_12, half_x_13 = fmul2((x[12], x[13]), (0.5, 0.5))
                    half_x_14, half_x_15 = fmul2((x[14], x[15]), (0.5, 0.5))
                    half_x_16, half_x_17 = fmul2((x[16], x[17]), (0.5, 0.5))
                    half_x_18, half_x_19 = fmul2((x[18], x[19]), (0.5, 0.5))
                    half_x_20, half_x_21 = fmul2((x[20], x[21]), (0.5, 0.5))
                    half_x_22, half_x_23 = fmul2((x[22], x[23]), (0.5, 0.5))
                    half_x_24, half_x_25 = fmul2((x[24], x[25]), (0.5, 0.5))
                    half_x_26, half_x_27 = fmul2((x[26], x[27]), (0.5, 0.5))
                    half_x_28, half_x_29 = fmul2((x[28], x[29]), (0.5, 0.5))
                    half_x_30, half_x_31 = fmul2((x[30], x[31]), (0.5, 0.5))

                    if cutlass.const_expr(self.m == 512 and self.n == 4096):
                        tanh_0, tanh_1 = (
                            cute.math.tanh(half_x_0, fastmath=True),
                            cute.math.tanh(half_x_1, fastmath=True),
                        )
                        tanh_2, tanh_3 = (
                            cute.math.tanh(half_x_2, fastmath=True),
                            cute.math.tanh(half_x_3, fastmath=True),
                        )
                        tanh_4, tanh_5 = (
                            cute.math.tanh(half_x_4, fastmath=True),
                            cute.math.tanh(half_x_5, fastmath=True),
                        )
                        tanh_6, tanh_7 = (
                            cute.math.tanh(half_x_6, fastmath=True),
                            cute.math.tanh(half_x_7, fastmath=True),
                        )
                        tanh_8, tanh_9 = (
                            cute.math.tanh(half_x_8, fastmath=True),
                            cute.math.tanh(half_x_9, fastmath=True),
                        )
                        tanh_10, tanh_11 = (
                            cute.math.tanh(half_x_10, fastmath=True),
                            cute.math.tanh(half_x_11, fastmath=True),
                        )
                        tanh_12, tanh_13 = (
                            cute.math.tanh(half_x_12, fastmath=True),
                            cute.math.tanh(half_x_13, fastmath=True),
                        )
                        tanh_14, tanh_15 = (
                            cute.math.tanh(half_x_14, fastmath=True),
                            cute.math.tanh(half_x_15, fastmath=True),
                        )
                        tanh_16, tanh_17 = (
                            cute.math.tanh(half_x_16, fastmath=True),
                            cute.math.tanh(half_x_17, fastmath=True),
                        )
                        tanh_18, tanh_19 = (
                            cute.math.tanh(half_x_18, fastmath=True),
                            cute.math.tanh(half_x_19, fastmath=True),
                        )
                        tanh_20, tanh_21 = (
                            cute.math.tanh(half_x_20, fastmath=True),
                            cute.math.tanh(half_x_21, fastmath=True),
                        )
                        tanh_22, tanh_23 = (
                            cute.math.tanh(half_x_22, fastmath=True),
                            cute.math.tanh(half_x_23, fastmath=True),
                        )
                        tanh_24, tanh_25 = (
                            cute.math.tanh(half_x_24, fastmath=True),
                            cute.math.tanh(half_x_25, fastmath=True),
                        )
                        tanh_26, tanh_27 = (
                            cute.math.tanh(half_x_26, fastmath=True),
                            cute.math.tanh(half_x_27, fastmath=True),
                        )
                        tanh_28, tanh_29 = (
                            cute.math.tanh(half_x_28, fastmath=True),
                            cute.math.tanh(half_x_29, fastmath=True),
                        )
                        tanh_30, tanh_31 = (
                            cute.math.tanh(half_x_30, fastmath=True),
                            cute.math.tanh(half_x_31, fastmath=True),
                        )
                    else:
                        tanh_0, tanh_1 = (
                            tanh(half_x_0),
                            tanh(half_x_1),
                        )
                        tanh_2, tanh_3 = (
                            tanh(half_x_2),
                            tanh(half_x_3),
                        )
                        tanh_4, tanh_5 = (
                            tanh(half_x_4),
                            tanh(half_x_5),
                        )
                        tanh_6, tanh_7 = (
                            tanh(half_x_6),
                            tanh(half_x_7),
                        )
                        tanh_8, tanh_9 = (
                            tanh(half_x_8),
                            tanh(half_x_9),
                        )
                        tanh_10, tanh_11 = (
                            tanh(half_x_10),
                            tanh(half_x_11),
                        )
                        tanh_12, tanh_13 = (
                            tanh(half_x_12),
                            tanh(half_x_13),
                        )
                        tanh_14, tanh_15 = (
                            tanh(half_x_14),
                            tanh(half_x_15),
                        )
                        tanh_16, tanh_17 = (
                            tanh(half_x_16),
                            tanh(half_x_17),
                        )
                        tanh_18, tanh_19 = (
                            tanh(half_x_18),
                            tanh(half_x_19),
                        )
                        tanh_20, tanh_21 = (
                            tanh(half_x_20),
                            tanh(half_x_21),
                        )
                        tanh_22, tanh_23 = (
                            tanh(half_x_22),
                            tanh(half_x_23),
                        )
                        tanh_24, tanh_25 = (
                            tanh(half_x_24),
                            tanh(half_x_25),
                        )
                        tanh_26, tanh_27 = (
                            tanh(half_x_26),
                            tanh(half_x_27),
                        )
                        tanh_28, tanh_29 = (
                            tanh(half_x_28),
                            tanh(half_x_29),
                        )
                        tanh_30, tanh_31 = (
                            tanh(half_x_30),
                            tanh(half_x_31),
                        )

                    # scaled = half_x * (1 + tanh) = half_x * tanh + half_x
                    scaled_0, scaled_1 = ffma2(
                        (half_x_0, half_x_1), (tanh_0, tanh_1), (half_x_0, half_x_1)
                    )
                    scaled_2, scaled_3 = ffma2(
                        (half_x_2, half_x_3), (tanh_2, tanh_3), (half_x_2, half_x_3)
                    )
                    scaled_4, scaled_5 = ffma2(
                        (half_x_4, half_x_5), (tanh_4, tanh_5), (half_x_4, half_x_5)
                    )
                    scaled_6, scaled_7 = ffma2(
                        (half_x_6, half_x_7), (tanh_6, tanh_7), (half_x_6, half_x_7)
                    )
                    scaled_8, scaled_9 = ffma2(
                        (half_x_8, half_x_9), (tanh_8, tanh_9), (half_x_8, half_x_9)
                    )
                    scaled_10, scaled_11 = ffma2(
                        (half_x_10, half_x_11),
                        (tanh_10, tanh_11),
                        (half_x_10, half_x_11),
                    )
                    scaled_12, scaled_13 = ffma2(
                        (half_x_12, half_x_13),
                        (tanh_12, tanh_13),
                        (half_x_12, half_x_13),
                    )
                    scaled_14, scaled_15 = ffma2(
                        (half_x_14, half_x_15),
                        (tanh_14, tanh_15),
                        (half_x_14, half_x_15),
                    )
                    scaled_16, scaled_17 = ffma2(
                        (half_x_16, half_x_17),
                        (tanh_16, tanh_17),
                        (half_x_16, half_x_17),
                    )
                    scaled_18, scaled_19 = ffma2(
                        (half_x_18, half_x_19),
                        (tanh_18, tanh_19),
                        (half_x_18, half_x_19),
                    )
                    scaled_20, scaled_21 = ffma2(
                        (half_x_20, half_x_21),
                        (tanh_20, tanh_21),
                        (half_x_20, half_x_21),
                    )
                    scaled_22, scaled_23 = ffma2(
                        (half_x_22, half_x_23),
                        (tanh_22, tanh_23),
                        (half_x_22, half_x_23),
                    )
                    scaled_24, scaled_25 = ffma2(
                        (half_x_24, half_x_25),
                        (tanh_24, tanh_25),
                        (half_x_24, half_x_25),
                    )
                    scaled_26, scaled_27 = ffma2(
                        (half_x_26, half_x_27),
                        (tanh_26, tanh_27),
                        (half_x_26, half_x_27),
                    )
                    scaled_28, scaled_29 = ffma2(
                        (half_x_28, half_x_29),
                        (tanh_28, tanh_29),
                        (half_x_28, half_x_29),
                    )
                    scaled_30, scaled_31 = ffma2(
                        (half_x_30, half_x_31),
                        (tanh_30, tanh_31),
                        (half_x_30, half_x_31),
                    )

                    acc_res[0], acc_res[1] = fmul2((scaled_0, scaled_1), (y[0], y[1]))
                    acc_res[2], acc_res[3] = fmul2((scaled_2, scaled_3), (y[2], y[3]))
                    acc_res[4], acc_res[5] = fmul2((scaled_4, scaled_5), (y[4], y[5]))
                    acc_res[6], acc_res[7] = fmul2((scaled_6, scaled_7), (y[6], y[7]))
                    acc_res[8], acc_res[9] = fmul2((scaled_8, scaled_9), (y[8], y[9]))
                    acc_res[10], acc_res[11] = fmul2(
                        (scaled_10, scaled_11), (y[10], y[11])
                    )
                    acc_res[12], acc_res[13] = fmul2(
                        (scaled_12, scaled_13), (y[12], y[13])
                    )
                    acc_res[14], acc_res[15] = fmul2(
                        (scaled_14, scaled_15), (y[14], y[15])
                    )
                    acc_res[16], acc_res[17] = fmul2(
                        (scaled_16, scaled_17), (y[16], y[17])
                    )
                    acc_res[18], acc_res[19] = fmul2(
                        (scaled_18, scaled_19), (y[18], y[19])
                    )
                    acc_res[20], acc_res[21] = fmul2(
                        (scaled_20, scaled_21), (y[20], y[21])
                    )
                    acc_res[22], acc_res[23] = fmul2(
                        (scaled_22, scaled_23), (y[22], y[23])
                    )
                    acc_res[24], acc_res[25] = fmul2(
                        (scaled_24, scaled_25), (y[24], y[25])
                    )
                    acc_res[26], acc_res[27] = fmul2(
                        (scaled_26, scaled_27), (y[26], y[27])
                    )
                    acc_res[28], acc_res[29] = fmul2(
                        (scaled_28, scaled_29), (y[28], y[29])
                    )
                    acc_res[30], acc_res[31] = fmul2(
                        (scaled_30, scaled_31), (y[30], y[31])
                    )

                    # acc_vec1 = 0.5 * x * y
                    # acc_vec2 = 0.5 * x * cute.math.tanh(0.5 * x, fastmath=True) * y
                    # acc_res = acc_vec1 + acc_vec2
                    tRS_rC.store(acc_res.load().to(self.c_dtype))

                    #
                    # Store C to shared memory
                    #
                    c_buffer = (num_prev_subtiles + real_subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    self.epilog_sync_barrier.arrive_and_wait()

                    #
                    # TMA store C to global memory
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, real_subtile_idx)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing (tma_atom_c, bSG_sC, bSG_gC) where:
            - tma_atom_c: The TMA copy atom
            - bSG_sC: The partitioned shared memory tensor C
            - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of Scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Scale factor vector size.
        :type sf_vec_size: int
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # ACC stages
        num_acc_stage = 1

        # Default C stages
        num_c_stage = 2

        # Calculate smem layout and size for one stage of A, B, SFA, SFB and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one) * 2  # Dual in B
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
            * 2  # Dual in SFB
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B/SFA/SFB stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, swizzle_size=2
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid


# --------------------------------------------------------------------------------------
# Compilation and Execution Interface
# --------------------------------------------------------------------------------------

_compiled_kernel_cache = {}


def compile_kernel(problem_size):
    global _compiled_kernel_cache
    if problem_size in _compiled_kernel_cache:
        return _compiled_kernel_cache[problem_size]

    m, n, k, l = problem_size  # noqa: E741
    # Create pointers for compiling (dummy pointers)
    a_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    b1_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    b2_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb1_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb2_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)

    m, n, k, l = problem_size  # noqa: E741
    mma_tiler_m = 256
    mma_tiler_n = 128 if m > 256 else 64
    cluster_m = 2
    cluster_n = 2
    mma_tiler_mn = (mma_tiler_m, mma_tiler_n)
    cluster_shape_mn = (cluster_m, cluster_n)
    prefetch_dist = 3 if m == 256 and n == 3072 else (1 if m == 512 else 0)
    gemm = Sm100BlockScaledPersistentDualDenseGemmKernel(
        sf_vec_size, mma_tiler_mn, cluster_shape_mn, prefetch_dist
    )

    max_active_clusters = 148
    _compiled_kernel_cache[problem_size] = cute.compile(
        gemm,
        a_ptr,
        b1_ptr,
        b2_ptr,
        sfa_ptr,
        sfb1_ptr,
        sfb2_ptr,
        c_ptr,
        problem_size,
        max_active_clusters,
        options="--opt-level 2",
    )

    return _compiled_kernel_cache[problem_size]


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled Persistent Dual GEMM kernel.
    Input args match better_baseline.py logic but mapped to input_t.
    """
    # Unpack based on input_t from baseline logic
    # data: (a, b1, b2, sfa_ref, sfb1_ref, sfb2_ref, sfa_permuted, sfb1_permuted, sfb2_permuted, c)
    a, b1, b2, _, _, _, sfa_permuted, sfb1_permuted, sfb2_permuted, c = data

    m, k, l = a.shape  # noqa: E741
    n, _, _ = b1.shape
    k = k * 2  # Torch uses e2m1_x2
    problem_size = m, n, k, l
    compiled_func = compile_kernel(problem_size)

    a_ptr = make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b1_ptr = make_ptr(ab_dtype, b1.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b2_ptr = make_ptr(ab_dtype, b2.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(
        sf_dtype, sfa_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb1_ptr = make_ptr(
        sf_dtype, sfb1_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb2_ptr = make_ptr(
        sf_dtype, sfb2_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    compiled_func(a_ptr, b1_ptr, b2_ptr, sfa_ptr, sfb1_ptr, sfb2_ptr, c_ptr)

    return c
