// Only ever constructed and read from inside the `#[spirv(compute(..))]` entry
// points that `warp_kernel_spec!` generates, and those exist only on the SPIR-V
// target. On the host build nothing references it, so gate the allow on the
// target rather than silencing it everywhere -- that way genuinely dead code
// here is still reported when compiling the shader.
#[cfg_attr(not(target_arch = "spirv"), allow(dead_code))]
pub struct GpuMatrix<'a> {
    diagonal: &'a mut [f32],
    diagonal_offset: u32,
    mask: u32,
}

// All indices here are deliberately 32-bit. `diag_offset` is signed and routinely
// negative; masking a negative with `mask` (always `2^n - 1`) selects the same low
// bits whether the wrap happens in 32 or 64 bits, so this is exactly equivalent to
// the previous `isize`/`usize` form -- minus the `Int64` capability that the 64-bit
// version forced onto every device. See `KernelConstants`.
#[cfg_attr(not(target_arch = "spirv"), allow(dead_code))]
impl GpuMatrix<'_> {
    #[inline(always)]
    fn get_diagonal_cell(&self, _diag_row: u32, diag_offset: i32) -> f32 {
        self.diagonal[(self.diagonal_offset + (diag_offset as u32 & self.mask)) as usize]
    }

    #[inline(always)]
    fn set_diagonal_cell(&mut self, _diag_row: u32, diag_offset: i32, value: f32) {
        self.diagonal[(self.diagonal_offset + (diag_offset as u32 & self.mask)) as usize] = value;
    }
}

macro_rules! warp_kernel_spec {
    ($(
        fn $name:ident[$impl_struct:ident](
            $a:ident[$a_offset:ident],
            $b:ident[$b_offset:ident],
            $i:ident,
            $j:ident,
            $x:ident,
            $y:ident,
            $z:ident,
            [$($param1:ident: $ty1:ty)?],
            [$($param2:ident: $ty2:ty)?],
            [$($param3:ident: $ty3:ty)?],
            [$($param4:ident: $ty4:ty)?],
            [$($vec5:ident: $ty5:ty)?]
        ) $body:block
    )*) => {
        $(
            pub mod $name {
                #[cfg(not(target_arch = "spirv"))]
                pub mod cpu {
                    use std::sync::Arc;
                    use vulkano::buffer::Subbuffer;
                    use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
                    use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
                    use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
                    use vulkano::device::Device;
                    use crate::{kernels::kernel_trait::{GpuKernelImpl}, utils::SubBuffersAllocator};
                    use vulkano::pipeline::{Pipeline, PipelineBindPoint};

                    pub struct $impl_struct {
                        $(pub $param1: $ty1,)?
                        $(pub $param2: $ty2,)?
                        $(pub $param3: $ty3,)?
                        $(pub $param4: $ty4,)?
                        $(pub $vec5:  Vec<$ty5>,)?
                    }

                    pub struct KernelParams {
                        $(pub $vec5:  Subbuffer<[$ty5]>)?
                    }

                    impl GpuKernelImpl for $impl_struct {

                        type KernelParams = KernelParams;

                        fn build_kernel_params(
                            &self,
                            _allocator: SubBuffersAllocator,
                            _builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                        ) -> Self::KernelParams {
                            $(
                                use crate::utils::SubBufferPair;
                                let buffers = SubBufferPair::new(&_allocator, self.$vec5.len() as u64);
                            )?
                            KernelParams {
                                $($vec5: buffers.move_gpu_data(&self.$vec5, _builder))?
                            }
                        }

                        fn dispatch(
                            &self,
                            device: Arc<Device>,
                            dsa: Arc<StandardDescriptorSetAllocator>,
                            builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                            first_coord: i64,
                            row: u64,
                            tile_count: u64,
                            a_start: u64,
                            b_start: u64,
                            a_len: u64,
                            b_len: u64,
                            a_stride: u64,
                            b_stride: u64,
                            max_subgroup_threads: u64,
                            a: &Subbuffer<[f32]>,
                            b: &Subbuffer<[f32]>,
                            diagonal: &mut Subbuffer<[f32]>,
                            _kernel_params: &Self::KernelParams,
                        ) {

                            let shader_name = concat!("kernels::", stringify!($name), "::batch_call");
                            // Series count comes from the *stride*: that is the physical row
                            // pitch of the flattened buffer. Dividing by the logical length
                            // would undercount whenever the series are padded.
                            let a_count = a.len() as u64 / a_stride;
                            let b_count = b.len() as u64 / b_stride;
                            let threads_count = (a_count * b_count * tile_count * max_subgroup_threads) as u32;
                            let diag_len = diagonal.len() as u64 / (a_count * b_count);


                            // One definition, used both to patch the shader's LocalSize and to
                            // size the dispatch below -- these must not drift apart.
                            let workgroup_size = crate::utils::compute_workgroup_size(
                                &device,
                                max_subgroup_threads as usize,
                            );

                            let pipeline = crate::shader_load::get_shader_entry_pipeline(device.clone(), shader_name, workgroup_size);
                            let layout = &pipeline.layout().set_layouts()[0];

                            let set = DescriptorSet::new(
                                dsa.clone(),
                                layout.clone(),
                                [
                                    WriteDescriptorSet::buffer(0, diagonal.clone()),
                                    WriteDescriptorSet::buffer(1, a.clone()),
                                    WriteDescriptorSet::buffer(2, b.clone()),
                                    $(WriteDescriptorSet::buffer(3, _kernel_params.$vec5.clone()),)?
                                ],
                                [],
                            )
                            .unwrap();

                            // The host keeps its arithmetic in u64/usize; the push-constant
                            // block is 32-bit (see `KernelConstants`). Every value here is an
                            // index or a length bounded by `max_storage_buffer_range`, which is
                            // itself a u32, so these casts cannot truncate -- asserted in debug.
                            debug_assert!(i32::try_from(first_coord).is_ok());
                            debug_assert!(
                                [row, tile_count, a_start, b_start, a_len, b_len, a_stride,
                                 b_stride, a_count, b_count, diag_len, max_subgroup_threads]
                                    .iter()
                                    .all(|v| u32::try_from(*v).is_ok())
                            );
                            let kernel_constants = super::KernelConstants {
                                    first_coord: first_coord as i32,
                                    row: row as u32,
                                    tile_count: tile_count as u32,
                                    a_start: a_start as u32,
                                    b_start: b_start as u32,
                                    a_len: a_len as u32,
                                    b_len: b_len as u32,
                                    a_stride: a_stride as u32,
                                    b_stride: b_stride as u32,
                                    a_count: a_count as u32,
                                    b_count: b_count as u32,
                                    diag_len: diag_len as u32,
                                    max_subgroup_threads: max_subgroup_threads as u32,
                                    $(param1: self.$param1,)?
                                    $(param2: self.$param2,)?
                                    $(param3: self.$param3,)?
                                    $(param4: self.$param4,)?
                            };

                            builder
                                .bind_pipeline_compute(pipeline.clone())
                                .unwrap()
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Compute,
                                    pipeline.layout().clone(),
                                    0,
                                    set,
                                )
                                .unwrap()
                                .push_constants(
                                    pipeline.layout().clone(),
                                    0,
                                    kernel_constants
                                )
                                .unwrap();

                            unsafe { builder.dispatch([threads_count.div_ceil(workgroup_size), 1u32, 1u32]) }.unwrap();
                        }
                    }
                }

                // 32-bit on purpose. These were `i64`/`u64`, which made the shader declare
                // `OpCapability Int64` and forced `shader_int64` on at device creation -- a
                // feature MoltenVK/Metal does not expose at all, so the crate could not run
                // on Apple. Every field is an index or a length bounded by
                // `max_storage_buffer_range` (a u32), so none of them ever needed 64 bits,
                // and 64-bit integer ALU is emulated as 32-bit pairs on essentially every
                // GPU -- this block is the wavefront inner loop's index math.
                //
                // The all-32-bit layout is also why there is no longer a trailing `_padding`
                // field: with `i32`/`u32`/`f32` members the struct is 4-aligned throughout,
                // so host `#[repr(C)]` offsets and the SPIR-V push-constant block offsets
                // coincide trivially, where the old `u64`-then-`f32` mix did not.
                #[derive(Clone, Copy, bytemuck::AnyBitPattern)]
                #[repr(C)]
                #[allow(unused)]
                pub struct KernelConstants {
                    first_coord: i32,
                    row: u32,
                    tile_count: u32,
                    a_start: u32,
                    b_start: u32,
                    a_len: u32,
                    b_len: u32,
                    a_stride: u32,
                    b_stride: u32,
                    a_count: u32,
                    b_count: u32,
                    diag_len: u32,
                    max_subgroup_threads: u32,
                    $(param1: $ty1,)?
                    $(param2: $ty2,)?
                    $(param3: $ty3,)?
                    $(param4: $ty4,)?
                }

                #[cfg(target_arch = "spirv")]
                // `num_traits::Float` used to be imported here for `.powi()`. With that gone
                // the kernels reach for no `num_traits` extension trait at all -- `.abs()`,
                // `.min()` and `.max()` resolve as inherent `f32` methods on this target.
                use spirv_std::{glam::UVec3, spirv};

                #[cfg(target_arch = "spirv")]
                #[inline(always)]
                fn warp_kernel_inner(
                    mut matrix: super::GpuMatrix,
                    active: bool,
                    d_offset: u32,
                    a_start: u32,
                    b_start: u32,
                    diag_mid: i32,
                    diag_count: u32,
                    warp: u32,
                    max_subgroup_threads: u32,
                    $a: &[f32],
                    $b: &[f32],
                    $a_offset: usize,
                    $b_offset: usize,
                    $($param1: $ty1,)?
                    $($param2: $ty2,)?
                    $($param3: $ty3,)?
                    $($param4: $ty4,)?
                    $($vec5: &[$ty5],)?
                ) {
                    let mut i = a_start;
                    let mut j = b_start;
                    let mut s = diag_mid;
                    let mut e = diag_mid;

                    // The trip count must be identical for every invocation that shares the
                    // barrier below (Workgroup execution scope). `diag_count` varies from
                    // diamond to diamond, so this is only sound because
                    // `utils::compute_workgroup_size` puts exactly one diamond in each
                    // workgroup -- every invocation here belongs to the same diamond and so
                    // computes the same bound. Widening the workgroup without revisiting
                    // this loop reintroduces undefined behaviour.
                    for d in 2..diag_count {
                        let k = (warp * 2) as i32 + s;
                        if active && k <= e {
                            let $i = i - warp;
                            let $j = j + warp;

                            let $x = matrix.get_diagonal_cell(d_offset + d - 1, k - 1);
                            let $y = matrix.get_diagonal_cell(d_offset + d - 2, k);
                            let $z = matrix.get_diagonal_cell(d_offset + d - 1, k + 1);


                            let value = {
                                $body
                            };

                            matrix.set_diagonal_cell(d_offset + d, k, value);
                        }
                        // Warp synchronize.
                        //
                        // The cells being exchanged here live in a *storage buffer*, whose
                        // SPIR-V memory-semantics bit is `UniformMemory`. The obvious helper,
                        // `workgroup_memory_barrier_with_group_sync()`, only declares
                        // `WorkgroupMemory` -- so it synchronized execution but placed no
                        // ordering at all on the writes this loop actually depends on. That
                        // worked purely by driver luck.
                        //
                        // Spelled out rather than using `all_memory_barrier_with_group_sync()`
                        // because that helper escalates the *memory scope* to `Device`; every
                        // participant here is in one workgroup, so `Workgroup` scope is both
                        // sufficient and cheaper. This is the SPIR-V equivalent of GLSL's
                        // `barrier(); memoryBarrierBuffer();` fused into one instruction.
                        spirv_std::arch::control_barrier::<
                            { spirv_std::memory::Scope::Workgroup as u32 },
                            { spirv_std::memory::Scope::Workgroup as u32 },
                            {
                                spirv_std::memory::Semantics::UNIFORM_MEMORY.bits()
                                    | spirv_std::memory::Semantics::WORKGROUP_MEMORY.bits()
                                    | spirv_std::memory::Semantics::ACQUIRE_RELEASE.bits()
                            },
                        >();

                        if d <= max_subgroup_threads {
                            i += 1;
                            s -= 1;
                            e += 1;
                        } else {
                            j += 1;
                            s += 1;
                            e -= 1;
                        }
                    }
                }

                #[cfg(target_arch = "spirv")]
                #[inline(always)]
                fn warp_kernel(
                    global_id: u32,
                    active: bool,
                    first_coord: i32,
                    row: u32,
                    a_start: u32,
                    b_start: u32,
                    a_len: u32,
                    b_len: u32,
                    max_subgroup_threads: u32,
                    diagonal: &mut [f32],
                    diagonal_offset: u32,
                    diagonal_len: u32,
                    $a: &[f32],
                    $b: &[f32],
                    $a_offset: usize,
                    $b_offset: usize,
                    $($param1: $ty1,)?
                    $($param2: $ty2,)?
                    $($param3: $ty3,)?
                    $($param4: $ty4,)?
                    $($vec5: &[$ty5],)?
                ) {
                    let warp_id: u32 = global_id % max_subgroup_threads;
                    let diamond_id = global_id / max_subgroup_threads;

                    // There used to be an `if diamond_id >= tile_count { return; }` here.
                    // It was dead code -- this function is called with
                    // `instance_id = global_id % (tile_count * max_subgroup_threads)`, so
                    // `diamond_id` is always below `tile_count` -- and had it ever fired it
                    // would have been worse than useless: returning early puts the barrier
                    // below in divergent control flow, which is undefined for a Workgroup
                    // execution scope. Excess invocations are handled by `active` instead,
                    // which gates the *writes* while every invocation still reaches every
                    // barrier. See `batch_call`.

                    let diag_start = first_coord + ((diamond_id * max_subgroup_threads) as i32) * 2;
                    let d_a_start = a_start - diamond_id * max_subgroup_threads;
                    let d_b_start = b_start + diamond_id * max_subgroup_threads;

                    let alen = a_len - d_a_start;
                    let blen = b_len - d_b_start;

                    let matrix = super::GpuMatrix {
                        diagonal,
                        diagonal_offset,
                        mask: diagonal_len - 1,
                    };

                    warp_kernel_inner(
                        matrix,
                        active,
                        row * max_subgroup_threads,
                        d_a_start,
                        d_b_start,
                        diag_start + (max_subgroup_threads as i32),
                        (max_subgroup_threads * 2 + 1).min(alen + blen + 1),
                        warp_id,
                        max_subgroup_threads,
                        $a,
                        $b,
                        $a_offset,
                        $b_offset,
                        $($param1,)?
                        $($param2,)?
                        $($param3,)?
                        $($param4,)?
                        $($vec5,)?
                    );
                }

                #[cfg(target_arch = "spirv")]
                #[spirv(compute(threads(1)))]
                pub fn batch_call(
                    #[spirv(global_invocation_id)] global_id: UVec3,
                    #[spirv(push_constant)] constants: &KernelConstants,
                    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] diagonal: &mut [f32],
                    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] $a: &[f32],
                    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] $b: &[f32],
                    $(#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] vec5: &[$ty5],)?
                ) {

                    $(let $param1 = constants.param1;)?
                    $(let $param2 = constants.param2;)?
                    $(let $param3 = constants.param3;)?
                    $(let $param4 = constants.param4;)?
                    $(let $vec5 = vec5;)?


                    // `global_invocation_id` is natively u32; it used to be widened to u64
                    // here purely to match the old 64-bit constants block.
                    let global_id = global_id.x;
                    let threads_stride = constants.tile_count * constants.max_subgroup_threads;

                    // The dispatch rounds the thread count up to whole workgroups, so the
                    // trailing invocations address pairs that do not exist. Left unguarded
                    // they computed `diagonal_offset = pair_index * diag_len` past the end
                    // of the diagonal buffer and wrote there. Clamp the index so all
                    // addressing stays in range, and carry `active` so those invocations
                    // still reach every barrier but never store.
                    let pair_count = constants.a_count * constants.b_count;
                    let raw_pair_index = global_id / threads_stride;
                    let active = raw_pair_index < pair_count;
                    let pair_index = if active { raw_pair_index } else { 0 };
                    let instance_id = global_id % threads_stride;

                    let a_index = pair_index / constants.b_count;
                    let b_index = pair_index % constants.b_count;

                    let diagonal_offset = pair_index * constants.diag_len;

                    // Row pitch of the flattened buffer is the *stride* (length rounded up
                    // to a whole tile), not the logical length. Using `a_len` here would
                    // read into the previous series once the input needs padding.
                    let $a_offset = a_index as usize * constants.a_stride as usize;
                    let $b_offset = b_index as usize * constants.b_stride as usize;

                    warp_kernel(
                        instance_id,
                        active,
                        constants.first_coord,
                        constants.row,
                        constants.a_start,
                        constants.b_start,
                        constants.a_len,
                        constants.b_len,
                        constants.max_subgroup_threads,
                        diagonal,
                        diagonal_offset,
                        constants.diag_len,
                        $a,
                        $b,
                        $a_offset,
                        $b_offset,
                        $($param1,)?
                        $($param2,)?
                        $($param3,)?
                        $($param4,)?
                        $($vec5,)?
                    );
                }
            }
        )*
    };
}

#[cfg(not(target_arch = "spirv"))]
pub mod kernel_trait {
    use crate::utils::SubBuffersAllocator;
    use std::sync::Arc;
    use vulkano::buffer::Subbuffer;
    use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
    use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
    use vulkano::device::Device;

    pub trait GpuKernelImpl {
        type KernelParams;

        fn build_kernel_params(
            &self,
            allocator: SubBuffersAllocator,
            builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        ) -> Self::KernelParams;

        fn dispatch(
            &self,
            device: Arc<Device>,
            stsa: Arc<StandardDescriptorSetAllocator>,
            builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
            first_coord: i64,
            row: u64,
            tile_count: u64,
            a_start: u64,
            b_start: u64,
            // `a_len`/`b_len` are the true (unpadded) series lengths and bound the DP.
            // `a_stride`/`b_stride` are those lengths rounded up to a whole number of
            // tiles -- the row pitch of the flattened buffers. The padding exists only to
            // keep tile-aligned reads in bounds and must never be treated as data.
            a_len: u64,
            b_len: u64,
            a_stride: u64,
            b_stride: u64,
            max_subgroup_threads: u64,
            a: &Subbuffer<[f32]>,
            b: &Subbuffer<[f32]>,
            diagonal: &mut Subbuffer<[f32]>,
            kernel_params: &Self::KernelParams,
        );
    }
}

#[inline(always)]
fn min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}
#[inline(always)]
fn max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

const MSM_C: f32 = 1.0;
/// MSM split/merge cost: `c` when `x` lies between `y` and `z`, otherwise `c` plus the
/// distance to the nearer of them.
///
/// The second term was `x - max(z, x)`, which is `<= 0` for every input -- `max(z, x)` is
/// never below `x` -- so it could never contribute and the cost only ever penalised `x`
/// falling *below* both neighbours, never above. That made GPU MSM disagree with this
/// crate's own CPU backend (`tsdistances::utils::msm_cost_function`, which has always been
/// correct) on essentially every input. The identical typo in the test-local reference
/// implementation is why the existing tests ratified it instead of catching it.
#[inline(always)]
pub fn msm_cost_function(x: f32, y: f32, z: f32) -> f32 {
    MSM_C + max(max(min(y, z) - x, x - max(y, z)), 0.0)
}

warp_kernel_spec! {
    fn erp_distance[ERPImpl](a[a_offset], b[b_offset], i, j, x, y, z, [gap_penalty: f32], [], [], [], []) {
        (y + (a[a_offset + i as usize] - b[b_offset + j as usize]).abs())
        .min((z + (a[a_offset + i as usize] - gap_penalty).abs()).min(x + (b[b_offset + j as usize] - gap_penalty).abs()))
    }
    fn lcss_distance[LCSSImpl](a[a_offset], b[b_offset], i, j, x, y, z, [epsilon: f32], [], [], [], []) {
        let dist = (a[a_offset + i as usize] - b[b_offset + j as usize]).abs();
        (dist <= epsilon) as i32 as f32 * (y + 1.0) + (dist > epsilon) as i32 as f32 * x.max(z)
    }
    fn dtw_distance[DTWImpl](a[a_offset], b[b_offset], i, j, x, y, z, [], [], [], [], []) {
        // NOT `.powi(2)`: rust-gpu lowers `powi` to `GLSL.std.450 Pow` (PR#518), and
        // `Pow(x, y)` is *undefined for x < 0* -- which this base is, about half the
        // time. See `msm_cost_function` above for the crate's no-transcendentals style.
        let diff = a[a_offset + i as usize] - b[b_offset + j as usize];
        let dist = diff * diff;
        dist + z.min(x.min(y))
    }
    fn wdtw_distance[WDTWImpl](a[a_offset], b[b_offset], i, j, x, y, z, [], [], [], [], [weights: f32]) {
        let diff = a[a_offset + i as usize] - b[b_offset + j as usize];
        let dist = diff * diff * weights[(i as i32 - j as i32).abs() as usize];
        dist + x.min(y.min(z))
    }
    fn msm_distance[MSMImpl](a[a_offset], b[b_offset], i, j, x, y, z, [], [], [], [], []) {
        (y + (a[a_offset + i as usize] - b[b_offset + j as usize]).abs())
        .min(
            z + super::msm_cost_function(a[a_offset + i as usize], if i == 0 {0.0} else {a[a_offset + i as usize - 1]}, b[b_offset + j as usize]),
        )
        .min(
            x + super::msm_cost_function(b[b_offset + j as usize], a[a_offset + i as usize], if j == 0 {0.0} else {b[b_offset + j as usize - 1]}),
        )
    }
    fn twe_distance[TWEImpl](a[a_offset], b[b_offset], i, j, x, y, z, [stiffness: f32], [penalty: f32], [], [], []) {
        let delete_addition = penalty + stiffness;
        // deletion in a
        let del_a =
        z + (if i == 0 {0.0} else {a[a_offset + i as usize - 1]} - a[a_offset + i as usize]).abs() + delete_addition;

        // deletion in b
        let del_b =
            x + (if j == 0 {0.0} else {b[b_offset + j as usize - 1]} - b[b_offset + j as usize]).abs() + delete_addition;

        // match
        let match_current = (a[a_offset + i as usize] - b[b_offset + j as usize]).abs();
        let match_previous = (if i == 0 {0.0} else {a[a_offset + i as usize - 1]}
            - if j == 0 {0.0} else {b[b_offset + j as usize - 1]})
        .abs();
        let match_a_b = y
            + match_current
            + match_previous
            + stiffness * (2.0 * (i as isize - j as isize).abs() as f32);

        del_a.min(del_b.min(match_a_b))
    }
    fn adtw_distance[ADTWImpl](a[a_offset], b[b_offset], i, j, x, y, z, [w: f32], [], [], [], []) {
        let diff = a[a_offset + i as usize] - b[b_offset + j as usize];
        let dist = diff * diff;
                dist + (z + w).min((x + w).min(y))
    }
}
