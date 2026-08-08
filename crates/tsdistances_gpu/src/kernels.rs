pub struct GpuMatrix<'a> {
    diagonal: &'a mut [f32],
    diagonal_offset: usize,
    mask: usize,
}

impl GpuMatrix<'_> {
    #[inline(always)]
    fn get_diagonal_cell(&self, _diag_row: usize, diag_offset: isize) -> f32 {
        self.diagonal[self.diagonal_offset + (diag_offset as usize & self.mask)]
    }

    #[inline(always)]
    fn set_diagonal_cell(&mut self, _diag_row: usize, diag_offset: isize, value: f32) {
        self.diagonal[self.diagonal_offset + (diag_offset as usize & self.mask)] = value;
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
                            max_subgroup_threads: u64,
                            a: &Subbuffer<[f32]>,
                            b: &Subbuffer<[f32]>,
                            diagonal: &mut Subbuffer<[f32]>,
                            _kernel_params: &Self::KernelParams,
                        ) {

                            let shader_name = concat!("kernels::", stringify!($name), "::batch_call");
                            let a_count = a.len() as u64 / a_len;
                            let b_count = b.len() as u64 / b_len;
                            let threads_count = (a_count * b_count * tile_count * max_subgroup_threads) as u32;
                            let diag_len = diagonal.len() as u64 / (a_count * b_count);


                            let pipeline = crate::shader_load::get_shader_entry_pipeline(device.clone(), shader_name);
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

                            let kernel_constants = super::KernelConstants {
                                    first_coord,
                                    row,
                                    tile_count,
                                    a_start,
                                    b_start,
                                    a_len,
                                    b_len,
                                    a_count,
                                    b_count,
                                    diag_len,
                                    max_subgroup_threads,
                                    $(param1: self.$param1,)?
                                    $(param2: self.$param2,)?
                                    $(param3: self.$param3,)?
                                    $(param4: self.$param4,)?
                                    _padding: 0,
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

                            let max_threads_x = device
                                .physical_device()
                                .properties()
                                .max_compute_work_group_size[0];

                            unsafe { builder.dispatch([threads_count.div_ceil(max_threads_x), 1u32, 1u32]) }.unwrap();
                        }
                    }
                }

                #[derive(Clone, Copy, bytemuck::AnyBitPattern)]
                #[repr(C)]
                #[allow(unused)]
                pub struct KernelConstants {
                    first_coord: i64,
                    row: u64,
                    tile_count: u64,
                    a_start: u64,
                    b_start: u64,
                    a_len: u64,
                    b_len: u64,
                    a_count: u64,
                    b_count: u64,
                    diag_len: u64,
                    max_subgroup_threads: u64,
                    $(param1: $ty1,)?
                    $(param2: $ty2,)?
                    $(param3: $ty3,)?
                    $(param4: $ty4,)?
                    _padding: u64
                }

                #[cfg(target_arch = "spirv")]
                use spirv_std::{glam::UVec3, spirv, num_traits::Float};

                #[cfg(target_arch = "spirv")]
                #[inline(always)]
                fn warp_kernel_inner(
                    mut matrix: super::GpuMatrix,
                    d_offset: u64,
                    a_start: u64,
                    b_start: u64,
                    diag_mid: i64,
                    diag_count: u64,
                    warp: u64,
                    max_subgroup_threads: u64,
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

                    for d in 2..diag_count {
                        let k = (warp * 2) as i64 + s;
                        if k <= e {
                            let $i = i - warp;
                            let $j = j + warp;

                            let $x = matrix.get_diagonal_cell((d_offset + d - 1) as usize, (k - 1) as isize);
                            let $y = matrix.get_diagonal_cell((d_offset + d - 2) as usize, k as isize);
                            let $z = matrix.get_diagonal_cell((d_offset + d - 1) as usize, (k + 1) as isize);


                            let value = {
                                $body
                            };

                            matrix.set_diagonal_cell((d_offset + d) as usize, k as isize, value);
                        }
                        // Warp synchronize
                        unsafe { spirv_std::arch::workgroup_memory_barrier_with_group_sync() };

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
                    global_id: u64,
                    first_coord: i64,
                    row: u64,
                    tile_count: u64,
                    a_start: u64,
                    b_start: u64,
                    a_len: u64,
                    b_len: u64,
                    max_subgroup_threads: u64,
                    diagonal: &mut [f32],
                    diagonal_offset: u64,
                    diagonal_len: u64,
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
                    let warp_id: u64 = global_id % max_subgroup_threads;
                    let diamond_id = global_id / max_subgroup_threads;

                    if diamond_id >= tile_count {
                        return;
                    }

                    let diag_start = first_coord + ((diamond_id * max_subgroup_threads) as i64) * 2;
                    let d_a_start = a_start - diamond_id * max_subgroup_threads;
                    let d_b_start = b_start + diamond_id * max_subgroup_threads;

                    let alen = a_len - d_a_start;
                    let blen = b_len - d_b_start;

                    let matrix = super::GpuMatrix {
                        diagonal,
                        diagonal_offset: diagonal_offset as usize,
                        mask: diagonal_len as usize - 1,
                    };

                    warp_kernel_inner(
                        matrix,
                        row * max_subgroup_threads,
                        d_a_start,
                        d_b_start,
                        diag_start + (max_subgroup_threads as i64),
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


                    let global_id = global_id.x as u64;
                    let threads_stride = constants.tile_count * constants.max_subgroup_threads;

                    let pair_index = global_id / threads_stride;
                    let instance_id = global_id % threads_stride;

                    let a_index = pair_index / constants.b_count as u64;
                    let b_index = pair_index % constants.b_count as u64;

                    let diagonal_offset = pair_index * constants.diag_len;

                    let $a_offset = a_index as usize * constants.a_len as usize;
                    let $b_offset = b_index as usize * constants.b_len as usize;

                    warp_kernel(
                        instance_id,
                        constants.first_coord,
                        constants.row,
                        constants.tile_count,
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
            a_len: u64,
            b_len: u64,
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
#[inline(always)]
pub fn msm_cost_function(x: f32, y: f32, z: f32) -> f32 {
    MSM_C + max(max(min(y, z) - x, x - max(z, x)), 0.0)
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
        let dist = (a[a_offset + i as usize] - b[b_offset + j as usize]).powi(2);
        dist + z.min(x.min(y))
    }
    fn wdtw_distance[WDTWImpl](a[a_offset], b[b_offset], i, j, x, y, z, [], [], [], [], [weights: f32]) {
        let dist = (a[a_offset + i as usize] - b[b_offset + j as usize]).powi(2) * weights[(i as i32 - j as i32).abs() as usize];
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
        let dist = (a[a_offset + i as usize] - b[b_offset + j as usize]).powi(2);
                dist + (z + w).min((x + w).min(y))
    }
}
