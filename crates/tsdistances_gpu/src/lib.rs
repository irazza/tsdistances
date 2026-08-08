#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(unexpected_cfgs)]

pub mod kernels;

#[cfg(not(target_arch = "spirv"))]
mod shader_load;
#[cfg(not(target_arch = "spirv"))]
pub mod utils;
#[cfg(not(target_arch = "spirv"))]
pub mod warps;
#[cfg(not(target_arch = "spirv"))]
pub mod timing;

#[cfg(not(target_arch = "spirv"))]
pub mod cpu {
    use crate::kernels::adtw_distance::cpu::ADTWImpl;
    use crate::kernels::dtw_distance::cpu::DTWImpl;
    use crate::kernels::erp_distance::cpu::ERPImpl;
    use crate::kernels::lcss_distance::cpu::LCSSImpl;
    use crate::kernels::msm_distance::cpu::MSMImpl;
    use crate::kernels::twe_distance::cpu::TWEImpl;
    use crate::kernels::wdtw_distance::cpu::WDTWImpl;
    use crate::utils::SubBuffersAllocator;
    use crate::warps::diamond_partitioning_gpu;
    use std::cmp::min;
    use std::sync::Arc;

    use vulkano::device::Queue;
    use vulkano::{
        command_buffer::allocator::StandardCommandBufferAllocator,
        descriptor_set::allocator::StandardDescriptorSetAllocator, device::Device,
    };

    pub fn erp(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        sa: SubBuffersAllocator,
        a: &Vec<Vec<f32>>,
        b: &Vec<Vec<f32>>,
        gap_penalty: f32,
    ) -> Vec<Vec<f32>> {
        diamond_partitioning_gpu::<_>(
            device,
            queue,
            sba,
            dsa,
            sa,
            ERPImpl {
                gap_penalty: gap_penalty as f32,
            },
            a,
            b,
            f32::INFINITY,
        )
    }

    pub fn lcss(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        sa: SubBuffersAllocator,
        a: &Vec<Vec<f32>>,
        b: &Vec<Vec<f32>>,
        epsilon: f32,
    ) -> Vec<Vec<f32>> {
        let a_len = a.first().unwrap().len();
        let b_len = b.first().unwrap().len();
        let similarity = diamond_partitioning_gpu::<_>(
            device,
            queue,
            sba,
            dsa,
            sa,
            LCSSImpl { epsilon },
            a,
            b,
            0.0,
        );
        let min_len = min(a_len, b_len) as f32;
        similarity
            .iter()
            .map(|row| row.iter().map(|&s| 1.0 - s / min_len).collect::<Vec<f32>>())
            .collect::<Vec<Vec<f32>>>()
    }

    pub fn dtw(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        sa: SubBuffersAllocator,
        a: &Vec<Vec<f32>>,
        b: &Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        diamond_partitioning_gpu::<_>(device, queue, sba, dsa, sa, DTWImpl {}, a, b, f32::INFINITY)
    }

    pub fn wdtw(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        sa: SubBuffersAllocator,
        a: &Vec<Vec<f32>>,
        b: &Vec<Vec<f32>>,
        weights: &[f32],
    ) -> Vec<Vec<f32>> {
        diamond_partitioning_gpu::<_>(
            device,
            queue,
            sba,
            dsa,
            sa,
            WDTWImpl {
                weights: weights.to_vec(),
            },
            a,
            b,
            f32::INFINITY,
        )
    }

    pub fn msm(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        sa: SubBuffersAllocator,
        a: &Vec<Vec<f32>>,
        b: &Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        diamond_partitioning_gpu::<_>(device, queue, sba, dsa, sa, MSMImpl {}, a, b, f32::INFINITY)
    }

    pub fn twe(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        sa: SubBuffersAllocator,
        a: &Vec<Vec<f32>>,
        b: &Vec<Vec<f32>>,
        stiffness: f32,
        penalty: f32,
    ) -> Vec<Vec<f32>> {
        diamond_partitioning_gpu::<_>(
            device,
            queue,
            sba,
            dsa,
            sa,
            TWEImpl { stiffness, penalty },
            a,
            b,
            f32::INFINITY,
        )
    }

    pub fn adtw(
        device: Arc<Device>,
        queue: Arc<Queue>,
        sba: Arc<StandardCommandBufferAllocator>,
        dsa: Arc<StandardDescriptorSetAllocator>,
        sa: SubBuffersAllocator,
        a: &Vec<Vec<f32>>,
        b: &Vec<Vec<f32>>,
        w: f32,
    ) -> Vec<Vec<f32>> {
        diamond_partitioning_gpu::<_>(
            device,
            queue,
            sba,
            dsa,
            sa,
            ADTWImpl { w },
            a,
            b,
            f32::INFINITY,
        )
    }
}
