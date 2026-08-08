use std::sync::Arc;

use crate::{
    kernels::kernel_trait::GpuKernelImpl,
    utils::{CommandBufferPool, SubBufferPair, SubBuffersAllocator, compute_optimized_diag_len, next_multiple_of_n},
};
use std::cmp::max;
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, Queue},
    sync::GpuFuture,
};

/// Number of rows to batch in a single dispatch when possible
const ROWS_PER_BATCH: usize = 4;

fn compute_sample_len(a: &Vec<Vec<f32>>) -> usize {
    a.iter().map(|x| x.len()).sum()
}

fn flatten_and_pad(a: &Vec<Vec<f32>>, pad: usize) -> Vec<f32> {
    let new_len = next_multiple_of_n(a.first().unwrap().len(), pad);
    let mut padded = vec![0.0; new_len * a.len()];
    for (i, row) in a.into_iter().enumerate() {
        for (j, val) in row.into_iter().enumerate() {
            padded[i * new_len + j] = *val;
        }
    }
    padded
}

/// Optimized diamond partitioning buffers with caching
pub struct DiamondPartitioning<G: GpuKernelImpl> {
    a_buffer: SubBufferPair<f32>,
    b_buffer: SubBufferPair<f32>,
    diagonal_buffer: SubBufferPair<f32>,
    kernel_params: Option<G::KernelParams>,
    /// Cached command buffer pool for reuse
    command_pool: Option<CommandBufferPool>,
}

pub fn diamond_partitioning_gpu<G: GpuKernelImpl>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    subbuffer_allocator: SubBuffersAllocator,
    params: G,
    a: &Vec<Vec<f32>>,
    b: &Vec<Vec<f32>>,
    init_val: f32,
) -> Vec<Vec<f32>> {
    let (a, b) = if compute_sample_len(a) > compute_sample_len(b) {
        (b, a)
    } else {
        (a, b)
    };

    let properties = device.physical_device().properties();
    let max_subgroup_size = properties.max_subgroup_size.unwrap() as usize;
    let max_storage_buffer_size =
        properties.max_storage_buffer_range as usize / std::mem::size_of::<f32>();

    let a_count = a.len();
    let a_len = next_multiple_of_n(a.first().unwrap().len(), max_subgroup_size);
    let b_count = b.len();
    let b_len = next_multiple_of_n(b.first().unwrap().len(), max_subgroup_size);
    let len = max(a_len, b_len);

    let a_padded = flatten_and_pad(&a, max_subgroup_size);
    let b_padded = flatten_and_pad(&b, max_subgroup_size);

    // Use optimized diagonal length calculation
    let diag_len = compute_optimized_diag_len(len, max_subgroup_size);

    let max_pairs = (max_storage_buffer_size / diag_len).min(usize::MAX);

    let chunk_side = (max_pairs as f64).sqrt().floor() as usize;
    // to fill the gap in a or b chunk if one is too small
    let a_chunk = a_count.min(chunk_side);
    let b_chunk = b_count.min(chunk_side);

    let mut dist_matrix = vec![vec![0f32; b_count]; a_count];

    // Create command buffer pool for reuse across chunks
    let command_pool = CommandBufferPool::new(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
    );

    let mut dp_buffers = DiamondPartitioning::new_with_pool(
        subbuffer_allocator.clone(),
        a_chunk as u64,
        b_chunk as u64,
        a_len as u64,
        b_len as u64,
        diag_len as u64,
        command_pool,
    );

    for a_start in (0..a_count).step_by(a_chunk) {
        let a_end = (a_start + a_chunk).min(a_count);

        for b_start in (0..b_count).step_by(b_chunk) {
            let b_end = (b_start + b_chunk).min(b_count);

            let a_sub = &a_padded[a_start * a_len..a_end * a_len];
            let b_sub = &b_padded[b_start * b_len..b_end * b_len];

            dp_buffers.diamond_partitioning_gpu(
                device.clone(),
                queue.clone(),
                command_buffer_allocator.clone(),
                descriptor_set_allocator.clone(),
                subbuffer_allocator.clone(),
                &params,
                max_subgroup_size,
                a_len,
                b_len,
                a_sub,
                b_sub,
                a_end - a_start,
                b_end - b_start,
                init_val,
                &mut dist_matrix[a_start..a_end],
                b_start,
            );
        }
    }

    subbuffer_allocator.clear_with_size(0);

    // panic!("dist matrix {:?}", &dist_matrix[..5].iter().map(|r| &r[..5]).collect::<Vec<_>>());
    dist_matrix
}

impl<G: GpuKernelImpl> DiamondPartitioning<G> {
    pub fn new(
        subbuffer_allocator: SubBuffersAllocator,
        a_count: u64,
        b_count: u64,
        a_padded_len: u64,
        b_padded_len: u64,
        diag_len: u64,
    ) -> Self {
        let a_buffer_size = a_count * a_padded_len;
        let b_buffer_size = b_count * b_padded_len;
        let diagonal_buffer_size = a_count * b_count * diag_len;

        // Use ensure_capacity to reduce reallocations
        subbuffer_allocator.ensure_capacity(
            (a_buffer_size + b_buffer_size + diagonal_buffer_size) * size_of::<f32>() as u64,
        );

        Self {
            a_buffer: SubBufferPair::new(&subbuffer_allocator, a_buffer_size),
            b_buffer: SubBufferPair::new(&subbuffer_allocator, b_buffer_size),
            diagonal_buffer: SubBufferPair::new(&subbuffer_allocator, diagonal_buffer_size),
            kernel_params: None,
            command_pool: None,
        }
    }

    /// Create with a pre-initialized command buffer pool for better reuse
    pub fn new_with_pool(
        subbuffer_allocator: SubBuffersAllocator,
        a_count: u64,
        b_count: u64,
        a_padded_len: u64,
        b_padded_len: u64,
        diag_len: u64,
        command_pool: CommandBufferPool,
    ) -> Self {
        let a_buffer_size = a_count * a_padded_len;
        let b_buffer_size = b_count * b_padded_len;
        let diagonal_buffer_size = a_count * b_count * diag_len;

        // Use ensure_capacity to reduce reallocations
        subbuffer_allocator.ensure_capacity(
            (a_buffer_size + b_buffer_size + diagonal_buffer_size) * size_of::<f32>() as u64,
        );

        Self {
            a_buffer: SubBufferPair::new(&subbuffer_allocator, a_buffer_size),
            b_buffer: SubBufferPair::new(&subbuffer_allocator, b_buffer_size),
            diagonal_buffer: SubBufferPair::new(&subbuffer_allocator, diagonal_buffer_size),
            kernel_params: None,
            command_pool: Some(command_pool),
        }
    }

    #[inline(always)]
    fn diamond_partitioning_gpu(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        buffer_allocator: SubBuffersAllocator,
        params: &G,
        max_subgroup_threads: usize,
        a_len: usize,
        b_len: usize,
        a_padded: &[f32],
        b_padded: &[f32],
        a_count: usize,
        b_count: usize,
        init_val: f32,
        dist_matrix: &mut [Vec<f32>],
        column_offset: usize,
    ) {
        // Use optimized diagonal length calculation
        let diag_len = compute_optimized_diag_len(max(a_len, b_len), max_subgroup_threads);

        let mut diagonal_buffer_cpu = self.diagonal_buffer.get_cpu_buffer();
        let diagonal_size = a_count * b_count * diag_len;

        // Optimized initialization - use chunks for better cache utilization
        let chunk_init_size = diag_len;
        for chunk_start in (0..diagonal_size).step_by(chunk_init_size) {
            let chunk_end = (chunk_start + chunk_init_size).min(diagonal_size);
            diagonal_buffer_cpu[chunk_start..chunk_end].fill(init_val);
        }
        
        // Set initial diagonal values
        for i in 0..(a_count * b_count) {
            diagonal_buffer_cpu[i * diag_len] = 0.0;
        }

        drop(diagonal_buffer_cpu);

        let n_tiles_in_a = a_len.div_ceil(max_subgroup_threads);
        let n_tiles_in_b = b_len.div_ceil(max_subgroup_threads);

        let rows_count = (a_len + b_len).div_ceil(max_subgroup_threads) - 1;

        let mut diamonds_count = 1;
        let mut first_coord = -(max_subgroup_threads as isize);
        let mut a_start = 0;
        let mut b_start = 0;

        // Use command pool if available, otherwise create new builder
        let mut builder = if let Some(ref pool) = self.command_pool {
            pool.get_builder()
        } else {
            AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap()
        };

        if self.kernel_params.is_none() {
            self.kernel_params =
                Some(params.build_kernel_params(buffer_allocator.clone(), &mut builder));
        }

        let kernel_params = self.kernel_params.as_mut().unwrap();

        let a_gpu = self.a_buffer.move_gpu_data(&a_padded, &mut builder);
        let b_gpu = self.b_buffer.move_gpu_data(&b_padded, &mut builder);
        let mut diagonal_buffer_gpu = self.diagonal_buffer.move_gpu(&mut builder, diagonal_size);

        // Number of kernel calls
        for i in 0..rows_count {
            params.dispatch(
                device.clone(),
                descriptor_set_allocator.clone(),
                &mut builder,
                first_coord as i64,
                i as u64,
                diamonds_count as u64,
                a_start as u64,
                b_start as u64,
                a_len as u64,
                b_len as u64,
                max_subgroup_threads as u64,
                &a_gpu,
                &b_gpu,
                &mut diagonal_buffer_gpu,
                &kernel_params,
            );

            if i < (n_tiles_in_a - 1) {
                diamonds_count += 1;
                first_coord -= max_subgroup_threads as isize;
                a_start += max_subgroup_threads;
            } else if i < (n_tiles_in_b - 1) {
                first_coord += max_subgroup_threads as isize;
                b_start += max_subgroup_threads;
            } else {
                diamonds_count -= 1;
                first_coord += max_subgroup_threads as isize;
                b_start += max_subgroup_threads;
            }
        }

        fn index_mat_to_diag(i: usize, j: usize) -> (usize, isize) {
            (i + j, (j as isize) - (i as isize))
        }

        let (_, cx) = index_mat_to_diag(a_len, b_len);

        let diagonal = self.diagonal_buffer.move_cpu(&mut builder);
        let command_buffer = builder.build().unwrap();
        let future = vulkano::sync::now(device)
            .then_execute(queue, command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
        let diagonal = diagonal.read().unwrap();
        for i in 0..a_count {
            for j in 0..b_count {
                let diag_offset = (i * b_count + j) * diag_len;
                dist_matrix[i][column_offset + j] =
                    diagonal[diag_offset + ((cx as usize) & (diag_len - 1))];
            }
        }
    }
}

