use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use vulkano::{
    VulkanLibrary,
    buffer::{
        BufferContents, BufferUsage, BufferWriteGuard, Subbuffer,
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
    },
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::{
        MemoryPropertyFlags,
        allocator::{MemoryTypeFilter, StandardMemoryAllocator},
    },
};

/// Threshold for direct upload via host-visible memory (in bytes)
const DIRECT_UPLOAD_THRESHOLD: usize = 64 * 1024; // 64KB

/// Cache line size for alignment
const CACHE_LINE_SIZE: u64 = 64;

#[macro_export]
macro_rules! assert_eq_with_tol {
    ($a:expr, $b:expr, $tol:expr) => {
        if ($a - $b).abs() > $tol {
            panic!(
                "assertion failed: `(left == right)`\n  left: `{:?}`\n right: `{:?}`",
                $a, $b
            );
        }
    };
    ($a:expr, $b:expr) => {
        assert_eq_with_tol!($a, $b, 1e-6);
    };
}

pub struct SubBuffersAllocator {
    gpu: Arc<SubbufferAllocator>,
    cpu: Arc<SubbufferAllocator>,
    /// Host-visible GPU memory for direct uploads (small data)
    host_visible: Arc<SubbufferAllocator>,
    current_size: Arc<std::sync::atomic::AtomicU64>,
    /// Buffer pool for reusing allocations
    buffer_pool: Arc<BufferPool>,
}

/// Buffer pool that caches allocations by size bucket
pub struct BufferPool {
    /// Cached buffers keyed by size bucket (rounded up to power of 2)
    cached: Mutex<HashMap<u64, Vec<CachedBuffer>>>,
    memory_allocator: Arc<StandardMemoryAllocator>,
}

struct CachedBuffer {
    gpu: Subbuffer<[f32]>,
    cpu: Subbuffer<[f32]>,
}

impl BufferPool {
    pub fn new(memory_allocator: Arc<StandardMemoryAllocator>) -> Self {
        Self {
            cached: Mutex::new(HashMap::new()),
            memory_allocator,
        }
    }

    /// Get a size bucket for caching (round up to nearest power of 2)
    fn size_bucket(size: u64) -> u64 {
        if size == 0 {
            return 0;
        }
        size.next_power_of_two()
    }

    /// Try to get a cached buffer of at least the requested size
    pub fn try_get(&self, min_size: u64) -> Option<(Subbuffer<[f32]>, Subbuffer<[f32]>)> {
        let bucket = Self::size_bucket(min_size);
        let mut cache = self.cached.lock().unwrap();
        if let Some(buffers) = cache.get_mut(&bucket) {
            if let Some(cached) = buffers.pop() {
                return Some((cached.gpu, cached.cpu));
            }
        }
        None
    }

    /// Return a buffer to the pool for reuse
    pub fn return_buffer(&self, gpu: Subbuffer<[f32]>, cpu: Subbuffer<[f32]>) {
        let bucket = Self::size_bucket(gpu.len());
        let mut cache = self.cached.lock().unwrap();
        let buffers = cache.entry(bucket).or_insert_with(Vec::new);
        // Limit cache size per bucket to prevent memory bloat
        if buffers.len() < 8 {
            buffers.push(CachedBuffer { gpu, cpu });
        }
    }

    /// Clear all cached buffers
    pub fn clear(&self) {
        let mut cache = self.cached.lock().unwrap();
        cache.clear();
    }
}

impl Clone for SubBuffersAllocator {
    fn clone(&self) -> Self {
        Self {
            gpu: self.gpu.clone(),
            cpu: self.cpu.clone(),
            host_visible: self.host_visible.clone(),
            current_size: self.current_size.clone(),
            buffer_pool: self.buffer_pool.clone(),
        }
    }
}

impl SubBuffersAllocator {
    /// Clear and resize the arena. Only resizes if the new size is larger or significantly smaller.
    /// This reduces allocation overhead for repeated calls with similar sizes.
    pub fn clear_with_size(&self, size: u64) -> () {
        let current = self.current_size.load(std::sync::atomic::Ordering::Relaxed);
        
        // Only resize if:
        // 1. New size is larger than current, OR
        // 2. New size is less than 25% of current (to reclaim memory)
        // 3. size is 0 (explicit cleanup)
        if size > current || size == 0 || (current > 0 && size < current / 4) {
            self.gpu.set_arena_size(size);
            self.cpu.set_arena_size(size);
            self.host_visible.set_arena_size(size.min(DIRECT_UPLOAD_THRESHOLD as u64 * 4));
            self.current_size.store(size, std::sync::atomic::Ordering::Relaxed);
        }
        
        // Clear buffer pool when size is 0
        if size == 0 {
            self.buffer_pool.clear();
        }
    }
    
    /// Ensure the allocator has at least the specified capacity.
    /// Pre-allocate with extra headroom to reduce future resizes.
    pub fn ensure_capacity(&self, required_size: u64) {
        let current = self.current_size.load(std::sync::atomic::Ordering::Relaxed);
        if required_size > current {
            // Allocate 1.5x the required size to reduce future resizes
            let new_size = required_size + required_size / 2;
            self.gpu.set_arena_size(new_size);
            self.cpu.set_arena_size(new_size);
            self.host_visible.set_arena_size((new_size / 4).min(DIRECT_UPLOAD_THRESHOLD as u64 * 8));
            self.current_size.store(new_size, std::sync::atomic::Ordering::Relaxed);
        }
    }
    
    /// Get the buffer pool for caching allocations
    pub fn buffer_pool(&self) -> &Arc<BufferPool> {
        &self.buffer_pool
    }
    
    /// Allocate host-visible GPU buffer for direct uploads
    pub fn allocate_host_visible(&self, length: u64) -> Subbuffer<[f32]> {
        self.host_visible
            .allocate_slice(length)
            .expect("failed to allocate host-visible buffer")
    }
    
    /// Check if data size is suitable for direct upload
    pub fn should_use_direct_upload(data_size: usize) -> bool {
        data_size * std::mem::size_of::<f32>() <= DIRECT_UPLOAD_THRESHOLD
    }
}

type CachedCore = (
    Arc<Device>,
    Arc<Queue>,
    Arc<StandardCommandBufferAllocator>,
    Arc<StandardDescriptorSetAllocator>,
    Arc<StandardMemoryAllocator>, // memory allocator is Sync
);

static DEVICE_CORE: LazyLock<CachedCore> = LazyLock::new(|| {
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap();

    let device_extensions = DeviceExtensions::empty();

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            enabled_features: {
                let mut features = DeviceFeatures::default();
                features.shader_int8 = true;
                features.shader_int64 = true;
                features
            },
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    (
        device,
        queues.next().unwrap(),
        command_buffer_allocator,
        descriptor_set_allocator,
        memory_allocator,
    )
});

pub fn get_device() -> (
    Arc<Device>,
    Arc<Queue>,
    Arc<StandardCommandBufferAllocator>,
    Arc<StandardDescriptorSetAllocator>,
    SubBuffersAllocator,
) {
    let (device, queue, command_buffer_allocator, descriptor_set_allocator, memory_allocator) =
        DEVICE_CORE.clone();

    let gpu_buffer_allocator = Arc::new(SubbufferAllocator::new(
        memory_allocator.clone(),
        SubbufferAllocatorCreateInfo {
            buffer_usage: BufferUsage::TRANSFER_DST
                | BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_SRC,
            memory_type_filter: MemoryTypeFilter {
                required_flags: MemoryPropertyFlags::DEVICE_LOCAL,
                preferred_flags: MemoryPropertyFlags::empty(),
                not_preferred_flags: MemoryPropertyFlags::empty(),
            },
            ..Default::default()
        },
    ));

    let cpu_buffer_allocator = Arc::new(SubbufferAllocator::new(
        memory_allocator.clone(),
        SubbufferAllocatorCreateInfo {
            buffer_usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
    ));

    // Host-visible GPU memory for direct uploads (small data optimization)
    let host_visible_allocator = Arc::new(SubbufferAllocator::new(
        memory_allocator.clone(),
        SubbufferAllocatorCreateInfo {
            buffer_usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
            memory_type_filter: MemoryTypeFilter {
                // Prefer host-visible device-local for unified memory architectures
                required_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
                preferred_flags: MemoryPropertyFlags::DEVICE_LOCAL,
                not_preferred_flags: MemoryPropertyFlags::empty(),
            },
            ..Default::default()
        },
    ));

    // Create buffer pool for reusing allocations
    let buffer_pool = Arc::new(BufferPool::new(memory_allocator.clone()));

    (
        device,
        queue,
        command_buffer_allocator,
        descriptor_set_allocator,
        SubBuffersAllocator {
            gpu: gpu_buffer_allocator,
            cpu: cpu_buffer_allocator,
            host_visible: host_visible_allocator,
            current_size: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            buffer_pool,
        },
    )
}

pub struct SubBufferPair<T> {
    cpu: Subbuffer<[T]>,
    gpu: Subbuffer<[T]>,
}

impl<T: BufferContents + Copy> SubBufferPair<T> {
    pub fn new(subbuffer_allocator: &SubBuffersAllocator, length: u64) -> Self {
        let cpu = subbuffer_allocator
            .cpu
            .allocate_slice(length)
            .expect("failed to allocate cpu buffer");
        let gpu = subbuffer_allocator
            .gpu
            .allocate_slice(length)
            .expect("failed to allocate gpu buffer");
        Self { cpu, gpu }
    }
}

impl<T: BufferContents + Copy> SubBufferPair<T> {
    pub fn get_cpu_buffer(&self) -> BufferWriteGuard<'_, [T]> {
        self.cpu.write().unwrap()
    }

    pub fn move_gpu<L>(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<L>,
        size: usize,
    ) -> Subbuffer<[T]> {
        command_buffer
            .copy_buffer(CopyBufferInfo::buffers(
                self.cpu.clone().slice(0..size as u64),
                self.gpu.clone().slice(0..size as u64),
            ))
            .unwrap();

        self.gpu.clone().slice(0..size as u64)
    }

    pub fn move_gpu_data<L>(
        &self,
        data: &[T],
        command_buffer: &mut AutoCommandBufferBuilder<L>,
    ) -> Subbuffer<[T]> {
        self.cpu.write().unwrap()[0..data.len()].copy_from_slice(&data);

        command_buffer
            .copy_buffer(CopyBufferInfo::buffers(
                self.cpu.clone().slice(0..data.len() as u64),
                self.gpu.clone().slice(0..data.len() as u64),
            ))
            .unwrap();

        self.gpu.clone().slice(0..data.len() as u64)
    }

    pub fn move_cpu<L>(&self, command_buffer: &mut AutoCommandBufferBuilder<L>) -> Subbuffer<[T]> {
        command_buffer
            .copy_buffer(CopyBufferInfo::buffers(self.gpu.clone(), self.cpu.clone()))
            .unwrap();
        self.cpu.clone()
    }
}
/// Command buffer pool for reusing command buffers across operations
pub struct CommandBufferPool {
    queue_family_index: u32,
    allocator: Arc<StandardCommandBufferAllocator>,
}

impl CommandBufferPool {
    pub fn new(allocator: Arc<StandardCommandBufferAllocator>, queue_family_index: u32) -> Self {
        Self {
            queue_family_index,
            allocator,
        }
    }

    /// Get a new primary command buffer builder
    pub fn get_builder(&self) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            self.allocator.clone(),
            self.queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Failed to create command buffer builder")
    }
}

/// Align a size to cache line boundary for better memory access patterns
#[inline]
pub fn align_to_cache_line(size: u64) -> u64 {
    (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1)
}

/// Compute optimized diagonal buffer length
/// Uses cache-aligned size instead of power-of-two to reduce over-allocation
#[inline]
pub fn compute_optimized_diag_len(len: usize, max_subgroup_size: usize) -> usize {
    let base_len = 2 * (next_multiple_of_n(len, max_subgroup_size) + 1);
    // Align to power of 2 for efficient indexing, but use smaller multiplier
    // Use next power of 2 for efficient mask-based indexing
    base_len.next_power_of_two()
}

/// Compute next multiple of n (utility function)
#[inline]
pub fn next_multiple_of_n(x: usize, n: usize) -> usize {
    (x + n - 1) / n * n
}