use std::sync::{Arc, LazyLock};

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
    current_size: Arc<std::sync::atomic::AtomicU64>,
}

impl Clone for SubBuffersAllocator {
    fn clone(&self) -> Self {
        Self {
            gpu: self.gpu.clone(),
            cpu: self.cpu.clone(),
            current_size: self.current_size.clone(),
        }
    }
}

impl SubBuffersAllocator {
    /// Clear and resize the arena. Only resizes if the new size is larger or significantly smaller.
    /// This reduces allocation overhead for repeated calls with similar sizes.
    pub fn clear_with_size(&self, size: u64) {
        let current = self.current_size.load(std::sync::atomic::Ordering::Relaxed);

        // Only resize if:
        // 1. New size is larger than current, OR
        // 2. New size is less than 25% of current (to reclaim memory)
        // 3. size is 0 (explicit cleanup)
        if size > current || size == 0 || (current > 0 && size < current / 4) {
            self.gpu.set_arena_size(size);
            self.cpu.set_arena_size(size);
            self.current_size
                .store(size, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Allocate a device-local buffer with no host mirror.
    ///
    /// [`SubBufferPair`] always allocates a host-visible twin, which is right for data
    /// that crosses the bus but pure waste for the diagonal buffer: it is written and read
    /// only by the GPU (filled with `fill_buffer`, seeded and harvested by
    /// `kernels::init_diagonal` / `kernels::gather_results`), and on the ACSF1 workload its
    /// twin alone was 163.8 MB of host memory that nothing ever touched.
    pub fn allocate_gpu<T: BufferContents>(&self, length: u64) -> Subbuffer<[T]> {
        self.gpu
            .allocate_slice(length)
            .expect("failed to allocate device-local buffer")
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
            self.current_size
                .store(new_size, std::sync::atomic::Ordering::Relaxed);
        }
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
    // The bare `unwrap()` here reported "the requested version of Vulkan is not supported
    // by the driver", which sounds like a driver-too-old problem but is what the loader
    // returns when it finds *no* ICD at all -- including when `VK_ICD_FILENAMES` points at
    // a path that does not exist. Say so, since that is the common cause.
    .expect(
        "could not create a Vulkan instance. The loader found no usable driver (ICD). \
         Check that a Vulkan driver is installed, and that VK_ICD_FILENAMES / VK_DRIVER_FILES, \
         if set, point at a file that exists. `vulkaninfo --summary` should list a device.",
    );

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
            // No `shader_int64` / `shader_int8`. The kernels' constants block used to be
            // 64-bit, which made the shader declare `OpCapability Int64` and so required
            // the matching device feature -- one MoltenVK/Metal does not expose, meaning
            // this crate could not open a device on Apple at all. `Int8` was never even
            // declared by the emitted module; it was requested and paid for regardless.
            enabled_features: DeviceFeatures::default(),
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

    (
        device,
        queue,
        command_buffer_allocator,
        descriptor_set_allocator,
        SubBuffersAllocator {
            gpu: gpu_buffer_allocator,
            cpu: cpu_buffer_allocator,
            current_size: Arc::new(std::sync::atomic::AtomicU64::new(0)),
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

    /// The device-local half, for kernels that write it directly rather than receiving it
    /// from a host copy.
    pub fn gpu_buffer(&self) -> Subbuffer<[T]> {
        self.gpu.clone()
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
        self.cpu.write().unwrap()[0..data.len()].copy_from_slice(data);

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
    x.div_ceil(n) * n
}

/// Width of one diamond tile, in invocations.
///
/// Reads `subgroup_size` (`VkPhysicalDeviceVulkan11Properties`, i.e. Vulkan 1.1) in
/// preference to `max_subgroup_size` (`VkPhysicalDeviceVulkan13Properties`). The value
/// is only ever used as a tile width -- the diamond barrier is workgroup-scoped, so it
/// need not match any hardware subgroup -- and the 1.3 property is `None` on older
/// devices, where the previous `.unwrap()` panicked outright instead of degrading. The
/// two agree on every driver checked (RADV 64/64, lavapipe 8/8).
pub fn compute_tile_width(device: &Device) -> usize {
    let properties = device.physical_device().properties();
    properties
        .subgroup_size
        .or(properties.max_subgroup_size)
        .unwrap_or(32) as usize
}

/// Workgroup size to dispatch kernels with, for a given diamond tile width.
///
/// `shader_load::load` patches this into the shader's `LocalSize` execution mode and the
/// `dispatch` in `kernels.rs` divides the thread count by it to get the workgroup count.
/// Those two numbers MUST agree, which is why there is exactly one definition of it.
///
/// **Exactly one diamond per workgroup.** This used to be
/// `max_compute_work_group_size[0]` (1024), packing 16 independent diamonds into one
/// workgroup. That is not merely wasteful -- the kernel's barrier has Workgroup execution
/// scope, and the loop it sits in runs for `diag_count` iterations, which differs between
/// diamonds. Invocations belonging to different diamonds therefore reached different
/// numbers of barriers: undefined behaviour, which AMD tolerated and lavapipe did not.
///
/// One diamond per workgroup makes `diag_count` (and the `active` flag) uniform across
/// every invocation that shares a barrier, which is what makes the tight loop bound in
/// `warp_kernel_inner` legal. It also lets a diamond map onto a single hardware subgroup.
pub fn compute_workgroup_size(_device: &Device, tile_width: usize) -> u32 {
    u32::try_from(tile_width).expect("tile width must fit in u32")
}
