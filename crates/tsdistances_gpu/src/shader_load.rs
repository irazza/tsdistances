use std::sync::{Arc, OnceLock};

use dashmap::DashMap;
use vulkano::{
    Validated, VulkanError,
    device::Device,
    pipeline::{
        ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
};

static SHADE_PIPELINES: OnceLock<DashMap<&'static str, Arc<ComputePipeline>>> = OnceLock::new();

/// The compiled SPIR-V module, byte for byte as it is handed to the driver.
///
/// Public so `tests/shader_invariants.rs` can assert module-level invariants without a
/// Vulkan device. Those invariants are not decoration: the `Int64` capability and the
/// `GLSL.std.450 Pow` this crate used to emit were both introduced by a green build and
/// caught only by disassembling this blob by hand.
pub const SHADER_CODE: &[u8] = include_bytes!(env!("tsdistances_gpu.spv"));

use rspirv::binary::Assemble;
use rspirv::spirv::{ExecutionMode, Op};

fn load(
    entry_point: &str,
    device: Arc<Device>,
    shader: &[u8],
    workgroup_size: u32,
) -> Result<Arc<ShaderModule>, Validated<VulkanError>> {
    // Load the SPIR-V module
    let mut spirv_module = rspirv::dr::load_bytes(shader).expect("Failed to load SPIR-V module");

    // Find the entry point ID for the given entry point name
    let entry_point_id = spirv_module
        .entry_points
        .iter()
        .find(|entry| entry.operands[2].unwrap_literal_string() == entry_point)
        .expect("Entry point not found")
        .operands[1]
        .unwrap_id_ref(); // Operand[1] is the function ID

    // Remove existing LocalSize if needed
    spirv_module.execution_modes.retain(|inst| {
        !(inst.class.opcode == Op::ExecutionMode
            && inst.operands[0].unwrap_id_ref() == entry_point_id
            && inst.operands[1].unwrap_execution_mode() == ExecutionMode::LocalSize)
    });

    // Add a new LocalSize mode. `workgroup_size` comes from `utils::compute_workgroup_size`
    // and MUST be the same value the dispatch in `kernels.rs` divides its thread count by,
    // which is why it is passed in rather than recomputed here.
    spirv_module
        .execution_modes
        .push(rspirv::dr::Instruction::new(
            Op::ExecutionMode,
            None,
            None,
            vec![
                rspirv::dr::Operand::IdRef(entry_point_id),
                rspirv::dr::Operand::ExecutionMode(ExecutionMode::LocalSize),
                rspirv::dr::Operand::LiteralBit32(workgroup_size), // x dimension
                rspirv::dr::Operand::LiteralBit32(1),              // y dimension
                rspirv::dr::Operand::LiteralBit32(1),              // z dimension
            ],
        ));

    let spirv = spirv_module.assemble();

    // Create the ShaderModule with the optimized SPIR-V
    unsafe { ShaderModule::new(device, ShaderModuleCreateInfo::new(&spirv)) }
}

/// `workgroup_size` is baked into the pipeline's `LocalSize`. The cache is keyed by entry
/// point alone because that size is a pure function of the process-wide `get_device()`
/// singleton, so every caller necessarily passes the same value.
pub fn get_shader_entry_pipeline(
    device: Arc<Device>,
    name: &'static str,
    workgroup_size: u32,
) -> Arc<ComputePipeline> {
    let pipelines = SHADE_PIPELINES.get_or_init(Default::default);

    match pipelines.entry(name) {
        dashmap::Entry::Occupied(entry) => entry.get().clone(),
        dashmap::Entry::Vacant(vacant_entry) => {
            let shader_module = load(name, device.clone(), SHADER_CODE, workgroup_size).unwrap();
            let Some(entry_point) = shader_module.entry_point(name) else {
                panic!("Entry point {} not found in shader module", name);
            };
            let stage = PipelineShaderStageCreateInfo::new(entry_point);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            let pipeline = ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap();
            vacant_entry.insert(pipeline.clone());
            pipeline
        }
    }
}
