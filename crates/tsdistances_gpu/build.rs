use spirv_builder::{SpirvBuilder, SpirvMetadata};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "spirv" {
        return Ok(());
    }

    // Stay on a `spv*` target. Switching to `spirv-unknown-vulkan1.2` looks free -- it
    // emits the same SPIR-V 1.5 -- but rust-gpu ties the memory model to the target
    // family with no opt-out (`rustc_codegen_spirv/src/target.rs`: `SpirvTarget::Vulkan`
    // => `MemoryModel::Vulkan`), so every `vulkan1.x` target additionally declares
    // `OpCapability VulkanMemoryModel`. Two consequences, both bad here:
    //
    //  1. vulkano rejects the module unless the `vulkan_memory_model` *device feature* is
    //     enabled (VUID-VkShaderModuleCreateInfo-pCode-08742). That feature is optional
    //     even on Vulkan 1.2, so requiring it costs device compatibility.
    //  2. Under the Vulkan memory model, a barrier's memory semantics only order accesses
    //     marked `NonPrivatePointer` -- and rust-gpu marks none of them. That would
    //     silently reduce the diamond barrier in kernels.rs to a no-op for the storage
    //     buffer it exists to protect, i.e. exactly the bug that barrier was fixed for.
    //
    // Universal targets get `MemoryModel::Simple`, under which the barrier's semantics
    // apply to all accesses, which is what the wavefront relies on.
    //
    // No `.capability(Int8)` / `.capability(Int64)`: the constants block is 32-bit now,
    // so neither is declared by the emitted module, and requiring them cost real device
    // compatibility (Metal has no 64-bit integers). See `KernelConstants` in kernels.rs.
    let mut builder =
        SpirvBuilder::new(".", "spirv-unknown-spv1.5").spirv_metadata(SpirvMetadata::NameVariables);
    // Newer spirv-builder dropped `print_metadata(MetadataPrintout::Full)` in favor of
    // BuildScriptConfig. Emitting the shader-path env var is now opt-in (off by
    // default), so enable it explicitly: this prints
    // `cargo::rustc-env=tsdistances_gpu.spv=<path>`, which src/shader_load.rs consumes
    // via `include_bytes!(env!("tsdistances_gpu.spv"))`. `defaults` re-enables the
    // dependency info so the shader rebuilds when its sources change.
    builder.build_script.defaults = true;
    builder.build_script.env_shader_spv_path = Some(true);
    builder.build()?;
    Ok(())
}
