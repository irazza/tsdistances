use spirv_builder::{Capability, SpirvBuilder, SpirvMetadata};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "spirv" {
        return Ok(());
    }

    let mut builder = SpirvBuilder::new(".", "spirv-unknown-spv1.5")
        .spirv_metadata(SpirvMetadata::NameVariables)
        .capability(Capability::Int8)
        .capability(Capability::Int64);
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
