//! Module-level assertions on the compiled SPIR-V.
//!
//! Every one of these guards a defect that a green build did not catch. When rust-gpu
//! was bumped across the 0.9 -> 0.10.0-alpha.1 release, `build.rs` was migrated to the
//! new API, everything compiled and every GPU test passed -- while the shader had
//! quietly started emitting `GLSL.std.450 Pow` for `powi(2)` (undefined for a negative
//! base, and the base here is `a - b`), kept requiring an `Int64` capability no device
//! needs, and synchronised a storage buffer with a barrier that declared only
//! `WorkgroupMemory`. All three are invisible to `cargo test` on a driver that happens
//! to be forgiving, and all three are one instruction to spot in the module.
//!
//! Deliberately needs no Vulkan device, so it runs in CI where the GPU tests cannot.

use rspirv::binary::Disassemble as _;
use rspirv::dr::Operand;
use rspirv::spirv::{Capability, Op};
use std::collections::HashMap;
use tsdistances_gpu::SHADER_CODE;

/// GLSL.std.450 `Pow`. See <https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html>.
const GLSL_STD_450_POW: u32 = 26;

/// SPIR-V memory semantics bits (SPIR-V spec, "Memory Semantics").
const ACQUIRE_RELEASE: u32 = 0x8;
const UNIFORM_MEMORY: u32 = 0x40;
const WORKGROUP_MEMORY: u32 = 0x100;

/// `Scope::Workgroup`.
const SCOPE_WORKGROUP: u32 = 2;

const EXPECTED_ENTRY_POINTS: &[&str] = &[
    "kernels::adtw_distance::batch_call",
    "kernels::dtw_distance::batch_call",
    "kernels::erp_distance::batch_call",
    "kernels::lcss_distance::batch_call",
    "kernels::msm_distance::batch_call",
    "kernels::twe_distance::batch_call",
    "kernels::wdtw_distance::batch_call",
];

fn module() -> rspirv::dr::Module {
    rspirv::dr::load_bytes(SHADER_CODE).expect("compiled shader is not valid SPIR-V")
}

/// `result id -> value` for every `OpConstant`, so barrier operands can be resolved.
fn scalar_constants(module: &rspirv::dr::Module) -> HashMap<u32, u32> {
    module
        .types_global_values
        .iter()
        .filter(|inst| inst.class.opcode == Op::Constant)
        .filter_map(|inst| match inst.operands.first() {
            Some(Operand::LiteralBit32(value)) => Some((inst.result_id?, *value)),
            _ => None,
        })
        .collect()
}

/// The shader must not require any optional device capability.
///
/// `Int64` was required for years by a push-constant block whose every field is an index
/// or a length bounded by `maxStorageBufferRange` -- itself a u32. The cost was real:
/// MoltenVK does not expose `shaderInt64` at all (Metal has no 64-bit integers), so the
/// crate could not open a device on Apple. `Int8` was requested in `build.rs` and paid
/// for at device creation without ever being declared by the module.
#[test]
fn declares_no_optional_capabilities() {
    let capabilities: Vec<Capability> = module()
        .capabilities
        .iter()
        .filter_map(|inst| match inst.operands.first() {
            Some(Operand::Capability(cap)) => Some(*cap),
            _ => None,
        })
        .collect();

    assert_eq!(
        capabilities,
        vec![Capability::Shader],
        "shader gained a capability beyond `Shader`, which becomes a device requirement \
         every user must satisfy; if it is genuinely needed, enable the matching feature \
         in `utils::get_device` in the same commit"
    );
}

/// `Pow(x, y)` is undefined for `x < 0` in GLSL.std.450, and rust-gpu lowers
/// `num_traits::Float::powi` straight onto it (rust-gpu PR#518). The squared-difference
/// kernels (`dtw`, `wdtw`, `adtw`) feed it `a - b`, which is negative about half the
/// time, so `.powi(2)` there is a latent NaN generator on any driver that does not
/// happen to constant-fold it back into a multiply. Write `d * d`.
#[test]
fn uses_no_pow_intrinsic() {
    let module = module();

    let glsl_std_450: Vec<u32> = module
        .ext_inst_imports
        .iter()
        .filter(|inst| {
            matches!(inst.operands.first(),
                Some(Operand::LiteralString(name)) if name == "GLSL.std.450")
        })
        .filter_map(|inst| inst.result_id)
        .collect();

    let pow_calls = module
        .all_inst_iter()
        .filter(|inst| inst.class.opcode == Op::ExtInst)
        .filter(|inst| match (inst.operands.first(), inst.operands.get(1)) {
            (Some(Operand::IdRef(set)), Some(Operand::LiteralExtInstInteger(op))) => {
                glsl_std_450.contains(set) && *op == GLSL_STD_450_POW
            }
            _ => false,
        })
        .count();

    assert_eq!(
        pow_calls, 0,
        "shader emits GLSL.std.450 Pow, which is undefined for negative bases; \
         `x.powi(2)` lowers to it -- use `let d = ..; d * d` instead"
    );
}

/// The diamond wavefront exchanges cells through a *storage buffer*, whose SPIR-V
/// memory-semantics bit is `UniformMemory`. `workgroup_memory_barrier_with_group_sync()`
/// declares only `WorkgroupMemory`, so it synchronised execution while placing no
/// ordering whatsoever on the writes the algorithm depends on.
#[test]
fn barriers_order_storage_buffer_traffic() {
    let module = module();
    let constants = scalar_constants(&module);

    let barriers: Vec<_> = module
        .all_inst_iter()
        .filter(|inst| inst.class.opcode == Op::ControlBarrier)
        .collect();

    assert!(
        !barriers.is_empty(),
        "no OpControlBarrier in the module -- the diamond wavefront cannot be correct \
         without one; did an entry point stop being generated?"
    );

    for barrier in barriers {
        // rspirv lifts these to the typed `IdScope` / `IdMemorySemantics` variants rather
        // than a bare `IdRef`; all three are ids of an `OpConstant`.
        let operand = |i: usize| {
            let id = match barrier.operands.get(i) {
                Some(
                    Operand::IdRef(id) | Operand::IdScope(id) | Operand::IdMemorySemantics(id),
                ) => *id,
                other => panic!("unexpected barrier operand {i}: {other:?}"),
            };
            *constants
                .get(&id)
                .unwrap_or_else(|| panic!("barrier operand {i} (%{id}) is not an OpConstant"))
        };
        let (execution, memory, semantics) = (operand(0), operand(1), operand(2));

        assert_eq!(
            execution,
            SCOPE_WORKGROUP,
            "barrier execution scope must be Workgroup: {}",
            barrier.disassemble()
        );
        assert_eq!(
            memory,
            SCOPE_WORKGROUP,
            "barrier memory scope must be Workgroup -- Device scope is broader than the \
             workgroup that participates, and costs more: {}",
            barrier.disassemble()
        );
        for (bit, name) in [
            (UNIFORM_MEMORY, "UniformMemory (i.e. the storage buffer)"),
            (WORKGROUP_MEMORY, "WorkgroupMemory"),
            (ACQUIRE_RELEASE, "AcquireRelease"),
        ] {
            assert_ne!(
                semantics & bit,
                0,
                "barrier memory semantics 0x{semantics:x} is missing {name}; the diagonal \
                 cells live in a storage buffer, so a barrier without UniformMemory orders \
                 nothing that matters: {}",
                barrier.disassemble()
            );
        }
    }
}

/// Every kernel the host dispatches by name must exist. `get_shader_entry_pipeline`
/// panics at run time on a missing entry point, and only on the code path that uses it.
#[test]
fn exposes_every_expected_entry_point() {
    let module = module();
    let mut names: Vec<String> = module
        .entry_points
        .iter()
        .filter_map(|inst| match inst.operands.get(2) {
            Some(Operand::LiteralString(name)) => Some(name.clone()),
            _ => None,
        })
        .collect();
    names.sort();

    assert_eq!(names, EXPECTED_ENTRY_POINTS);
}
