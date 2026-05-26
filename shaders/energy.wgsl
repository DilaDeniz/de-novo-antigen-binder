// WGSL compute shader: AMBER LJ + Coulomb + hydrophobic energy for one candidate
// per workgroup.
//
// Dispatch: (N_CANDIDATES, 1, 1) workgroups of (128, 1, 1) threads.
// One workgroup evaluates the total non-bonded interaction energy between the
// antigen and one antibody candidate.
//
// Thread layout:
//   Each thread strides over antigen atoms in chunks of WRKGRP (128), accumulates
//   partial energies into shared memory, then a parallel reduction writes the
//   final scalar to energies[cand].
//
// Buffer layout (8 × f32 = 32 bytes per atom, cache-line friendly):
//   struct GpuAtom { x, y, z, q, r_min_half, epsilon, hydrophobic_f32, _pad }

const CUTOFF_SQ : f32 = 100.0;   // 10 Å cutoff
const MIN_R_SQ  : f32 = 0.25;    // 0.5 Å singularity guard
const COULOMB_K : f32 = 332.0;   // kcal/mol·Å·e⁻²
const WRKGRP    : u32 = 128u;

struct GpuAtom {
    x         : f32,
    y         : f32,
    z         : f32,
    q         : f32,
    r_min_half: f32,
    epsilon   : f32,
    hydrophobic: f32,  // 1.0 = hydrophobic, 0.0 = not
    _pad      : f32,
}

struct Uniforms {
    n_ag : u32,
    n_ab : u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read>       antigen    : array<GpuAtom>;
@group(0) @binding(1) var<storage, read>       antibodies : array<GpuAtom>;
@group(0) @binding(2) var<uniform>             uniforms   : Uniforms;
@group(0) @binding(3) var<storage, read_write> energies   : array<f32>;

var<workgroup> shared_e : array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id)        wid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let cand   = wid.x;
    let tid    = lid.x;
    let n_ag   = uniforms.n_ag;
    let n_ab   = uniforms.n_ab;
    let ab_off = cand * n_ab;

    var acc : f32 = 0.0;

    // Each thread handles antigen atoms [tid, tid+WRKGRP, tid+2*WRKGRP, ...]
    var i : u32 = tid;
    loop {
        if i >= n_ag { break; }

        let ag = antigen[i];

        // Inner loop over all antibody atoms for this candidate
        for (var j : u32 = 0u; j < n_ab; j = j + 1u) {
            let ab = antibodies[ab_off + j];
            let dx = ab.x - ag.x;
            let dy = ab.y - ag.y;
            let dz = ab.z - ag.z;
            let r2 = dx * dx + dy * dy + dz * dz;

            if r2 <= CUTOFF_SQ && r2 >= MIN_R_SQ {
                // Lorentz-Berthelot mixing (AMBER additive Rmin + geometric eps)
                let r_ij  = ag.r_min_half + ab.r_min_half;
                let eps   = sqrt(ag.epsilon * ab.epsilon);

                // AMBER LJ: ε[(R/r)^12 − 2(R/r)^6]
                let r2_ratio = (r_ij * r_ij) / r2;
                let r6  = r2_ratio * r2_ratio * r2_ratio;
                let r12 = r6 * r6;
                acc += eps * (r12 - 2.0 * r6);

                // Coulomb
                if ag.q != 0.0 && ab.q != 0.0 {
                    acc += COULOMB_K * ag.q * ab.q / sqrt(r2);
                }

                // Hydrophobic bonus: −0.5 kcal/mol per hydrophobic pair < 6 Å
                if ag.hydrophobic > 0.5 && ab.hydrophobic > 0.5 && r2 < 36.0 {
                    acc += -0.5;
                }
            }
        }

        i = i + WRKGRP;
    }

    // Store thread partial sum
    shared_e[tid] = acc;
    workgroupBarrier();

    // Parallel reduction (halving)
    var stride : u32 = WRKGRP / 2u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            shared_e[tid] += shared_e[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if tid == 0u {
        energies[cand] = shared_e[0u];
    }
}
