/// Generative reverse-diffusion engine.
///
/// Two paths share the same outer interface (`run` / `run_allatom`):
///
/// **Coarse-grained (Cα-only)** — original path, unchanged.  One point per
/// residue, residue-level LJ/Coulomb.  64 candidates, 800 steps.
///
/// **All-atom hybrid** — new path.  Full AMBER99SB heavy atoms, rotamer MC
/// moves (NERF side-chain rebuild), optional GPU broad-sampling phase.
///   - GPU phase  : 1024 candidates × 200 gradient-only steps (no noise).
///     Top-64 survivors selected by GPU energy.
///   - CPU phase  : Top-64 candidates × 600 Langevin + rotamer MC steps
///     (Rayon parallel, no shared state).
///   Without GPU  : 64 candidates × 800 CPU steps (identical to CG path but
///     with all-atom energy).
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::allatom::{protein_from_ca_trace, AtomProtein};
use crate::atom::{AminoAcid, ResidueCloud, AA_COUNT};
use crate::energy;
use crate::rotamer::rotamers;
use crate::spatial::SpatialHashGrid;

// ── Hyperparameters ───────────────────────────────────────────────────────────

pub const POPULATION:   usize = 64;
pub const ITERATIONS:   usize = 800;

/// All-atom population (used only by run_allatom).
pub const AA_POPULATION: usize = 1024;
/// All-atom GPU phase steps.
pub const GPU_STEPS:     usize = 200;
/// All-atom CPU refinement steps.
pub const CPU_STEPS:     usize = 600;
/// Survivors from GPU phase.
pub const TOP_K:         usize = 64;

const STEP:       f32 = 0.08;
const NOISE_BASE: f32 = 0.25;
const T_HOT:      f32 = 3.0;
const T_COLD:     f32 = 0.02;
const MUTATION_P: f32 = 0.08;
/// Probability of a rotamer MC move vs. amino-acid mutation.
const ROTAMER_MOVE_P: f32 = 0.12;
/// Probability of a backbone phi/psi torsion MC move per residue per step.
const BACKBONE_MOVE_P: f32 = 0.04;
/// Maximum phi/psi perturbation per MC move (degrees).
const MAX_TORSION_DEG: f32 = 15.0;
const INIT_RADIUS:    f32 = 20.0;
const MAX_DISP:       f32 = 2.0;
const RESTRAINT_K:    f32 = 0.02;

// ── Public result types ───────────────────────────────────────────────────────

pub struct DiffusionResult {
    pub antibody: ResidueCloud,
    pub energy:   f32,
    pub sequence: String,
}

pub struct AllAtomResult {
    pub antibody: AtomProtein,
    pub energy:   f32,
    pub sequence: String,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline(always)]
fn temperature(iter: usize, total: usize) -> f32 {
    T_HOT * (T_COLD / T_HOT).powf(iter as f32 / total as f32)
}

#[inline(always)]
fn noise(rng: &mut SmallRng, scale: f32) -> f32 {
    rng.gen_range(-scale..scale)
}

fn random_antibody(n: usize, center: [f32; 3], rng: &mut SmallRng) -> ResidueCloud {
    let mut cloud = ResidueCloud::with_capacity(n);
    for i in 0..n {
        let angle = i as f32 * 2.399_f32;
        let cx = center[0] + INIT_RADIUS * angle.cos() + noise(rng, 5.0);
        let cy = center[1] + noise(rng, INIT_RADIUS * 0.5);
        let cz = center[2] + INIT_RADIUS * angle.sin() + noise(rng, 5.0);
        let aa = AminoAcid::from_index(rng.gen_range(0..AA_COUNT));
        cloud.push(cx, cy, cz, aa);
    }
    cloud
}

fn seed_u64(seed: usize) -> u64 {
    (seed as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ── Coarse-grained diffusion step (unchanged) ─────────────────────────────────

fn diffusion_step(
    antibody: &mut ResidueCloud,
    antigen:  &ResidueCloud,
    grid:     &SpatialHashGrid,
    ag_center: [f32; 3],
    temp:     f32,
    rng:      &mut SmallRng,
    fx_buf:   &mut Vec<f32>,
    fy_buf:   &mut Vec<f32>,
    fz_buf:   &mut Vec<f32>,
) {
    let n = antibody.len();

    if fx_buf.len() < n { fx_buf.resize(n, 0.0); }
    if fy_buf.len() < n { fy_buf.resize(n, 0.0); }
    if fz_buf.len() < n { fz_buf.resize(n, 0.0); }

    energy::compute_forces_with_grid(antigen, grid, antibody, fx_buf, fy_buf, fz_buf);

    let noise_sigma = NOISE_BASE * temp.sqrt();

    for j in 0..n {
        let dx_c = antibody.x[j] - ag_center[0];
        let dy_c = antibody.y[j] - ag_center[1];
        let dz_c = antibody.z[j] - ag_center[2];
        let r = (dx_c * dx_c + dy_c * dy_c + dz_c * dz_c).sqrt().max(0.001);
        let excess = r - INIT_RADIUS;
        let f_r = -RESTRAINT_K * excess / r;
        fx_buf[j] += f_r * dx_c;
        fy_buf[j] += f_r * dy_c;
        fz_buf[j] += f_r * dz_c;

        let disp_x = (STEP * fx_buf[j]).clamp(-MAX_DISP, MAX_DISP);
        let disp_y = (STEP * fy_buf[j]).clamp(-MAX_DISP, MAX_DISP);
        let disp_z = (STEP * fz_buf[j]).clamp(-MAX_DISP, MAX_DISP);
        antibody.x[j] += disp_x + noise(rng, noise_sigma);
        antibody.y[j] += disp_y + noise(rng, noise_sigma);
        antibody.z[j] += disp_z + noise(rng, noise_sigma);

        if rng.gen::<f32>() < MUTATION_P {
            let old_aa = antibody.amino_acid[j];
            let new_aa = AminoAcid::from_index(rng.gen_range(0..AA_COUNT));
            let old_e  = single_residue_energy(j, antibody, antigen);
            antibody.set_aa(j, new_aa);
            let new_e  = single_residue_energy(j, antibody, antigen);
            let delta_e = new_e - old_e;
            let accept = delta_e <= 0.0 || rng.gen::<f32>() < (-delta_e / temp).exp();
            if !accept { antibody.set_aa(j, old_aa); }
        }
    }
}

fn single_residue_energy(j: usize, antibody: &ResidueCloud, antigen: &ResidueCloud) -> f32 {
    let bx = antibody.x[j];
    let by = antibody.y[j];
    let bz = antibody.z[j];
    let bq = antibody.charge[j];
    let be = antibody.epsilon[j];
    let bs = antibody.sigma[j];
    let bh = antibody.hydrophobic[j];
    let mut e = 0.0_f32;
    for i in 0..antigen.len() {
        let dx = bx - antigen.x[i];
        let dy = by - antigen.y[i];
        let dz = bz - antigen.z[i];
        let r_sq = dx * dx + dy * dy + dz * dz;
        if r_sq > energy::CUTOFF_SQ || r_sq < 0.25 { continue; }
        let eps_ij    = (be * antigen.epsilon[i]).sqrt();
        let sig_ij    = 0.5 * (bs + antigen.sigma[i]);
        let sigma_sq  = sig_ij * sig_ij;
        let s2 = sigma_sq / r_sq;
        let s6 = s2 * s2 * s2;
        let s12 = s6 * s6;
        e += 4.0 * eps_ij * (s12 - s6);
        if bq != 0.0 && antigen.charge[i] != 0.0 {
            e += 332.0 * bq * antigen.charge[i] / r_sq.sqrt();
        }
        if bh == 1 && antigen.hydrophobic[i] == 1 && r_sq < 36.0 {
            e += -0.5;
        }
    }
    e
}

// ── Coarse-grained public interface ──────────────────────────────────────────

pub fn run(antigen: &ResidueCloud, ab_length: usize) -> DiffusionResult {
    let center = antigen.center_of_mass();
    let mut grid = SpatialHashGrid::new(10.0);
    grid.build(&antigen.x, &antigen.y, &antigen.z);

    let best = (0..POPULATION)
        .into_par_iter()
        .map(|seed| {
            let mut rng = SmallRng::seed_from_u64(seed_u64(seed));
            let mut antibody = random_antibody(ab_length, center, &mut rng);
            let mut fx_buf = Vec::with_capacity(ab_length);
            let mut fy_buf = Vec::with_capacity(ab_length);
            let mut fz_buf = Vec::with_capacity(ab_length);
            for iter in 0..ITERATIONS {
                let temp = temperature(iter, ITERATIONS);
                diffusion_step(&mut antibody, antigen, &grid, center, temp, &mut rng,
                               &mut fx_buf, &mut fy_buf, &mut fz_buf);
            }
            let e = energy::interaction_energy(antigen, &antibody);
            (antibody, e)
        })
        .reduce_with(|a, b| if a.1 <= b.1 { a } else { b })
        .expect("population is non-empty");

    let (antibody, energy) = best;
    let sequence = antibody.sequence();
    DiffusionResult { antibody, energy, sequence }
}

// ── All-atom helpers ──────────────────────────────────────────────────────────

/// Build a random all-atom antibody around `center`.
fn random_allatom_antibody(n: usize, center: [f32; 3], rng: &mut SmallRng) -> AtomProtein {
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n);
    let mut aas = Vec::with_capacity(n);
    for i in 0..n {
        let angle = i as f32 * 2.399_f32;
        xs.push(center[0] + INIT_RADIUS * angle.cos() + noise(rng, 5.0));
        ys.push(center[1] + noise(rng, INIT_RADIUS * 0.5));
        zs.push(center[2] + INIT_RADIUS * angle.sin() + noise(rng, 5.0));
        aas.push(AminoAcid::from_index(rng.gen_range(0..AA_COUNT)));
    }
    protein_from_ca_trace(&xs, &ys, &zs, &aas)
}

/// Compute total AMBER LJ + Coulomb + hydrophobic energy between two AtomProteins.
fn allatom_energy(ag: &AtomProtein, ab: &AtomProtein) -> f32 {
    energy::interaction_energy_atoms(&ag.atoms, &ab.atoms)
}

/// Gradient + noise step on the Cα positions only, then rebuild side-chains.
///
/// `ag_ca` / `ag_ca_grid` — antigen Cα cloud + hash, pre-computed once, used
///   for the fast force computation.
/// `ag` — full antigen AtomProtein, used for AMBER-consistent MC energy evals.
fn allatom_diffusion_step(
    ab: &mut AtomProtein,
    ag: &AtomProtein,
    ag_ca: &ResidueCloud,
    ag_ca_grid: &SpatialHashGrid,
    ag_center: [f32; 3],
    temp: f32,
    rng: &mut SmallRng,
    with_noise: bool,
) {
    let n = ab.n_residues();

    let mut fx_buf = vec![0.0_f32; n];
    let mut fy_buf = vec![0.0_f32; n];
    let mut fz_buf = vec![0.0_f32; n];

    // Build antibody Cα cloud for force computation (residue-level approximation)
    let mut ca_cloud = ResidueCloud::with_capacity(n);
    for r in 0..n {
        let p = ab.ca_pos(r);
        ca_cloud.push(p[0], p[1], p[2], ab.amino_acid[r]);
    }

    energy::compute_forces_with_grid(ag_ca, ag_ca_grid, &ca_cloud, &mut fx_buf, &mut fy_buf, &mut fz_buf);

    let noise_sigma = if with_noise { NOISE_BASE * temp.sqrt() } else { 0.0 };

    for r in 0..n {
        let ca = ab.ca_pos(r);
        let dx_c = ca[0] - ag_center[0];
        let dy_c = ca[1] - ag_center[1];
        let dz_c = ca[2] - ag_center[2];
        let d = (dx_c * dx_c + dy_c * dy_c + dz_c * dz_c).sqrt().max(0.001);
        let excess = d - INIT_RADIUS;
        let f_r = -RESTRAINT_K * excess / d;
        fx_buf[r] += f_r * dx_c;
        fy_buf[r] += f_r * dy_c;
        fz_buf[r] += f_r * dz_c;

        let nx = if with_noise { noise(rng, noise_sigma) } else { 0.0 };
        let ny = if with_noise { noise(rng, noise_sigma) } else { 0.0 };
        let nz = if with_noise { noise(rng, noise_sigma) } else { 0.0 };

        let new_ca = [
            ca[0] + (STEP * fx_buf[r]).clamp(-MAX_DISP, MAX_DISP) + nx,
            ca[1] + (STEP * fy_buf[r]).clamp(-MAX_DISP, MAX_DISP) + ny,
            ca[2] + (STEP * fz_buf[r]).clamp(-MAX_DISP, MAX_DISP) + nz,
        ];
        ab.set_ca_pos(r, new_ca);
    }

    // Metropolis AA mutation or rotamer move
    for r in 0..n {
        let p = rng.gen::<f32>();
        if p < MUTATION_P {
            let move_type = rng.gen::<f32>();
            if move_type < ROTAMER_MOVE_P {
                // Rotamer MC move
                let aa = ab.amino_acid[r];
                let rots = rotamers(aa);
                if !rots.is_empty() {
                    let old_chi = ab.chi[r];
                    let rot_idx = rng.gen_range(0..rots.len());
                    let new_rot = rots[rot_idx];

                    let old_e = residue_contribution_allatom(r, ab, ag);
                    ab.apply_rotamer(r, &new_rot);
                    let new_e = residue_contribution_allatom(r, ab, ag);
                    let delta_e = new_e - old_e;
                    let accept = delta_e <= 0.0 || rng.gen::<f32>() < (-delta_e / temp).exp();
                    if !accept {
                        ab.chi[r] = old_chi;
                        ab.rebuild_side_chain(r);
                    }
                }
            } else {
                // AA mutation MC move
                let old_aa  = ab.amino_acid[r];
                let new_aa  = AminoAcid::from_index(rng.gen_range(0..AA_COUNT));
                let old_chi = ab.chi[r];

                let old_e = residue_contribution_allatom(r, ab, ag);
                // Use first rotamer of new AA as proposal
                let rots = rotamers(new_aa);
                let new_chi = if rots.is_empty() { [0.0f32; 4] } else { rots[0].chi };
                ab.mutate_residue(r, new_aa, new_chi);
                let new_e = residue_contribution_allatom(r, ab, ag);
                let delta_e = new_e - old_e;
                let accept = delta_e <= 0.0 || rng.gen::<f32>() < (-delta_e / temp).exp();
                if !accept {
                    ab.mutate_residue(r, old_aa, old_chi);
                }
            }
        }
    }

    // Backbone phi/psi torsion MC moves — propagating: residues r+1..n also move,
    // so we evaluate the energy of the entire affected region (r..n).
    for r in 0..n {
        if rng.gen::<f32>() < BACKBONE_MOVE_P {
            let delta = rng.gen_range(-MAX_TORSION_DEG..MAX_TORSION_DEG).to_radians();
            let do_phi = rng.gen::<bool>();
            let old_e: f32 = (r..n).map(|i| residue_contribution_allatom(i, ab, ag)).sum();
            if do_phi { ab.perturb_phi(r, delta); } else { ab.perturb_psi(r, delta); }
            let new_e: f32 = (r..n).map(|i| residue_contribution_allatom(i, ab, ag)).sum();
            let accept = (new_e - old_e) <= 0.0
                || rng.gen::<f32>() < (-(new_e - old_e) / temp).exp();
            if !accept {
                if do_phi { ab.perturb_phi(r, -delta); } else { ab.perturb_psi(r, -delta); }
            }
        }
    }
}

/// Fast per-residue energy contribution using AMBER LJ at the Cα level.
///
/// Uses real AMBER r_min_half + epsilon from the Cα atoms (atom type CT) of
/// both the antibody and antigen, matching the convention used for final scoring.
fn residue_contribution_allatom(r: usize, ab: &AtomProtein, ag: &AtomProtein) -> f32 {
    let ab_ca = ab.ca_atom_idx[r] as usize;
    let p     = ab.ca_pos(r);
    let brm   = ab.atoms.r_min_half[ab_ca];
    let be    = ab.atoms.epsilon[ab_ca];
    let bq    = ab.atoms.charge[ab_ca];
    let bh    = ab.atoms.hydrophobic[ab_ca];
    let mut e = 0.0_f32;
    for i in 0..ag.n_residues() {
        let ag_ca = ag.ca_atom_idx[i] as usize;
        let p_ag  = ag.ca_pos(i);
        let dx = p[0] - p_ag[0];
        let dy = p[1] - p_ag[1];
        let dz = p[2] - p_ag[2];
        let r_sq = dx * dx + dy * dy + dz * dz;
        if r_sq > energy::CUTOFF_SQ || r_sq < 0.25 { continue; }
        let arm  = ag.atoms.r_min_half[ag_ca];
        let ae   = ag.atoms.epsilon[ag_ca];
        let aq   = ag.atoms.charge[ag_ca];
        let ah   = ag.atoms.hydrophobic[ag_ca];
        let r_ij = brm + arm;
        let eps  = (be * ae).sqrt();
        let r2   = (r_ij * r_ij) / r_sq;
        let r6   = r2 * r2 * r2;
        e += eps * (r6 * r6 - 2.0 * r6);
        if bq != 0.0 && aq != 0.0 {
            e += 332.0 * bq * aq / r_sq.sqrt();
        }
        if bh == 1 && ah == 1 && r_sq < 36.0 {
            e -= 0.5;
        }
    }
    e
}

// ── All-atom public interface ─────────────────────────────────────────────────

/// Run the all-atom hybrid diffusion engine.
///
/// If a GPU context is provided, runs 1024-candidate GPU broad-sampling first
/// (gradient-only, 200 steps), then refines the top-64 with CPU Langevin +
/// rotamer MC (600 steps).  Without GPU, runs 64 candidates × 800 CPU steps.
pub fn run_allatom(
    antigen: &AtomProtein,
    ab_length: usize,
    #[cfg(feature = "gpu")] gpu: Option<&crate::gpu::GpuContext>,
) -> AllAtomResult {
    let center = antigen.ca_center_of_mass();

    // Pre-compute antigen Cα cloud and grid once; shared read-only across all steps.
    let ag_ca = {
        let n = antigen.n_residues();
        let mut cloud = ResidueCloud::with_capacity(n);
        for r in 0..n {
            let p = antigen.ca_pos(r);
            cloud.push(p[0], p[1], p[2], antigen.amino_acid[r]);
        }
        cloud
    };
    let mut ag_ca_grid = SpatialHashGrid::new(10.0);
    ag_ca_grid.build(&ag_ca.x, &ag_ca.y, &ag_ca.z);

    // ── GPU broad-sampling phase ──────────────────────────────────────────────
    #[cfg(feature = "gpu")]
    let survivors: Vec<AtomProtein> = if let Some(gpu_ctx) = gpu {
        // Build all 1024 candidates
        let mut all_cands: Vec<AtomProtein> = (0..AA_POPULATION)
            .map(|seed| {
                let mut rng = SmallRng::seed_from_u64(seed_u64(seed));
                random_allatom_antibody(ab_length, center, &mut rng)
            })
            .collect();

        // GPU gradient steps
        for iter in 0..GPU_STEPS {
            let temp = temperature(iter, GPU_STEPS + CPU_STEPS);
            // Move each candidate with gradient (no noise, no MC) in parallel
            all_cands.par_iter_mut().enumerate().for_each(|(seed, ab)| {
                let mut rng = SmallRng::seed_from_u64(seed_u64(seed + iter * AA_POPULATION));
                allatom_diffusion_step(ab, antigen, &ag_ca, &ag_ca_grid, center, temp, &mut rng, false);
            });

            // Score and prune to TOP_K once at the end of the GPU phase
            if iter == GPU_STEPS - 1 {
                let n_ab = all_cands[0].n_atoms();
                let gpu_atoms: Vec<crate::gpu::GpuAtom> = all_cands.iter().flat_map(|ab| {
                    (0..ab.n_atoms()).map(move |k| crate::gpu::GpuAtom {
                        x:           ab.atoms.x[k],
                        y:           ab.atoms.y[k],
                        z:           ab.atoms.z[k],
                        q:           ab.atoms.charge[k],
                        r_min_half:  ab.atoms.r_min_half[k],
                        epsilon:     ab.atoms.epsilon[k],
                        hydrophobic: ab.atoms.hydrophobic[k] as f32,
                        _pad:        0.0,
                    })
                }).collect();

                let energies = gpu_ctx.score_batch(&antigen.atoms, &gpu_atoms, n_ab);

                // Keep only TOP_K lowest-energy candidates
                let mut indexed: Vec<(usize, f32)> = energies.into_iter().enumerate().collect();
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(TOP_K);
                let keep_idxs: Vec<usize> = indexed.into_iter().map(|(i, _)| i).collect();

                let mut survivors_tmp: Vec<AtomProtein> = Vec::with_capacity(TOP_K);
                for idx in keep_idxs {
                    // Move the protein out; fill hole with default (will be overwritten or dropped)
                    let placeholder = AtomProtein::new();
                    let cand = std::mem::replace(&mut all_cands[idx], placeholder);
                    survivors_tmp.push(cand);
                }
                all_cands = survivors_tmp;
            }
        }
        // Ensure exactly TOP_K
        all_cands.truncate(TOP_K);
        all_cands
    } else {
        // No GPU: create TOP_K candidates for CPU-only path
        (0..TOP_K)
            .map(|seed| {
                let mut rng = SmallRng::seed_from_u64(seed_u64(seed));
                random_allatom_antibody(ab_length, center, &mut rng)
            })
            .collect()
    };

    // When compiled without GPU feature, always create TOP_K candidates directly
    #[cfg(not(feature = "gpu"))]
    let survivors: Vec<AtomProtein> = (0..TOP_K)
        .map(|seed| {
            let mut rng = SmallRng::seed_from_u64(seed_u64(seed));
            random_allatom_antibody(ab_length, center, &mut rng)
        })
        .collect();

    // ── CPU refinement phase (Rayon over survivors) ───────────────────────────
    let best = survivors
        .into_par_iter()
        .enumerate()
        .map(|(seed, mut ab)| {
            let mut rng = SmallRng::seed_from_u64(seed_u64(seed + 999_999));
            for iter in 0..CPU_STEPS {
                let temp = temperature(iter, CPU_STEPS);
                allatom_diffusion_step(&mut ab, antigen, &ag_ca, &ag_ca_grid, center, temp, &mut rng, true);
            }
            let e = allatom_energy(antigen, &ab);
            (ab, e)
        })
        .reduce_with(|a, b| if a.1 <= b.1 { a } else { b })
        .expect("survivors non-empty");

    let (antibody, energy) = best;
    let sequence = antibody.sequence();
    AllAtomResult { antibody, energy, sequence }
}
