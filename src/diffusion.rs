/// Generative reverse-diffusion engine.
///
/// Algorithm (per candidate):
///   1. Initialise antibody as random noise around the antigen CoM.
///   2. For N steps with annealing temperature T(t):
///      a. Compute LJ/Coulomb/hydrophobic force on every antibody residue.
///      b. Gradient step: x += η·F  (moves toward energy minimum).
///      c. Langevin noise:  x += σ_T · ξ   (σ_T = √(2·T·dt)).
///      d. Metropolis Monte Carlo: propose AA mutation; accept with P = min(1, exp(-ΔE/T)).
///   3. Return candidate with lowest interaction energy across the whole population.
///
/// Parallelism: a population of `POPULATION` independent candidates is mapped
/// over Rayon's thread pool.  Each worker owns its own ResidueCloud and SmallRng.
/// Zero shared mutable state → no Mutex, no Arc, no locks.
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::atom::{AminoAcid, ResidueCloud, AA_COUNT};
use crate::energy;
use crate::spatial::SpatialHashGrid;

// ── Hyperparameters ──────────────────────────────────────────────────────────

/// Number of parallel design candidates explored simultaneously.
pub const POPULATION: usize = 64;

/// Diffusion iterations per candidate.
pub const ITERATIONS: usize = 800;

/// Gradient-descent step size (Å per kcal/mol/Å of force).
const STEP: f32 = 0.08;

/// Langevin noise scale at T=1:  σ = NOISE_BASE * √T.
const NOISE_BASE: f32 = 0.25;

/// Initial and final simulated-annealing temperatures.
const T_HOT: f32 = 3.0;
const T_COLD: f32 = 0.02;

/// Per-residue MC mutation probability per iteration.
const MUTATION_P: f32 = 0.08;

/// Sphere radius around antigen CoM where the antibody is initialised (Å).
const INIT_RADIUS: f32 = 20.0;

// ── Public types ─────────────────────────────────────────────────────────────

pub struct DiffusionResult {
    pub antibody: ResidueCloud,
    pub energy: f32,
    pub sequence: String,
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Exponential annealing schedule: T(t) = T_HOT · (T_COLD/T_HOT)^(t/N).
#[inline(always)]
fn temperature(iter: usize) -> f32 {
    T_HOT * (T_COLD / T_HOT).powf(iter as f32 / ITERATIONS as f32)
}

/// Sample a uniform float in [−scale, +scale] using a SmallRng.
#[inline(always)]
fn noise(rng: &mut SmallRng, scale: f32) -> f32 {
    rng.gen_range(-scale..scale)
}

/// Build a randomised antibody starting cloud with `n` residues.
/// Residues are scattered on a shell around `center` with Gaussian scatter.
fn random_antibody(n: usize, center: [f32; 3], rng: &mut SmallRng) -> ResidueCloud {
    let mut cloud = ResidueCloud::with_capacity(n);
    for i in 0..n {
        // Distribute residues on a rough spiral around the antigen surface
        let angle = i as f32 * 2.399_f32; // golden angle
        let cx = center[0] + INIT_RADIUS * angle.cos() + noise(rng, 5.0);
        let cy = center[1] + noise(rng, INIT_RADIUS * 0.5);
        let cz = center[2] + INIT_RADIUS * angle.sin() + noise(rng, 5.0);
        let aa = AminoAcid::from_index(rng.gen_range(0..AA_COUNT));
        cloud.push(cx, cy, cz, aa);
    }
    cloud
}

/// Maximum displacement per atom per step (Å).  Clamps runaway LJ forces.
const MAX_DISP: f32 = 2.0;

/// Harmonic restraint spring constant (kcal/mol/Å²) keeping antibody
/// atoms near the antigen surface.  Prevents drifting into empty space.
const RESTRAINT_K: f32 = 0.02;

/// Single Langevin + Metropolis MC step for one candidate.
///
/// Uses the pre-built `SpatialHashGrid` of the antigen for O(n) force lookup
/// instead of the O(n·m) brute-force path.
fn diffusion_step(
    antibody: &mut ResidueCloud,
    antigen: &ResidueCloud,
    grid: &SpatialHashGrid,
    ag_center: [f32; 3],
    temp: f32,
    rng: &mut SmallRng,
    fx_buf: &mut Vec<f32>,
    fy_buf: &mut Vec<f32>,
    fz_buf: &mut Vec<f32>,
) {
    let n = antibody.len();

    // Ensure force buffers are large enough (no realloc after first iteration)
    if fx_buf.len() < n {
        fx_buf.resize(n, 0.0);
        fy_buf.resize(n, 0.0);
        fz_buf.resize(n, 0.0);
    }

    energy::compute_forces_with_grid(antigen, grid, antibody, fx_buf, fy_buf, fz_buf);

    let noise_sigma = NOISE_BASE * temp.sqrt();

    for j in 0..n {
        // Harmonic restraint: pull toward binding shell (r = INIT_RADIUS from CoM)
        let dx_c = antibody.x[j] - ag_center[0];
        let dy_c = antibody.y[j] - ag_center[1];
        let dz_c = antibody.z[j] - ag_center[2];
        let r = (dx_c * dx_c + dy_c * dy_c + dz_c * dz_c).sqrt().max(0.001);
        let excess = r - INIT_RADIUS;
        let f_r = -RESTRAINT_K * excess / r; // radial spring force / r
        fx_buf[j] += f_r * dx_c;
        fy_buf[j] += f_r * dy_c;
        fz_buf[j] += f_r * dz_c;

        // Gradient step with displacement clamping to prevent LJ explosion
        let disp_x = (STEP * fx_buf[j]).clamp(-MAX_DISP, MAX_DISP);
        let disp_y = (STEP * fy_buf[j]).clamp(-MAX_DISP, MAX_DISP);
        let disp_z = (STEP * fz_buf[j]).clamp(-MAX_DISP, MAX_DISP);
        antibody.x[j] += disp_x;
        antibody.y[j] += disp_y;
        antibody.z[j] += disp_z;

        // Langevin thermal noise
        antibody.x[j] += noise(rng, noise_sigma);
        antibody.y[j] += noise(rng, noise_sigma);
        antibody.z[j] += noise(rng, noise_sigma);

        // Metropolis MC sequence mutation
        if rng.gen::<f32>() < MUTATION_P {
            let old_aa = antibody.amino_acid[j];
            let new_aa = AminoAcid::from_index(rng.gen_range(0..AA_COUNT));

            // Compute ΔE of the mutation by swapping and re-evaluating a
            // single-residue contribution (cheap residue-level estimate).
            let old_e = single_residue_energy(j, antibody, antigen);
            antibody.set_aa(j, new_aa);
            let new_e = single_residue_energy(j, antibody, antigen);
            let delta_e = new_e - old_e;

            // Metropolis criterion: accept downhill moves always, uphill with exp(-ΔE/T)
            let accept = delta_e <= 0.0 || rng.gen::<f32>() < (-delta_e / temp).exp();
            if !accept {
                antibody.set_aa(j, old_aa);
            }
        }
    }
}

/// Fast single-residue energy contribution (only loops over antigen, O(|antigen|)).
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

        if r_sq > energy::CUTOFF_SQ || r_sq < 0.25 {
            continue;
        }

        let eps_ij = (be * antigen.epsilon[i]).sqrt();
        let sig_ij = 0.5 * (bs + antigen.sigma[i]);
        let sigma_sq = sig_ij * sig_ij;
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

// ── Public interface ─────────────────────────────────────────────────────────

/// Run the full population-based reverse diffusion and return the best binder.
///
/// * `antigen`    — the target protein (read-only, shared across all threads).
/// * `ab_length`  — desired antibody length in residues.
pub fn run(antigen: &ResidueCloud, ab_length: usize) -> DiffusionResult {
    let center = antigen.center_of_mass();

    // Build antigen spatial hash once; all Rayon threads share it read-only.
    // SpatialHashGrid is Sync (HashMap<_, Vec<_>> with Sync key/value types).
    let mut grid = SpatialHashGrid::new(10.0); // cell size = cutoff (Å)
    grid.build(&antigen.x, &antigen.y, &antigen.z);

    // Parallel population — each element is fully independent (no shared state)
    let best = (0..POPULATION)
        .into_par_iter()
        .map(|seed| {
            // Splitmix64-style scramble to spread seeds across the u64 range
            let seed_u64 = (seed as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let mut rng = SmallRng::seed_from_u64(seed_u64);

            let mut antibody = random_antibody(ab_length, center, &mut rng);

            let mut fx_buf: Vec<f32> = Vec::with_capacity(ab_length);
            let mut fy_buf: Vec<f32> = Vec::with_capacity(ab_length);
            let mut fz_buf: Vec<f32> = Vec::with_capacity(ab_length);

            for iter in 0..ITERATIONS {
                let temp = temperature(iter);
                diffusion_step(
                    &mut antibody,
                    antigen,
                    &grid,
                    center,
                    temp,
                    &mut rng,
                    &mut fx_buf,
                    &mut fy_buf,
                    &mut fz_buf,
                );
            }

            let energy = energy::interaction_energy(antigen, &antibody);
            (antibody, energy)
        })
        .reduce_with(|a, b| if a.1 <= b.1 { a } else { b })
        .expect("population is non-empty");

    let (antibody, energy) = best;
    let sequence = antibody.sequence();

    DiffusionResult {
        antibody,
        energy,
        sequence,
    }
}
