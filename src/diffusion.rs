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

use crate::allatom::{cross3, norm3, protein_from_ca_trace, AtomProtein};
use crate::amber::{HBOND_ACCEPTOR, HBOND_DONOR};
use crate::atom::{AminoAcid, ResidueCloud, ALL_AA, AA_COUNT};
use crate::energy;
use crate::germline;
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

/// Fv (two-chain VH/VL) population and iteration counts. Smaller than the
/// single-chain all-atom population because each candidate now carries two
/// full chains plus inter-chain (VH-VL) scoring.
pub const FV_POPULATION: usize = 8;
pub const FV_ITERATIONS: usize = 350;

/// Scale factor for each Fv chain's initial packing radius: `radius =
/// FV_LOCAL_RADIUS_SCALE * sqrt(n_residues)`. Chosen so initial Cα density
/// is comparable to a real globular domain rather than the thin-ring packing
/// used for the single-chain paths (which is fine for one short chain against
/// a fixed antigen, but produces severe self- and cross-chain overlap once
/// two full ~110-residue chains share the same ring).
const FV_LOCAL_RADIUS_SCALE: f32 = 1.3;
/// Extra clearance (Å) added on top of the two chains' own packing radii
/// when placing their initial centers, so the heavy and light starting
/// spheres never overlap by construction regardless of chain length.
const FV_CHAIN_MARGIN: f32 = 6.0;

/// Steepest-descent clash-relief step size and per-step displacement cap.
/// Deliberately smaller than the MC `STEP`/`MAX_DISP` since minimization
/// should settle clashes, not re-explore conformational space.
const MIN_STEP:     f32 = 0.04;
const MIN_MAX_DISP: f32 = 0.5;

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

/// Interface-adaptive MC: residues within this distance (Å²) of any antigen Cα
/// are sampled 3× more aggressively — they are the CDR-equivalent region.
const IFACE_CUTOFF_SQ: f32 = 64.0; // 8 Å
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

/// Run the coarse-grained diffusion engine, returning the `top_n` lowest-energy
/// candidates (sorted ascending) out of `population` independent trajectories.
pub fn run(
    antigen: &ResidueCloud,
    ab_length: usize,
    population: usize,
    iterations: usize,
    top_n: usize,
) -> Vec<DiffusionResult> {
    let center = antigen.center_of_mass();
    let mut grid = SpatialHashGrid::new(10.0);
    grid.build(&antigen.x, &antigen.y, &antigen.z);

    let mut results: Vec<(ResidueCloud, f32)> = (0..population)
        .into_par_iter()
        .map(|seed| {
            let mut rng = SmallRng::seed_from_u64(seed_u64(seed));
            let mut antibody = random_antibody(ab_length, center, &mut rng);
            let mut fx_buf = Vec::with_capacity(ab_length);
            let mut fy_buf = Vec::with_capacity(ab_length);
            let mut fz_buf = Vec::with_capacity(ab_length);
            for iter in 0..iterations {
                let temp = temperature(iter, iterations);
                diffusion_step(&mut antibody, antigen, &grid, center, temp, &mut rng,
                               &mut fx_buf, &mut fy_buf, &mut fz_buf);
            }
            let e = energy::interaction_energy(antigen, &antibody);
            (antibody, e)
        })
        .collect();

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_n.max(1));

    results
        .into_iter()
        .map(|(antibody, energy)| {
            let sequence = antibody.sequence();
            DiffusionResult { antibody, energy, sequence }
        })
        .collect()
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

/// Compute total AMBER LJ + Coulomb + H-bond + hydrophobic + solvation energy
/// between two AtomProteins, plus the antibody's intramolecular disulfide term.
pub fn allatom_energy(ag: &AtomProtein, ab: &AtomProtein) -> f32 {
    energy::interaction_energy_atoms(&ag.atoms, &ab.atoms) + ab.disulfide_energy()
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

    // Interface mask: residues within 8 Å of any antigen Cα get boosted MC rates.
    // These are the CDR-equivalent positions — the primary binding determinants.
    let interface: Vec<bool> = (0..n).map(|r| {
        let p = ab.ca_pos(r);
        (0..ag_ca.len()).any(|i| {
            let dx = p[0] - ag_ca.x[i];
            let dy = p[1] - ag_ca.y[i];
            let dz = p[2] - ag_ca.z[i];
            dx * dx + dy * dy + dz * dz < IFACE_CUTOFF_SQ
        })
    }).collect();

    // Metropolis AA mutation or rotamer move — interface-adaptive rates.
    for r in 0..n {
        // Interface residues: 3× higher mutation rate, 2× higher rotamer rate.
        // Non-interface (framework): 0.4× mutation rate to preserve structure.
        let eff_mut_p = if interface[r] { MUTATION_P * 3.0 } else { MUTATION_P * 0.4 };
        let eff_rot_p = if interface[r] { ROTAMER_MOVE_P * 2.0 } else { ROTAMER_MOVE_P };
        let p = rng.gen::<f32>();
        if p < eff_mut_p {
            let move_type = rng.gen::<f32>();
            if move_type < eff_rot_p {
                // Rotamer MC move
                let aa = ab.amino_acid[r];
                let rots = rotamers(aa);
                if !rots.is_empty() {
                    let old_chi = ab.chi[r];
                    let rot_idx = rng.gen_range(0..rots.len());
                    let new_rot = rots[rot_idx];

                    let old_e = residue_contribution_allatom(r, ab, ag) + ab.disulfide_energy();
                    ab.apply_rotamer(r, &new_rot);
                    let new_e = residue_contribution_allatom(r, ab, ag) + ab.disulfide_energy();
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

                let old_e = residue_contribution_allatom(r, ab, ag) + ab.disulfide_energy();
                let rots = rotamers(new_aa);
                let new_chi = if rots.is_empty() { [0.0f32; 4] } else { rots[0].chi };
                ab.mutate_residue(r, new_aa, new_chi);
                let new_e = residue_contribution_allatom(r, ab, ag) + ab.disulfide_energy();
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
            let old_e: f32 = (r..n).map(|i| residue_contribution_allatom(i, ab, ag)).sum::<f32>()
                + ab.disulfide_energy();
            if do_phi { ab.perturb_phi(r, delta); } else { ab.perturb_psi(r, delta); }
            let new_e: f32 = (r..n).map(|i| residue_contribution_allatom(i, ab, ag)).sum::<f32>()
                + ab.disulfide_energy();
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
    let bt    = ab.atoms.atom_type[ab_ca] as usize;
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
        let at   = ag.atoms.atom_type[ag_ca] as usize;
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
        if (HBOND_DONOR[bt] && HBOND_ACCEPTOR[at]) || (HBOND_ACCEPTOR[bt] && HBOND_DONOR[at]) {
            e += energy::hbond_energy(r_sq);
        }
    }
    e
}

/// Build a residue-level Cα cloud from an all-atom protein, for use with the
/// fast residue-level force routines (`compute_forces`/`compute_forces_with_grid`).
fn residue_ca_cloud(prot: &AtomProtein) -> ResidueCloud {
    let n = prot.n_residues();
    let mut cloud = ResidueCloud::with_capacity(n);
    for r in 0..n {
        let p = prot.ca_pos(r);
        cloud.push(p[0], p[1], p[2], prot.amino_acid[r]);
    }
    cloud
}

// ── Fv (two-chain VH/VL) engine ───────────────────────────────────────────────
//
// Full de novo Ig-fold prediction is out of scope; instead we hold a
// realistic human germline framework fixed (see `germline.rs`) and only
// actively design the CDR1-3 loops, mirroring real CDR-grafting workflows.
// Both chains are scored against the antigen AND against each other
// (VH-VL packing), reusing the existing generic `residue_contribution_allatom`.

/// Per-chain result from the Fv search.
pub struct FvResult {
    pub heavy: AtomProtein,
    pub light: AtomProtein,
    pub heavy_regions: Vec<germline::Region>,
    pub light_regions: Vec<germline::Region>,
    pub heavy_sequence: String,
    pub light_sequence: String,
    pub energy: f32,
}

/// Evenly distribute point `i` of `n` across a sphere of `radius` centered at
/// the origin (Fibonacci/golden-angle sphere mapping). Unlike a single-axis
/// ring spiral, density grows with the sphere's surface area (∝ radius²)
/// rather than its circumference, so a ~100+ residue chain doesn't end up
/// packed into a thin overlapping ring.
fn fib_sphere_offset(i: usize, n: usize, radius: f32) -> [f32; 3] {
    let n_f = (n.max(1)) as f32;
    let y = 1.0 - 2.0 * (i as f32 + 0.5) / n_f; // (-1, 1)
    let r_y = (1.0 - y * y).max(0.0).sqrt();
    let theta = 2.399_963_f32 * i as f32;
    [theta.cos() * r_y * radius, y * radius, theta.sin() * r_y * radius]
}

/// Build one Fv chain (heavy or light) as a compact globule centered at
/// `chain_center`, with germline framework residues fixed and CDR loops
/// randomized. Residues are spread over a sphere surface sized to the
/// chain's own length so initial Cα density resembles a folded domain,
/// rather than wrapping the entire antigen on a shared thin ring.
fn random_fv_chain(
    fw: &germline::FrameworkLayout,
    chain_center: [f32; 3],
    rng: &mut SmallRng,
) -> (AtomProtein, Vec<germline::Region>) {
    let cdr1 = germline::random_cdr(fw.cdr1_len, rng);
    let cdr2 = germline::random_cdr(fw.cdr2_len, rng);
    let cdr3 = germline::random_cdr(fw.cdr3_len, rng);
    let (seq, regions) = germline::assemble_chain(fw, &cdr1, &cdr2, &cdr3);

    let n = seq.len();
    let local_radius = FV_LOCAL_RADIUS_SCALE * (n as f32).sqrt();
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n);
    for i in 0..n {
        let off = fib_sphere_offset(i, n, local_radius);
        xs.push(chain_center[0] + off[0] + noise(rng, 1.5));
        ys.push(chain_center[1] + off[1] + noise(rng, 1.5));
        zs.push(chain_center[2] + off[2] + noise(rng, 1.5));
    }
    let prot = protein_from_ca_trace(&xs, &ys, &zs, &seq);
    (prot, regions)
}

/// Total Fv binding energy: antigen-VH + antigen-VL + VH-VL packing, plus
/// each chain's intradomain disulfide bonus.
pub fn fv_energy(ag: &AtomProtein, h: &AtomProtein, l: &AtomProtein) -> f32 {
    energy::interaction_energy_atoms(&ag.atoms, &h.atoms)
        + energy::interaction_energy_atoms(&ag.atoms, &l.atoms)
        + energy::interaction_energy_atoms(&h.atoms, &l.atoms)
        + h.disulfide_energy()
        + l.disulfide_energy()
}

/// Per-residue energy contribution against both the antigen and the partner
/// chain (e.g. heavy residue `r` scored vs. antigen and vs. the light chain).
fn fv_residue_contribution(r: usize, ab: &AtomProtein, ag: &AtomProtein, partner: &AtomProtein) -> f32 {
    residue_contribution_allatom(r, ab, ag) + residue_contribution_allatom(r, ab, partner)
}

/// One MC step for a single Fv chain. Identical in structure to
/// `allatom_diffusion_step`, except: (a) energy is scored against the
/// antigen AND the partner chain, and (b) amino-acid mutation moves are
/// restricted to CDR-labeled positions — the germline framework sequence is
/// held fixed, matching real CDR-grafting design.
fn fv_chain_step(
    ab: &mut AtomProtein,
    regions: &[germline::Region],
    ag: &AtomProtein,
    ag_ca: &ResidueCloud,
    ag_ca_grid: &SpatialHashGrid,
    partner: &AtomProtein,
    ag_center: [f32; 3],
    target_radius: f32,
    temp: f32,
    rng: &mut SmallRng,
    with_noise: bool,
) {
    let n = ab.n_residues();

    let mut fx_buf = vec![0.0_f32; n];
    let mut fy_buf = vec![0.0_f32; n];
    let mut fz_buf = vec![0.0_f32; n];

    let ca_cloud = residue_ca_cloud(ab);
    energy::compute_forces_with_grid(ag_ca, ag_ca_grid, &ca_cloud, &mut fx_buf, &mut fy_buf, &mut fz_buf);

    // Inter-chain (VH-VL packing) force. Fv chains are short, so a direct
    // O(n²) pass against the partner's Cα cloud is cheap — no grid needed.
    let partner_ca = residue_ca_cloud(partner);
    let mut pfx = vec![0.0_f32; n];
    let mut pfy = vec![0.0_f32; n];
    let mut pfz = vec![0.0_f32; n];
    energy::compute_forces(&partner_ca, &ca_cloud, &mut pfx, &mut pfy, &mut pfz);
    for r in 0..n {
        fx_buf[r] += pfx[r];
        fy_buf[r] += pfy[r];
        fz_buf[r] += pfz[r];
    }

    let noise_sigma = if with_noise { NOISE_BASE * temp.sqrt() } else { 0.0 };

    for r in 0..n {
        let ca = ab.ca_pos(r);
        let dx_c = ca[0] - ag_center[0];
        let dy_c = ca[1] - ag_center[1];
        let dz_c = ca[2] - ag_center[2];
        let d = (dx_c * dx_c + dy_c * dy_c + dz_c * dz_c).sqrt().max(0.001);
        let excess = d - target_radius;
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

    // Sequence-changing MC moves are restricted to CDR loops (germline
    // framework identity is held fixed); framework residues only get
    // rotamer relaxation.
    for r in 0..n {
        let is_cdr = regions[r].is_cdr();

        if is_cdr && rng.gen::<f32>() < MUTATION_P * 3.0 {
            if rng.gen::<f32>() < ROTAMER_MOVE_P * 2.0 {
                let aa = ab.amino_acid[r];
                let rots = rotamers(aa);
                if !rots.is_empty() {
                    let old_chi = ab.chi[r];
                    let rot_idx = rng.gen_range(0..rots.len());
                    let new_rot = rots[rot_idx];
                    let old_e = fv_residue_contribution(r, ab, ag, partner) + ab.disulfide_energy();
                    ab.apply_rotamer(r, &new_rot);
                    let new_e = fv_residue_contribution(r, ab, ag, partner) + ab.disulfide_energy();
                    let delta_e = new_e - old_e;
                    let accept = delta_e <= 0.0 || rng.gen::<f32>() < (-delta_e / temp).exp();
                    if !accept {
                        ab.chi[r] = old_chi;
                        ab.rebuild_side_chain(r);
                    }
                }
            } else {
                let old_aa  = ab.amino_acid[r];
                let new_aa  = AminoAcid::from_index(rng.gen_range(0..AA_COUNT));
                let old_chi = ab.chi[r];
                let old_e = fv_residue_contribution(r, ab, ag, partner) + ab.disulfide_energy();
                let rots = rotamers(new_aa);
                let new_chi = if rots.is_empty() { [0.0f32; 4] } else { rots[0].chi };
                ab.mutate_residue(r, new_aa, new_chi);
                let new_e = fv_residue_contribution(r, ab, ag, partner) + ab.disulfide_energy();
                let delta_e = new_e - old_e;
                let accept = delta_e <= 0.0 || rng.gen::<f32>() < (-delta_e / temp).exp();
                if !accept {
                    ab.mutate_residue(r, old_aa, old_chi);
                }
            }
        } else if rng.gen::<f32>() < ROTAMER_MOVE_P * 0.3 {
            // Framework: rotamer-only relaxation, no identity change.
            let aa = ab.amino_acid[r];
            let rots = rotamers(aa);
            if !rots.is_empty() {
                let old_chi = ab.chi[r];
                let rot_idx = rng.gen_range(0..rots.len());
                let new_rot = rots[rot_idx];
                let old_e = fv_residue_contribution(r, ab, ag, partner) + ab.disulfide_energy();
                ab.apply_rotamer(r, &new_rot);
                let new_e = fv_residue_contribution(r, ab, ag, partner) + ab.disulfide_energy();
                let delta_e = new_e - old_e;
                let accept = delta_e <= 0.0 || rng.gen::<f32>() < (-delta_e / temp).exp();
                if !accept {
                    ab.chi[r] = old_chi;
                    ab.rebuild_side_chain(r);
                }
            }
        }
    }

    // Backbone phi/psi torsion MC moves — propagating, so evaluate r..n.
    for r in 0..n {
        if rng.gen::<f32>() < BACKBONE_MOVE_P {
            let delta = rng.gen_range(-MAX_TORSION_DEG..MAX_TORSION_DEG).to_radians();
            let do_phi = rng.gen::<bool>();
            let old_e: f32 = (r..n).map(|i| fv_residue_contribution(i, ab, ag, partner)).sum::<f32>()
                + ab.disulfide_energy();
            if do_phi { ab.perturb_phi(r, delta); } else { ab.perturb_psi(r, delta); }
            let new_e: f32 = (r..n).map(|i| fv_residue_contribution(i, ab, ag, partner)).sum::<f32>()
                + ab.disulfide_energy();
            let accept = (new_e - old_e) <= 0.0
                || rng.gen::<f32>() < (-(new_e - old_e) / temp).exp();
            if !accept {
                if do_phi { ab.perturb_phi(r, -delta); } else { ab.perturb_psi(r, -delta); }
            }
        }
    }
}

/// Run the two-chain Fv (heavy + light) diffusion engine, returning the
/// `top_n` lowest-energy candidates. `h3_len`/`l3_len` set the designed
/// CDR-H3/CDR-L3 loop lengths (the most variable, typically binding-dominant
/// loops); CDR1/2 lengths follow the germline framework.
pub fn run_fv(
    antigen: &AtomProtein,
    h3_len: usize,
    l3_len: usize,
    population: usize,
    iterations: usize,
    top_n: usize,
) -> Vec<FvResult> {
    let center = antigen.ca_center_of_mass();
    let ag_ca = residue_ca_cloud(antigen);
    let mut ag_ca_grid = SpatialHashGrid::new(10.0);
    ag_ca_grid.build(&ag_ca.x, &ag_ca.y, &ag_ca.z);

    let mut vh_fw = germline::VH_FRAMEWORK;
    vh_fw.cdr3_len = h3_len.max(1);
    let mut vl_fw = germline::VL_FRAMEWORK;
    vl_fw.cdr3_len = l3_len.max(1);

    // Initial heavy/light packing-sphere radii and the center separation
    // that keeps them just clear of each other (see FV_CHAIN_MARGIN).
    let radius_h = FV_LOCAL_RADIUS_SCALE * (vh_fw.chain_len() as f32).sqrt();
    let radius_l = FV_LOCAL_RADIUS_SCALE * (vl_fw.chain_len() as f32).sqrt();
    let half_sep = (radius_h + radius_l + FV_CHAIN_MARGIN) * 0.5;
    // Push the approach point far enough out that even the near edge of
    // each chain's own packing sphere clears the antigen's bulk (whose
    // atoms can sit well inside the thin INIT_RADIUS shell used for the
    // single-chain paths) — otherwise the local cluster's near side can
    // land inside the antigen itself.
    let approach_radius = INIT_RADIUS + radius_h.max(radius_l) + 5.0;
    // The radial restraint (which pulls every residue back toward a fixed
    // distance from the antigen center) must target the chain's actual mean
    // distance once offset by `half_sep`, or it will fight the placement
    // above and drag the chain back into the antigen as iterations proceed.
    let target_radius = (approach_radius * approach_radius + half_sep * half_sep).sqrt();

    let mut results: Vec<(AtomProtein, AtomProtein, Vec<germline::Region>, Vec<germline::Region>, f32)> =
        (0..population)
            .into_par_iter()
            .map(|seed| {
                let mut rng = SmallRng::seed_from_u64(seed_u64(seed));

                // Pick one random approach direction toward the antigen and
                // place both chains as two separated clusters near that same
                // face (mirroring how a real Fv presents both CDR sets to one
                // epitope), rather than wrapping the whole antigen on a
                // shared ring (which, for two full-length chains, guarantees
                // severe initial overlap).
                let theta = rng.gen_range(0.0..std::f32::consts::TAU);
                let cos_phi = rng.gen_range(-1.0_f32..1.0_f32);
                let sin_phi = (1.0 - cos_phi * cos_phi).max(0.0).sqrt();
                let dir = [sin_phi * theta.cos(), cos_phi, sin_phi * theta.sin()];
                let approach = [
                    center[0] + dir[0] * approach_radius,
                    center[1] + dir[1] * approach_radius,
                    center[2] + dir[2] * approach_radius,
                ];
                let up = if dir[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
                let perp = norm3(cross3(dir, up));
                let heavy_center = [
                    approach[0] + perp[0] * half_sep,
                    approach[1] + perp[1] * half_sep,
                    approach[2] + perp[2] * half_sep,
                ];
                let light_center = [
                    approach[0] - perp[0] * half_sep,
                    approach[1] - perp[1] * half_sep,
                    approach[2] - perp[2] * half_sep,
                ];

                let (mut heavy, h_regions) = random_fv_chain(&vh_fw, heavy_center, &mut rng);
                let (mut light, l_regions) = random_fv_chain(&vl_fw, light_center, &mut rng);

                // The per-iteration Cα step below is gradient + noise on a
                // coarse, generic-per-residue force field, while `fv_energy`
                // (the actual ranking criterion) is the full all-atom AMBER
                // energy with real per-atom radii. Since the position update
                // itself carries no accept/reject, an unlucky gradient
                // direction (especially once two whole chains are pulling on
                // each other) can drift the true all-atom energy far worse
                // over many iterations with nothing to pull it back. Track
                // the best all-atom state seen and fall back to it whenever
                // an iteration makes the true energy worse, so the search is
                // elitist with respect to ground truth even though the
                // underlying move proposal is coarse.
                let mut best_heavy = heavy.clone();
                let mut best_light = light.clone();
                let mut best_e = fv_energy(antigen, &heavy, &light);

                for iter in 0..iterations {
                    let temp = temperature(iter, iterations);
                    let light_snapshot = light.clone();
                    fv_chain_step(&mut heavy, &h_regions, antigen, &ag_ca, &ag_ca_grid,
                                  &light_snapshot, center, target_radius, temp, &mut rng, true);
                    let heavy_snapshot = heavy.clone();
                    fv_chain_step(&mut light, &l_regions, antigen, &ag_ca, &ag_ca_grid,
                                  &heavy_snapshot, center, target_radius, temp, &mut rng, true);

                    let e = fv_energy(antigen, &heavy, &light);
                    if e < best_e {
                        best_e = e;
                        best_heavy = heavy.clone();
                        best_light = light.clone();
                    } else if e > best_e {
                        heavy = best_heavy.clone();
                        light = best_light.clone();
                    }
                }

                (best_heavy, best_light, h_regions, l_regions, best_e)
            })
            .collect();

    results.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_n.max(1));

    results
        .into_iter()
        .map(|(heavy, light, heavy_regions, light_regions, energy)| {
            let heavy_sequence = heavy.sequence();
            let light_sequence = light.sequence();
            FvResult { heavy, light, heavy_regions, light_regions, heavy_sequence, light_sequence, energy }
        })
        .collect()
}

// ── Energy minimization (clash relief) ────────────────────────────────────────

/// Total all-atom energy of `ab` against all `partners` (interaction energy
/// summed over each partner, plus `ab`'s own disulfide term) — the same
/// quantity `minimize` is meant to be lowering, used as its ground-truth
/// acceptance check.
fn partners_energy(ab: &AtomProtein, partners: &[&AtomProtein]) -> f32 {
    partners
        .iter()
        .map(|p| energy::interaction_energy_atoms(&p.atoms, &ab.atoms))
        .sum::<f32>()
        + ab.disulfide_energy()
}

/// Steepest-descent clash relief on Cα positions against all `partners` (no
/// noise, no identity/rotamer changes), with side chains rebuilt after each
/// step. Step size decays linearly to zero over `steps`.
///
/// The Cα-level gradient is only a coarse proxy for the full all-atom AMBER
/// energy actually used for scoring (real per-atom radii vs. generic
/// per-residue ones), so a raw gradient step can occasionally *increase* the
/// true all-atom energy (e.g. a side-chain rebuild after a Cα move lands
/// atoms in a worse clash than before). Each step is therefore checked
/// against the real all-atom energy and rolled back if it didn't actually
/// help, guaranteeing `minimize` never leaves `ab` worse than it found it.
pub fn minimize(ab: &mut AtomProtein, partners: &[&AtomProtein], steps: usize) {
    let n = ab.n_residues();
    if n == 0 || partners.is_empty() || steps == 0 {
        return;
    }

    let mut best_e = partners_energy(ab, partners);

    let partner_grids: Vec<(ResidueCloud, SpatialHashGrid)> = partners
        .iter()
        .map(|p| {
            let cloud = residue_ca_cloud(p);
            let mut grid = SpatialHashGrid::new(10.0);
            grid.build(&cloud.x, &cloud.y, &cloud.z);
            (cloud, grid)
        })
        .collect();

    let mut fx_buf = vec![0.0_f32; n];
    let mut fy_buf = vec![0.0_f32; n];
    let mut fz_buf = vec![0.0_f32; n];
    let mut pfx = vec![0.0_f32; n];
    let mut pfy = vec![0.0_f32; n];
    let mut pfz = vec![0.0_f32; n];

    for step in 0..steps {
        let frac = 1.0 - step as f32 / steps as f32;
        let step_size = MIN_STEP * frac;

        let before = ab.clone();
        let ca_cloud = residue_ca_cloud(ab);
        fx_buf.iter_mut().for_each(|v| *v = 0.0);
        fy_buf.iter_mut().for_each(|v| *v = 0.0);
        fz_buf.iter_mut().for_each(|v| *v = 0.0);

        for (cloud, grid) in &partner_grids {
            energy::compute_forces_with_grid(cloud, grid, &ca_cloud, &mut pfx, &mut pfy, &mut pfz);
            for r in 0..n {
                fx_buf[r] += pfx[r];
                fy_buf[r] += pfy[r];
                fz_buf[r] += pfz[r];
            }
        }

        for r in 0..n {
            let ca = ab.ca_pos(r);
            let disp = [
                (step_size * fx_buf[r]).clamp(-MIN_MAX_DISP, MIN_MAX_DISP),
                (step_size * fy_buf[r]).clamp(-MIN_MAX_DISP, MIN_MAX_DISP),
                (step_size * fz_buf[r]).clamp(-MIN_MAX_DISP, MIN_MAX_DISP),
            ];
            ab.set_ca_pos(r, [ca[0] + disp[0], ca[1] + disp[1], ca[2] + disp[2]]);
            ab.rebuild_side_chain(r);
        }

        let new_e = partners_energy(ab, partners);
        if new_e < best_e {
            best_e = new_e;
        } else {
            *ab = before;
        }
    }
}

// ── Affinity maturation (point mutation scan) ─────────────────────────────────

/// Greedy single-pass affinity maturation: for each mutable position, try all
/// 20 amino acids (best rotamer for each) and keep the first strictly
/// improving mutation found, scored by the caller-supplied `score` closure
/// (lower is better). Returns the final score.
fn affinity_maturation<F: Fn(&AtomProtein) -> f32>(
    ab: &mut AtomProtein,
    mutable: &[usize],
    score: F,
) -> f32 {
    let mut best = score(ab);
    for &r in mutable {
        let old_aa  = ab.amino_acid[r];
        let old_chi = ab.chi[r];
        for &candidate_aa in ALL_AA.iter() {
            if candidate_aa == old_aa {
                continue;
            }
            let rots = rotamers(candidate_aa);
            let new_chi = if rots.is_empty() { [0.0f32; 4] } else { rots[0].chi };
            ab.mutate_residue(r, candidate_aa, new_chi);
            let e = score(ab);
            if e < best {
                best = e;
                // Keep this mutation and stop scanning candidates for `r`:
                // trying further candidates against the *new* `best` would
                // otherwise revert to `old_aa` on a later non-improving
                // candidate, silently discarding this accepted improvement
                // while `best` kept reporting the lower score.
                break;
            } else {
                ab.mutate_residue(r, old_aa, old_chi);
            }
        }
    }
    best
}

/// Affinity maturation for the legacy single-chain all-atom path: scans
/// interface (CDR-equivalent) positions against the antigen.
pub fn affinity_maturation_allatom(ab: &mut AtomProtein, ag: &AtomProtein, mutable: &[usize]) -> f32 {
    affinity_maturation(ab, mutable, |cand| allatom_energy(ag, cand))
}

/// Affinity maturation for the Fv path: scans CDR positions of one chain,
/// scoring against both the antigen and the (fixed) partner chain.
pub fn affinity_maturation_fv(
    ab: &mut AtomProtein,
    ag: &AtomProtein,
    partner: &AtomProtein,
    mutable: &[usize],
) -> f32 {
    affinity_maturation(ab, mutable, |cand| fv_energy(ag, cand, partner))
}

// ── All-atom public interface ─────────────────────────────────────────────────

/// Run the all-atom hybrid diffusion engine, returning the `top_n` lowest-energy
/// candidates (sorted ascending).
///
/// If a GPU context is provided, runs 1024-candidate GPU broad-sampling first
/// (gradient-only, 200 steps), then refines the top `population` with CPU
/// Langevin + rotamer MC (`cpu_iterations` steps).  Without GPU, runs
/// `population` candidates × `cpu_iterations` CPU steps directly.
pub fn run_allatom(
    antigen: &AtomProtein,
    ab_length: usize,
    population: usize,
    cpu_iterations: usize,
    top_n: usize,
    #[cfg(feature = "gpu")] gpu: Option<&crate::gpu::GpuContext>,
) -> Vec<AllAtomResult> {
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
            let temp = temperature(iter, GPU_STEPS + cpu_iterations);
            // Move each candidate with gradient (no noise, no MC) in parallel
            all_cands.par_iter_mut().enumerate().for_each(|(seed, ab)| {
                let mut rng = SmallRng::seed_from_u64(seed_u64(seed + iter * AA_POPULATION));
                allatom_diffusion_step(ab, antigen, &ag_ca, &ag_ca_grid, center, temp, &mut rng, false);
            });

            // Score and prune to `population` once at the end of the GPU phase
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

                // Keep only the `population` lowest-energy candidates
                let mut indexed: Vec<(usize, f32)> = energies.into_iter().enumerate().collect();
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(population);
                let keep_idxs: Vec<usize> = indexed.into_iter().map(|(i, _)| i).collect();

                let mut survivors_tmp: Vec<AtomProtein> = Vec::with_capacity(population);
                for idx in keep_idxs {
                    // Move the protein out; fill hole with default (will be overwritten or dropped)
                    let placeholder = AtomProtein::new();
                    let cand = std::mem::replace(&mut all_cands[idx], placeholder);
                    survivors_tmp.push(cand);
                }
                all_cands = survivors_tmp;
            }
        }
        // Ensure exactly `population`
        all_cands.truncate(population);
        all_cands
    } else {
        // No GPU: create `population` candidates for CPU-only path
        (0..population)
            .map(|seed| {
                let mut rng = SmallRng::seed_from_u64(seed_u64(seed));
                random_allatom_antibody(ab_length, center, &mut rng)
            })
            .collect()
    };

    // When compiled without GPU feature, always create `population` candidates directly
    #[cfg(not(feature = "gpu"))]
    let survivors: Vec<AtomProtein> = (0..population)
        .map(|seed| {
            let mut rng = SmallRng::seed_from_u64(seed_u64(seed));
            random_allatom_antibody(ab_length, center, &mut rng)
        })
        .collect();

    // ── CPU refinement phase (Rayon over survivors) ───────────────────────────
    let mut results: Vec<(AtomProtein, f32)> = survivors
        .into_par_iter()
        .enumerate()
        .map(|(seed, mut ab)| {
            let mut rng = SmallRng::seed_from_u64(seed_u64(seed + 999_999));
            for iter in 0..cpu_iterations {
                let temp = temperature(iter, cpu_iterations);
                allatom_diffusion_step(&mut ab, antigen, &ag_ca, &ag_ca_grid, center, temp, &mut rng, true);
            }
            let e = allatom_energy(antigen, &ab);
            (ab, e)
        })
        .collect();

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_n.max(1));

    results
        .into_iter()
        .map(|(antibody, energy)| {
            let sequence = antibody.sequence();
            AllAtomResult { antibody, energy, sequence }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::AminoAcid;

    /// A small all-atom protein with `n` Ala residues placed on a straight
    /// Cα trace starting at `start`, spaced 3.8 Å apart along x.
    fn straight_chain_at(start: [f32; 3], n: usize) -> AtomProtein {
        let xs: Vec<f32> = (0..n).map(|i| start[0] + i as f32 * 3.8).collect();
        let ys = vec![start[1]; n];
        let zs = vec![start[2]; n];
        let seq = vec![AminoAcid::Ala; n];
        protein_from_ca_trace(&xs, &ys, &zs, &seq)
    }

    #[test]
    fn minimize_never_increases_true_partner_energy() {
        // Deliberately overlapping placement: antibody starts right on top
        // of the partner, guaranteeing severe initial clashes.
        let partner = straight_chain_at([0.0, 0.0, 0.0], 6);
        let mut ab = straight_chain_at([0.2, 0.2, 0.2], 6);

        let partners: Vec<&AtomProtein> = vec![&partner];
        let before = partners_energy(&ab, &partners);

        minimize(&mut ab, &partners, 50);

        let after = partners_energy(&ab, &partners);
        assert!(
            after <= before + 1e-3,
            "minimize made the true energy worse: before={before}, after={after}"
        );
    }

    #[test]
    fn minimize_is_a_noop_with_zero_steps() {
        let partner = straight_chain_at([0.0, 0.0, 0.0], 4);
        let mut ab = straight_chain_at([20.0, 20.0, 20.0], 4);
        let partners: Vec<&AtomProtein> = vec![&partner];

        let before = partners_energy(&ab, &partners);
        minimize(&mut ab, &partners, 0);
        let after = partners_energy(&ab, &partners);

        assert_eq!(before, after);
    }

    #[test]
    fn affinity_maturation_never_worsens_the_score() {
        let ag = straight_chain_at([0.0, 0.0, 0.0], 6);
        let mut ab = straight_chain_at([8.0, 0.0, 0.0], 6);
        let mutable: Vec<usize> = (0..ab.n_residues()).collect();

        let before = allatom_energy(&ag, &ab);
        let after = affinity_maturation_allatom(&mut ab, &ag, &mutable);

        assert!(
            after <= before + 1e-3,
            "affinity maturation worsened the score: before={before}, after={after}"
        );
        // The returned score must also match what re-scoring the mutated
        // structure actually gives.
        assert!((after - allatom_energy(&ag, &ab)).abs() < 1e-3);
    }

    #[test]
    fn affinity_maturation_fv_never_worsens_the_score() {
        let ag = straight_chain_at([0.0, 0.0, 0.0], 6);
        let partner = straight_chain_at([12.0, 0.0, 0.0], 6);
        let mut heavy = straight_chain_at([6.0, 6.0, 0.0], 6);
        let mutable: Vec<usize> = (0..heavy.n_residues()).collect();

        let before = fv_energy(&ag, &heavy, &partner);
        let after = affinity_maturation_fv(&mut heavy, &ag, &partner, &mutable);

        assert!(
            after <= before + 1e-3,
            "Fv affinity maturation worsened the score: before={before}, after={after}"
        );
    }

    #[test]
    fn fib_sphere_offsets_stay_within_their_target_radius() {
        let radius = 5.0;
        let n = 20;
        for i in 0..n {
            let off = fib_sphere_offset(i, n, radius);
            let d = (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]).sqrt();
            assert!((d - radius).abs() < 1e-3, "offset {i} has radius {d}, expected {radius}");
        }
    }

    #[test]
    fn run_fv_produces_a_physically_sane_energy() {
        // Tiny, fast smoke test: 1 candidate, few iterations. Mirrors the
        // pop=1/iter=150 manual check that confirmed the energy-explosion
        // fix end-to-end, just cheap enough to run in CI.
        let antigen = straight_chain_at([0.0, 0.0, 0.0], 20);
        let results = run_fv(&antigen, 4, 4, 1, 5, 1);

        assert_eq!(results.len(), 1);
        let e = results[0].energy;
        assert!(e.is_finite(), "Fv energy is not finite: {e}");
        // A real binder may be modestly negative or near zero, but a
        // construction-level bug (e.g. chains overlapping the antigen or
        // each other at placement time) explodes into the millions.
        assert!(
            e.abs() < 10_000.0,
            "Fv energy is not physically sane: {e}"
        );
    }
}
