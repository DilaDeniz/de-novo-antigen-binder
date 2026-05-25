/// Physics-based scoring engine.
///
/// All distances passed in as r² to avoid sqrt wherever the cutoff check
/// or full formula permits (LJ, hydrophobic).  Coulomb requires 1/r so sqrt
/// is computed once per pair and reused.
use crate::atom::ResidueCloud;
use crate::spatial::SpatialHashGrid;

// Coulomb constant in kcal/mol·Å·e⁻² (= 332.0 in common AMBER/CHARMM units)
const COULOMB_K: f32 = 332.0;

// Non-bonded interaction cutoff (Å²); only pairs closer than √CUTOFF are evaluated.
pub const CUTOFF_SQ: f32 = 100.0; // 10 Å
const MIN_R_SQ: f32 = 0.25; // 0.5 Å — avoid singularity

// Hydrophobic bonus cutoff (Å²)
const HYDRO_CUTOFF_SQ: f32 = 36.0; // 6 Å
const HYDRO_BONUS: f32 = -0.5; // kcal/mol per hydrophobic–hydrophobic pair

/// Combined Lennard-Jones energy for a residue pair using pre-squared distance.
///
/// V_LJ = 4ε [ (σ/r)^12 − (σ/r)^6 ]
///
/// With s2 = σ²/r², s6 = s2³, s12 = s6²:
///   V_LJ = 4ε (s12 − s6)
///
/// Entire inner loop is branch-free and auto-vectorizable.
#[inline(always)]
fn lj_energy(eps_ij: f32, sigma_sq_ij: f32, r_sq: f32) -> f32 {
    let s2 = sigma_sq_ij / r_sq;
    let s6 = s2 * s2 * s2;
    let s12 = s6 * s6;
    4.0 * eps_ij * (s12 - s6)
}

/// Coulomb energy: V = k·q1·q2 / r.  Requires one sqrt.
#[inline(always)]
fn coulomb_energy(q1: f32, q2: f32, r_sq: f32) -> f32 {
    let r = r_sq.sqrt();
    COULOMB_K * q1 * q2 / r
}

/// Total non-bonded interaction energy between antigen and antibody (kcal/mol).
///
/// Fallback brute-force path (used for final scoring and testing).
/// Inner loops over antigen atoms are written to be auto-vectorized by LLVM:
/// antigen coordinate arrays are contiguous f32 slices, the antibody scalar
/// (bx, by, bz, …) stays in registers, and there are no pointer aliasing
/// concerns between antigen and antibody arrays.
pub fn interaction_energy(antigen: &ResidueCloud, antibody: &ResidueCloud) -> f32 {
    let ag_n = antigen.len();
    let ab_n = antibody.len();

    let mut total = 0.0_f32;

    for j in 0..ab_n {
        let bx = antibody.x[j];
        let by = antibody.y[j];
        let bz = antibody.z[j];
        let bq = antibody.charge[j];
        let be = antibody.epsilon[j];
        let bs = antibody.sigma[j];
        let bh = antibody.hydrophobic[j];

        // Inner loop — LLVM can vectorize dx/dy/dz/r_sq computation over i
        for i in 0..ag_n {
            let dx = bx - antigen.x[i];
            let dy = by - antigen.y[i];
            let dz = bz - antigen.z[i];
            let r_sq = dx * dx + dy * dy + dz * dz;

            if r_sq > CUTOFF_SQ || r_sq < MIN_R_SQ {
                continue;
            }

            // Lorentz-Berthelot mixing
            let eps_ij = (be * antigen.epsilon[i]).sqrt();
            let sig_ij = 0.5 * (bs + antigen.sigma[i]);
            let sigma_sq_ij = sig_ij * sig_ij;

            total += lj_energy(eps_ij, sigma_sq_ij, r_sq);

            let q1 = bq;
            let q2 = antigen.charge[i];
            if q1 != 0.0 && q2 != 0.0 {
                total += coulomb_energy(q1, q2, r_sq);
            }

            if bh == 1 && antigen.hydrophobic[i] == 1 && r_sq < HYDRO_CUTOFF_SQ {
                total += HYDRO_BONUS;
            }
        }
    }

    total
}

/// Compute force on every antibody atom due to all antigen atoms.
///
/// Force = −∇V.  For atom j:
///   F_j = Σ_i [ (24ε_ij/r²) (2(σ/r)^12 − (σ/r)^6) · r̂_ij ]
///         + Σ_i [ k·q_i·q_j / r³ · r_ij ]
///         + hydrophobic attraction
///
/// Outer loop over antibody (j) keeps j-scalars in registers.
/// Inner loop over antigen (i) reads contiguous SoA slices → SIMD friendly.
///
/// Brute-force O(|ag|·|ab|) reference path; used in tests to validate the
/// grid-accelerated version.
#[cfg_attr(not(test), allow(dead_code))]
pub fn compute_forces(
    antigen: &ResidueCloud,
    antibody: &ResidueCloud,
    fx: &mut [f32],
    fy: &mut [f32],
    fz: &mut [f32],
) {
    let ag_n = antigen.len();
    let ab_n = antibody.len();

    // Zero output buffers
    fx[..ab_n].fill(0.0);
    fy[..ab_n].fill(0.0);
    fz[..ab_n].fill(0.0);

    for j in 0..ab_n {
        let bx = antibody.x[j];
        let by = antibody.y[j];
        let bz = antibody.z[j];
        let bq = antibody.charge[j];
        let be = antibody.epsilon[j];
        let bs = antibody.sigma[j];
        let bh = antibody.hydrophobic[j];

        let mut acc_fx = 0.0_f32;
        let mut acc_fy = 0.0_f32;
        let mut acc_fz = 0.0_f32;

        for i in 0..ag_n {
            // r_vec points from antigen[i] → antibody[j]
            let dx = bx - antigen.x[i];
            let dy = by - antigen.y[i];
            let dz = bz - antigen.z[i];
            let r_sq = dx * dx + dy * dy + dz * dz;

            if r_sq > CUTOFF_SQ || r_sq < MIN_R_SQ {
                continue;
            }

            // Lorentz-Berthelot mixing
            let eps_ij = (be * antigen.epsilon[i]).sqrt();
            let sig_ij = 0.5 * (bs + antigen.sigma[i]);
            let sigma_sq_ij = sig_ij * sig_ij;

            // LJ force magnitude / r²
            // F_LJ = (24 ε / r²) · [ 2(σ/r)^12 − (σ/r)^6 ] · r_vec
            let s2 = sigma_sq_ij / r_sq;
            let s6 = s2 * s2 * s2;
            let s12 = s6 * s6;
            let f_lj = (24.0 * eps_ij / r_sq) * (2.0 * s12 - s6);

            // Coulomb force: F_C = k·q1·q2 / r³ · r_vec  =>  scale = k·q1·q2 / r_sq·√r_sq
            let f_coul = {
                let q1 = bq;
                let q2 = antigen.charge[i];
                if q1 != 0.0 && q2 != 0.0 {
                    let r = r_sq.sqrt();
                    COULOMB_K * q1 * q2 / (r_sq * r)
                } else {
                    0.0
                }
            };

            // Soft hydrophobic attraction toward antigen hydrophobics
            let f_hydro = if bh == 1 && antigen.hydrophobic[i] == 1 && r_sq < HYDRO_CUTOFF_SQ {
                -0.1 / r_sq.sqrt() // mild attractive pull, magnitude ∝ 1/r
            } else {
                0.0
            };

            let f_total = f_lj + f_coul + f_hydro;
            acc_fx += f_total * dx;
            acc_fy += f_total * dy;
            acc_fz += f_total * dz;
        }

        fx[j] = acc_fx;
        fy[j] = acc_fy;
        fz[j] = acc_fz;
    }
}

/// Grid-accelerated force computation using the antigen's SpatialHashGrid.
///
/// For each antibody atom j the spatial hash returns only the antigen atoms
/// within the cutoff cell neighbourhood, reducing work from O(|ag|·|ab|) to
/// O(|ab|·avg_neighbours).  The antigen grid is built once and shared as an
/// immutable reference across all Rayon threads.
pub fn compute_forces_with_grid(
    antigen: &ResidueCloud,
    grid: &SpatialHashGrid,
    antibody: &ResidueCloud,
    fx: &mut [f32],
    fy: &mut [f32],
    fz: &mut [f32],
) {
    let ab_n = antibody.len();

    fx[..ab_n].fill(0.0);
    fy[..ab_n].fill(0.0);
    fz[..ab_n].fill(0.0);

    for j in 0..ab_n {
        let bx = antibody.x[j];
        let by = antibody.y[j];
        let bz = antibody.z[j];
        let bq = antibody.charge[j];
        let be = antibody.epsilon[j];
        let bs = antibody.sigma[j];
        let bh = antibody.hydrophobic[j];

        let mut acc_fx = 0.0_f32;
        let mut acc_fy = 0.0_f32;
        let mut acc_fz = 0.0_f32;

        // Query only the 27 neighbouring cells — O(avg_density) per atom
        grid.query_neighbors(bx, by, bz, |raw_i| {
            let i = raw_i as usize;

            let dx = bx - antigen.x[i];
            let dy = by - antigen.y[i];
            let dz = bz - antigen.z[i];
            let r_sq = dx * dx + dy * dy + dz * dz;

            if r_sq > CUTOFF_SQ || r_sq < MIN_R_SQ {
                return;
            }

            let eps_ij = (be * antigen.epsilon[i]).sqrt();
            let sig_ij = 0.5 * (bs + antigen.sigma[i]);
            let sigma_sq_ij = sig_ij * sig_ij;

            let s2 = sigma_sq_ij / r_sq;
            let s6 = s2 * s2 * s2;
            let s12 = s6 * s6;
            let f_lj = (24.0 * eps_ij / r_sq) * (2.0 * s12 - s6);

            let f_coul = {
                let q1 = bq;
                let q2 = antigen.charge[i];
                if q1 != 0.0 && q2 != 0.0 {
                    let r = r_sq.sqrt();
                    COULOMB_K * q1 * q2 / (r_sq * r)
                } else {
                    0.0
                }
            };

            let f_hydro = if bh == 1 && antigen.hydrophobic[i] == 1 && r_sq < HYDRO_CUTOFF_SQ {
                -0.1 / r_sq.sqrt()
            } else {
                0.0
            };

            let f_total = f_lj + f_coul + f_hydro;
            acc_fx += f_total * dx;
            acc_fy += f_total * dy;
            acc_fz += f_total * dz;
        });

        fx[j] = acc_fx;
        fy[j] = acc_fy;
        fz[j] = acc_fz;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::AminoAcid;
    use crate::spatial::SpatialHashGrid;

    fn two_residue_clouds() -> (ResidueCloud, ResidueCloud) {
        let mut ag = ResidueCloud::with_capacity(2);
        ag.push(0.0, 0.0, 0.0, AminoAcid::Lys);
        ag.push(5.0, 0.0, 0.0, AminoAcid::Leu);

        let mut ab = ResidueCloud::with_capacity(1);
        ab.push(3.0, 0.0, 0.0, AminoAcid::Asp);

        (ag, ab)
    }

    /// Brute-force and grid-accelerated forces must agree to f32 tolerance.
    #[test]
    fn grid_forces_match_brute_force() {
        let (ag, ab) = two_residue_clouds();

        let n = ab.len();
        let mut fx_bf = vec![0.0f32; n];
        let mut fy_bf = vec![0.0f32; n];
        let mut fz_bf = vec![0.0f32; n];
        compute_forces(&ag, &ab, &mut fx_bf, &mut fy_bf, &mut fz_bf);

        let mut grid = SpatialHashGrid::new(10.0);
        grid.build(&ag.x, &ag.y, &ag.z);

        let mut fx_gr = vec![0.0f32; n];
        let mut fy_gr = vec![0.0f32; n];
        let mut fz_gr = vec![0.0f32; n];
        compute_forces_with_grid(&ag, &grid, &ab, &mut fx_gr, &mut fy_gr, &mut fz_gr);

        for j in 0..n {
            assert!((fx_bf[j] - fx_gr[j]).abs() < 1e-4, "fx mismatch at {j}");
            assert!((fy_bf[j] - fy_gr[j]).abs() < 1e-4, "fy mismatch at {j}");
            assert!((fz_bf[j] - fz_gr[j]).abs() < 1e-4, "fz mismatch at {j}");
        }
    }

    /// Lys (+1) at origin and Asp (−1) at 3 Å → net Coulomb attraction.
    #[test]
    fn opposite_charges_attract() {
        let (ag, ab) = two_residue_clouds();

        let n = ab.len();
        let mut fx = vec![0.0f32; n];
        let mut fy = vec![0.0f32; n];
        let mut fz = vec![0.0f32; n];
        compute_forces(&ag, &ab, &mut fx, &mut fy, &mut fz);

        // antibody Asp is +x from antigen Lys; attraction means fx[0] < 0
        assert!(fx[0] < 0.0, "expected attractive Coulomb force, got fx={}", fx[0]);
        let _ = (fy, fz);
    }

    /// Two Gly at r = 1 Å (well inside σ = 2.5 Å) → strong LJ repulsion.
    #[test]
    fn lj_repulsion_at_close_range() {
        let mut ag = ResidueCloud::with_capacity(1);
        ag.push(0.0, 0.0, 0.0, AminoAcid::Gly);

        let mut ab = ResidueCloud::with_capacity(1);
        ab.push(1.0, 0.0, 0.0, AminoAcid::Gly);

        let mut fx = vec![0.0f32; 1];
        let mut fy = vec![0.0f32; 1];
        let mut fz = vec![0.0f32; 1];
        compute_forces(&ag, &ab, &mut fx, &mut fy, &mut fz);
        assert!(fx[0] > 0.0, "expected LJ repulsion, got fx={}", fx[0]);
    }
}
