/// EEF1 implicit solvation (Lazaridis & Karplus, 1999, Proteins 35:133).
///
/// Models the free energy cost/gain of burial in the context of protein–protein
/// interaction.  For a pair of atoms (i from antibody, j from antigen) within
/// the cutoff, the cross-burial contribution to the binding ΔG_solvation is:
///
///   ΔG_ij = −ΔG_ref_i · V_j · f(r_ij, λ_i)
///           −ΔG_ref_j · V_i · f(r_ij, λ_j)
///
/// where f(r, λ) = exp(−(r/λ)²) / (π^(3/2) · λ³).
///
/// Sign convention consistent with energy minimisation:
///   • ΔG_ref > 0  (hydrophobic) → burial lowers energy  → ΔG_ij < 0  (favourable)
///   • ΔG_ref < 0  (polar)       → burial raises energy  → ΔG_ij > 0  (unfavourable)
///
/// This is added as a cross-term to `interaction_energy_atoms`, so it is
/// seamlessly included in every force-field energy evaluation.
use crate::amber::{Eef1Params, EEF1};
use crate::allatom::AtomCloud;
use rayon::prelude::*;

// π^(3/2) pre-computed
const PI32: f32 = 5.568_328; // π^(3/2)

/// Cutoff for EEF1 burial calculation (Å).  Beyond this the Gaussian is < 0.1 %.
pub const SOLV_CUTOFF: f32 = 9.0;
const SOLV_CUTOFF_SQ: f32 = SOLV_CUTOFF * SOLV_CUTOFF;

/// EEF1 cross-term solvation energy for one atom pair at distance r.
///
/// Returns the ΔΔG_solvation contribution from mutual burial of atoms i and j,
/// where i belongs to cloud_a and j belongs to cloud_b.
#[inline(always)]
fn pair_solvation(ei: &Eef1Params, ej: &Eef1Params, r_sq: f32) -> f32 {
    let r = r_sq.sqrt();

    // burial of atom i by atom j
    let fi = (-(r / ei.lambda).powi(2)).exp() / (PI32 * ei.lambda.powi(3));
    // burial of atom j by atom i
    let fj = (-(r / ej.lambda).powi(2)).exp() / (PI32 * ej.lambda.powi(3));

    // Negate because burial reduces the solvation free energy by dg_ref * vol * f
    -(ei.dg_ref * ej.vol * fi + ej.dg_ref * ei.vol * fj)
}

/// EEF1 cross-term solvation free energy of the antigen–antibody interface.
///
/// Iterates over all (ag_i, ab_j) pairs within `SOLV_CUTOFF` and accumulates
/// the mutual burial contribution.  Same O(|ag|·|ab|) loop as LJ/Coulomb but
/// called only once at final scoring; the energy gradient step uses the cheaper
/// residue-level MC energy instead.
pub fn solvation_interaction(ag: &AtomCloud, ab: &AtomCloud) -> f32 {
    let ag_n = ag.len();
    let ab_n = ab.len();

    (0..ab_n)
        .into_par_iter()
        .map(|j| {
            let bx = ab.x[j];
            let by = ab.y[j];
            let bz = ab.z[j];
            let ej = &EEF1[ab.atom_type[j] as usize];
            if ej.vol == 0.0 && ej.dg_ref == 0.0 {
                return 0.0;
            }

            let mut local = 0.0_f32;
            for i in 0..ag_n {
                let dx = bx - ag.x[i];
                let dy = by - ag.y[i];
                let dz = bz - ag.z[i];
                let r_sq = dx * dx + dy * dy + dz * dz;

                if r_sq >= SOLV_CUTOFF_SQ || r_sq < 0.25 {
                    continue;
                }

                let ei = &EEF1[ag.atom_type[i] as usize];
                local += pair_solvation(ei, ej, r_sq);
            }
            local
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amber::AtomType;
    use crate::allatom::AtomCloud;

    fn make_cloud(x: f32, atype: AtomType) -> AtomCloud {
        let mut c = AtomCloud::new();
        c.x.push(x); c.y.push(0.0); c.z.push(0.0);
        c.charge.push(0.0); c.r_min_half.push(1.9); c.epsilon.push(0.1);
        c.hydrophobic.push(0); c.atom_type.push(atype as u8); c.residue_idx.push(0);
        c
    }

    #[test]
    fn hydrophobic_burial_favorable() {
        // Two CT (aliphatic C, dg_ref = -0.187... wait, for purely hydrophobic we
        // need dg_ref > 0.  Actually CT has dg_ref = -0.187 (slightly prefers exposure).
        // Use the conceptual test: pair_solvation sign check.
        let ei = Eef1Params { dg_ref: 1.0, vol: 10.0, lambda: 3.5 };
        let ej = Eef1Params { dg_ref: 1.0, vol: 10.0, lambda: 3.5 };
        let e = pair_solvation(&ei, &ej, 9.0); // r = 3 Å
        assert!(e < 0.0, "hydrophobic burial should lower energy, got {e}");
    }

    #[test]
    fn polar_burial_unfavorable() {
        let ei = Eef1Params { dg_ref: -5.0, vol: 14.0, lambda: 3.15 };
        let ej = Eef1Params { dg_ref: -5.0, vol: 14.0, lambda: 3.15 };
        let e = pair_solvation(&ei, &ej, 9.0);
        assert!(e > 0.0, "polar burial should raise energy, got {e}");
    }
}
