/// Lightweight structure validation: steric clash detection and a generous
/// Ramachandran-basin outlier count, computed directly from the backbone
/// N-CA-C(-O) geometry already present in `AtomProtein`.
///
/// This is not a full Ramachandran statistical-potential check (that would
/// require empirical φ/ψ density maps, out of scope for this hand-rolled
/// engine) — it's a coarse sanity filter: does this backbone avoid steric
/// clashes, and do its torsion angles fall within generously-sized
/// alpha/beta/left-handed-alpha basins?
use crate::allatom::{cross3, norm3, sub3, AtomProtein};

/// Fraction of (r_min_half_i + r_min_half_j) below which two non-bonded
/// atoms are considered clashing (steric overlap).
const CLASH_FACTOR: f32 = 0.7;

pub struct ValidationReport {
    pub n_residues: usize,
    /// Non-bonded atom pairs closer than `CLASH_FACTOR` × (sum of radii).
    pub clashes: usize,
    /// Residues whose (φ, ψ) fall outside the generous allowed basins.
    pub rama_outliers: usize,
}

impl ValidationReport {
    pub fn assess(prot: &AtomProtein) -> Self {
        ValidationReport {
            n_residues: prot.n_residues(),
            clashes: clash_score(prot),
            rama_outliers: ramachandran_outliers(prot),
        }
    }

    pub fn is_clean(&self) -> bool {
        self.clashes == 0 && self.rama_outliers == 0
    }
}

/// Count non-bonded atom pairs in steric clash, skipping same-residue and
/// adjacent-residue pairs (which are expected to sit close together).
pub fn clash_score(prot: &AtomProtein) -> usize {
    let n_atoms = prot.n_atoms();
    let mut clashes = 0usize;

    for i in 0..n_atoms {
        let ri = prot.atoms.residue_idx[i];
        for j in (i + 1)..n_atoms {
            let rj = prot.atoms.residue_idx[j];
            if ri.abs_diff(rj) <= 1 {
                continue;
            }
            let dx = prot.atoms.x[i] - prot.atoms.x[j];
            let dy = prot.atoms.y[i] - prot.atoms.y[j];
            let dz = prot.atoms.z[i] - prot.atoms.z[j];
            let d_sq = dx * dx + dy * dy + dz * dz;
            let r_sum = prot.atoms.r_min_half[i] + prot.atoms.r_min_half[j];
            let thresh = CLASH_FACTOR * r_sum;
            if d_sq < thresh * thresh {
                clashes += 1;
            }
        }
    }
    clashes
}

/// Backbone atom position at `offset` within residue `r` (0=N, 1=CA, 2=C).
#[inline]
fn backbone_atom(prot: &AtomProtein, r: usize, offset: usize) -> [f32; 3] {
    let idx = prot.atom_range(r).start + offset;
    [prot.atoms.x[idx], prot.atoms.y[idx], prot.atoms.z[idx]]
}

/// Dihedral angle (radians) for four points, standard formula via the two
/// half-plane normals.
fn dihedral(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]) -> f32 {
    let b1 = sub3(p1, p0);
    let b2 = sub3(p2, p1);
    let b3 = sub3(p3, p2);

    let n1 = norm3(cross3(b1, b2));
    let n2 = norm3(cross3(b2, b3));
    let m1 = cross3(n1, norm3(b2));

    let x = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
    let y = m1[0] * n2[0] + m1[1] * n2[1] + m1[2] * n2[2];
    y.atan2(x)
}

/// φ = C(r-1)-N(r)-CA(r)-C(r); undefined (None) for the first residue.
fn phi_angle(prot: &AtomProtein, r: usize) -> Option<f32> {
    if r == 0 {
        return None;
    }
    let c_prev = backbone_atom(prot, r - 1, 2);
    let n = backbone_atom(prot, r, 0);
    let ca = backbone_atom(prot, r, 1);
    let c = backbone_atom(prot, r, 2);
    Some(dihedral(c_prev, n, ca, c).to_degrees())
}

/// ψ = N(r)-CA(r)-C(r)-N(r+1); undefined (None) for the last residue.
fn psi_angle(prot: &AtomProtein, r: usize) -> Option<f32> {
    if r + 1 >= prot.n_residues() {
        return None;
    }
    let n = backbone_atom(prot, r, 0);
    let ca = backbone_atom(prot, r, 1);
    let c = backbone_atom(prot, r, 2);
    let n_next = backbone_atom(prot, r + 1, 0);
    Some(dihedral(n, ca, c, n_next).to_degrees())
}

/// Generously-sized allowed basins: right-handed alpha helix, beta sheet,
/// and left-handed alpha helix. Anything else is flagged as an outlier.
fn in_allowed_basin(phi: f32, psi: f32) -> bool {
    let alpha_r  = (-100.0..=-30.0).contains(&phi) && (-80.0..=-5.0).contains(&psi);
    let beta     = (-170.0..=-60.0).contains(&phi) && (90.0..=180.0).contains(&psi)
        || (-170.0..=-60.0).contains(&phi) && (-180.0..=-150.0).contains(&psi);
    let alpha_l  = (30.0..=100.0).contains(&phi) && (-10.0..=80.0).contains(&psi);
    alpha_r || beta || alpha_l
}

/// Count residues (excluding the two chain termini, which lack a full
/// φ/ψ pair) whose backbone torsions fall outside the allowed basins.
pub fn ramachandran_outliers(prot: &AtomProtein) -> usize {
    let n = prot.n_residues();
    (0..n)
        .filter(|&r| {
            match (phi_angle(prot, r), psi_angle(prot, r)) {
                (Some(phi), Some(psi)) => !in_allowed_basin(phi, psi),
                _ => false,
            }
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allatom::protein_from_ca_trace;
    use crate::atom::AminoAcid;

    fn straight_chain(n: usize) -> AtomProtein {
        let xs: Vec<f32> = (0..n).map(|i| i as f32 * 3.8).collect();
        let ys = vec![0.0f32; n];
        let zs = vec![0.0f32; n];
        let seq = vec![AminoAcid::Ala; n];
        protein_from_ca_trace(&xs, &ys, &zs, &seq)
    }

    #[test]
    fn no_clashes_in_a_well_spaced_extended_chain() {
        let prot = straight_chain(6);
        assert_eq!(clash_score(&prot), 0);
    }

    #[test]
    fn overlapping_residues_are_flagged_as_clashing() {
        // Two residues placed on top of each other (non-adjacent in sequence
        // once a third spacer residue is inserted) must register a clash.
        let xs = vec![0.0_f32, 3.8, 0.05];
        let ys = vec![0.0_f32, 0.0, 0.05];
        let zs = vec![0.0_f32, 0.0, 0.05];
        let seq = vec![AminoAcid::Ala; 3];
        let prot = protein_from_ca_trace(&xs, &ys, &zs, &seq);
        assert!(clash_score(&prot) > 0);
    }

    #[test]
    fn validation_report_assesses_residue_count() {
        let prot = straight_chain(5);
        let report = ValidationReport::assess(&prot);
        assert_eq!(report.n_residues, 5);
    }
}
