/// Biophysical quality assessment for a candidate antibody.
///
/// Reports sequence-level metrics (net charge, aggregation risk) and
/// a residue-level interface map that identifies which residues are in
/// contact with the antigen — the CDR-equivalent region in our model.
///
/// Also estimates the −TΔS_bind entropy penalty for a MM-GBSA-style
/// corrected ΔG:
///
///   ΔG_bind ≈ E_MM + ΔΔG_solv + ΔG_entropy
///
///   E_MM + ΔΔG_solv  — returned by interaction_energy_atoms (includes EEF1)
///   ΔG_entropy        — returned here (translational/rotational + chi freeze)
use crate::allatom::AtomProtein;
use crate::rotamer::rotamers;

/// Interface distance cutoff (Å²): residues closer than √64 = 8 Å are
/// considered interface-facing (CDR-equivalent).
pub const IFACE_SQ: f32 = 64.0;

/// Translational + rotational entropy loss upon binding at 300 K (kcal/mol).
/// Empirical constant from MM-GBSA literature (Gilson & Zhou 2007).
const BASE_ENTROPY: f32 = 5.4;

/// Side-chain entropy cost per frozen χ angle at the interface (kcal/mol).
/// Derived from Doig & Sternberg 1995 (≈0.3–0.5 kcal/mol per frozen bond).
const PER_CHI: f32 = 0.3;

/// Biophysical quality metrics for a designed antibody candidate.
pub struct SequenceQuality {
    /// Formal net charge at pH 7.4 (Arg/Lys = +1, Asp/Glu = −1).
    pub net_charge: i32,
    /// Longest consecutive run of hydrophobic residues.
    pub max_hydro_run: usize,
    /// True if max_hydro_run > 4 (aggregation-prone patch flagged).
    pub aggregation_risk: bool,
    /// Number of residues within 8 Å of any antigen Cα (CDR-equivalent).
    pub n_interface: usize,
    /// −TΔS_bind estimate (kcal/mol); always positive (entropic cost of binding).
    pub entropy_penalty: f32,
}

impl SequenceQuality {
    /// Assess a designed antibody `ab` against antigen `ag`.
    pub fn assess(ab: &AtomProtein, ag: &AtomProtein) -> Self {
        // Formal net charge
        let net_charge: i32 = ab.amino_acid.iter()
            .map(|&aa| aa.charge().round() as i32)
            .sum();

        // Longest hydrophobic run
        let mut max_hydro_run = 0usize;
        let mut run = 0usize;
        for &aa in &ab.amino_acid {
            if aa.is_hydrophobic() {
                run += 1;
                if run > max_hydro_run { max_hydro_run = run; }
            } else {
                run = 0;
            }
        }

        // Interface residues + frozen chi count
        let mut n_interface = 0usize;
        let mut n_frozen_chi = 0usize;
        for r in 0..ab.n_residues() {
            if is_interface(r, ab, ag) {
                n_interface += 1;
                if let Some(rot) = rotamers(ab.amino_acid[r]).first() {
                    n_frozen_chi += rot.chi.iter().filter(|&&c| c != 0.0).count();
                }
            }
        }

        let entropy_penalty = BASE_ENTROPY + n_frozen_chi as f32 * PER_CHI;

        SequenceQuality {
            net_charge,
            max_hydro_run,
            aggregation_risk: max_hydro_run > 4,
            n_interface,
            entropy_penalty,
        }
    }

    /// Per-residue interface labels: 'I' = interface (CDR-like), 'F' = framework.
    pub fn interface_labels(ab: &AtomProtein, ag: &AtomProtein) -> String {
        (0..ab.n_residues())
            .map(|r| if is_interface(r, ab, ag) { 'I' } else { 'F' })
            .collect()
    }
}

/// True if antibody residue `r` has its Cα within IFACE_SQ of any antigen Cα.
#[inline]
pub fn is_interface(r: usize, ab: &AtomProtein, ag: &AtomProtein) -> bool {
    let p = ab.ca_pos(r);
    (0..ag.n_residues()).any(|i| {
        let pi = ag.ca_pos(i);
        let dx = p[0] - pi[0];
        let dy = p[1] - pi[1];
        let dz = p[2] - pi[2];
        dx * dx + dy * dy + dz * dz < IFACE_SQ
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allatom::protein_from_ca_trace;
    use crate::atom::AminoAcid;

    fn protein(seq: &[AminoAcid], spacing: f32) -> AtomProtein {
        let n = seq.len();
        let xs: Vec<f32> = (0..n).map(|i| i as f32 * spacing).collect();
        let ys = vec![0.0f32; n];
        let zs = vec![0.0f32; n];
        protein_from_ca_trace(&xs, &ys, &zs, seq)
    }

    /// Arg + Lys (+1 each) minus Asp + Glu (−1 each) nets to zero.
    #[test]
    fn net_charge_balances() {
        use AminoAcid::*;
        let ab = protein(&[Arg, Lys, Asp, Glu], 3.8);
        let ag = protein(&[Gly], 3.8);

        let q = SequenceQuality::assess(&ab, &ag);
        assert_eq!(q.net_charge, 0);
    }

    /// Five consecutive hydrophobic residues should trigger the aggregation flag.
    #[test]
    fn long_hydrophobic_run_flags_aggregation() {
        use AminoAcid::*;
        let ab = protein(&[Leu, Ile, Val, Phe, Met], 3.8);
        let ag = protein(&[Gly], 3.8);

        let q = SequenceQuality::assess(&ab, &ag);
        assert_eq!(q.max_hydro_run, 5);
        assert!(q.aggregation_risk);
    }

    /// A polar/charged sequence with no hydrophobic run should not be flagged.
    #[test]
    fn polar_sequence_is_not_aggregation_prone() {
        use AminoAcid::*;
        let ab = protein(&[Asp, Lys, Glu, Arg], 3.8);
        let ag = protein(&[Gly], 3.8);

        let q = SequenceQuality::assess(&ab, &ag);
        assert!(!q.aggregation_risk);
    }

    /// A residue placed right on top of the antigen is interface; one placed
    /// far away (well beyond the 8 Å cutoff) is framework.
    #[test]
    fn interface_detection_uses_distance_cutoff() {
        use AminoAcid::*;
        let ab = protein(&[Gly, Gly], 100.0); // second residue is 100 Å away
        let ag = protein(&[Gly], 3.8);

        let labels = SequenceQuality::interface_labels(&ab, &ag);
        assert_eq!(labels.chars().next(), Some('I'));
        assert_eq!(labels.chars().nth(1), Some('F'));
    }

    /// Entropy penalty must always exceed the base translational/rotational term.
    #[test]
    fn entropy_penalty_is_at_least_base() {
        use AminoAcid::*;
        let ab = protein(&[Trp, Arg], 3.8);
        let ag = protein(&[Gly], 3.8);

        let q = SequenceQuality::assess(&ab, &ag);
        assert!(q.entropy_penalty >= BASE_ENTROPY);
    }
}
