/// Germline-consensus antibody framework scaffolds.
///
/// Full de novo Ig-fold prediction is out of scope for this hand-rolled
/// physics engine, so we take the same approach used in real CDR-grafting
/// workflows: hold a realistic human germline framework (FR1-FR4) fixed and
/// only actively design the CDR1/2/3 loops. The heavy-chain scaffold below
/// approximates IGHV3-23 (a common VH3 germline), the light-chain scaffold
/// approximates IGKV1-39 (a common Vκ1 germline). Both frameworks carry their
/// natural Cys residues (Kabat ~22/~92-96) so the conserved intradomain
/// disulfide forms via the existing `disulfide_energy()` bonus.
use crate::atom::{AminoAcid, AA_COUNT};
use rand::rngs::SmallRng;
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chain {
    Heavy,
    Light,
}

/// Fixed framework regions (FR1-FR4) flanking three CDR loops of variable
/// (designed) length.
pub struct FrameworkLayout {
    pub fr1: &'static str,
    pub cdr1_len: usize,
    pub fr2: &'static str,
    pub cdr2_len: usize,
    pub fr3: &'static str,
    pub cdr3_len: usize,
    pub fr4: &'static str,
}

impl FrameworkLayout {
    /// Total assembled chain length (framework + all three designed CDRs).
    pub fn chain_len(&self) -> usize {
        self.fr1.len() + self.cdr1_len + self.fr2.len() + self.cdr2_len
            + self.fr3.len() + self.cdr3_len + self.fr4.len()
    }
}

/// VH3-23-like human germline heavy-chain framework.
pub const VH_FRAMEWORK: FrameworkLayout = FrameworkLayout {
    fr1: "EVQLLESGGGLVQPGGSLRLSCAAS",
    cdr1_len: 8,
    fr2: "WVRQAPGKGLEWVS",
    cdr2_len: 8,
    fr3: "RFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK",
    cdr3_len: 12,
    fr4: "WGQGTLVTVSS",
};

/// IGKV1-39-like human germline kappa light-chain framework.
pub const VL_FRAMEWORK: FrameworkLayout = FrameworkLayout {
    fr1: "DIQMTQSPSSLSASVGDRVTITC",
    cdr1_len: 11,
    fr2: "WYQQKPGKAPKLLIY",
    cdr2_len: 7,
    fr3: "GVPSRFSGSGSGTDFTLTISSLQPEDFATYYC",
    cdr3_len: 9,
    fr4: "FGQGTKVEIK",
};

/// Per-residue region label, used for reporting and to restrict mutation /
/// affinity-maturation moves to the CDR loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Fr,
    Cdr1,
    Cdr2,
    Cdr3,
}

impl Region {
    pub fn label(self) -> char {
        match self {
            Region::Fr => 'F',
            Region::Cdr1 => '1',
            Region::Cdr2 => '2',
            Region::Cdr3 => '3',
        }
    }

    #[inline]
    pub fn is_cdr(self) -> bool {
        !matches!(self, Region::Fr)
    }
}

/// Assemble a full chain sequence (framework + designed CDRs) and the
/// parallel per-residue region map.
pub fn assemble_chain(
    fw: &FrameworkLayout,
    cdr1: &[AminoAcid],
    cdr2: &[AminoAcid],
    cdr3: &[AminoAcid],
) -> (Vec<AminoAcid>, Vec<Region>) {
    let mut seq = Vec::new();
    let mut regions = Vec::new();

    fn push_fr(s: &str, seq: &mut Vec<AminoAcid>, regions: &mut Vec<Region>) {
        for c in s.chars() {
            seq.push(AminoAcid::from_char(c).expect("germline framework contains only standard amino acids"));
            regions.push(Region::Fr);
        }
    }
    fn push_cdr(loop_aa: &[AminoAcid], region: Region, seq: &mut Vec<AminoAcid>, regions: &mut Vec<Region>) {
        for &aa in loop_aa {
            seq.push(aa);
            regions.push(region);
        }
    }

    push_fr(fw.fr1, &mut seq, &mut regions);
    push_cdr(cdr1, Region::Cdr1, &mut seq, &mut regions);
    push_fr(fw.fr2, &mut seq, &mut regions);
    push_cdr(cdr2, Region::Cdr2, &mut seq, &mut regions);
    push_fr(fw.fr3, &mut seq, &mut regions);
    push_cdr(cdr3, Region::Cdr3, &mut seq, &mut regions);
    push_fr(fw.fr4, &mut seq, &mut regions);

    (seq, regions)
}

/// A uniformly random CDR loop sequence of length `n`.
pub fn random_cdr(n: usize, rng: &mut SmallRng) -> Vec<AminoAcid> {
    (0..n).map(|_| AminoAcid::from_index(rng.gen_range(0..AA_COUNT))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembled_chain_length_matches_framework_plus_cdrs() {
        let cdr1 = vec![AminoAcid::Ala; VH_FRAMEWORK.cdr1_len];
        let cdr2 = vec![AminoAcid::Ala; VH_FRAMEWORK.cdr2_len];
        let cdr3 = vec![AminoAcid::Ala; VH_FRAMEWORK.cdr3_len];
        let (seq, regions) = assemble_chain(&VH_FRAMEWORK, &cdr1, &cdr2, &cdr3);

        let expected_len = VH_FRAMEWORK.fr1.len()
            + VH_FRAMEWORK.cdr1_len
            + VH_FRAMEWORK.fr2.len()
            + VH_FRAMEWORK.cdr2_len
            + VH_FRAMEWORK.fr3.len()
            + VH_FRAMEWORK.cdr3_len
            + VH_FRAMEWORK.fr4.len();

        assert_eq!(seq.len(), expected_len);
        assert_eq!(regions.len(), expected_len);
    }

    #[test]
    fn region_map_flags_cdr_loops_correctly() {
        let cdr1 = vec![AminoAcid::Trp; VH_FRAMEWORK.cdr1_len];
        let cdr2 = vec![AminoAcid::Trp; VH_FRAMEWORK.cdr2_len];
        let cdr3 = vec![AminoAcid::Trp; VH_FRAMEWORK.cdr3_len];
        let (_, regions) = assemble_chain(&VH_FRAMEWORK, &cdr1, &cdr2, &cdr3);

        let n_cdr = regions.iter().filter(|r| r.is_cdr()).count();
        assert_eq!(n_cdr, VH_FRAMEWORK.cdr1_len + VH_FRAMEWORK.cdr2_len + VH_FRAMEWORK.cdr3_len);
    }

    #[test]
    fn frameworks_carry_a_cysteine_for_the_conserved_disulfide() {
        assert!(VH_FRAMEWORK.fr1.contains('C'));
        assert!(VH_FRAMEWORK.fr3.contains('C'));
        assert!(VL_FRAMEWORK.fr1.contains('C'));
        assert!(VL_FRAMEWORK.fr3.contains('C'));
    }
}
