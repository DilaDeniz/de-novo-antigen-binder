/// AMBER99SB-ildn non-bonded parameters and partial charges.
///
/// Parameters taken from the AMBER99SB parameter set (parm99.dat + frcmod.ff99SB).
/// All values are in kcal/mol (epsilon) and Å (r_min_half = Rmin/2).
/// The standard AMBER LJ form uses r_min_half: V_LJ = ε[(R_ij/r)^12 - 2(R_ij/r)^6]
/// where R_ij = r_min_half_i + r_min_half_j.
use crate::atom::AminoAcid;

// ── Atom types ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AtomType {
    C  = 0,  // carbonyl C (backbone C=O)
    CA = 1,  // alpha-carbon / aromatic C
    CB = 2,  // aliphatic/aromatic Cβ
    CC = 3,  // conjugated C (HIS ring)
    CT = 4,  // sp3 C (aliphatic chains)
    N  = 5,  // backbone amide N
    N2 = 6,  // guanidinium N (ARG)
    N3 = 7,  // charged amino N (LYS NZ)
    NA = 8,  // aromatic N–H (HIS, TRP)
    NB = 9,  // aromatic N (HIS ND1 neutral)
    O  = 10, // carbonyl O
    O2 = 11, // carboxylate O (ASP, GLU)
    OH = 12, // hydroxyl O (SER, THR, TYR)
    OS = 13, // ether O (no H)
    S  = 14, // thioether S (MET)
    SH = 15, // thiol S (CYS)
}

pub const N_ATOM_TYPES: usize = 16;

/// Non-bonded LJ parameters for each atom type.
#[derive(Debug, Clone, Copy)]
pub struct AmberNbParams {
    /// Rmin/2 (Å) — half the Lennard-Jones minimum-energy distance.
    pub r_min_half: f32,
    /// LJ well depth (kcal/mol).
    pub epsilon: f32,
}

/// AMBER99SB non-bonded table, indexed by `AtomType as usize`.
///
/// Source: AMBER parm99.dat (NONBON section), Hornak et al. 2006.
pub static AMBER_NB: [AmberNbParams; N_ATOM_TYPES] = [
    AmberNbParams { r_min_half: 1.9080, epsilon: 0.0860 }, // C
    AmberNbParams { r_min_half: 1.9080, epsilon: 0.0860 }, // CA
    AmberNbParams { r_min_half: 1.9080, epsilon: 0.0860 }, // CB
    AmberNbParams { r_min_half: 1.9080, epsilon: 0.0860 }, // CC
    AmberNbParams { r_min_half: 1.9080, epsilon: 0.1094 }, // CT
    AmberNbParams { r_min_half: 1.8240, epsilon: 0.1700 }, // N
    AmberNbParams { r_min_half: 1.8240, epsilon: 0.1700 }, // N2
    AmberNbParams { r_min_half: 1.8240, epsilon: 0.1700 }, // N3
    AmberNbParams { r_min_half: 1.8240, epsilon: 0.1700 }, // NA
    AmberNbParams { r_min_half: 1.8240, epsilon: 0.1700 }, // NB
    AmberNbParams { r_min_half: 1.6612, epsilon: 0.2100 }, // O
    AmberNbParams { r_min_half: 1.6612, epsilon: 0.2100 }, // O2
    AmberNbParams { r_min_half: 1.7210, epsilon: 0.2104 }, // OH
    AmberNbParams { r_min_half: 1.6837, epsilon: 0.1700 }, // OS
    AmberNbParams { r_min_half: 2.0000, epsilon: 0.2500 }, // S
    AmberNbParams { r_min_half: 2.0000, epsilon: 0.2500 }, // SH
];

// ── Per-residue atom lists ────────────────────────────────────────────────────

/// A single heavy atom within a residue.
#[derive(Clone, Copy)]
pub struct ResAtom {
    /// PDB atom name (4-char, space-padded like " CA ").
    pub name: &'static str,
    pub atom_type: AtomType,
    /// AMBER99SB partial charge (elementary units).
    pub charge: f32,
    /// 1 if this atom contributes to hydrophobic scoring, 0 otherwise.
    pub hydrophobic: u8,
}

/// All heavy atoms for one amino acid in standard order:
/// backbone first (N, CA, C, O, CB), then side-chain.
pub struct ResTopology {
    pub atoms: &'static [ResAtom],
    /// Index of the Cα atom within `atoms` (always 1 for standard residues).
    pub ca_idx: u8,
    /// Number of chi angles this residue has (0–4).
    pub n_chi: u8,
}

// ── Residue topologies (AMBER99SB charges, Hornak 2006) ──────────────────────

// Backbone shared atoms: N, CA, C, O are the same type across all residues
// (charges vary slightly per residue).  We embed them inline for clarity.

// Helper to keep the tables readable
macro_rules! atom {
    ($name:literal, $atype:ident, $q:literal) => {
        ResAtom { name: $name, atom_type: AtomType::$atype, charge: $q, hydrophobic: 0 }
    };
    ($name:literal, $atype:ident, $q:literal, H) => {
        ResAtom { name: $name, atom_type: AtomType::$atype, charge: $q, hydrophobic: 1 }
    };
}

// ── ALA ──────────────────────────────────────────────────────────────────────
static ALA_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.4157),
    atom!(" CA ", CT,  0.0337),
    atom!(" C  ", C,   0.5973),
    atom!(" O  ", O,  -0.5679),
    atom!(" CB ", CT, -0.1825, H),
];
// ── ARG ──────────────────────────────────────────────────────────────────────
static ARG_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.3479),
    atom!(" CA ", CT, -0.2637),
    atom!(" C  ", C,   0.7341),
    atom!(" O  ", O,  -0.5894),
    atom!(" CB ", CT, -0.0007, H),
    atom!(" CG ", CT,  0.0390, H),
    atom!(" CD ", CT,  0.0486),
    atom!(" NE ", N2, -0.5295),
    atom!(" CZ ", CA,  0.8076),
    atom!(" NH1", N2, -0.8627),
    atom!(" NH2", N2, -0.8627),
];
// ── ASN ──────────────────────────────────────────────────────────────────────
static ASN_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.4157),
    atom!(" CA ", CT, -0.0301),
    atom!(" C  ", C,   0.5973),
    atom!(" O  ", O,  -0.5679),
    atom!(" CB ", CT, -0.2041, H),
    atom!(" CG ", C,   0.7130),
    atom!(" OD1", O,  -0.5931),
    atom!(" ND2", N,  -0.9191),
];
// ── ASP ──────────────────────────────────────────────────────────────────────
static ASP_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.5163),
    atom!(" CA ", CT,  0.0381),
    atom!(" C  ", C,   0.5366),
    atom!(" O  ", O,  -0.5819),
    atom!(" CB ", CT, -0.0303, H),
    atom!(" CG ", C,   0.7994),
    atom!(" OD1", O2, -0.8014),
    atom!(" OD2", O2, -0.8014),
];
// ── CYS ──────────────────────────────────────────────────────────────────────
static CYS_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.4157),
    atom!(" CA ", CT,  0.0213),
    atom!(" C  ", C,   0.5973),
    atom!(" O  ", O,  -0.5679),
    atom!(" CB ", CT, -0.1231, H),
    atom!(" SG ", SH, -0.3119),
];
// ── GLN ──────────────────────────────────────────────────────────────────────
static GLN_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.4157),
    atom!(" CA ", CT, -0.0031),
    atom!(" C  ", C,   0.5973),
    atom!(" O  ", O,  -0.5679),
    atom!(" CB ", CT, -0.0036, H),
    atom!(" CG ", CT, -0.0645, H),
    atom!(" CD ", C,   0.6951),
    atom!(" OE1", O,  -0.6086),
    atom!(" NE2", N,  -0.9407),
];
// ── GLU ──────────────────────────────────────────────────────────────────────
static GLU_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.5163),
    atom!(" CA ", CT,  0.0397),
    atom!(" C  ", C,   0.5366),
    atom!(" O  ", O,  -0.5819),
    atom!(" CB ", CT,  0.0560, H),
    atom!(" CG ", CT, -0.0173, H),
    atom!(" CD ", C,   0.8054),
    atom!(" OE1", O2, -0.8188),
    atom!(" OE2", O2, -0.8188),
];
// ── GLY ──────────────────────────────────────────────────────────────────────
static GLY_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.4157),
    atom!(" CA ", CT, -0.0252),
    atom!(" C  ", C,   0.5973),
    atom!(" O  ", O,  -0.5679),
];
// ── HIS ──────────────────────────────────────────────────────────────────────
static HIS_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.4157),
    atom!(" CA ", CT,  0.0188),
    atom!(" C  ", C,   0.5973),
    atom!(" O  ", O,  -0.5679),
    atom!(" CB ", CT, -0.0462, H),
    atom!(" CG ", CC, -0.0266),
    atom!(" ND1", NB, -0.3811),
    atom!(" CD2", CC,  0.1292),
    atom!(" CE1", CC,  0.2057),
    atom!(" NE2", NA, -0.5727),
];
// ── ILE ──────────────────────────────────────────────────────────────────────
static ILE_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.4157),
    atom!(" CA ", CT, -0.0597),
    atom!(" C  ", C,   0.5973),
    atom!(" O  ", O,  -0.5679),
    atom!(" CB ", CT,  0.1303, H),
    atom!(" CG1", CT, -0.0430, H),
    atom!(" CG2", CT, -0.3204, H),
    atom!(" CD1", CT, -0.0660, H),
];
// ── LEU ──────────────────────────────────────────────────────────────────────
static LEU_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.4157),
    atom!(" CA ", CT, -0.0518),
    atom!(" C  ", C,   0.5973),
    atom!(" O  ", O,  -0.5679),
    atom!(" CB ", CT, -0.1102, H),
    atom!(" CG ", CT,  0.3531, H),
    atom!(" CD1", CT, -0.4121, H),
    atom!(" CD2", CT, -0.4121, H),
];
// ── LYS ──────────────────────────────────────────────────────────────────────
static LYS_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.3479),
    atom!(" CA ", CT, -0.2400),
    atom!(" C  ", C,   0.7341),
    atom!(" O  ", O,  -0.5894),
    atom!(" CB ", CT, -0.0094, H),
    atom!(" CG ", CT,  0.0187, H),
    atom!(" CD ", CT, -0.0479, H),
    atom!(" CE ", CT, -0.0143, H),
    atom!(" NZ ", N3, -0.3854),
];
// ── MET ──────────────────────────────────────────────────────────────────────
static MET_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.4157),
    atom!(" CA ", CT, -0.0237),
    atom!(" C  ", C,   0.5973),
    atom!(" O  ", O,  -0.5679),
    atom!(" CB ", CT,  0.0342, H),
    atom!(" CG ", CT,  0.0018, H),
    atom!(" SD ", S,  -0.2737),
    atom!(" CE ", CT, -0.0536, H),
];
// ── PHE ──────────────────────────────────────────────────────────────────────
static PHE_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.4157),
    atom!(" CA ", CT, -0.0024),
    atom!(" C  ", C,   0.5973),
    atom!(" O  ", O,  -0.5679),
    atom!(" CB ", CT, -0.0343, H),
    atom!(" CG ", CA,  0.0118, H),
    atom!(" CD1", CA, -0.1256, H),
    atom!(" CD2", CA, -0.1256, H),
    atom!(" CE1", CA, -0.1704, H),
    atom!(" CE2", CA, -0.1704, H),
    atom!(" CZ ", CA, -0.1072, H),
];
// ── PRO ──────────────────────────────────────────────────────────────────────
static PRO_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.2548),
    atom!(" CA ", CT, -0.0266),
    atom!(" C  ", C,   0.5896),
    atom!(" O  ", O,  -0.5748),
    atom!(" CB ", CT, -0.0070, H),
    atom!(" CG ", CT,  0.0189, H),
    atom!(" CD ", CT,  0.0192, H),
];
// ── SER ──────────────────────────────────────────────────────────────────────
static SER_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.4157),
    atom!(" CA ", CT, -0.0249),
    atom!(" C  ", C,   0.5973),
    atom!(" O  ", O,  -0.5679),
    atom!(" CB ", CT,  0.2117),
    atom!(" OG ", OH, -0.6546),
];
// ── THR ──────────────────────────────────────────────────────────────────────
static THR_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.4157),
    atom!(" CA ", CT,  0.0764),
    atom!(" C  ", C,   0.5973),
    atom!(" O  ", O,  -0.5679),
    atom!(" CB ", CT,  0.3654),
    atom!(" OG1", OH, -0.6761),
    atom!(" CG2", CT, -0.2438, H),
];
// ── TRP ──────────────────────────────────────────────────────────────────────
static TRP_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.4157),
    atom!(" CA ", CT, -0.0275),
    atom!(" C  ", C,   0.5973),
    atom!(" O  ", O,  -0.5679),
    atom!(" CB ", CT, -0.0050, H),
    atom!(" CG ", CB, -0.1415, H),
    atom!(" CD1", CB, -0.1638, H),
    atom!(" CD2", CA,  0.1243, H),
    atom!(" NE1", NA, -0.3418),
    atom!(" CE2", CA,  0.1380, H),
    atom!(" CE3", CA, -0.2387, H),
    atom!(" CZ2", CA, -0.2601, H),
    atom!(" CZ3", CA, -0.1972, H),
    atom!(" CH2", CA, -0.1134, H),
];
// ── TYR ──────────────────────────────────────────────────────────────────────
static TYR_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.4157),
    atom!(" CA ", CT, -0.0014),
    atom!(" C  ", C,   0.5973),
    atom!(" O  ", O,  -0.5679),
    atom!(" CB ", CT, -0.0152, H),
    atom!(" CG ", CA, -0.0011, H),
    atom!(" CD1", CA, -0.1906, H),
    atom!(" CD2", CA, -0.1906, H),
    atom!(" CE1", CA, -0.2341, H),
    atom!(" CE2", CA, -0.2341, H),
    atom!(" CZ ", CA,  0.3226, H),
    atom!(" OH ", OH, -0.5579),
];
// ── VAL ──────────────────────────────────────────────────────────────────────
static VAL_ATOMS: &[ResAtom] = &[
    atom!(" N  ", N,  -0.4157),
    atom!(" CA ", CT,  0.0145),
    atom!(" C  ", C,   0.5973),
    atom!(" O  ", O,  -0.5679),
    atom!(" CB ", CT,  0.2985, H),
    atom!(" CG1", CT, -0.3192, H),
    atom!(" CG2", CT, -0.3192, H),
];

// ── Topology table ────────────────────────────────────────────────────────────

/// Per-residue topology: atom list, Cα index, and chi-angle count.
/// Indexed by `AminoAcid as usize`.
pub static RESIDUE_TOPOLOGY: [ResTopology; 20] = [
    ResTopology { atoms: ALA_ATOMS, ca_idx: 1, n_chi: 0 }, // Ala
    ResTopology { atoms: ARG_ATOMS, ca_idx: 1, n_chi: 4 }, // Arg
    ResTopology { atoms: ASN_ATOMS, ca_idx: 1, n_chi: 2 }, // Asn
    ResTopology { atoms: ASP_ATOMS, ca_idx: 1, n_chi: 2 }, // Asp
    ResTopology { atoms: CYS_ATOMS, ca_idx: 1, n_chi: 1 }, // Cys
    ResTopology { atoms: GLN_ATOMS, ca_idx: 1, n_chi: 3 }, // Gln
    ResTopology { atoms: GLU_ATOMS, ca_idx: 1, n_chi: 3 }, // Glu
    ResTopology { atoms: GLY_ATOMS, ca_idx: 1, n_chi: 0 }, // Gly
    ResTopology { atoms: HIS_ATOMS, ca_idx: 1, n_chi: 2 }, // His
    ResTopology { atoms: ILE_ATOMS, ca_idx: 1, n_chi: 2 }, // Ile
    ResTopology { atoms: LEU_ATOMS, ca_idx: 1, n_chi: 2 }, // Leu
    ResTopology { atoms: LYS_ATOMS, ca_idx: 1, n_chi: 4 }, // Lys
    ResTopology { atoms: MET_ATOMS, ca_idx: 1, n_chi: 3 }, // Met
    ResTopology { atoms: PHE_ATOMS, ca_idx: 1, n_chi: 2 }, // Phe
    ResTopology { atoms: PRO_ATOMS, ca_idx: 1, n_chi: 2 }, // Pro
    ResTopology { atoms: SER_ATOMS, ca_idx: 1, n_chi: 1 }, // Ser
    ResTopology { atoms: THR_ATOMS, ca_idx: 1, n_chi: 1 }, // Thr
    ResTopology { atoms: TRP_ATOMS, ca_idx: 1, n_chi: 2 }, // Trp
    ResTopology { atoms: TYR_ATOMS, ca_idx: 1, n_chi: 2 }, // Tyr
    ResTopology { atoms: VAL_ATOMS, ca_idx: 1, n_chi: 1 }, // Val
];

// ── Public helpers ────────────────────────────────────────────────────────────

/// Return the `ResTopology` for an amino acid.
#[inline(always)]
pub fn topology(aa: AminoAcid) -> &'static ResTopology {
    &RESIDUE_TOPOLOGY[aa as usize]
}

/// Lorentz-Berthelot mixing for AMBER r_min_half (additive) and epsilon (geometric).
#[inline(always)]
pub fn mix_lj(a: &AmberNbParams, b: &AmberNbParams) -> (f32, f32) {
    let r_ij = a.r_min_half + b.r_min_half;  // R_min_ij
    let eps_ij = (a.epsilon * b.epsilon).sqrt();
    (r_ij, eps_ij)
}
