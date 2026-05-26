/// Backbone-independent rotamer library (Dunbrack 1993) + NERF atom placement.
///
/// Each amino acid has up to 5 rotamers represented as chi-angle tuples (degrees).
/// GLY and ALA have no side-chain and return empty slices.
///
/// NERF (Natural Extension Reference Frame) places a new atom D given three
/// reference atoms A-B-C and the internal coordinates (bond length, bond angle,
/// dihedral).  This is the standard algorithm used in all molecular modelling
/// toolkits for reconstructing Cartesian coordinates from torsion angles.
use crate::atom::AminoAcid;

// ── Rotamer data ──────────────────────────────────────────────────────────────

/// A single rotamer state: up to 4 chi angles (degrees) and a prior probability.
#[derive(Clone, Copy, Debug)]
pub struct Rotamer {
    /// Chi angles in degrees.  Unused chi slots are 0.0.
    pub chi: [f32; 4],
    /// Backbone-independent probability (sum ≈ 1 per residue).
    pub probability: f32,
}

impl Rotamer {
    const fn new(chi1: f32, chi2: f32, chi3: f32, chi4: f32, prob: f32) -> Self {
        Self { chi: [chi1, chi2, chi3, chi4], probability: prob }
    }
}

// Top-5 backbone-independent rotamers per AA (Dunbrack & Cohen 1997).
// For residues with <5 distinct rotamers the remaining slots repeat the last.
// Angles in degrees (positive = gauche+, negative = gauche−, ±180 = trans).

static ROT_ALA: &[Rotamer] = &[];
static ROT_ARG: &[Rotamer] = &[
    Rotamer::new(-67.0,  180.0,  65.0,  85.0, 0.22),
    Rotamer::new(-67.0,  180.0, 180.0,  85.0, 0.18),
    Rotamer::new(-67.0,  180.0,  65.0, -85.0, 0.12),
    Rotamer::new(-67.0,  180.0, 180.0, 180.0, 0.10),
    Rotamer::new( 62.0,  180.0,  65.0,  85.0, 0.08),
];
static ROT_ASN: &[Rotamer] = &[
    Rotamer::new(-65.0,  -10.0, 0.0, 0.0, 0.36),
    Rotamer::new(-65.0,  130.0, 0.0, 0.0, 0.18),
    Rotamer::new(-174.0,  10.0, 0.0, 0.0, 0.12),
    Rotamer::new(-174.0, -10.0, 0.0, 0.0, 0.10),
    Rotamer::new(  62.0,  10.0, 0.0, 0.0, 0.08),
];
static ROT_ASP: &[Rotamer] = &[
    Rotamer::new(-70.0,  -15.0, 0.0, 0.0, 0.38),
    Rotamer::new(-70.0,  170.0, 0.0, 0.0, 0.22),
    Rotamer::new(-174.0,  25.0, 0.0, 0.0, 0.16),
    Rotamer::new( 62.0,   25.0, 0.0, 0.0, 0.09),
    Rotamer::new( 62.0,  -15.0, 0.0, 0.0, 0.06),
];
static ROT_CYS: &[Rotamer] = &[
    Rotamer::new(-65.0, 0.0, 0.0, 0.0, 0.44),
    Rotamer::new(-179.0, 0.0, 0.0, 0.0, 0.30),
    Rotamer::new( 62.0, 0.0, 0.0, 0.0, 0.21),
    Rotamer::new( 62.0, 0.0, 0.0, 0.0, 0.05),
    Rotamer::new(-65.0, 0.0, 0.0, 0.0, 0.00),
];
static ROT_GLN: &[Rotamer] = &[
    Rotamer::new(-67.0,  180.0,  20.0, 0.0, 0.25),
    Rotamer::new(-67.0,  180.0, -20.0, 0.0, 0.18),
    Rotamer::new(-67.0,  -65.0,  40.0, 0.0, 0.12),
    Rotamer::new(-174.0,  65.0, -30.0, 0.0, 0.10),
    Rotamer::new(  62.0, 180.0,  20.0, 0.0, 0.08),
];
static ROT_GLU: &[Rotamer] = &[
    Rotamer::new(-67.0,  180.0,  20.0, 0.0, 0.26),
    Rotamer::new(-67.0,  180.0, -20.0, 0.0, 0.20),
    Rotamer::new(-67.0,  -65.0,  40.0, 0.0, 0.12),
    Rotamer::new(-174.0,  65.0, -20.0, 0.0, 0.10),
    Rotamer::new(  62.0, 180.0, -20.0, 0.0, 0.08),
];
static ROT_GLY: &[Rotamer] = &[];
static ROT_HIS: &[Rotamer] = &[
    Rotamer::new(-62.0, -75.0, 0.0, 0.0, 0.28),
    Rotamer::new(-62.0,  80.0, 0.0, 0.0, 0.19),
    Rotamer::new(-174.0,-80.0, 0.0, 0.0, 0.16),
    Rotamer::new(-174.0, 80.0, 0.0, 0.0, 0.13),
    Rotamer::new(  62.0,-80.0, 0.0, 0.0, 0.08),
];
static ROT_ILE: &[Rotamer] = &[
    Rotamer::new(-60.0,  170.0, 0.0, 0.0, 0.38),
    Rotamer::new( 60.0,  170.0, 0.0, 0.0, 0.20),
    Rotamer::new(-60.0,  -60.0, 0.0, 0.0, 0.17),
    Rotamer::new(-174.0, 170.0, 0.0, 0.0, 0.12),
    Rotamer::new( 60.0,  -60.0, 0.0, 0.0, 0.05),
];
static ROT_LEU: &[Rotamer] = &[
    Rotamer::new(-60.0,  180.0, 0.0, 0.0, 0.46),
    Rotamer::new(-60.0,   60.0, 0.0, 0.0, 0.20),
    Rotamer::new(180.0,   60.0, 0.0, 0.0, 0.14),
    Rotamer::new( 60.0,  180.0, 0.0, 0.0, 0.10),
    Rotamer::new(-174.0,  60.0, 0.0, 0.0, 0.06),
];
static ROT_LYS: &[Rotamer] = &[
    Rotamer::new(-67.0,  180.0,  68.0, 180.0, 0.21),
    Rotamer::new(-67.0,  180.0, 180.0, 180.0, 0.17),
    Rotamer::new(-67.0,  180.0, -68.0, 180.0, 0.12),
    Rotamer::new(-174.0, 180.0,  68.0, 180.0, 0.09),
    Rotamer::new(  62.0, 180.0,  68.0, 180.0, 0.07),
];
static ROT_MET: &[Rotamer] = &[
    Rotamer::new(-67.0,  180.0, 75.0, 0.0, 0.24),
    Rotamer::new(-67.0,  180.0,-75.0, 0.0, 0.18),
    Rotamer::new(-67.0,  -65.0, 75.0, 0.0, 0.12),
    Rotamer::new(-174.0,  65.0, 75.0, 0.0, 0.10),
    Rotamer::new(  62.0, 180.0, 75.0, 0.0, 0.08),
];
static ROT_PHE: &[Rotamer] = &[
    Rotamer::new(-65.0,  90.0, 0.0, 0.0, 0.37),
    Rotamer::new(-65.0, -85.0, 0.0, 0.0, 0.22),
    Rotamer::new(-174.0, 80.0, 0.0, 0.0, 0.16),
    Rotamer::new(  62.0, 90.0, 0.0, 0.0, 0.12),
    Rotamer::new(-174.0,-80.0, 0.0, 0.0, 0.06),
];
static ROT_PRO: &[Rotamer] = &[
    Rotamer::new(-65.0, 0.0, 0.0, 0.0, 0.58),
    Rotamer::new( 30.0, 0.0, 0.0, 0.0, 0.28),
    Rotamer::new(-30.0, 0.0, 0.0, 0.0, 0.14),
    Rotamer::new(-65.0, 0.0, 0.0, 0.0, 0.00),
    Rotamer::new(-65.0, 0.0, 0.0, 0.0, 0.00),
];
static ROT_SER: &[Rotamer] = &[
    Rotamer::new( 62.0, 0.0, 0.0, 0.0, 0.37),
    Rotamer::new(-65.0, 0.0, 0.0, 0.0, 0.35),
    Rotamer::new(-174.0,0.0, 0.0, 0.0, 0.23),
    Rotamer::new( 62.0, 0.0, 0.0, 0.0, 0.05),
    Rotamer::new(-65.0, 0.0, 0.0, 0.0, 0.00),
];
static ROT_THR: &[Rotamer] = &[
    Rotamer::new( 62.0, 0.0, 0.0, 0.0, 0.45),
    Rotamer::new(-60.0, 0.0, 0.0, 0.0, 0.32),
    Rotamer::new(-174.0,0.0, 0.0, 0.0, 0.18),
    Rotamer::new( 62.0, 0.0, 0.0, 0.0, 0.05),
    Rotamer::new(-60.0, 0.0, 0.0, 0.0, 0.00),
];
static ROT_TRP: &[Rotamer] = &[
    Rotamer::new(-65.0, 96.0, 0.0, 0.0, 0.32),
    Rotamer::new(-65.0,-84.0, 0.0, 0.0, 0.20),
    Rotamer::new(-174.0,96.0, 0.0, 0.0, 0.15),
    Rotamer::new(  62.0,96.0, 0.0, 0.0, 0.12),
    Rotamer::new(-174.0,-84.0,0.0, 0.0, 0.08),
];
static ROT_TYR: &[Rotamer] = &[
    Rotamer::new(-65.0,  90.0, 0.0, 0.0, 0.35),
    Rotamer::new(-65.0, -85.0, 0.0, 0.0, 0.22),
    Rotamer::new(-174.0, 80.0, 0.0, 0.0, 0.15),
    Rotamer::new(  62.0, 90.0, 0.0, 0.0, 0.12),
    Rotamer::new(-174.0,-80.0, 0.0, 0.0, 0.06),
];
static ROT_VAL: &[Rotamer] = &[
    Rotamer::new(-60.0, 0.0, 0.0, 0.0, 0.47),
    Rotamer::new(180.0, 0.0, 0.0, 0.0, 0.36),
    Rotamer::new( 60.0, 0.0, 0.0, 0.0, 0.14),
    Rotamer::new(-60.0, 0.0, 0.0, 0.0, 0.03),
    Rotamer::new(180.0, 0.0, 0.0, 0.0, 0.00),
];

/// Indexed by `AminoAcid as usize`.  Empty slice means no side-chain (GLY/ALA).
pub static ROTAMER_LIB: [&[Rotamer]; 20] = [
    ROT_ALA, // Ala = 0
    ROT_ARG,
    ROT_ASN,
    ROT_ASP,
    ROT_CYS,
    ROT_GLN,
    ROT_GLU,
    ROT_GLY, // Gly = 7
    ROT_HIS,
    ROT_ILE,
    ROT_LEU,
    ROT_LYS,
    ROT_MET,
    ROT_PHE,
    ROT_PRO,
    ROT_SER,
    ROT_THR,
    ROT_TRP,
    ROT_TYR,
    ROT_VAL, // Val = 19
];

// ── NERF algorithm ────────────────────────────────────────────────────────────

/// Place atom D given reference atoms A, B, C and internal coordinates.
///
/// Uses the Natural Extension Reference Frame (NERF) algorithm:
///   Parsons et al. (2005) "Practical Conversion from Torsion Space to Cartesian
///   Space for In Silico Protein Synthesis", J. Comput. Chem. 26(10).
///
/// `bond_len`    : |C-D| in Å
/// `angle_deg`   : bond angle ∠B-C-D in degrees
/// `dihedral_deg`: dihedral A-B-C-D in degrees
#[inline]
pub fn place_atom(
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    bond_len: f32,
    angle_deg: f32,
    dihedral_deg: f32,
) -> [f32; 3] {
    let angle = angle_deg.to_radians();
    let dihedral = dihedral_deg.to_radians();

    // bc = unit vector C → B
    let bc = norm3(sub3(b, c));
    // n  = unit normal to plane A-B-C
    let ba = sub3(b, a);
    let n = norm3(cross3(ba, bc));
    // m  = bc × n  (completes right-handed frame)
    let m = cross3(bc, n);

    // D in local frame of C:
    //   along bc:     -cos(angle)
    //   along m:       sin(angle)*cos(dihedral)
    //   along n:       sin(angle)*sin(dihedral)
    let sa = angle.sin();
    let ca = angle.cos();
    let sd = dihedral.sin();
    let cd = dihedral.cos();

    let d = [
        c[0] + bond_len * (-ca * bc[0] + sa * cd * m[0] + sa * sd * n[0]),
        c[1] + bond_len * (-ca * bc[1] + sa * cd * m[1] + sa * sd * n[1]),
        c[2] + bond_len * (-ca * bc[2] + sa * cd * m[2] + sa * sd * n[2]),
    ];
    d
}

#[inline(always)]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
fn norm3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-8);
    [v[0] / len, v[1] / len, v[2] / len]
}

// ── Side-chain build instructions ─────────────────────────────────────────────

/// One bond in the side-chain build sequence.
/// ref_a/b/c are indices into the growing per-residue atom list (backbone = 0-4).
/// dihedral_chi = which chi index drives this dihedral (0-3), or 255 for fixed.
#[derive(Clone, Copy)]
pub struct BondDef {
    pub ref_a: u8,
    pub ref_b: u8,
    pub ref_c: u8,
    pub bond_len: f32,
    pub angle: f32,     // degrees
    pub dihedral_fixed: f32, // degrees; used only if chi_idx == 255
    pub chi_idx: u8,    // 0-3 = use chi[chi_idx], 255 = use dihedral_fixed
}

impl BondDef {
    const fn chi(ra: u8, rb: u8, rc: u8, bl: f32, ang: f32, ci: u8) -> Self {
        BondDef { ref_a: ra, ref_b: rb, ref_c: rc, bond_len: bl, angle: ang,
                  dihedral_fixed: 0.0, chi_idx: ci }
    }
    const fn fixed(ra: u8, rb: u8, rc: u8, bl: f32, ang: f32, dih: f32) -> Self {
        BondDef { ref_a: ra, ref_b: rb, ref_c: rc, bond_len: bl, angle: ang,
                  dihedral_fixed: dih, chi_idx: 255 }
    }
}

// Backbone atom layout (indices 0-4):
//   0=N  1=CA  2=C  3=O  4=CB  (GLY has no CB, ALA stops at CB)
// Side-chain atoms start at index 5.

// ─────── build defs per AA ────────────────────────────────────────────────────
// For ALA/GLY the slices are empty.

static BD_ALA: &[BondDef] = &[];
static BD_GLY: &[BondDef] = &[];

//              ref_a ref_b ref_c bond   angle   chi
static BD_CYS: &[BondDef] = &[
    // SG: N(0)-CA(1)-CB(4)-SG, chi1
    BondDef::chi(0, 1, 4, 1.81, 114.0, 0),
];
static BD_SER: &[BondDef] = &[
    BondDef::chi(0, 1, 4, 1.42, 110.8, 0),
];
static BD_THR: &[BondDef] = &[
    // OG1 chi1
    BondDef::chi(0, 1, 4, 1.43, 109.1, 0),
    // CG2 fixed ~120° from OG1
    BondDef::fixed(0, 1, 4, 1.52, 111.5, 120.0),
];
static BD_VAL: &[BondDef] = &[
    // CG1 chi1
    BondDef::chi(0, 1, 4, 1.52, 111.0, 0),
    // CG2 fixed
    BondDef::fixed(0, 1, 4, 1.52, 111.0, 120.0),
];
static BD_ILE: &[BondDef] = &[
    // CG1 chi1
    BondDef::chi(0, 1, 4, 1.53, 111.5, 0),
    // CG2 fixed
    BondDef::fixed(0, 1, 4, 1.52, 111.5, 120.0),
    // CD1 chi2 off CG1(5)
    BondDef::chi(1, 4, 5, 1.52, 114.0, 1),
];
static BD_LEU: &[BondDef] = &[
    // CG chi1
    BondDef::chi(0, 1, 4, 1.53, 116.1, 0),
    // CD1 chi2
    BondDef::chi(1, 4, 5, 1.52, 111.0, 1),
    // CD2 fixed from CG
    BondDef::fixed(1, 4, 5, 1.52, 111.0, 120.0),
];
static BD_MET: &[BondDef] = &[
    BondDef::chi(0, 1, 4, 1.53, 114.1, 0), // CG chi1
    BondDef::chi(1, 4, 5, 1.81, 112.7, 1), // SD chi2
    BondDef::chi(4, 5, 6, 1.79, 100.9, 2), // CE chi3
];
static BD_PRO: &[BondDef] = &[
    BondDef::chi(0, 1, 4, 1.53, 104.0, 0), // CG chi1 (ring puckering)
    BondDef::fixed(1, 4, 5, 1.50, 106.0, 0.0), // CD closes ring
];
static BD_PHE: &[BondDef] = &[
    BondDef::chi(0, 1, 4, 1.51, 113.8, 0),  // CG chi1
    BondDef::chi(1, 4, 5, 1.39, 120.7, 1),  // CD1 chi2
    BondDef::fixed(1, 4, 5, 1.39, 120.7, 180.0), // CD2
    BondDef::fixed(4, 5, 6, 1.39, 120.0, 0.0),   // CE1
    BondDef::fixed(4, 6, 7, 1.39, 120.0, 0.0),   // CE2
    BondDef::fixed(5, 6, 8, 1.38, 120.0, 0.0),   // CZ
];
static BD_TYR: &[BondDef] = &[
    BondDef::chi(0, 1, 4, 1.51, 113.8, 0),  // CG chi1
    BondDef::chi(1, 4, 5, 1.39, 120.7, 1),  // CD1 chi2
    BondDef::fixed(1, 4, 5, 1.39, 120.7, 180.0),
    BondDef::fixed(4, 5, 6, 1.39, 120.0, 0.0),
    BondDef::fixed(4, 6, 7, 1.39, 120.0, 0.0),
    BondDef::fixed(5, 6, 8, 1.38, 120.0, 0.0),   // CZ
    BondDef::fixed(6, 7, 9, 1.38, 119.8, 180.0), // OH
];
static BD_TRP: &[BondDef] = &[
    BondDef::chi(0, 1, 4, 1.50, 114.1, 0),  // CG chi1
    BondDef::chi(1, 4, 5, 1.37, 126.8, 1),  // CD1 chi2
    BondDef::fixed(1, 4, 5, 1.43, 126.8, 180.0), // CD2
    BondDef::fixed(4, 5, 6, 1.37, 108.0, 0.0),   // NE1
    BondDef::fixed(4, 6, 7, 1.40, 126.0, 0.0),   // CE2
    BondDef::fixed(6, 7, 8, 1.40, 120.0, 0.0),   // CE3
    BondDef::fixed(7, 8, 9, 1.37, 120.0, 0.0),   // CZ2
    BondDef::fixed(8, 9, 10,1.37, 120.0, 0.0),   // CZ3
    BondDef::fixed(9, 10,11,1.40, 120.0, 0.0),   // CH2
];
static BD_HIS: &[BondDef] = &[
    BondDef::chi(0, 1, 4, 1.50, 113.8, 0),  // CG chi1
    BondDef::chi(1, 4, 5, 1.38, 122.7, 1),  // ND1 chi2
    BondDef::fixed(1, 4, 5, 1.36, 131.5, 180.0), // CD2
    BondDef::fixed(4, 5, 6, 1.32, 108.0, 0.0),   // CE1
    BondDef::fixed(4, 6, 7, 1.36, 108.0, 0.0),   // NE2
];
static BD_ASP: &[BondDef] = &[
    BondDef::chi(0, 1, 4, 1.52, 113.4, 0),  // CG chi1
    BondDef::chi(1, 4, 5, 1.25, 118.4, 1),  // OD1 chi2
    BondDef::fixed(1, 4, 5, 1.25, 118.4, 180.0), // OD2
];
static BD_ASN: &[BondDef] = &[
    BondDef::chi(0, 1, 4, 1.52, 112.6, 0),  // CG chi1
    BondDef::chi(1, 4, 5, 1.23, 120.8, 1),  // OD1 chi2
    BondDef::fixed(1, 4, 5, 1.33, 116.5, 180.0), // ND2
];
static BD_GLU: &[BondDef] = &[
    BondDef::chi(0, 1, 4, 1.53, 114.1, 0),  // CG chi1
    BondDef::chi(1, 4, 5, 1.52, 116.6, 1),  // CD chi2
    BondDef::chi(4, 5, 6, 1.25, 118.4, 2),  // OE1 chi3
    BondDef::fixed(4, 5, 6, 1.25, 118.4, 180.0), // OE2
];
static BD_GLN: &[BondDef] = &[
    BondDef::chi(0, 1, 4, 1.53, 114.1, 0),  // CG chi1
    BondDef::chi(1, 4, 5, 1.52, 111.3, 1),  // CD chi2
    BondDef::chi(4, 5, 6, 1.23, 120.8, 2),  // OE1 chi3
    BondDef::fixed(4, 5, 6, 1.33, 116.5, 180.0), // NE2
];
static BD_LYS: &[BondDef] = &[
    BondDef::chi(0, 1, 4, 1.53, 114.1, 0),  // CG chi1
    BondDef::chi(1, 4, 5, 1.52, 111.3, 1),  // CD chi2
    BondDef::chi(4, 5, 6, 1.52, 111.3, 2),  // CE chi3
    BondDef::chi(5, 6, 7, 1.49, 112.0, 3),  // NZ chi4
];
static BD_ARG: &[BondDef] = &[
    BondDef::chi(0, 1, 4, 1.53, 114.1, 0),  // CG chi1
    BondDef::chi(1, 4, 5, 1.52, 111.3, 1),  // CD chi2
    BondDef::chi(4, 5, 6, 1.46, 112.3, 2),  // NE chi3
    BondDef::chi(5, 6, 7, 1.33, 124.2, 3),  // CZ chi4
    BondDef::fixed(6, 7, 8, 1.33, 120.0, 0.0),   // NH1
    BondDef::fixed(6, 7, 8, 1.33, 120.0, 180.0), // NH2
];

/// Build instructions for each AA, indexed by `AminoAcid as usize`.
pub static SIDECHAIN_DEFS: [&[BondDef]; 20] = [
    BD_ALA, // Ala
    BD_ARG,
    BD_ASN,
    BD_ASP,
    BD_CYS,
    BD_GLN,
    BD_GLU,
    BD_GLY, // Gly
    BD_HIS,
    BD_ILE,
    BD_LEU,
    BD_LYS,
    BD_MET,
    BD_PHE,
    BD_PRO,
    BD_SER,
    BD_THR,
    BD_TRP,
    BD_TYR,
    BD_VAL,
];

// ── Side-chain builder ────────────────────────────────────────────────────────

/// Place all side-chain heavy atoms given the backbone frame and chi angles.
///
/// `backbone` contains [N, CA, C, O, CB] (5 atoms; GLY: 4, ALA: 5 with no call needed).
/// `chi_deg`  contains chi angles in degrees (chi1..chi_n_chi, rest ignored).
/// Appended atom positions are pushed into `out` in the order defined by `SIDECHAIN_DEFS`.
pub fn build_side_chain(
    aa: AminoAcid,
    backbone: &[[f32; 3]],
    chi_deg: &[f32; 4],
    out: &mut Vec<[f32; 3]>,
) {
    let defs = SIDECHAIN_DEFS[aa as usize];
    // working buffer: backbone + side-chain atoms built so far
    let mut atoms: Vec<[f32; 3]> = backbone.to_vec();

    for bd in defs {
        let a = atoms[bd.ref_a as usize];
        let b = atoms[bd.ref_b as usize];
        let c = atoms[bd.ref_c as usize];
        let dih = if bd.chi_idx == 255 {
            bd.dihedral_fixed
        } else {
            chi_deg[bd.chi_idx as usize]
        };
        let pos = place_atom(a, b, c, bd.bond_len, bd.angle, dih);
        atoms.push(pos);
        out.push(pos);
    }
}

// ── Accessors ─────────────────────────────────────────────────────────────────

/// Return the rotamer library for a given amino acid.
#[inline(always)]
pub fn rotamers(aa: AminoAcid) -> &'static [Rotamer] {
    ROTAMER_LIB[aa as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nerf_places_atom_at_correct_bond_length() {
        let a = [0.0_f32, 0.0, 0.0];
        let b = [1.5, 0.0, 0.0];
        let c = [1.5, 1.4, 0.0];
        let d = place_atom(a, b, c, 1.52, 112.0, -60.0);
        let dx = d[0] - c[0];
        let dy = d[1] - c[1];
        let dz = d[2] - c[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!((dist - 1.52).abs() < 1e-4, "bond length {dist:.4} ≠ 1.52");
    }

    #[test]
    fn rotamer_lib_non_empty_for_ile() {
        let rots = rotamers(AminoAcid::Ile);
        assert!(!rots.is_empty());
        assert_eq!(rots.len(), 5);
    }
}
