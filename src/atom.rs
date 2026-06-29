/// All 20 standard amino acids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AminoAcid {
    Ala = 0,
    Arg,
    Asn,
    Asp,
    Cys,
    Gln,
    Glu,
    Gly,
    His,
    Ile,
    Leu,
    Lys,
    Met,
    Phe,
    Pro,
    Ser,
    Thr,
    Trp,
    Tyr,
    Val,
}

pub const AA_COUNT: usize = 20;

pub const ALL_AA: [AminoAcid; AA_COUNT] = [
    AminoAcid::Ala,
    AminoAcid::Arg,
    AminoAcid::Asn,
    AminoAcid::Asp,
    AminoAcid::Cys,
    AminoAcid::Gln,
    AminoAcid::Glu,
    AminoAcid::Gly,
    AminoAcid::His,
    AminoAcid::Ile,
    AminoAcid::Leu,
    AminoAcid::Lys,
    AminoAcid::Met,
    AminoAcid::Phe,
    AminoAcid::Pro,
    AminoAcid::Ser,
    AminoAcid::Thr,
    AminoAcid::Trp,
    AminoAcid::Tyr,
    AminoAcid::Val,
];

impl AminoAcid {
    /// Net partial charge (elementary units, pH 7.4 approximation).
    #[inline(always)]
    pub fn charge(self) -> f32 {
        match self {
            AminoAcid::Arg | AminoAcid::Lys => 1.0,
            AminoAcid::Asp | AminoAcid::Glu => -1.0,
            AminoAcid::His => 0.1,
            _ => 0.0,
        }
    }

    /// Lennard-Jones well depth ε (kcal/mol), residue-level C-alpha approximation.
    #[inline(always)]
    pub fn lj_epsilon(self) -> f32 {
        match self {
            AminoAcid::Gly => 0.05,
            AminoAcid::Ala => 0.10,
            AminoAcid::Val | AminoAcid::Ile | AminoAcid::Leu => 0.15,
            AminoAcid::Phe | AminoAcid::Trp | AminoAcid::Tyr => 0.20,
            AminoAcid::Arg | AminoAcid::Lys => 0.12,
            AminoAcid::Asp | AminoAcid::Glu => 0.13,
            _ => 0.11,
        }
    }

    /// Lennard-Jones radius σ (Å), C-alpha effective radius by residue size.
    #[inline(always)]
    pub fn lj_sigma(self) -> f32 {
        match self {
            AminoAcid::Gly => 2.5,
            AminoAcid::Ala => 3.0,
            AminoAcid::Val => 3.5,
            AminoAcid::Ile | AminoAcid::Leu => 3.8,
            AminoAcid::Phe | AminoAcid::Trp | AminoAcid::Tyr => 4.2,
            AminoAcid::Arg | AminoAcid::Lys => 4.0,
            _ => 3.2,
        }
    }

    #[inline(always)]
    pub fn is_hydrophobic(self) -> bool {
        matches!(
            self,
            AminoAcid::Ala
                | AminoAcid::Val
                | AminoAcid::Ile
                | AminoAcid::Leu
                | AminoAcid::Met
                | AminoAcid::Phe
                | AminoAcid::Trp
        )
    }

    pub fn to_char(self) -> char {
        const CHARS: [char; AA_COUNT] = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
            'T', 'W', 'Y', 'V',
        ];
        CHARS[self as usize]
    }

    pub fn three_letter(self) -> &'static str {
        const NAMES: [&str; AA_COUNT] = [
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
            "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        ];
        NAMES[self as usize]
    }

    #[inline(always)]
    pub fn from_index(idx: usize) -> Self {
        ALL_AA[idx % AA_COUNT]
    }

    pub fn from_three_letter(s: &str) -> Self {
        match s {
            "ALA" => AminoAcid::Ala,
            "ARG" => AminoAcid::Arg,
            "ASN" => AminoAcid::Asn,
            "ASP" => AminoAcid::Asp,
            "CYS" => AminoAcid::Cys,
            "GLN" => AminoAcid::Gln,
            "GLU" => AminoAcid::Glu,
            "GLY" => AminoAcid::Gly,
            "HIS" | "HID" | "HIE" | "HIP" => AminoAcid::His,
            "ILE" => AminoAcid::Ile,
            "LEU" => AminoAcid::Leu,
            "LYS" => AminoAcid::Lys,
            "MET" => AminoAcid::Met,
            "PHE" => AminoAcid::Phe,
            "PRO" => AminoAcid::Pro,
            "SER" => AminoAcid::Ser,
            "THR" => AminoAcid::Thr,
            "TRP" => AminoAcid::Trp,
            "TYR" => AminoAcid::Tyr,
            "VAL" => AminoAcid::Val,
            _ => AminoAcid::Gly,
        }
    }

    /// Parse a one-letter FASTA code. Returns `None` for unrecognized letters.
    pub fn from_char(c: char) -> Option<Self> {
        Some(match c.to_ascii_uppercase() {
            'A' => AminoAcid::Ala,
            'R' => AminoAcid::Arg,
            'N' => AminoAcid::Asn,
            'D' => AminoAcid::Asp,
            'C' => AminoAcid::Cys,
            'Q' => AminoAcid::Gln,
            'E' => AminoAcid::Glu,
            'G' => AminoAcid::Gly,
            'H' => AminoAcid::His,
            'I' => AminoAcid::Ile,
            'L' => AminoAcid::Leu,
            'K' => AminoAcid::Lys,
            'M' => AminoAcid::Met,
            'F' => AminoAcid::Phe,
            'P' => AminoAcid::Pro,
            'S' => AminoAcid::Ser,
            'T' => AminoAcid::Thr,
            'W' => AminoAcid::Trp,
            'Y' => AminoAcid::Tyr,
            'V' => AminoAcid::Val,
            _ => return None,
        })
    }
}

/// Residue-level protein representation using Structure-of-Arrays layout.
///
/// Each Vec has exactly `len()` elements. Separate coordinate arrays let the
/// compiler issue SIMD loads across all x (or all y/z) values in a tight loop.
pub struct ResidueCloud {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    pub charge: Vec<f32>,
    pub epsilon: Vec<f32>,
    pub sigma: Vec<f32>,
    /// 1 = hydrophobic, 0 = hydrophilic — stored as u8 for branchless arithmetic.
    pub hydrophobic: Vec<u8>,
    pub amino_acid: Vec<AminoAcid>,
}

impl ResidueCloud {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            x: Vec::with_capacity(cap),
            y: Vec::with_capacity(cap),
            z: Vec::with_capacity(cap),
            charge: Vec::with_capacity(cap),
            epsilon: Vec::with_capacity(cap),
            sigma: Vec::with_capacity(cap),
            hydrophobic: Vec::with_capacity(cap),
            amino_acid: Vec::with_capacity(cap),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    pub fn push(&mut self, x: f32, y: f32, z: f32, aa: AminoAcid) {
        self.x.push(x);
        self.y.push(y);
        self.z.push(z);
        self.charge.push(aa.charge());
        self.epsilon.push(aa.lj_epsilon());
        self.sigma.push(aa.lj_sigma());
        self.hydrophobic.push(aa.is_hydrophobic() as u8);
        self.amino_acid.push(aa);
    }

    /// Mutate the amino acid at position `idx`, updating all derived fields.
    #[inline]
    pub fn set_aa(&mut self, idx: usize, aa: AminoAcid) {
        self.charge[idx] = aa.charge();
        self.epsilon[idx] = aa.lj_epsilon();
        self.sigma[idx] = aa.lj_sigma();
        self.hydrophobic[idx] = aa.is_hydrophobic() as u8;
        self.amino_acid[idx] = aa;
    }

    pub fn center_of_mass(&self) -> [f32; 3] {
        let n = self.len() as f32;
        if n == 0.0 {
            return [0.0, 0.0, 0.0];
        }
        let cx = self.x.iter().sum::<f32>() / n;
        let cy = self.y.iter().sum::<f32>() / n;
        let cz = self.z.iter().sum::<f32>() / n;
        [cx, cy, cz]
    }

    /// One-letter FASTA sequence.
    pub fn sequence(&self) -> String {
        self.amino_acid.iter().map(|a| a.to_char()).collect()
    }
}
