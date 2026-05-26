/// PDB file I/O and inline peptide parsing.
///
/// Reader: extracts CA (C-alpha) atoms → one residue per ATOM/HETATM record.
/// Writer: outputs antibody result in valid PDB ATOM format.
/// Peptide: parses a one-letter FASTA string into a ResidueCloud with a
///          straight-chain backbone geometry (3.8 Å Cα–Cα bonds).
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{self, BufRead};

use crate::allatom::AtomProtein;
use crate::amber::topology;
use crate::atom::{AminoAcid, ResidueCloud};
use crate::error::BinderError;

// ── PDB reader ────────────────────────────────────────────────────────────────

/// Parse a PDB file and return a residue-level cloud (CA only).
pub fn parse_pdb(path: &str) -> Result<ResidueCloud, BinderError> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut cloud = ResidueCloud::with_capacity(256);

    for raw in reader.lines() {
        let line = raw?;
        if !line.starts_with("ATOM") && !line.starts_with("HETATM") {
            continue;
        }

        let atom_name = line.get(12..16).unwrap_or("    ").trim();
        if atom_name != "CA" {
            continue;
        }

        let res_name = line.get(17..20).unwrap_or("   ").trim();
        let x: f32 = line
            .get(30..38)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);
        let y: f32 = line
            .get(38..46)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);
        let z: f32 = line
            .get(46..54)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        let aa = AminoAcid::from_three_letter(res_name);
        cloud.push(x, y, z, aa);
    }

    if cloud.is_empty() {
        return Err(BinderError::EmptyInput);
    }
    Ok(cloud)
}

// ── FASTA peptide builder ─────────────────────────────────────────────────────

/// Build a ResidueCloud from a one-letter peptide string.
/// Residues are laid out in an idealized extended chain (Cα–Cα = 3.8 Å)
/// along the X axis, centred at the origin.
pub fn parse_peptide(seq: &str) -> Result<ResidueCloud, BinderError> {
    if seq.is_empty() {
        return Err(BinderError::EmptyInput);
    }

    const CA_BOND: f32 = 3.8; // Å, standard Cα–Cα distance

    let n = seq.len();
    let mut cloud = ResidueCloud::with_capacity(n);

    for (i, ch) in seq.chars().enumerate() {
        let aa = aa_from_char(ch).ok_or_else(|| {
            BinderError::Parse(format!("unknown one-letter code '{ch}' in peptide"))
        })?;
        let x = (i as f32 - n as f32 * 0.5) * CA_BOND;
        cloud.push(x, 0.0, 0.0, aa);
    }

    Ok(cloud)
}

fn aa_from_char(c: char) -> Option<AminoAcid> {
    let aa = match c.to_ascii_uppercase() {
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
    };
    Some(aa)
}

// ── PDB writer ────────────────────────────────────────────────────────────────

/// Serialise a ResidueCloud to PDB ATOM records (CA only).
///
/// Returns a `String` — no file I/O so the caller decides where it goes.
pub fn write_pdb(cloud: &ResidueCloud, chain_id: char) -> String {
    let mut out = String::with_capacity(cloud.len() * 80);

    for i in 0..cloud.len() {
        let aa = cloud.amino_acid[i];
        let serial = (i + 1).min(99_999);
        let res_seq = (i + 1).min(9_999);

        // PDB ATOM record — fixed-width column format
        let _ = write!(
            out,
            "ATOM  {:5} {:<4} {:3} {:1}{:4}    {:8.3}{:8.3}{:8.3}  1.00  0.00           C\n",
            serial,
            " CA ",
            aa.three_letter(),
            chain_id,
            res_seq,
            cloud.x[i],
            cloud.y[i],
            cloud.z[i],
        );
    }
    let _ = write!(out, "END\n");
    out
}

/// Serialise an `AtomProtein` to PDB ATOM records (all heavy atoms).
pub fn write_pdb_allatom(prot: &AtomProtein, chain_id: char) -> String {
    let n_atoms = prot.n_atoms();
    let mut out = String::with_capacity(n_atoms * 80);
    let mut serial = 0usize;

    for r in 0..prot.n_residues() {
        let aa      = prot.amino_acid[r];
        let topo    = topology(aa);
        let range   = prot.atom_range(r);
        let res_seq = (r + 1).min(9_999);

        for (k, atom_idx) in range.enumerate() {
            if k >= topo.atoms.len() { break; }
            serial += 1;
            let atom_name = topo.atoms[k].name;
            let _ = write!(
                out,
                "ATOM  {:5} {:<4} {:3} {:1}{:4}    {:8.3}{:8.3}{:8.3}  1.00  0.00           C\n",
                serial.min(99_999),
                atom_name,
                aa.three_letter(),
                chain_id,
                res_seq,
                prot.atoms.x[atom_idx],
                prot.atoms.y[atom_idx],
                prot.atoms.z[atom_idx],
            );
        }
    }
    let _ = write!(out, "END\n");
    out
}
