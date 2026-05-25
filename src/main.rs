mod atom;
mod diffusion;
mod energy;
mod error;
mod pdb;
mod spatial;

use error::BinderError;
use std::env;
use std::fs;
use std::time::Instant;

fn usage() {
    eprintln!(
        "USAGE\n\
         \n  binder --pdb <antigen.pdb> [--length N] [--out antibody.pdb]\
         \n  binder --seq <PEPTIDE>    [--length N] [--out antibody.pdb]\
         \n\
         \nARGUMENTS\
         \n  --pdb   PATH    Antigen input as PDB file (CA atoms used)\
         \n  --seq   STRING  Antigen input as one-letter FASTA string\
         \n  --length N      Desired antibody length in residues [default: same as antigen]\
         \n  --out   PATH    Write resulting antibody PDB [default: stdout]\
         \n  --pop   N       Parallel population size   [default: {POPULATION}]\
         \n  --iter  N       Diffusion iterations       [default: {ITERATIONS}]",
        POPULATION = diffusion::POPULATION,
        ITERATIONS = diffusion::ITERATIONS,
    );
}

fn main() -> Result<(), BinderError> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        usage();
        std::process::exit(1);
    }

    // ── Argument parsing ─────────────────────────────────────────────────────
    let mut pdb_path: Option<String> = None;
    let mut seq_str: Option<String> = None;
    let mut ab_length: Option<usize> = None;
    let mut out_path: Option<String> = None;

    let mut idx = 1;
    while idx < args.len() {
        match args[idx].as_str() {
            "--pdb" => {
                idx += 1;
                pdb_path = args.get(idx).cloned();
            }
            "--seq" => {
                idx += 1;
                seq_str = args.get(idx).cloned();
            }
            "--length" => {
                idx += 1;
                ab_length = args
                    .get(idx)
                    .and_then(|s| s.parse().ok())
                    .or(Some(20));
            }
            "--out" => {
                idx += 1;
                out_path = args.get(idx).cloned();
            }
            "--help" | "-h" => {
                usage();
                return Ok(());
            }
            other => {
                eprintln!("Unknown argument: {other}");
                usage();
                std::process::exit(1);
            }
        }
        idx += 1;
    }

    // ── Load antigen ─────────────────────────────────────────────────────────
    let antigen = if let Some(path) = pdb_path {
        eprintln!("[binder] Loading antigen PDB: {path}");
        pdb::parse_pdb(&path)?
    } else if let Some(seq) = seq_str {
        eprintln!("[binder] Building antigen from peptide: {seq}");
        pdb::parse_peptide(&seq)?
    } else {
        eprintln!("[binder] Error: provide --pdb or --seq");
        usage();
        std::process::exit(1);
    };

    let ag_n = antigen.len();
    let ab_n = ab_length.unwrap_or(ag_n);

    eprintln!(
        "[binder] Antigen: {ag_n} residues  |  sequence: {}",
        antigen.sequence()
    );
    eprintln!("[binder] Designing antibody of {ab_n} residues");
    eprintln!(
        "[binder] Population: {}  |  Iterations: {}",
        diffusion::POPULATION,
        diffusion::ITERATIONS,
    );

    // ── Run diffusion engine ─────────────────────────────────────────────────
    let t0 = Instant::now();
    let result = diffusion::run(&antigen, ab_n);
    let elapsed = t0.elapsed();

    eprintln!(
        "[binder] Done in {:.2?}  |  best energy = {:.3} kcal/mol",
        elapsed, result.energy,
    );

    // ── Output ───────────────────────────────────────────────────────────────
    println!("=== De Novo Antibody Design Result ===");
    println!("Antigen sequence  : {}", antigen.sequence());
    println!("Antibody sequence : {}", result.sequence);
    println!("Binding energy    : {:.4} kcal/mol", result.energy);
    println!("Residues          : {}", result.antibody.len());
    println!("Elapsed           : {:.2?}", elapsed);
    println!();

    let pdb_data = pdb::write_pdb(&result.antibody, 'B');

    if let Some(path) = out_path {
        fs::write(&path, &pdb_data)
            .map_err(BinderError::Io)?;
        eprintln!("[binder] Antibody PDB written to {path}");
    } else {
        print!("{pdb_data}");
    }

    Ok(())
}
