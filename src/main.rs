mod allatom;
mod amber;
mod atom;
mod diffusion;
mod energy;
mod error;
#[cfg(feature = "gpu")]
mod gpu;
mod pdb;
mod rotamer;
mod solvation;
mod spatial;

use error::BinderError;
use std::env;
use std::fs;
use std::time::Instant;

fn usage() {
    eprintln!(
        "USAGE\n\
         \n  binder --pdb <antigen.pdb> [OPTIONS]\
         \n  binder --seq <PEPTIDE>    [OPTIONS]\
         \n\
         \nOPTIONS\
         \n  --pdb   PATH      Antigen input as PDB file\
         \n  --seq   STRING    Antigen input as one-letter FASTA string\
         \n  --length N        Desired antibody length in residues [default: same as antigen]\
         \n  --out   PATH      Write resulting antibody PDB [default: stdout]\
         \n  --allatom         Use all-atom AMBER engine (default: Cα coarse-grained)\
         \n  --gpu             Enable GPU acceleration [requires --allatom]\
         \n  --no-gpu          Force CPU-only [default]\
         \n  --pop   N         Population size [default: {POP}]\
         \n  --iter  N         Diffusion iterations [default: {ITER}]",
        POP  = diffusion::POPULATION,
        ITER = diffusion::ITERATIONS,
    );
}

fn main() -> Result<(), BinderError> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        usage();
        std::process::exit(1);
    }

    // ── Argument parsing ─────────────────────────────────────────────────────
    let mut pdb_path:  Option<String> = None;
    let mut seq_str:   Option<String> = None;
    let mut ab_length: Option<usize>  = None;
    let mut out_path:  Option<String> = None;
    let mut use_allatom = false;
    let mut want_gpu    = false;

    let mut idx = 1;
    while idx < args.len() {
        match args[idx].as_str() {
            "--pdb" => { idx += 1; pdb_path = args.get(idx).cloned(); }
            "--seq" => { idx += 1; seq_str  = args.get(idx).cloned(); }
            "--length" => {
                idx += 1;
                ab_length = args.get(idx).and_then(|s| s.parse().ok()).or(Some(20));
            }
            "--out"     => { idx += 1; out_path = args.get(idx).cloned(); }
            "--allatom" => { use_allatom = true; }
            "--gpu"     => { want_gpu = true; use_allatom = true; }
            "--no-gpu"  => { want_gpu = false; }
            "--help" | "-h" => { usage(); return Ok(()); }
            other => {
                eprintln!("Unknown argument: {other}");
                usage();
                std::process::exit(1);
            }
        }
        idx += 1;
    }

    // ── Load antigen ─────────────────────────────────────────────────────────
    let antigen_cg = if let Some(path) = pdb_path.as_deref() {
        eprintln!("[binder] Loading antigen PDB: {path}");
        pdb::parse_pdb(path)?
    } else if let Some(seq) = seq_str.as_deref() {
        eprintln!("[binder] Building antigen from peptide: {seq}");
        pdb::parse_peptide(seq)?
    } else {
        eprintln!("[binder] Error: provide --pdb or --seq");
        usage();
        std::process::exit(1);
    };

    let ag_n = antigen_cg.len();
    let ab_n = ab_length.unwrap_or(ag_n);

    eprintln!(
        "[binder] Antigen: {ag_n} residues | sequence: {}",
        antigen_cg.sequence()
    );
    eprintln!("[binder] Antibody length: {ab_n} residues");

    let t0 = Instant::now();

    if use_allatom {
        // ── All-atom path ─────────────────────────────────────────────────────
        let antigen_aa = allatom::protein_from_ca_trace(
            &antigen_cg.x, &antigen_cg.y, &antigen_cg.z,
            &antigen_cg.amino_acid,
        );
        eprintln!(
            "[binder] All-atom antigen: {} atoms | population: {} GPU + {} CPU",
            antigen_aa.n_atoms(),
            if want_gpu { diffusion::AA_POPULATION } else { 0 },
            diffusion::TOP_K,
        );

        #[cfg(feature = "gpu")]
        let gpu_ctx = if want_gpu {
            let ctx = crate::gpu::GpuContext::try_init();
            if ctx.is_none() {
                eprintln!("[binder] WARNING: No GPU adapter found, falling back to CPU");
            }
            ctx
        } else {
            None
        };

        let result = {
            #[cfg(feature = "gpu")]
            { diffusion::run_allatom(&antigen_aa, ab_n, gpu_ctx.as_ref()) }
            #[cfg(not(feature = "gpu"))]
            { diffusion::run_allatom(&antigen_aa, ab_n) }
        };
        let elapsed = t0.elapsed();

        eprintln!(
            "[binder] Done in {:.2?} | best energy = {:.3} kcal/mol (all-atom AMBER)",
            elapsed, result.energy,
        );

        println!("=== De Novo Antibody Design Result (All-Atom AMBER) ===");
        println!("Antigen sequence  : {}", antigen_cg.sequence());
        println!("Antibody sequence : {}", result.sequence);
        println!("Binding energy    : {:.4} kcal/mol", result.energy);
        println!("Residues          : {}", result.antibody.n_residues());
        println!("Atoms             : {}", result.antibody.n_atoms());
        println!("Elapsed           : {:.2?}", elapsed);
        println!();

        let pdb_data = pdb::write_pdb_allatom(&result.antibody, 'B');
        if let Some(path) = out_path {
            fs::write(&path, &pdb_data).map_err(BinderError::Io)?;
            eprintln!("[binder] Antibody PDB written to {path}");
        } else {
            print!("{pdb_data}");
        }
    } else {
        // ── Coarse-grained path (original) ────────────────────────────────────
        eprintln!(
            "[binder] Population: {} | Iterations: {}",
            diffusion::POPULATION,
            diffusion::ITERATIONS,
        );

        let result = diffusion::run(&antigen_cg, ab_n);
        let elapsed = t0.elapsed();

        eprintln!(
            "[binder] Done in {:.2?} | best energy = {:.3} kcal/mol",
            elapsed, result.energy,
        );

        println!("=== De Novo Antibody Design Result ===");
        println!("Antigen sequence  : {}", antigen_cg.sequence());
        println!("Antibody sequence : {}", result.sequence);
        println!("Binding energy    : {:.4} kcal/mol", result.energy);
        println!("Residues          : {}", result.antibody.len());
        println!("Elapsed           : {:.2?}", elapsed);
        println!();

        let pdb_data = pdb::write_pdb(&result.antibody, 'B');
        if let Some(path) = out_path {
            fs::write(&path, &pdb_data).map_err(BinderError::Io)?;
            eprintln!("[binder] Antibody PDB written to {path}");
        } else {
            print!("{pdb_data}");
        }
    }

    Ok(())
}
