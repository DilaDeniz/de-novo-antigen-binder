mod allatom;
mod amber;
mod atom;
mod diffusion;
mod energy;
mod error;
mod filters;
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
         \n  --pop   N         Population size [default: {POP} CG / {AAPOP} all-atom]\
         \n  --iter  N         Diffusion iterations [default: {ITER} CG / {CPUIT} all-atom]\
         \n  --top   N         Report the N lowest-energy candidates [default: 1]\
         \n  --fasta-only      Skip PDB generation; print only sequence + energy summary",
        POP   = diffusion::POPULATION,
        ITER  = diffusion::ITERATIONS,
        AAPOP = diffusion::TOP_K,
        CPUIT = diffusion::CPU_STEPS,
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
    let mut pop_override:  Option<usize> = None;
    let mut iter_override: Option<usize> = None;
    let mut top_n = 1usize;
    let mut fasta_only = false;

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
            "--pop"     => { idx += 1; pop_override  = args.get(idx).and_then(|s| s.parse().ok()); }
            "--iter"    => { idx += 1; iter_override = args.get(idx).and_then(|s| s.parse().ok()); }
            "--top"     => { idx += 1; top_n = args.get(idx).and_then(|s| s.parse().ok()).unwrap_or(1).max(1); }
            "--fasta-only" => { fasta_only = true; }
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
        let population = pop_override.unwrap_or(diffusion::TOP_K);
        let iterations = iter_override.unwrap_or(diffusion::CPU_STEPS);
        eprintln!(
            "[binder] All-atom antigen: {} atoms | population: {} GPU + {} CPU | iterations: {}",
            antigen_aa.n_atoms(),
            if want_gpu { diffusion::AA_POPULATION } else { 0 },
            population,
            iterations,
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

        let results = {
            #[cfg(feature = "gpu")]
            { diffusion::run_allatom(&antigen_aa, ab_n, population, iterations, top_n, gpu_ctx.as_ref()) }
            #[cfg(not(feature = "gpu"))]
            { diffusion::run_allatom(&antigen_aa, ab_n, population, iterations, top_n) }
        };
        let elapsed = t0.elapsed();

        eprintln!(
            "[binder] Done in {:.2?} | {} candidate(s) | best E_MM+solv = {:.3} kcal/mol",
            elapsed, results.len(), results[0].energy,
        );

        println!("=== De Novo Antibody Design Result (All-Atom AMBER) ===");
        println!("Antigen sequence  : {}", antigen_cg.sequence());
        println!();

        for (rank, result) in results.iter().enumerate() {
            let quality   = filters::SequenceQuality::assess(&result.antibody, &antigen_aa);
            let iface_lb  = filters::SequenceQuality::interface_labels(&result.antibody, &antigen_aa);
            let breakdown = energy::interaction_energy_atoms_breakdown(&antigen_aa.atoms, &result.antibody.atoms);
            let dg_corr   = result.energy + quality.entropy_penalty;

            println!("--- Candidate #{} ---", rank + 1);
            println!("Antibody sequence : {}", result.sequence);
            if !fasta_only {
                println!("CDR/FR map        : {iface_lb}  (I=interface, F=framework)");
                println!("Interface residues: {}/{}", quality.n_interface, result.antibody.n_residues());
            }
            println!();
            println!("LJ                : {:.4} kcal/mol", breakdown.lj);
            println!("Coulomb           : {:.4} kcal/mol", breakdown.coulomb);
            println!("Hydrophobic       : {:.4} kcal/mol", breakdown.hydrophobic);
            println!("H-bond            : {:.4} kcal/mol", breakdown.hbond);
            println!("Solvation (EEF1)  : {:.4} kcal/mol", breakdown.solvation);
            println!("Disulfide         : {:.4} kcal/mol", result.antibody.disulfide_energy());
            println!("E_MM + ΔΔG_solv   : {:.4} kcal/mol", result.energy);
            println!("−TΔS_bind (est.)  : +{:.2} kcal/mol  (transl/rot + {n_chi} frozen χ)",
                quality.entropy_penalty,
                n_chi = ((quality.entropy_penalty - 5.4) / 0.3).round() as usize,
            );
            println!("ΔG_bind (corrected): {:.4} kcal/mol", dg_corr);
            println!();
            println!("Net charge        : {:+}", quality.net_charge);
            println!("Aggregation risk  : {}  (max hydrophobic run: {})",
                if quality.aggregation_risk { "HIGH ⚠" } else { "Low" },
                quality.max_hydro_run,
            );
            if !fasta_only {
                println!();
                println!("Residues          : {}", result.antibody.n_residues());
                println!("Atoms             : {}", result.antibody.n_atoms());
            }
            println!();
        }
        println!("Elapsed           : {:.2?}", elapsed);
        println!();

        if !fasta_only {
            write_allatom_outputs(&results, out_path.as_deref())?;
        }
    } else {
        // ── Coarse-grained path (original) ────────────────────────────────────
        let population = pop_override.unwrap_or(diffusion::POPULATION);
        let iterations = iter_override.unwrap_or(diffusion::ITERATIONS);
        eprintln!(
            "[binder] Population: {} | Iterations: {}",
            population, iterations,
        );

        let results = diffusion::run(&antigen_cg, ab_n, population, iterations, top_n);
        let elapsed = t0.elapsed();

        eprintln!(
            "[binder] Done in {:.2?} | {} candidate(s) | best energy = {:.3} kcal/mol",
            elapsed, results.len(), results[0].energy,
        );

        println!("=== De Novo Antibody Design Result ===");
        println!("Antigen sequence  : {}", antigen_cg.sequence());
        println!();

        for (rank, result) in results.iter().enumerate() {
            let breakdown = energy::interaction_energy_breakdown(&antigen_cg, &result.antibody);

            println!("--- Candidate #{} ---", rank + 1);
            println!("Antibody sequence : {}", result.sequence);
            println!();
            println!("LJ                : {:.4} kcal/mol", breakdown.lj);
            println!("Coulomb           : {:.4} kcal/mol", breakdown.coulomb);
            println!("Hydrophobic       : {:.4} kcal/mol", breakdown.hydrophobic);
            println!("Binding energy    : {:.4} kcal/mol", result.energy);
            if !fasta_only {
                println!("Residues          : {}", result.antibody.len());
            }
            println!();
        }
        println!("Elapsed           : {:.2?}", elapsed);
        println!();

        if !fasta_only {
            write_cg_outputs(&results, out_path.as_deref())?;
        }
    }

    Ok(())
}

/// Write one PDB file per candidate. When `top_n == 1`, `out_path` is used
/// verbatim; otherwise each candidate gets a `_rank{N}` suffix before the
/// extension (or appended if there is none).
fn write_allatom_outputs(
    results: &[diffusion::AllAtomResult],
    out_path: Option<&str>,
) -> Result<(), BinderError> {
    for (rank, result) in results.iter().enumerate() {
        let pdb_data = pdb::write_pdb_allatom(&result.antibody, 'B');
        match out_path {
            Some(path) => {
                let path = candidate_path(path, rank, results.len());
                fs::write(&path, &pdb_data).map_err(BinderError::Io)?;
                eprintln!("[binder] Antibody PDB written to {path}");
            }
            None => print!("{pdb_data}"),
        }
    }
    Ok(())
}

fn write_cg_outputs(
    results: &[diffusion::DiffusionResult],
    out_path: Option<&str>,
) -> Result<(), BinderError> {
    for (rank, result) in results.iter().enumerate() {
        let pdb_data = pdb::write_pdb(&result.antibody, 'B');
        match out_path {
            Some(path) => {
                let path = candidate_path(path, rank, results.len());
                fs::write(&path, &pdb_data).map_err(BinderError::Io)?;
                eprintln!("[binder] Antibody PDB written to {path}");
            }
            None => print!("{pdb_data}"),
        }
    }
    Ok(())
}

/// Suffix `path` with `_rank{N}` before its extension when there is more
/// than one candidate; returns `path` unchanged for a single candidate.
fn candidate_path(path: &str, rank: usize, total: usize) -> String {
    if total <= 1 {
        return path.to_string();
    }
    match path.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}_rank{}.{ext}", rank + 1),
        None => format!("{path}_rank{}", rank + 1),
    }
}
