mod allatom;
mod amber;
mod atom;
mod diffusion;
mod energy;
mod error;
mod filters;
mod germline;
#[cfg(feature = "gpu")]
mod gpu;
mod pdb;
mod report;
mod rotamer;
mod solvation;
mod spatial;
mod validate;

use error::BinderError;
use std::env;
use std::fs;
use std::time::Instant;

/// Steepest-descent clash-relief steps run after the MC search, when
/// minimization is enabled (default on for --allatom and --fv).
const MINIMIZE_STEPS: usize = 60;

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
         \n  --fv              Two-chain VH/VL Fv design: germline framework held\
         \n                    fixed, CDR1-3 loops actively designed [implies --allatom]\
         \n  --h3-len N        CDR-H3 loop length in Fv mode [default: {H3LEN}]\
         \n  --l3-len N        CDR-L3 loop length in Fv mode [default: {L3LEN}]\
         \n  --gpu             Enable GPU acceleration [requires --allatom]\
         \n  --no-gpu          Force CPU-only [default]\
         \n  --pop   N         Population size [default: {POP} CG / {AAPOP} all-atom / {FVPOP} Fv]\
         \n  --iter  N         Diffusion iterations [default: {ITER} CG / {CPUIT} all-atom / {FVIT} Fv]\
         \n  --top   N         Report the N lowest-energy candidates [default: 1]\
         \n  --no-minimize     Skip post-search energy minimization (clash relief)\
         \n  --no-maturation   Skip affinity maturation (point mutation scan)\
         \n  --no-validate     Skip structure validation report (clashes/Ramachandran)\
         \n  --fasta-out PATH  Write candidate sequences as FASTA\
         \n  --json-out  PATH  Write candidate report as JSON\
         \n  --csv-out   PATH  Write candidate report as CSV\
         \n  --fasta-only      Skip PDB generation; print only sequence + energy summary",
        POP   = diffusion::POPULATION,
        ITER  = diffusion::ITERATIONS,
        AAPOP = diffusion::TOP_K,
        CPUIT = diffusion::CPU_STEPS,
        FVPOP = diffusion::FV_POPULATION,
        FVIT  = diffusion::FV_ITERATIONS,
        H3LEN = germline::VH_FRAMEWORK.cdr3_len,
        L3LEN = germline::VL_FRAMEWORK.cdr3_len,
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
    let mut fv_mode = false;
    let mut want_gpu    = false;
    let mut pop_override:  Option<usize> = None;
    let mut iter_override: Option<usize> = None;
    let mut top_n = 1usize;
    let mut fasta_only = false;
    let mut h3_len: Option<usize> = None;
    let mut l3_len: Option<usize> = None;
    let mut do_minimize = true;
    let mut do_maturation = true;
    let mut do_validate = true;
    let mut fasta_out: Option<String> = None;
    let mut json_out:  Option<String> = None;
    let mut csv_out:   Option<String> = None;

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
            "--fv"      => { fv_mode = true; use_allatom = true; }
            "--h3-len"  => { idx += 1; h3_len = args.get(idx).and_then(|s| s.parse().ok()); }
            "--l3-len"  => { idx += 1; l3_len = args.get(idx).and_then(|s| s.parse().ok()); }
            "--gpu"     => { want_gpu = true; use_allatom = true; }
            "--no-gpu"  => { want_gpu = false; }
            "--pop"     => { idx += 1; pop_override  = args.get(idx).and_then(|s| s.parse().ok()); }
            "--iter"    => { idx += 1; iter_override = args.get(idx).and_then(|s| s.parse().ok()); }
            "--top"     => { idx += 1; top_n = args.get(idx).and_then(|s| s.parse().ok()).unwrap_or(1).max(1); }
            "--no-minimize"   => { do_minimize = false; }
            "--no-maturation" => { do_maturation = false; }
            "--no-validate"   => { do_validate = false; }
            "--fasta-out" => { idx += 1; fasta_out = args.get(idx).cloned(); }
            "--json-out"  => { idx += 1; json_out  = args.get(idx).cloned(); }
            "--csv-out"   => { idx += 1; csv_out   = args.get(idx).cloned(); }
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

    if fv_mode {
        // ── Fv (two-chain VH/VL) path ────────────────────────────────────────────
        let antigen_aa = allatom::protein_from_ca_trace(
            &antigen_cg.x, &antigen_cg.y, &antigen_cg.z,
            &antigen_cg.amino_acid,
        );
        let population = pop_override.unwrap_or(diffusion::FV_POPULATION);
        let iterations = iter_override.unwrap_or(diffusion::FV_ITERATIONS);
        let h3 = h3_len.unwrap_or(germline::VH_FRAMEWORK.cdr3_len);
        let l3 = l3_len.unwrap_or(germline::VL_FRAMEWORK.cdr3_len);
        eprintln!(
            "[binder] Fv mode: germline VH/VL framework fixed, CDR1-3 designed | \
             antigen atoms: {} | population: {} | iterations: {} | CDR-H3: {} | CDR-L3: {}",
            antigen_aa.n_atoms(), population, iterations, h3, l3,
        );

        let mut results = diffusion::run_fv(&antigen_aa, h3, l3, population, iterations, top_n);

        for result in results.iter_mut() {
            if do_minimize {
                let light_snapshot = result.light.clone();
                diffusion::minimize(&mut result.heavy, &[&antigen_aa, &light_snapshot], MINIMIZE_STEPS);
                let heavy_snapshot = result.heavy.clone();
                diffusion::minimize(&mut result.light, &[&antigen_aa, &heavy_snapshot], MINIMIZE_STEPS);
                result.energy = diffusion::fv_energy(&antigen_aa, &result.heavy, &result.light);
            }
            if do_maturation {
                let h_mutable: Vec<usize> = (0..result.heavy_regions.len())
                    .filter(|&r| result.heavy_regions[r].is_cdr())
                    .collect();
                let light_snapshot = result.light.clone();
                diffusion::affinity_maturation_fv(&mut result.heavy, &antigen_aa, &light_snapshot, &h_mutable);

                let l_mutable: Vec<usize> = (0..result.light_regions.len())
                    .filter(|&r| result.light_regions[r].is_cdr())
                    .collect();
                let heavy_snapshot = result.heavy.clone();
                result.energy = diffusion::affinity_maturation_fv(
                    &mut result.light, &antigen_aa, &heavy_snapshot, &l_mutable,
                );

                result.heavy_sequence = result.heavy.sequence();
                result.light_sequence = result.light.sequence();
            }
        }
        let elapsed = t0.elapsed();

        eprintln!(
            "[binder] Done in {:.2?} | {} candidate(s) | best E_Fv = {:.3} kcal/mol",
            elapsed, results.len(), results[0].energy,
        );

        println!("=== De Novo Antibody Design Result (Fv: VH + VL, All-Atom AMBER) ===");
        println!("Antigen sequence  : {}", antigen_cg.sequence());
        println!();

        let mut candidate_reports = Vec::with_capacity(results.len());

        for (rank, result) in results.iter().enumerate() {
            let h_quality = filters::SequenceQuality::assess(&result.heavy, &antigen_aa);
            let l_quality = filters::SequenceQuality::assess(&result.light, &antigen_aa);

            let combined_entropy = h_quality.entropy_penalty + l_quality.entropy_penalty;
            let dg_corr = result.energy + combined_entropy;
            let net_charge = h_quality.net_charge + l_quality.net_charge;
            let aggregation_risk = h_quality.aggregation_risk || l_quality.aggregation_risk;
            let n_interface = h_quality.n_interface + l_quality.n_interface;

            println!("--- Candidate #{} ---", rank + 1);
            println!("VH sequence       : {}", result.heavy_sequence);
            println!("VL sequence       : {}", result.light_sequence);
            if !fasta_only {
                let h_regions: String = result.heavy_regions.iter().map(|r| r.label()).collect();
                let l_regions: String = result.light_regions.iter().map(|r| r.label()).collect();
                let h_iface = filters::SequenceQuality::interface_labels(&result.heavy, &antigen_aa);
                let l_iface = filters::SequenceQuality::interface_labels(&result.light, &antigen_aa);
                println!("VH region map     : {h_regions}  (F=framework, 1/2/3=CDR1/2/3)");
                println!("VH Ag-interface   : {h_iface}  (I=interface, F=framework)");
                println!("VL region map     : {l_regions}  (F=framework, 1/2/3=CDR1/2/3)");
                println!("VL Ag-interface   : {l_iface}  (I=interface, F=framework)");
                println!("Interface residues: {}/{}", n_interface, result.heavy.n_residues() + result.light.n_residues());
            }
            println!();

            let ag_vh = energy::interaction_energy_atoms_breakdown(&antigen_aa.atoms, &result.heavy.atoms);
            let ag_vl = energy::interaction_energy_atoms_breakdown(&antigen_aa.atoms, &result.light.atoms);
            let vh_vl = energy::interaction_energy_atoms_breakdown(&result.heavy.atoms, &result.light.atoms);
            println!("Ag-VH  LJ/Coul/Hphob/Hbond/Solv : {:.3} / {:.3} / {:.3} / {:.3} / {:.3}",
                ag_vh.lj, ag_vh.coulomb, ag_vh.hydrophobic, ag_vh.hbond, ag_vh.solvation);
            println!("Ag-VL  LJ/Coul/Hphob/Hbond/Solv : {:.3} / {:.3} / {:.3} / {:.3} / {:.3}",
                ag_vl.lj, ag_vl.coulomb, ag_vl.hydrophobic, ag_vl.hbond, ag_vl.solvation);
            println!("VH-VL  LJ/Coul/Hphob/Hbond/Solv : {:.3} / {:.3} / {:.3} / {:.3} / {:.3}",
                vh_vl.lj, vh_vl.coulomb, vh_vl.hydrophobic, vh_vl.hbond, vh_vl.solvation);
            println!("VH disulfide      : {:.4} kcal/mol", result.heavy.disulfide_energy());
            println!("VL disulfide      : {:.4} kcal/mol", result.light.disulfide_energy());
            println!("E_Fv (total)      : {:.4} kcal/mol", result.energy);
            println!("−TΔS_bind (est.)  : +{:.2} kcal/mol  (VH + VL)", combined_entropy);
            println!("ΔG_bind (corrected): {:.4} kcal/mol", dg_corr);
            println!();
            println!("Net charge        : {:+}", net_charge);
            println!("Aggregation risk  : {}  (VH run: {}, VL run: {})",
                if aggregation_risk { "HIGH ⚠" } else { "Low" },
                h_quality.max_hydro_run, l_quality.max_hydro_run,
            );

            let (clashes, rama_outliers) = if do_validate {
                let h_val = validate::ValidationReport::assess(&result.heavy);
                let l_val = validate::ValidationReport::assess(&result.light);
                println!();
                println!("Validation (VH)   : {} clashes, {} Ramachandran outliers / {} residues",
                    h_val.clashes, h_val.rama_outliers, h_val.n_residues);
                println!("Validation (VL)   : {} clashes, {} Ramachandran outliers / {} residues",
                    l_val.clashes, l_val.rama_outliers, l_val.n_residues);
                (Some(h_val.clashes + l_val.clashes), Some(h_val.rama_outliers + l_val.rama_outliers))
            } else {
                (None, None)
            };

            if !fasta_only {
                println!();
                println!("VH residues/atoms : {}/{}", result.heavy.n_residues(), result.heavy.n_atoms());
                println!("VL residues/atoms : {}/{}", result.light.n_residues(), result.light.n_atoms());
            }
            println!();

            candidate_reports.push(report::CandidateReport {
                rank: rank + 1,
                sequence: result.heavy_sequence.clone(),
                light_sequence: Some(result.light_sequence.clone()),
                energy: result.energy,
                entropy_penalty: combined_entropy,
                dg_corrected: dg_corr,
                net_charge,
                aggregation_risk,
                n_interface,
                clashes,
                rama_outliers,
            });
        }
        println!("Elapsed           : {:.2?}", elapsed);
        println!();

        write_reports(&candidate_reports, fasta_out.as_deref(), json_out.as_deref(), csv_out.as_deref())?;

        if !fasta_only {
            write_fv_outputs(&results, out_path.as_deref())?;
        }
    } else if use_allatom {
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

        let mut results = {
            #[cfg(feature = "gpu")]
            { diffusion::run_allatom(&antigen_aa, ab_n, population, iterations, top_n, gpu_ctx.as_ref()) }
            #[cfg(not(feature = "gpu"))]
            { diffusion::run_allatom(&antigen_aa, ab_n, population, iterations, top_n) }
        };

        for result in results.iter_mut() {
            if do_minimize {
                diffusion::minimize(&mut result.antibody, &[&antigen_aa], MINIMIZE_STEPS);
                result.energy = diffusion::allatom_energy(&antigen_aa, &result.antibody);
            }
            if do_maturation {
                let mutable: Vec<usize> = (0..result.antibody.n_residues())
                    .filter(|&r| filters::is_interface(r, &result.antibody, &antigen_aa))
                    .collect();
                result.energy = diffusion::affinity_maturation_allatom(&mut result.antibody, &antigen_aa, &mutable);
                result.sequence = result.antibody.sequence();
            }
        }
        let elapsed = t0.elapsed();

        eprintln!(
            "[binder] Done in {:.2?} | {} candidate(s) | best E_MM+solv = {:.3} kcal/mol",
            elapsed, results.len(), results[0].energy,
        );

        println!("=== De Novo Antibody Design Result (All-Atom AMBER) ===");
        println!("Antigen sequence  : {}", antigen_cg.sequence());
        println!();

        let mut candidate_reports = Vec::with_capacity(results.len());

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

            let (clashes, rama_outliers) = if do_validate {
                let val = validate::ValidationReport::assess(&result.antibody);
                println!();
                println!("Validation        : {} clashes, {} Ramachandran outliers / {} residues",
                    val.clashes, val.rama_outliers, val.n_residues);
                (Some(val.clashes), Some(val.rama_outliers))
            } else {
                (None, None)
            };

            if !fasta_only {
                println!();
                println!("Residues          : {}", result.antibody.n_residues());
                println!("Atoms             : {}", result.antibody.n_atoms());
            }
            println!();

            candidate_reports.push(report::CandidateReport {
                rank: rank + 1,
                sequence: result.sequence.clone(),
                light_sequence: None,
                energy: result.energy,
                entropy_penalty: quality.entropy_penalty,
                dg_corrected: dg_corr,
                net_charge: quality.net_charge,
                aggregation_risk: quality.aggregation_risk,
                n_interface: quality.n_interface,
                clashes,
                rama_outliers,
            });
        }
        println!("Elapsed           : {:.2?}", elapsed);
        println!();

        write_reports(&candidate_reports, fasta_out.as_deref(), json_out.as_deref(), csv_out.as_deref())?;

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

        let mut candidate_reports = Vec::with_capacity(results.len());

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

            let net_charge: i32 = result.antibody.amino_acid.iter()
                .map(|aa| aa.charge().round() as i32)
                .sum();
            let mut run = 0usize;
            let mut max_run = 0usize;
            for aa in &result.antibody.amino_acid {
                if aa.is_hydrophobic() {
                    run += 1;
                    if run > max_run { max_run = run; }
                } else {
                    run = 0;
                }
            }

            candidate_reports.push(report::CandidateReport {
                rank: rank + 1,
                sequence: result.sequence.clone(),
                light_sequence: None,
                energy: result.energy,
                entropy_penalty: 0.0,
                dg_corrected: result.energy,
                net_charge,
                aggregation_risk: max_run > 4,
                n_interface: 0,
                clashes: None,
                rama_outliers: None,
            });
        }
        println!("Elapsed           : {:.2?}", elapsed);
        println!();

        write_reports(&candidate_reports, fasta_out.as_deref(), json_out.as_deref(), csv_out.as_deref())?;

        if !fasta_only {
            write_cg_outputs(&results, out_path.as_deref())?;
        }
    }

    Ok(())
}

/// Write FASTA/JSON/CSV candidate reports to whichever paths were requested.
fn write_reports(
    reports: &[report::CandidateReport],
    fasta_path: Option<&str>,
    json_path: Option<&str>,
    csv_path: Option<&str>,
) -> Result<(), BinderError> {
    if let Some(path) = fasta_path {
        fs::write(path, report::write_fasta(reports)).map_err(BinderError::Io)?;
        eprintln!("[binder] FASTA report written to {path}");
    }
    if let Some(path) = json_path {
        fs::write(path, report::write_json(reports)).map_err(BinderError::Io)?;
        eprintln!("[binder] JSON report written to {path}");
    }
    if let Some(path) = csv_path {
        fs::write(path, report::write_csv(reports)).map_err(BinderError::Io)?;
        eprintln!("[binder] CSV report written to {path}");
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

/// Write one two-chain (heavy 'H' + light 'L') PDB file per Fv candidate.
fn write_fv_outputs(
    results: &[diffusion::FvResult],
    out_path: Option<&str>,
) -> Result<(), BinderError> {
    for (rank, result) in results.iter().enumerate() {
        let pdb_data = pdb::write_pdb_multi(&[(&result.heavy, 'H'), (&result.light, 'L')]);
        match out_path {
            Some(path) => {
                let path = candidate_path(path, rank, results.len());
                fs::write(&path, &pdb_data).map_err(BinderError::Io)?;
                eprintln!("[binder] Fv PDB written to {path}");
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
