/// Hand-rolled FASTA/JSON/CSV report writers (no serde — keeps this crate's
/// zero-extra-dependency philosophy intact).
use std::fmt::Write as FmtWrite;

/// Summary of one designed candidate, generic over both the legacy
/// single-chain path and the two-chain Fv path (light-chain fields are
/// `None` for single-chain candidates).
pub struct CandidateReport {
    pub rank: usize,
    pub sequence: String,
    pub light_sequence: Option<String>,
    pub energy: f32,
    pub entropy_penalty: f32,
    pub dg_corrected: f32,
    pub net_charge: i32,
    pub aggregation_risk: bool,
    pub n_interface: usize,
    pub clashes: Option<usize>,
    pub rama_outliers: Option<usize>,
}

/// Escape a string for embedding in a JSON string literal.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out
}

/// Quote a CSV field, doubling any embedded quotes (RFC 4180).
fn csv_field(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

pub fn write_fasta(reports: &[CandidateReport]) -> String {
    let mut out = String::new();
    for r in reports {
        if let Some(light) = &r.light_sequence {
            let _ = writeln!(out, ">candidate_{}_VH E={:.4}", r.rank, r.energy);
            let _ = writeln!(out, "{}", r.sequence);
            let _ = writeln!(out, ">candidate_{}_VL E={:.4}", r.rank, r.energy);
            let _ = writeln!(out, "{}", light);
        } else {
            let _ = writeln!(out, ">candidate_{} E={:.4}", r.rank, r.energy);
            let _ = writeln!(out, "{}", r.sequence);
        }
    }
    out
}

pub fn write_json(reports: &[CandidateReport]) -> String {
    let mut out = String::from("[\n");
    for (i, r) in reports.iter().enumerate() {
        let _ = write!(out, "  {{\n");
        let _ = write!(out, "    \"rank\": {},\n", r.rank);
        let _ = write!(out, "    \"sequence\": \"{}\",\n", json_escape(&r.sequence));
        match &r.light_sequence {
            Some(light) => { let _ = write!(out, "    \"light_sequence\": \"{}\",\n", json_escape(light)); }
            None => { let _ = write!(out, "    \"light_sequence\": null,\n"); }
        }
        let _ = write!(out, "    \"energy\": {:.6},\n", r.energy);
        let _ = write!(out, "    \"entropy_penalty\": {:.6},\n", r.entropy_penalty);
        let _ = write!(out, "    \"dg_corrected\": {:.6},\n", r.dg_corrected);
        let _ = write!(out, "    \"net_charge\": {},\n", r.net_charge);
        let _ = write!(out, "    \"aggregation_risk\": {},\n", r.aggregation_risk);
        let _ = write!(out, "    \"n_interface\": {},\n", r.n_interface);
        match r.clashes {
            Some(c) => { let _ = write!(out, "    \"clashes\": {},\n", c); }
            None => { let _ = write!(out, "    \"clashes\": null,\n"); }
        }
        match r.rama_outliers {
            Some(c) => { let _ = write!(out, "    \"rama_outliers\": {}\n", c); }
            None => { let _ = write!(out, "    \"rama_outliers\": null\n"); }
        }
        let sep = if i + 1 < reports.len() { "," } else { "" };
        let _ = write!(out, "  }}{}\n", sep);
    }
    out.push_str("]\n");
    out
}

pub fn write_csv(reports: &[CandidateReport]) -> String {
    let mut out = String::from(
        "rank,sequence,light_sequence,energy,entropy_penalty,dg_corrected,net_charge,aggregation_risk,n_interface,clashes,rama_outliers\n",
    );
    for r in reports {
        let light = r.light_sequence.as_deref().unwrap_or("");
        let clashes = r.clashes.map(|c| c.to_string()).unwrap_or_default();
        let rama = r.rama_outliers.map(|c| c.to_string()).unwrap_or_default();
        let _ = writeln!(
            out,
            "{},{},{},{:.6},{:.6},{:.6},{},{},{},{},{}",
            r.rank,
            csv_field(&r.sequence),
            csv_field(light),
            r.energy,
            r.entropy_penalty,
            r.dg_corrected,
            r.net_charge,
            r.aggregation_risk,
            r.n_interface,
            clashes,
            rama,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CandidateReport {
        CandidateReport {
            rank: 1,
            sequence: "ACDEFG".to_string(),
            light_sequence: None,
            energy: -12.5,
            entropy_penalty: 5.7,
            dg_corrected: -6.8,
            net_charge: 2,
            aggregation_risk: false,
            n_interface: 4,
            clashes: Some(0),
            rama_outliers: Some(1),
        }
    }

    #[test]
    fn fasta_contains_header_and_sequence() {
        let out = write_fasta(&[sample()]);
        assert!(out.contains(">candidate_1"));
        assert!(out.contains("ACDEFG"));
    }

    #[test]
    fn fasta_emits_two_records_for_fv_candidates() {
        let mut r = sample();
        r.light_sequence = Some("HIKLMN".to_string());
        let out = write_fasta(&[r]);
        assert!(out.contains("_VH"));
        assert!(out.contains("_VL"));
        assert!(out.contains("HIKLMN"));
    }

    #[test]
    fn json_is_well_formed_brackets() {
        let out = write_json(&[sample(), sample()]);
        assert!(out.starts_with('['));
        assert!(out.trim_end().ends_with(']'));
        assert_eq!(out.matches('{').count(), 2);
        assert_eq!(out.matches('}').count(), 2);
    }

    #[test]
    fn csv_has_header_plus_one_row_per_candidate() {
        let out = write_csv(&[sample(), sample()]);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 rows
        assert!(lines[0].starts_with("rank,"));
    }
}
