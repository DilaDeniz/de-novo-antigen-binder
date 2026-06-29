/// All-atom protein representation using Structure-of-Arrays layout.
///
/// `AtomCloud`   — flat SoA of coordinates and force-field parameters for every
///                 heavy atom.  This is the buffer handed to the energy engine
///                 and (later) the GPU shader.
///
/// `AtomProtein` — residue bookkeeping layered on top of an `AtomCloud`.
///                 Owns chi angles and exposes residue-level mutation/rotamer APIs.
use crate::amber::{topology, AtomType, AMBER_NB};
use crate::atom::AminoAcid;
use crate::rotamer::{build_side_chain, rotamers, Rotamer};

// ── AtomCloud ─────────────────────────────────────────────────────────────────

/// Flat all-atom SoA.  All Vecs have length `n_atoms`.
#[derive(Clone)]
pub struct AtomCloud {
    pub x:          Vec<f32>,
    pub y:          Vec<f32>,
    pub z:          Vec<f32>,
    pub charge:     Vec<f32>,
    /// Rmin/2 (AMBER convention).
    pub r_min_half: Vec<f32>,
    pub epsilon:    Vec<f32>,
    /// 1 = hydrophobic C, 0 otherwise.
    pub hydrophobic: Vec<u8>,
    /// AMBER atom type index (AtomType as u8) — used for EEF1 solvation lookup.
    pub atom_type:  Vec<u8>,
    /// Which residue (in AtomProtein::amino_acid) this atom belongs to.
    pub residue_idx: Vec<u32>,
}

impl AtomCloud {
    pub fn new() -> Self {
        Self {
            x: Vec::new(), y: Vec::new(), z: Vec::new(),
            charge: Vec::new(), r_min_half: Vec::new(), epsilon: Vec::new(),
            hydrophobic: Vec::new(), atom_type: Vec::new(), residue_idx: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            x: Vec::with_capacity(cap), y: Vec::with_capacity(cap),
            z: Vec::with_capacity(cap), charge: Vec::with_capacity(cap),
            r_min_half: Vec::with_capacity(cap), epsilon: Vec::with_capacity(cap),
            hydrophobic: Vec::with_capacity(cap), atom_type: Vec::with_capacity(cap),
            residue_idx: Vec::with_capacity(cap),
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize { self.x.len() }

    #[inline(always)]
    pub fn is_empty(&self) -> bool { self.x.is_empty() }

    pub fn push_atom(&mut self, pos: [f32; 3], charge: f32, r_min_half: f32,
                     epsilon: f32, hydrophobic: u8, atype: AtomType, res_idx: u32) {
        self.x.push(pos[0]);
        self.y.push(pos[1]);
        self.z.push(pos[2]);
        self.charge.push(charge);
        self.r_min_half.push(r_min_half);
        self.epsilon.push(epsilon);
        self.hydrophobic.push(hydrophobic);
        self.atom_type.push(atype as u8);
        self.residue_idx.push(res_idx);
    }

    pub fn center_of_mass(&self) -> [f32; 3] {
        let n = self.len() as f32;
        if n == 0.0 { return [0.0; 3]; }
        [
            self.x.iter().sum::<f32>() / n,
            self.y.iter().sum::<f32>() / n,
            self.z.iter().sum::<f32>() / n,
        ]
    }
}

impl Default for AtomCloud {
    fn default() -> Self { Self::new() }
}

// ── AtomProtein ───────────────────────────────────────────────────────────────

/// All-atom protein with residue bookkeeping.
///
/// `atoms.x[ranges[i].clone()]` gives the x-coordinates for residue i.
#[derive(Clone)]
pub struct AtomProtein {
    pub atoms: AtomCloud,
    /// Byte ranges into `atoms` per residue: atoms[ranges[i]..ranges[i+1]].
    pub res_start: Vec<u32>,   // length = n_residues + 1  (sentinel at end)
    pub amino_acid: Vec<AminoAcid>,
    /// Cα index within the protein's flat atom array for each residue.
    pub ca_atom_idx: Vec<u32>,
    /// Current chi angles in degrees for each residue (chi1..chi4, unused = 0.0).
    pub chi: Vec<[f32; 4]>,
}

impl AtomProtein {
    pub fn new() -> Self {
        Self {
            atoms: AtomCloud::new(),
            res_start: vec![0],
            amino_acid: Vec::new(),
            ca_atom_idx: Vec::new(),
            chi: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn n_residues(&self) -> usize { self.amino_acid.len() }

    #[inline(always)]
    pub fn n_atoms(&self) -> usize { self.atoms.len() }

    /// Atom range for residue `r`.
    #[inline(always)]
    pub fn atom_range(&self, r: usize) -> std::ops::Range<usize> {
        self.res_start[r] as usize .. self.res_start[r + 1] as usize
    }

    /// Cα position for residue `r`.
    #[inline(always)]
    pub fn ca_pos(&self, r: usize) -> [f32; 3] {
        let i = self.ca_atom_idx[r] as usize;
        [self.atoms.x[i], self.atoms.y[i], self.atoms.z[i]]
    }

    /// One-letter sequence.
    pub fn sequence(&self) -> String {
        self.amino_acid.iter().map(|a| a.to_char()).collect()
    }

    /// SG (sulfhydryl) atom position for residue `r`, if it is a Cys.
    pub fn sg_pos(&self, r: usize) -> Option<[f32; 3]> {
        if self.amino_acid[r] != AminoAcid::Cys {
            return None;
        }
        let topo = topology(AminoAcid::Cys);
        for (k, idx) in self.atom_range(r).enumerate() {
            if k < topo.atoms.len() && topo.atoms[k].name.trim() == "SG" {
                return Some([self.atoms.x[idx], self.atoms.y[idx], self.atoms.z[idx]]);
            }
        }
        None
    }

    /// Disulfide bond energy: a Gaussian attractive well centred on the ideal
    /// S–S bond length (2.05 Å) between every pair of Cys SG atoms. Rewards the
    /// MC search for discovering Cys pairs positioned to form a disulfide bridge.
    pub fn disulfide_energy(&self) -> f32 {
        const D0: f32 = 2.05; // Å, ideal S–S bond length
        const SIGMA: f32 = 0.4;
        const DEPTH: f32 = 4.0; // kcal/mol stabilization for a formed disulfide
        const CUTOFF: f32 = 4.0; // Å, beyond this the well is negligible

        let sg_positions: Vec<[f32; 3]> = (0..self.n_residues())
            .filter_map(|r| self.sg_pos(r))
            .collect();

        let mut e = 0.0_f32;
        for i in 0..sg_positions.len() {
            for j in (i + 1)..sg_positions.len() {
                let dx = sg_positions[i][0] - sg_positions[j][0];
                let dy = sg_positions[i][1] - sg_positions[j][1];
                let dz = sg_positions[i][2] - sg_positions[j][2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                if d > CUTOFF {
                    continue;
                }
                let z = (d - D0) / SIGMA;
                e -= DEPTH * (-0.5 * z * z).exp();
            }
        }
        e
    }

    /// Center of mass of Cα atoms.
    pub fn ca_center_of_mass(&self) -> [f32; 3] {
        let n = self.n_residues();
        if n == 0 { return [0.0; 3]; }
        let mut cx = 0.0_f32;
        let mut cy = 0.0_f32;
        let mut cz = 0.0_f32;
        for r in 0..n {
            let p = self.ca_pos(r);
            cx += p[0]; cy += p[1]; cz += p[2];
        }
        let nf = n as f32;
        [cx / nf, cy / nf, cz / nf]
    }

    /// Append a new residue, placing all heavy atoms given a backbone frame.
    ///
    /// `backbone` = [N, CA, C, O, CB] in Cartesian coords (5 atoms; GLY uses 4).
    /// `chi_deg`  = initial chi angles.
    pub fn push_residue(&mut self, aa: AminoAcid, backbone: &[[f32; 3]],
                        chi_deg: [f32; 4]) {
        let topo = topology(aa);
        let res_idx = self.n_residues() as u32;
        let atom_start = self.atoms.len() as u32;

        // Push backbone atoms
        let n_backbone = backbone.len().min(topo.atoms.len());
        for k in 0..n_backbone {
            let rat = &topo.atoms[k];
            let nb = &AMBER_NB[rat.atom_type as usize];
            self.atoms.push_atom(
                backbone[k], rat.charge, nb.r_min_half, nb.epsilon,
                rat.hydrophobic, rat.atom_type, res_idx,
            );
        }

        // Record CA index (always index 1 in backbone per RESIDUE_TOPOLOGY)
        self.ca_atom_idx.push(atom_start + topo.ca_idx as u32);

        // Build and push side-chain atoms
        let mut sc_positions: Vec<[f32; 3]> = Vec::new();
        build_side_chain(aa, backbone, &chi_deg, &mut sc_positions);

        let n_sc_in_topo = topo.atoms.len().saturating_sub(n_backbone);
        for (k, pos) in sc_positions.iter().enumerate().take(n_sc_in_topo) {
            let atom_topo_idx = n_backbone + k;
            if atom_topo_idx >= topo.atoms.len() { break; }
            let rat = &topo.atoms[atom_topo_idx];
            let nb = &AMBER_NB[rat.atom_type as usize];
            self.atoms.push_atom(
                *pos, rat.charge, nb.r_min_half, nb.epsilon,
                rat.hydrophobic, rat.atom_type, res_idx,
            );
        }

        self.amino_acid.push(aa);
        self.chi.push(chi_deg);
        self.res_start.push(self.atoms.len() as u32);
    }

    /// Recompute side-chain positions for residue `r` using current chi angles
    /// without changing the backbone.  Overwrites atom positions in-place.
    pub fn rebuild_side_chain(&mut self, r: usize) {
        let aa = self.amino_acid[r];
        let range = self.atom_range(r);
        let topo = topology(aa);
        let n_backbone = 4 + (aa != AminoAcid::Gly) as usize; // 4 for Gly, 5 for rest

        // Collect current backbone positions
        let backbone: Vec<[f32; 3]> = (0..n_backbone)
            .map(|k| {
                let idx = range.start + k;
                [self.atoms.x[idx], self.atoms.y[idx], self.atoms.z[idx]]
            })
            .collect();

        // Re-build side-chain
        let mut sc_positions: Vec<[f32; 3]> = Vec::new();
        build_side_chain(aa, &backbone, &self.chi[r], &mut sc_positions);

        let n_sc_in_topo = topo.atoms.len().saturating_sub(n_backbone);
        for (k, pos) in sc_positions.iter().enumerate().take(n_sc_in_topo) {
            let atom_idx = range.start + n_backbone + k;
            if atom_idx >= range.end { break; }
            self.atoms.x[atom_idx] = pos[0];
            self.atoms.y[atom_idx] = pos[1];
            self.atoms.z[atom_idx] = pos[2];
        }
    }

    /// Sample a rotamer for residue `r` and rebuild.  Returns the rotamer used.
    pub fn apply_rotamer(&mut self, r: usize, rot: &Rotamer) {
        self.chi[r] = rot.chi;
        self.rebuild_side_chain(r);
    }

    /// Translate all atoms of residue `r` by `delta`.
    pub fn translate_residue(&mut self, r: usize, delta: [f32; 3]) {
        let range = self.atom_range(r);
        for i in range {
            self.atoms.x[i] += delta[0];
            self.atoms.y[i] += delta[1];
            self.atoms.z[i] += delta[2];
        }
    }

    /// Change the backbone Cα position for residue `r` (moves all atoms by the delta).
    pub fn set_ca_pos(&mut self, r: usize, new_pos: [f32; 3]) {
        let old = self.ca_pos(r);
        let delta = [new_pos[0] - old[0], new_pos[1] - old[1], new_pos[2] - old[2]];
        self.translate_residue(r, delta);
    }

    /// Change the amino acid of residue `r`, keeping the Cα position.
    ///
    /// Uses Vec::splice to atomically swap the old atom slice for the new one,
    /// then updates all bookkeeping in a single pass — no off-by-one possible.
    pub fn mutate_residue(&mut self, r: usize, new_aa: AminoAcid, chi_deg: [f32; 4]) {
        let old_aa = self.amino_acid[r];

        // If same type just update chi + rebuild side-chain
        if old_aa == new_aa {
            self.chi[r] = chi_deg;
            self.rebuild_side_chain(r);
            return;
        }

        let ca         = self.ca_pos(r);
        let old_range  = self.atom_range(r);
        let old_count  = old_range.len();
        let new_topo   = topology(new_aa);
        let n_new      = new_topo.atoms.len();
        let res_idx    = r as u32;

        // Build backbone positions around existing Cα
        let backbone = build_ideal_backbone_around_ca(ca, new_aa);
        let n_backbone = backbone.len();

        // Build side-chain positions
        let mut sc_positions: Vec<[f32; 3]> = Vec::new();
        build_side_chain(new_aa, &backbone, &chi_deg, &mut sc_positions);

        // Collect the new atom data in topology order
        let mut nx  = Vec::with_capacity(n_new);
        let mut ny  = Vec::with_capacity(n_new);
        let mut nz  = Vec::with_capacity(n_new);
        let mut nq  = Vec::with_capacity(n_new);
        let mut nrm = Vec::with_capacity(n_new);
        let mut ne  = Vec::with_capacity(n_new);
        let mut nh  = Vec::with_capacity(n_new);
        let mut nat = Vec::with_capacity(n_new);
        let mut nri = Vec::with_capacity(n_new);

        for k in 0..n_new {
            let rat = &new_topo.atoms[k];
            let nb  = &AMBER_NB[rat.atom_type as usize];
            let pos = if k < n_backbone {
                backbone[k]
            } else {
                let sc_k = k - n_backbone;
                if sc_k < sc_positions.len() { sc_positions[sc_k] } else { ca }
            };
            nx.push(pos[0]);  ny.push(pos[1]);  nz.push(pos[2]);
            nq.push(rat.charge);
            nrm.push(nb.r_min_half);
            ne.push(nb.epsilon);
            nh.push(rat.hydrophobic);
            nat.push(rat.atom_type as u8);
            nri.push(res_idx);
        }

        // Atomically replace old atom range with new atoms via splice
        let start = old_range.start;
        let end   = old_range.end;
        self.atoms.x.splice(start..end, nx);
        self.atoms.y.splice(start..end, ny);
        self.atoms.z.splice(start..end, nz);
        self.atoms.charge.splice(start..end, nq);
        self.atoms.r_min_half.splice(start..end, nrm);
        self.atoms.epsilon.splice(start..end, ne);
        self.atoms.hydrophobic.splice(start..end, nh);
        self.atoms.atom_type.splice(start..end, nat);
        self.atoms.residue_idx.splice(start..end, nri);

        // Update bookkeeping
        self.amino_acid[r] = new_aa;
        self.chi[r]        = chi_deg;

        let delta = n_new as i64 - old_count as i64;

        // Shift res_start for residues r+1 through the sentinel (inclusive)
        for i in (r + 1)..self.res_start.len() {
            self.res_start[i] = (self.res_start[i] as i64 + delta) as u32;
        }

        // Update CA atom index for residue r
        self.ca_atom_idx[r] = start as u32 + new_topo.ca_idx as u32;

        // Shift CA atom indices for all later residues
        for i in (r + 1)..self.n_residues() {
            self.ca_atom_idx[i] = (self.ca_atom_idx[i] as i64 + delta) as u32;
        }
    }
}

impl Default for AtomProtein {
    fn default() -> Self { Self::new() }
}

// ── Backbone torsion perturbation ─────────────────────────────────────────────

impl AtomProtein {
    /// Rotate all atoms after the N→CA bond by `delta_rad` and propagate the
    /// rotation to every subsequent residue.  Correctly models phi torsion: the
    /// entire chain from C_r onward pivots around the N_r–CA_r axis.
    pub fn perturb_phi(&mut self, r: usize, delta_rad: f32) {
        let range = self.atom_range(r);
        if range.len() < 3 { return; }
        let n_idx  = range.start;
        let ca_idx = range.start + 1;
        let n  = self.atom_xyz(n_idx);
        let ca = self.atom_xyz(ca_idx);
        let axis = norm3(sub3(ca, n));
        // Rotate atoms after CA within residue r
        for i in (ca_idx + 1)..range.end {
            let p = self.atom_xyz(i);
            let rp = rodrigues(p, ca, axis, delta_rad);
            self.atoms.x[i] = rp[0];
            self.atoms.y[i] = rp[1];
            self.atoms.z[i] = rp[2];
        }
        // Propagate: every subsequent residue pivots around the same axis
        for next_r in (r + 1)..self.n_residues() {
            let nr = self.atom_range(next_r);
            for i in nr {
                let p = self.atom_xyz(i);
                let rp = rodrigues(p, ca, axis, delta_rad);
                self.atoms.x[i] = rp[0];
                self.atoms.y[i] = rp[1];
                self.atoms.z[i] = rp[2];
            }
        }
    }

    /// Rotate the carbonyl O and all subsequent residues around the CA→C bond
    /// axis by `delta_rad`.  Correctly models psi torsion: O_r and the entire
    /// chain from N_{r+1} onward pivot around CA_r–C_r.
    pub fn perturb_psi(&mut self, r: usize, delta_rad: f32) {
        let range = self.atom_range(r);
        if range.len() < 4 { return; }
        let ca_idx = range.start + 1;
        let c_idx  = range.start + 2;
        let o_idx  = range.start + 3;
        let ca = self.atom_xyz(ca_idx);
        let c  = self.atom_xyz(c_idx);
        let axis = norm3(sub3(c, ca));
        // Rotate O of current residue
        let p = self.atom_xyz(o_idx);
        let rp = rodrigues(p, c, axis, delta_rad);
        self.atoms.x[o_idx] = rp[0];
        self.atoms.y[o_idx] = rp[1];
        self.atoms.z[o_idx] = rp[2];
        // Propagate: every subsequent residue pivots around the same axis
        for next_r in (r + 1)..self.n_residues() {
            let nr = self.atom_range(next_r);
            for i in nr {
                let p = self.atom_xyz(i);
                let rp = rodrigues(p, c, axis, delta_rad);
                self.atoms.x[i] = rp[0];
                self.atoms.y[i] = rp[1];
                self.atoms.z[i] = rp[2];
            }
        }
    }

    #[inline(always)]
    fn atom_xyz(&self, i: usize) -> [f32; 3] {
        [self.atoms.x[i], self.atoms.y[i], self.atoms.z[i]]
    }
}

// ── Rodrigues rotation ────────────────────────────────────────────────────────

/// Rotate `point` around `axis` (unit vector) through `origin` by `angle` radians.
/// Uses Rodrigues' rotation formula.
pub fn rodrigues(point: [f32; 3], origin: [f32; 3], axis: [f32; 3], angle: f32) -> [f32; 3] {
    let v    = sub3(point, origin);
    let ca   = angle.cos();
    let sa   = angle.sin();
    let dot  = v[0]*axis[0] + v[1]*axis[1] + v[2]*axis[2];
    let cx   = cross3(axis, v);
    let rotd = [
        v[0]*ca + cx[0]*sa + axis[0]*dot*(1.0 - ca),
        v[1]*ca + cx[1]*sa + axis[1]*dot*(1.0 - ca),
        v[2]*ca + cx[2]*sa + axis[2]*dot*(1.0 - ca),
    ];
    add3(rotd, origin)
}

#[inline(always)] pub(crate) fn sub3(a:[f32;3], b:[f32;3]) -> [f32;3] { [a[0]-b[0],a[1]-b[1],a[2]-b[2]] }
#[inline(always)] fn add3(a:[f32;3], b:[f32;3]) -> [f32;3] { [a[0]+b[0],a[1]+b[1],a[2]+b[2]] }
#[inline(always)] pub(crate) fn cross3(a:[f32;3], b:[f32;3]) -> [f32;3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
#[inline(always)] pub(crate) fn norm3(v:[f32;3]) -> [f32;3] {
    let l = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt().max(1e-8);
    [v[0]/l, v[1]/l, v[2]/l]
}

// ── Backbone geometry helpers ─────────────────────────────────────────────────

/// Ideal backbone geometry constants (AMBER99SB standard bond lengths/angles).
pub const N_CA_BOND:  f32 = 1.46;  // Å
pub const CA_C_BOND:  f32 = 1.52;
pub const C_O_BOND:   f32 = 1.23;
pub const CA_CB_BOND: f32 = 1.52;

/// Build approximate N, CA, C, O, CB positions given a Cα coordinate.
/// The returned geometry is ideal (standard AMBER bond lengths) but the
/// orientation is arbitrary — use this only as an initial placement for MC.
fn build_ideal_backbone_around_ca(ca: [f32; 3], aa: AminoAcid) -> Vec<[f32; 3]> {
    // Place N along -x from CA
    let n   = [ca[0] - N_CA_BOND, ca[1], ca[2]];
    // Place C along +x from CA
    let c   = [ca[0] + CA_C_BOND, ca[1], ca[2]];
    // Place O above C
    let o   = [c[0] + 0.5, c[1] + C_O_BOND * 0.9, c[2]];
    // Place CB below CA (for non-GLY)
    let cb  = [ca[0], ca[1] - CA_CB_BOND, ca[2]];

    if aa == AminoAcid::Gly {
        vec![n, ca, c, o]
    } else {
        vec![n, ca, c, o, cb]
    }
}

/// Build an AtomProtein from an extended-chain Cα trace (like ResidueCloud).
///
/// Each residue gets ideal backbone geometry around its Cα, with the most
/// probable rotamer for its amino acid.
pub fn protein_from_ca_trace(
    x: &[f32], y: &[f32], z: &[f32],
    aa_seq: &[AminoAcid],
) -> AtomProtein {
    let n = aa_seq.len();
    let mut prot = AtomProtein::new();
    for i in 0..n {
        let ca = [x[i], y[i], z[i]];
        let aa = aa_seq[i];
        let backbone = build_ideal_backbone_around_ca(ca, aa);
        // Use first (most probable) rotamer if available
        let chi = {
            let rots = rotamers(aa);
            if rots.is_empty() { [0.0f32; 4] } else { rots[0].chi }
        };
        prot.push_residue(aa, &backbone, chi);
    }
    prot
}

// ── Interaction energy for AtomCloud ─────────────────────────────────────────

/// AMBER LJ energy: V = ε[(R_ij/r)^12 − 2(R_ij/r)^6]
/// where R_ij = r_min_half_i + r_min_half_j.
#[inline(always)]
pub fn amber_lj(eps_ij: f32, r_min_ij: f32, r_sq: f32) -> f32 {
    let r2 = r_min_ij * r_min_ij / r_sq;
    let r6 = r2 * r2 * r2;
    let r12 = r6 * r6;
    eps_ij * (r12 - 2.0 * r6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_ala_residue() {
        let mut prot = AtomProtein::new();
        let ca = [10.0_f32, 0.0, 0.0];
        let backbone = build_ideal_backbone_around_ca(ca, AminoAcid::Ala);
        prot.push_residue(AminoAcid::Ala, &backbone, [0.0; 4]);
        assert_eq!(prot.n_residues(), 1);
        // ALA has 5 heavy atoms: N, CA, C, O, CB
        assert_eq!(prot.n_atoms(), 5);
    }

    #[test]
    fn build_gly_residue() {
        let mut prot = AtomProtein::new();
        let ca = [0.0_f32, 0.0, 0.0];
        let backbone = build_ideal_backbone_around_ca(ca, AminoAcid::Gly);
        prot.push_residue(AminoAcid::Gly, &backbone, [0.0; 4]);
        // GLY has 4 heavy atoms: N, CA, C, O
        assert_eq!(prot.n_atoms(), 4);
    }

    /// Only Cys residues expose an SG (sulfhydryl) atom.
    #[test]
    fn sg_pos_only_for_cysteine() {
        let prot = protein_from_ca_trace(&[0.0], &[0.0], &[0.0], &[AminoAcid::Cys]);
        assert!(prot.sg_pos(0).is_some());

        let prot = protein_from_ca_trace(&[0.0], &[0.0], &[0.0], &[AminoAcid::Gly]);
        assert!(prot.sg_pos(0).is_none());
    }

    /// A single Cys (no partner) has no pair to bond — energy is zero.
    #[test]
    fn disulfide_energy_zero_without_a_partner() {
        let prot = protein_from_ca_trace(&[0.0], &[0.0], &[0.0], &[AminoAcid::Cys]);
        assert_eq!(prot.disulfide_energy(), 0.0);
    }

    /// Two Cys residues placed far apart (CA trace 50 Å apart) have SG atoms
    /// far beyond the 4 Å cutoff — energy must be exactly zero.
    #[test]
    fn disulfide_energy_zero_beyond_cutoff() {
        let prot = protein_from_ca_trace(
            &[0.0, 50.0], &[0.0, 0.0], &[0.0, 0.0],
            &[AminoAcid::Cys, AminoAcid::Cys],
        );
        assert_eq!(prot.disulfide_energy(), 0.0);
    }

    /// `disulfide_energy` must equal the direct Gaussian-well sum over all
    /// SG pairs — i.e. it must not silently diverge from its own definition.
    #[test]
    fn disulfide_energy_matches_pairwise_gaussian_sum() {
        let prot = protein_from_ca_trace(
            &[0.0, 3.8, 7.6], &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0],
            &[AminoAcid::Cys, AminoAcid::Cys, AminoAcid::Cys],
        );

        const D0: f32 = 2.05;
        const SIGMA: f32 = 0.4;
        const DEPTH: f32 = 4.0;
        const CUTOFF: f32 = 4.0;

        let sg: Vec<[f32; 3]> = (0..3).filter_map(|r| prot.sg_pos(r)).collect();
        let mut expected = 0.0_f32;
        for i in 0..sg.len() {
            for j in (i + 1)..sg.len() {
                let dx = sg[i][0] - sg[j][0];
                let dy = sg[i][1] - sg[j][1];
                let dz = sg[i][2] - sg[j][2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                if d > CUTOFF {
                    continue;
                }
                let z = (d - D0) / SIGMA;
                expected -= DEPTH * (-0.5 * z * z).exp();
            }
        }

        assert!((prot.disulfide_energy() - expected).abs() < 1e-5);
    }
}
