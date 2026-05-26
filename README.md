# De Novo Antigen Binder

A generative de novo antibody design engine in pure Rust.
No PyTorch. No TensorFlow. No ONNX. No bioinformatics library dependencies.

Given a target antigen (PDB file or FASTA string), it searches for an antibody
candidate sequence and 3-D structure that binds it, driven entirely by a
physics-based energy function and a reverse-diffusion + Monte Carlo algorithm.

---

## Features

- **Two representation levels** — fast Cα coarse-grained mode for rapid
  exploration, and a full all-atom AMBER99SB-ildn mode for production-quality
  results.
- **In-house AMBER99SB force field** — partial charges and LJ parameters for
  all 20 amino acids, hard-coded as `const` tables; no file I/O, no parser.
- **Dunbrack rotamer library** — top-5 backbone-independent rotamers per amino
  acid, Metropolis MC acceptance, NERF side-chain reconstruction.
- **Optional GPU acceleration** — WGSL compute shader evaluates AMBER
  LJ + Coulomb for 1 024 candidates simultaneously via wgpu.
- **Lock-free parallelism** — Rayon parallel iterator over independent
  candidates; zero `Mutex`, zero `Arc`, zero atomics.
- **SIMD-friendly layout** — Structure-of-Arrays (coordinates in separate
  `Vec<f32>`) lets LLVM auto-vectorise inner force loops.
- **O(n) spatial lookup** — `SpatialHashGrid` built once on the antigen, shared
  read-only across all Rayon threads; 27-cell 3×3×3 neighbour probe.

---

## Quick Start

```bash
# Requires Rust 1.75+
git clone https://github.com/DilaDeniz/de-novo-antigen-binder
cd de-novo-antigen-binder
cargo build --release
```

```bash
# Design a 12-residue antibody from a peptide antigen (Cα mode, ~35 ms)
./target/release/binder --seq MKTAYIAKQRQISFVK --length 12

# Full all-atom AMBER engine (CPU)
./target/release/binder --seq MKTAYIAKQRQISFVK --length 12 --allatom

# All-atom with GPU acceleration
./target/release/binder --pdb antigen.pdb --length 20 --allatom --gpu --out antibody.pdb
```

---

## How It Works

### 1 — Physics-based scoring

The binding energy between antibody and antigen atoms is:

```
E = Σ  ε_ij [(R_ij/r)¹² − 2(R_ij/r)⁶]          ← AMBER LJ (all-atom)
     + 332 · q_i · q_j / r                        ← Coulomb (kcal/mol, AMBER units)
     − 0.5  [hydrophobic pair, r < 6 Å]           ← hydrophobic bonus
```

The coarse-grained mode uses `4ε[(σ/r)¹² − (σ/r)⁶]` with residue-level
parameters; the all-atom mode uses the AMBER convention with per-atom
`r_min_half` and partial charges from AMBER99SB-ildn.

### 2 — Reverse diffusion

The antibody is initialised as random noise on a sphere around the antigen
centre of mass.  Each step:

1. Compute force `F = −∇E` via the spatial hash grid (all-atom or Cα).
2. Gradient step: `x += η·F` (clamped to ≤ 2 Å to prevent LJ explosion).
3. Langevin noise: `x += σ_T · ξ` where `σ_T = NOISE_BASE · √T`.
4. Harmonic restraint keeps residues within 20 Å of the antigen surface.

### 3 — Metropolis Monte Carlo

At each step every residue has a chance to:

- **Mutate** to a different amino acid — ΔE evaluated at Cα level; accepted
  with probability `min(1, exp(−ΔE/T))`.
- **Rotamer move** (all-atom mode) — a new chi-angle combination is drawn from
  the Dunbrack library; the side chain is rebuilt via NERF; accepted by full
  Metropolis criterion.

Temperature follows an exponential annealing schedule: T(t) = 3·(0.02/3)^(t/N).

### 4 — Parallel population

64 (CPU) or 1 024 (GPU) independent candidates run in parallel.  Each owns its
own `ResidueCloud`/`AtomProtein` and `SmallRng`; there is no shared mutable
state.  The champion (lowest interaction energy) is returned.

### GPU hybrid loop (optional)

When `--gpu` is passed:

1. **GPU phase** — 1 024 candidates × 200 gradient-only steps; every 10 steps
   the WGSL shader scores all candidates and the top-64 survive.
2. **CPU refinement** — 64 survivors × 600 Langevin + rotamer MC steps on Rayon.

---

## NERF Algorithm

Side-chain atoms are placed with the Natural Extension Reference Frame (NERF)
algorithm (Parsons et al., 2005).  Given reference atoms A-B-C and internal
coordinates (bond length, bond angle, dihedral), atom D is placed as:

```
bc = normalise(C − B)
n  = normalise((B − A) × bc)
m  = bc × n
D  = C + L · [−cos(θ)·bc + sin(θ)·cos(φ)·m + sin(θ)·sin(φ)·n]
```

One function, ~20 lines, no matrix library.

---

## Architecture

```
src/
  amber.rs      AMBER99SB atom types, ε/Rmin per type, per-residue partial
                charges and heavy-atom topologies (all in-house const tables)
  rotamer.rs    Dunbrack rotamer library + NERF place_atom + BondDef tables
  allatom.rs    AtomCloud (flat all-atom SoA) · AtomProtein (residue bookkeeping,
                chi angles, mutate_residue via Vec::splice)
  atom.rs       AminoAcid enum · ResidueCloud (Cα-only SoA, coarse-grained path)
  energy.rs     LJ + Coulomb + hydrophobic — brute-force + grid-accelerated
                versions for both Cα and all-atom representations
  spatial.rs    SpatialHashGrid — build O(n), query O(avg_density), Sync
  diffusion.rs  run (CG) · run_allatom (hybrid GPU/CPU) — both lock-free Rayon
  pdb.rs        PDB reader (CA-only) · FASTA builder · PDB writers (Cα + all-atom)
  gpu.rs        wgpu GpuContext + score_batch (feature = "gpu")
  error.rs      BinderError — Io / Parse / EmptyInput, no panics
  main.rs       CLI: --pdb, --seq, --length, --out, --allatom, --gpu, --no-gpu
shaders/
  energy.wgsl   WGSL compute shader — AMBER LJ + Coulomb, stride loop,
                shared-memory parallel reduction
```

---

## CLI Reference

```
USAGE
  binder --pdb <antigen.pdb> [OPTIONS]
  binder --seq <PEPTIDE>    [OPTIONS]

OPTIONS
  --pdb   PATH     Antigen input as PDB file (CA atoms used as Cα trace)
  --seq   STRING   Antigen as one-letter FASTA string
  --length N       Desired antibody length in residues [default: same as antigen]
  --out   PATH     Write output PDB to file [default: stdout]
  --allatom        Use full AMBER99SB all-atom engine
  --gpu            GPU acceleration — implies --allatom
  --no-gpu         Force CPU-only (overrides --gpu)
```

### Example output (Cα mode)

```
=== De Novo Antibody Design Result ===
Antigen sequence  : MKTAYIAKQRQISFVK
Antibody sequence : RTHHRHAKVRGGQANN
Binding energy    : -518.50 kcal/mol
Residues          : 16
Elapsed           : 35ms

ATOM      1  CA  ARG B   1     ...
```

### Example output (all-atom mode)

```
=== De Novo Antibody Design Result (All-Atom AMBER) ===
Antigen sequence  : MKTAYIAKQRQISFVK
Antibody sequence : YEFPAYGYIKLTRDAW
Binding energy    : -42.31 kcal/mol
Residues          : 16
Atoms             : 136
Elapsed           : 120ms

ATOM      1  N   TYR B   1     ...
ATOM      2  CA  TYR B   1     ...
```

> The all-atom energy is on a different numerical scale from the Cα mode
> because the AMBER LJ minimum is at `R_min` rather than `σ`.

---

## Performance

| Mode | Antigen | Antibody | Candidates | Steps | Time (4 cores) |
|------|---------|----------|-----------|-------|---------------|
| Cα coarse-grained | 8 res | 8 res | 64 | 800 | ~35 ms |
| Cα coarse-grained | 65 res | 20 res | 64 | 800 | ~96 ms |
| All-atom CPU | 8 res | 8 res | 64 | 600 | ~60 ms |

Release build: `opt-level = 3`, `lto = "fat"`, `codegen-units = 1`.

---

## Running Tests

```bash
cargo test
```

8 unit tests:

| Test | What it checks |
|------|---------------|
| `lj_repulsion_at_close_range` | Two Gly at 1 Å → positive LJ force |
| `opposite_charges_attract` | Lys (+1) and Asp (−1) at 3 Å → negative Coulomb force |
| `grid_forces_match_brute_force` | Grid and O(n²) forces agree to 1e-4 |
| `query_finds_nearby_atom` | SpatialHashGrid finds atom 0, not atom 1 |
| `nerf_places_atom_at_correct_bond_length` | NERF output has correct bond length |
| `rotamer_lib_non_empty_for_ile` | Ile rotamer library has 5 entries |
| `build_ala_residue` | ALA has exactly 5 heavy atoms |
| `build_gly_residue` | GLY has exactly 4 heavy atoms (no Cβ) |

---

## Limitations

- Residue-level coarse-grained mode uses one Cα point per residue — it is a
  fast structural sketch, not all-atom accuracy.
- All-atom mode uses ideal backbone geometry (no backbone torsion sampling);
  backbone flexibility requires additional MD relaxation.
- No explicit solvent or GBSA solvation — complement with MM-GBSA or explicit
  MD for binding free energy estimates.
- For real drug discovery workflows, validate experimentally and with all-atom
  MD relaxation (e.g., OpenMM, GROMACS).

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
