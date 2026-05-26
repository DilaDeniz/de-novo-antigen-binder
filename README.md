# De Novo Antigen Binder

A generative de novo antibody design engine written in pure Rust — no PyTorch, no TensorFlow, no ONNX, no bioinformatics library dependencies.  Given a target antigen (PDB file or FASTA peptide) it outputs the 3-D coordinates and amino acid sequence of an antibody candidate optimised to bind it.

Two operating modes:

| Mode | Representation | Force field | Population | Typical time |
|------|---------------|-------------|-----------|-------------|
| Coarse-grained (default) | 1 point/residue (Cα) | Residue-level LJ + Coulomb | 64 candidates | ~35 ms |
| All-atom (`--allatom`) | All heavy atoms (AMBER) | AMBER99SB-ildn LJ + Coulomb | 64 CPU / 1024 GPU | ~60 ms CPU |

## How It Works

### Coarse-grained path

1. **Physics-based scoring** — interaction energy from Lennard-Jones (Van der Waals), Coulomb electrostatics, and a hydrophobic bonus term.
2. **Reverse diffusion** — the antibody starts as random noise.  Each step the energy gradient pulls residues toward the antigen surface.
3. **Monte Carlo sequence optimisation** — each residue has a chance to mutate; mutations are accepted or rejected via the Metropolis criterion under a simulated-annealing temperature schedule.
4. **Rayon parallelism** — 64 independent candidates run on all CPU cores with zero shared mutable state.

### All-atom path (`--allatom`)

Everything above, plus:

1. **AMBER99SB-ildn force field** — all heavy atoms with per-atom partial charges and LJ parameters (ε, Rmin/2) from the AMBER99SB parameter set.  Written entirely in-house with no external force-field library.
2. **NERF side-chain reconstruction** — the Natural Extension Reference Frame algorithm places every side-chain atom from backbone + chi angles in O(atoms) time.
3. **Rotamer MC moves** — Metropolis proposals draw from the Dunbrack backbone-independent rotamer library (top-5 rotamers per amino acid, all 20 AAs).  Accepted/rejected by full all-atom ΔE.
4. **Optional GPU compute** (`--gpu`) — WGSL compute shader evaluates AMBER LJ + Coulomb over 1024 candidates simultaneously.  Each workgroup handles one candidate; threads stride over antigen atoms; parallel reduction returns energy per candidate.

## Energy Function

### Coarse-grained

```
E = SUM over all (antigen_i, antibody_j) pairs:
      4*eps_ij * [(sigma_ij/r)^12 - (sigma_ij/r)^6]   (LJ)
    + 332 * q_i * q_j / r                               (Coulomb, AMBER units)
    - 0.5  [if both hydrophobic and r < 6 A]            (hydrophobic bonus)
```

### All-atom (AMBER)

```
E = SUM over all (antigen_i, antibody_j) atom pairs:
      eps_ij * [(R_ij/r)^12 - 2*(R_ij/r)^6]   (AMBER LJ, R_ij = rmin_i + rmin_j)
    + 332 * q_i * q_j / r                       (Coulomb)
    - 0.5  [hydrophobic pairs within 6 A]
```

Lorentz-Berthelot mixing: `eps_ij = sqrt(eps_i * eps_j)`, `R_ij = r_min_half_i + r_min_half_j`.

10 Å cutoff.  All r² comparisons avoid unnecessary sqrt calls.

## Architecture

```
src/
  amber.rs      AMBER99SB atom types, LJ parameters, partial charges, residue topologies
  rotamer.rs    Dunbrack rotamer library + NERF atom placement algorithm
  allatom.rs    AtomCloud (flat SoA, all heavy atoms) + AtomProtein (residue bookkeeping)
  atom.rs       AminoAcid enum + ResidueCloud (Cα-only SoA, coarse-grained path)
  energy.rs     LJ + Coulomb + hydrophobic for both Cα and all-atom representations
  spatial.rs    Lock-free SpatialHashGrid — O(n) neighbour lookup, shared across Rayon threads
  diffusion.rs  Population-based reverse diffusion + MC (both CG and all-atom loops)
  pdb.rs        PDB reader / FASTA peptide builder / PDB writer (Cα and all-atom)
  gpu.rs        wgpu GPU context + compute pipeline (feature = "gpu")
  error.rs      BinderError — no panics, full Result propagation
  main.rs       CLI entry point
shaders/
  energy.wgsl   WGSL compute shader: AMBER LJ + Coulomb reduction
```

### Key design decisions

- **Structure-of-Arrays layout** — `x`, `y`, `z`, `charge`, `epsilon`, `r_min_half` live in separate `Vec<f32>`.  Inner force loops read contiguous slices, enabling LLVM auto-vectorisation with SIMD loads.
- **SpatialHashGrid** — the antigen is hashed into 10 Å cells once.  Each force query inspects at most 27 neighbouring cells.  The grid is immutable after construction so all Rayon threads share it with zero synchronisation.
- **Lock-free parallelism** — each of the 64+ candidates owns its own data and `SmallRng`.  Rayon `into_par_iter().map(...).reduce_with(...)` produces the best candidate with no `Mutex`, no `Arc`, no atomics.
- **NERF algorithm** — side-chain atoms are placed by the Natural Extension Reference Frame algorithm from internal coordinates (bond length, bond angle, dihedral).  One function, ~20 lines, no matrix libraries.
- **In-house force field** — AMBER99SB parameters are hard-coded `const` tables in `amber.rs`.  No file I/O, no parser, no external library.
- **GPU-optional design** — `gpu.rs` is compiled only with `--features gpu` (default).  `GpuContext::try_init()` returns `Option<Self>`; the diffusion engine falls back to CPU Rayon when it is `None`.

## Installation

Requires Rust 1.75+ (stable).

```bash
git clone https://github.com/DilaDeniz/de-novo-antigen-binder
cd de-novo-antigen-binder
cargo build --release          # CPU-only (no wgpu compiled in)
cargo build --release          # GPU + CPU (default, compiles wgpu)
```

The binary lands at `target/release/binder`.

## Usage

```bash
# Coarse-grained (fast, Cα-only)
binder --seq ACDEFGHIKLMNPQRSTVWY --length 20 --out antibody.pdb

# All-atom AMBER engine, CPU only
binder --seq MKTAYIAKQRQISFVK --length 20 --allatom --out antibody.pdb

# All-atom with GPU acceleration (requires GPU hardware)
binder --pdb antigen.pdb --length 20 --allatom --gpu --out antibody.pdb

# Design against a PDB antigen
binder --pdb antigen.pdb --length 20 --out antibody.pdb
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--pdb PATH` | — | Antigen PDB file |
| `--seq STRING` | — | Antigen as one-letter FASTA string |
| `--length N` | same as antigen | Antibody length in residues |
| `--out PATH` | stdout | Output PDB file |
| `--allatom` | off | Use full AMBER99SB all-atom engine |
| `--gpu` | off | Enable GPU acceleration (implies `--allatom`) |
| `--no-gpu` | — | Force CPU-only even if GPU available |

One of `--pdb` or `--seq` is required.

### Example output (coarse-grained)

```
=== De Novo Antibody Design Result ===
Antigen sequence  : MKTAYIAKQRQISFVK
Antibody sequence : RTHHRHAKVRGGQANN
Binding energy    : -518.50 kcal/mol
Residues          : 16
Elapsed           : 35ms
```

### Example output (all-atom)

```
=== De Novo Antibody Design Result (All-Atom AMBER) ===
Antigen sequence  : MKTAYIAKQRQISFVK
Antibody sequence : YEFPAYGYIKLTRDAW
Binding energy    : -42.31 kcal/mol
Residues          : 16
Atoms             : 136
Elapsed           : 120ms
```

The all-atom binding energy is on a different scale from the Cα-only one (AMBER LJ minimum is at Rmin, not sigma).

## Performance

| Mode | Antigen res | Antibody res | Candidates | Steps | Wall time |
|------|------------|--------------|-----------|-------|-----------|
| Coarse-grained | 8 | 8 | 64 | 800 | ~35 ms |
| Coarse-grained | 65 | 20 | 64 | 800 | ~96 ms |
| All-atom CPU | 8 | 8 | 64 | 600 | ~60 ms |

Tested on 4 cores, release build (`opt-level=3`, `lto="fat"`).

## Running Tests

```bash
cargo test
```

Eight unit tests: LJ repulsion, Coulomb attraction, spatial hash lookup, brute-force/grid force agreement, NERF bond length, rotamer library contents, and all-atom residue construction for ALA and GLY.

## License

Apache 2.0 — see [LICENSE](LICENSE).
