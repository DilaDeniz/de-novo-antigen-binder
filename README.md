# De Novo Antigen Binder

A generative de novo antibody design engine written in pure Rust — no PyTorch, no TensorFlow, no ONNX. Given a target antigen (PDB file or FASTA peptide), it outputs the 3-D coordinates and amino acid sequence of an antibody candidate optimised to bind it.

## How It Works

The engine combines three ideas:

1. **Physics-based scoring** — interaction energy between the antibody and antigen is computed from first principles: Lennard-Jones (Van der Waals), Coulomb electrostatics, and a hydrophobic bonus term.

2. **Reverse diffusion** — the antibody starts as random noise in 3-D space. At each step, the energy gradient (force field) pulls residues toward the antigen surface, mimicking a reverse diffusion trajectory from chaos to a structured binder.

3. **Monte Carlo sequence optimisation** — at every step, each residue has a chance to mutate to a different amino acid. Mutations are accepted or rejected via the Metropolis criterion, so the sequence co-evolves with the structure under a simulated-annealing temperature schedule.

A population of 64 independent candidates runs in parallel on all CPU cores (via Rayon). The champion — lowest binding energy — is returned.

## Energy Function

```
E_total = SUM over all (antigen_i, antibody_j) pairs:
            V_LJ(r) + V_Coulomb(r) + V_hydro(r)
```

| Term | Expression | Notes |
|------|------------|-------|
| Lennard-Jones | `4·ε·[(σ/r)^12 − (σ/r)^6]` | sterics + weak attraction |
| Coulomb | `332·q1·q2 / r` kcal/mol | 332 = k in AMBER units (Å, e⁻) |
| Hydrophobic | `−0.5 kcal/mol` per pair | hydrophobic–hydrophobic pairs within 6 Å |

LJ parameters use Lorentz-Berthelot mixing rules: `ε_ij = sqrt(ε_i · ε_j)`, `σ_ij = (σ_i + σ_j) / 2`. Cutoff checks use `r²` throughout to avoid unnecessary square roots.

## Architecture

```
src/
  atom.rs       Structure-of-Arrays ResidueCloud (SoA Vec<f32> per field)
  energy.rs     LJ + Coulomb + hydrophobic scoring and force computation
  spatial.rs    Lock-free SpatialHashGrid — O(n) neighbour lookup
  diffusion.rs  Population-based reverse diffusion + MC mutation engine
  pdb.rs        PDB reader / FASTA peptide builder / PDB writer
  error.rs      BinderError — no panics, full Result propagation
  main.rs       CLI entry point
```

### Key design decisions

- **Structure-of-Arrays layout** — `x`, `y`, `z`, `charge`, `epsilon`, `sigma` live in separate `Vec<f32>`. The inner force loop reads a contiguous slice of all antigen x-coordinates, letting LLVM auto-vectorise with SIMD loads.

- **SpatialHashGrid** — the antigen is hashed into 10 Å cells once before the simulation. Each force query inspects at most 27 neighbouring cells (O(avg\_density)) rather than all antigen residues. The grid is immutable after construction so all Rayon threads share it with zero synchronisation.

- **Lock-free parallelism** — each of the 64 candidates owns its own `ResidueCloud` and `SmallRng`. The Rayon `into_par_iter().map(...).reduce_with(...)` pipeline produces the best candidate with no `Mutex`, no `Arc`, no atomics.

- **No unsafe, no `panic!`** — all arithmetic uses `wrapping_*` or `clamp` where needed; all I/O returns `Result<_, BinderError>`.

## Installation

Requires Rust 1.75+ (stable).

```bash
git clone https://github.com/DilaDeniz/de-novo-antigen-binder
cd de-novo-antigen-binder
cargo build --release
```

The binary lands at `target/release/binder`.

## Usage

```bash
# Design a 20-residue antibody against a peptide antigen
binder --seq ACDEFGHIKLMNPQRSTVWY --length 20 --out antibody.pdb

# Design against a PDB file (CA atoms are used)
binder --pdb antigen.pdb --length 20 --out antibody.pdb

# Print PDB to stdout instead of a file
binder --seq MKTAYIAK --length 12
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--pdb PATH` | — | Antigen PDB file |
| `--seq STRING` | — | Antigen as one-letter FASTA string |
| `--length N` | same as antigen | Antibody length in residues |
| `--out PATH` | stdout | Output PDB file |

One of `--pdb` or `--seq` is required.

### Example output

```
=== De Novo Antibody Design Result ===
Antigen sequence  : MKTAYIAKQRQISFVK
Antibody sequence : RTHHRHAKVRGGQANN
Binding energy    : -518.50 kcal/mol
Residues          : 16
Elapsed           : 96ms
```

Followed by a valid PDB `ATOM` record block (chain B, CA only) suitable for loading in PyMOL, ChimeraX, or any structural viewer.

## Performance

| Antigen residues | Antibody residues | Candidates | Steps | Wall time |
|-----------------|-------------------|-----------|-------|-----------|
| 65 | 20 | 64 | 800 | ~96 ms |
| 11 | 8 | 64 | 800 | ~95 ms |

Tested on 4 cores, release build (`opt-level=3`, `lto="fat"`).

## Running Tests

```bash
cargo test
```

Four unit tests cover LJ repulsion at close range, Coulomb attraction between opposite charges, spatial hash neighbour lookup, and agreement between the brute-force and grid-accelerated force paths.

## Limitations

This is a **residue-level coarse-grained model** (one point per residue at the Cα position). It is a proof-of-concept physics engine, not a production-ready antibody design tool. For real drug discovery workflows, complement it with all-atom MD relaxation and experimental validation.

## License

Apache 2.0 — see [LICENSE](LICENSE).
