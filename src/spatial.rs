/// Lock-free Spatial Hash Grid for O(n) neighbour lookup.
///
/// The grid is built once (read-only thereafter) from a set of 3-D positions.
/// Queries iterate only over the 27 neighbouring cells of the probe point,
/// capping work at O(avg_density) per query regardless of total atom count.
///
/// "Lock-free" here means: after `build`, all queries are purely read-only and
/// require no synchronisation — they can run concurrently from any number of
/// Rayon worker threads without atomics or mutexes.
use std::collections::HashMap;

pub struct SpatialHashGrid {
    /// Cell → list of atom indices.
    cells: HashMap<CellKey, Vec<u32>>,
    inv_cell_size: f32,
}

type CellKey = (i32, i32, i32);

impl SpatialHashGrid {
    /// Create an empty grid with the given cell edge length (Å).
    /// `cell_size` should equal the non-bonded interaction cutoff.
    pub fn new(cell_size: f32) -> Self {
        Self {
            cells: HashMap::new(),
            inv_cell_size: 1.0 / cell_size,
        }
    }

    /// (Re)build the grid from coordinate arrays.  O(n).
    pub fn build(&mut self, x: &[f32], y: &[f32], z: &[f32]) {
        self.cells.clear();
        let n = x.len().min(y.len()).min(z.len());
        self.cells.reserve(n);
        for i in 0..n {
            let key = self.cell_of(x[i], y[i], z[i]);
            self.cells.entry(key).or_default().push(i as u32);
        }
    }

    /// Call `callback` for every atom whose cell overlaps the probe point.
    /// Inspects at most 27 cells (3×3×3 neighbourhood).
    #[inline]
    pub fn query_neighbors(&self, px: f32, py: f32, pz: f32, mut callback: impl FnMut(u32)) {
        let (cx, cy, cz) = self.cell_of(px, py, pz);
        for dx in -1_i32..=1 {
            for dy in -1_i32..=1 {
                for dz in -1_i32..=1 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    if let Some(atoms) = self.cells.get(&key) {
                        for &idx in atoms {
                            callback(idx);
                        }
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn cell_of(&self, x: f32, y: f32, z: f32) -> CellKey {
        (
            (x * self.inv_cell_size).floor() as i32,
            (y * self.inv_cell_size).floor() as i32,
            (z * self.inv_cell_size).floor() as i32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_finds_nearby_atom() {
        let mut grid = SpatialHashGrid::new(10.0);
        let xs = vec![0.0_f32, 50.0, 100.0];
        let ys = vec![0.0_f32, 50.0, 100.0];
        let zs = vec![0.0_f32, 50.0, 100.0];
        grid.build(&xs, &ys, &zs);

        let mut found = Vec::new();
        // Probe near atom 0
        grid.query_neighbors(1.0, 1.0, 1.0, |i| found.push(i));
        assert!(found.contains(&0));
        assert!(!found.contains(&1)); // atom 1 is far away
    }
}
