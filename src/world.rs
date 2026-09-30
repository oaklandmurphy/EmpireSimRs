//! The simulated world: terrain, nations, and the per-step dynamics.
//!
//! All grids are row-major with `idx = y * width + x`.

use crate::terrain::PlateTectonics;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet, VecDeque};

const UNREACHABLE: u32 = u32::MAX;

/// All-pairs travel cost between land cells.
///
/// Only cells that were land when the matrix was built are stored, and costs
/// are `u32`, so a 100x100 world with ~50% land takes ~100 MB instead of the
/// 800 MB a dense `f64` `[w][h][w][h]` array needs.
///
/// Row `t` holds the cost of travelling *to* `t` from every land cell, so the
/// fitness sum for a cell reads one contiguous row.
pub struct DistanceMatrix {
    /// Grid cell -> compact land index, or `u32::MAX` for water.
    pub land_index: Vec<u32>,
    /// Compact land index -> grid cell.
    pub land_cells: Vec<u32>,
    dist: Vec<u32>,
}

impl DistanceMatrix {
    /// Cost to walk from `from` to `to` (grid indices), entering each cell at
    /// its terrain cost. `None` if either is not land or no path exists.
    pub fn get(&self, from: usize, to: usize) -> Option<u32> {
        let (s, t) = (self.land_index[from], self.land_index[to]);
        if s == UNREACHABLE || t == UNREACHABLE {
            return None;
        }
        let d = self.dist[t as usize * self.land_cells.len() + s as usize];
        (d != UNREACHABLE).then_some(d)
    }

    /// Dijkstra over the reversed grid graph, giving the cost from every cell to `target`.
    fn build_row(
        target: usize,
        width: usize,
        height: usize,
        cost: &[u32],
        walkable: &[bool],
    ) -> Vec<u32> {
        let mut d = vec![UNREACHABLE; width * height];
        let mut heap = BinaryHeap::new();
        d[target] = 0;
        heap.push(Reverse((0u32, target)));
        while let Some(Reverse((dv, v))) = heap.pop() {
            if dv > d[v] {
                continue;
            }
            let (x, y) = (v % width, v / width);
            let nd = dv + cost[v];
            let mut relax = |u: usize| {
                if walkable[u] && nd < d[u] {
                    d[u] = nd;
                    heap.push(Reverse((nd, u)));
                }
            };
            if x > 0 {
                relax(v - 1);
            }
            if x + 1 < width {
                relax(v + 1);
            }
            if y > 0 {
                relax(v - width);
            }
            if y + 1 < height {
                relax(v + width);
            }
        }
        d
    }

    /// Build the matrix for the cells where `terrain >= sea_level`; water is impassable.
    pub fn build(width: usize, height: usize, terrain: &[i32], sea_level: f64) -> Self {
        let walkable: Vec<bool> = terrain.iter().map(|&t| t as f64 >= sea_level).collect();
        let cost: Vec<u32> = terrain.iter().map(|&t| t.max(0) as u32).collect();

        let mut land_index = vec![UNREACHABLE; width * height];
        let mut land_cells = Vec::new();
        for (i, &ok) in walkable.iter().enumerate() {
            if ok {
                land_index[i] = land_cells.len() as u32;
                land_cells.push(i as u32);
            }
        }

        let l = land_cells.len();
        let mut dist = vec![UNREACHABLE; l * l];
        if l > 0 {
            dist.par_chunks_mut(l).enumerate().for_each(|(t, row)| {
                let d = Self::build_row(land_cells[t] as usize, width, height, &cost, &walkable);
                for (s, &cell) in land_cells.iter().enumerate() {
                    row[s] = d[cell as usize];
                }
            });
        }
        DistanceMatrix {
            land_index,
            land_cells,
            dist,
        }
    }
}

pub struct World {
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    pub(crate) rng: StdRng,

    pub influence: Vec<f64>,
    pub solidarity: Vec<f64>,
    pub terrain: Vec<i32>,
    pub fitness: Vec<f64>,
    pub nation: Vec<u32>,
    pub land_mask: Vec<bool>,

    pub tectonic_plates: Vec<i32>,
    /// `(dy, dx)` per cell.
    pub tectonic_plate_drift: Vec<[f64; 2]>,

    pub terrain_variability: f64,
    pub econ_growth_rate: f64,
    pub num_nations: u32,
    /// Indexed by nation id; id 0 is "unassigned".
    pub nation_colors: Vec<[u8; 3]>,

    /// Higher means harder to conquer.
    pub conquest_difficulty: f64,
    /// Chance to copy solidarity from the most influential neighbour.
    pub solidarity_spread_rate: f64,
    /// Higher means harder to rebel.
    pub nation_stability: f64,
    /// Chance to copy solidarity from the conquering nation.
    pub conquest_assimilation_rate: f64,
    pub sea_level: f64,

    /// The sea level the distance matrix was built with.
    pub distance_sea_level: f64,
    pub distance: DistanceMatrix,
}

impl World {
    pub fn new(width: usize, height: usize, seed: u64) -> Self {
        let terrain_variability = 20.0;
        let plates = PlateTectonics::new(width, height, 8, seed, 0.55);

        let min = plates
            .elevation
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = plates
            .elevation
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let range = if max > min { max - min } else { 1.0 };
        let terrain: Vec<i32> = plates
            .elevation
            .iter()
            .map(|e| ((e - min) / range * (terrain_variability - 1.0)) as i32 + 1)
            .collect();
        let sea_level = plates.sea_level * terrain_variability;

        Self::from_parts(
            width,
            height,
            seed,
            terrain,
            sea_level,
            plates.plate_id,
            plates.plate_motion_map,
            terrain_variability,
            None,
        )
    }

    /// Assemble a world around existing terrain. Used by [`World::new`] and by the loader.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        width: usize,
        height: usize,
        seed: u64,
        terrain: Vec<i32>,
        sea_level: f64,
        tectonic_plates: Vec<i32>,
        tectonic_plate_drift: Vec<[f64; 2]>,
        terrain_variability: f64,
        distance_sea_level: Option<f64>,
    ) -> Self {
        let n = width * height;
        let distance_sea_level = distance_sea_level.unwrap_or(sea_level);
        let distance = DistanceMatrix::build(width, height, &terrain, distance_sea_level);
        let land_mask = terrain.iter().map(|&t| t as f64 >= sea_level).collect();
        World {
            width,
            height,
            seed,
            rng: StdRng::seed_from_u64(seed),
            influence: vec![0.0; n],
            solidarity: vec![0.0; n],
            terrain,
            fitness: vec![1.0; n],
            nation: vec![0; n],
            land_mask,
            tectonic_plates,
            tectonic_plate_drift,
            terrain_variability,
            econ_growth_rate: 1.0,
            num_nations: 0,
            nation_colors: vec![[200, 200, 200]],
            conquest_difficulty: 3.0,
            solidarity_spread_rate: 0.25,
            nation_stability: 0.75,
            conquest_assimilation_rate: 0.5,
            sea_level,
            distance_sea_level,
            distance,
        }
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Orthogonal in-bounds neighbours of a cell index.
    fn neighbors(&self, i: usize) -> impl Iterator<Item = usize> {
        let (w, h) = (self.width, self.height);
        let (x, y) = (i % w, i / w);
        [
            (x + 1 < w).then(|| i + 1),
            (y > 0).then(|| i - w),
            (x > 0).then(|| i - 1),
            (y + 1 < h).then(|| i + w),
        ]
        .into_iter()
        .flatten()
    }

    /// Pick `count` distinct random land cells (fewer if the world lacks land).
    pub fn random_land_cells(&mut self, count: usize) -> Vec<(usize, usize)> {
        let land: Vec<usize> = (0..self.width * self.height)
            .filter(|&i| self.land_mask[i])
            .collect();
        let count = count.min(land.len());
        let mut chosen = HashSet::new();
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            let i = land[self.rng.gen_range(0..land.len())];
            if chosen.insert(i) {
                out.push((i % self.width, i / self.width));
            }
        }
        out
    }

    /// Found a nation at `(x, y)`. Returns its id.
    pub fn create_nation(
        &mut self,
        x: usize,
        y: usize,
        initial_influence: f64,
        initial_solidarity: f64,
    ) -> u32 {
        assert!(
            x < self.width && y < self.height,
            "center coordinates out of bounds"
        );
        self.create_nation_at(self.idx(x, y), initial_influence, initial_solidarity)
    }

    fn create_nation_at(
        &mut self,
        i: usize,
        initial_influence: f64,
        initial_solidarity: f64,
    ) -> u32 {
        self.num_nations += 1;
        self.nation[i] = self.num_nations;
        self.influence[i] = initial_influence;
        self.solidarity[i] = initial_solidarity;
        let color = [self.rng.gen(), self.rng.gen(), self.rng.gen()];
        self.nation_colors.push(color);
        self.num_nations
    }

    /// Flood fill from `start`, converting contiguous tyrant-owned cells with
    /// solidarity at or above `threshold` to the rebel nation.
    fn create_rebellion(&mut self, start: usize, threshold: f64, tyrant: u32) {
        let rebel = self.nation[start];
        let mut visited = HashSet::new();
        let mut queue = VecDeque::from([start]);
        while let Some(c) = queue.pop_front() {
            if !visited.insert(c) {
                continue;
            }
            if self.solidarity[c] >= threshold {
                self.nation[c] = rebel;
                let next: Vec<usize> = self
                    .neighbors(c)
                    .filter(|n| {
                        !visited.contains(n)
                            && self.solidarity[*n] >= threshold
                            && self.nation[*n] == tyrant
                    })
                    .collect();
                queue.extend(next);
            }
        }
    }

    /// Influence of each cell: its own weight plus what neighbours contribute.
    pub fn calc_influence(&mut self) {
        let (w, h) = (self.width, self.height);
        let (nation, sol) = (&self.nation, &self.solidarity);
        let infl = &mut self.influence;
        infl.iter_mut().for_each(|v| *v = 0.0);

        for i in 0..w * h {
            if nation[i] != 0 {
                infl[i] += 1.0 - sol[i];
            }
        }

        // Each adjacent pair (a, b) is handled once, contributing to both cells.
        let mut pair = |a: usize, b: usize| {
            if nation[a] == nation[b] {
                infl[b] += sol[a];
                infl[a] += sol[b];
            } else if nation[b] != 0 {
                infl[b] += sol[b] / 4.0;
                infl[a] += sol[a] / 4.0;
            }
        };
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if y + 1 < h {
                    pair(i, i + w);
                }
                if x + 1 < w {
                    pair(i, i + 1);
                }
            }
        }
    }

    /// Fitness of each cell: the influence*solidarity of every same-nation
    /// cell, weighted by inverse travel distance.
    pub fn calc_fitness(&mut self) {
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); self.num_nations as usize + 1];
        for (i, &n) in self.nation.iter().enumerate() {
            if n != 0 {
                members[n as usize].push(i as u32);
            }
        }
        let power: Vec<f64> = self
            .influence
            .iter()
            .zip(&self.solidarity)
            .map(|(a, b)| a * b)
            .collect();
        let (nation, dm) = (&self.nation, &self.distance);

        self.fitness.par_iter_mut().enumerate().for_each(|(t, f)| {
            let n = nation[t];
            if n == 0 {
                *f = 0.0;
                return;
            }
            let mut sum = 0.0;
            for &s in &members[n as usize] {
                let s = s as usize;
                if s == t {
                    sum += power[s];
                } else if let Some(d) = dm.get(s, t) {
                    sum += power[s] / d as f64;
                }
            }
            *f = sum;
        });
    }

    /// Cells adopt stronger neighbours' solidarity, mutate, and may rebel.
    #[allow(clippy::needless_range_loop)]
    pub fn update_solidarity(&mut self) {
        let (w, h) = (self.width, self.height);
        let n = w * h;
        let avg = self.avg_solidarity_by_nation();
        let mut new_sol = self.solidarity.clone();

        // Spread: copy solidarity from the most influential same-nation neighbour.
        for i in 0..n {
            let nat = self.nation[i];
            if nat == 0 {
                continue;
            }
            let (x, y) = (i % w, i / w);
            // Order matters for ties: up, down, left, right.
            let candidates = [
                (y > 0).then(|| i - w),
                (y + 1 < h).then(|| i + w),
                (x > 0).then(|| i - 1),
                (x + 1 < w).then(|| i + 1),
            ];
            let mut best = (0.0, None);
            for j in candidates.into_iter().flatten() {
                if self.nation[j] == nat && self.influence[j] > best.0 {
                    best = (self.influence[j], Some(j));
                }
            }
            if let (infl, Some(j)) = best {
                if infl > self.influence[i] && self.rng.gen::<f64>() < self.solidarity_spread_rate {
                    new_sol[i] = self.solidarity[j];
                }
            }
        }

        // Random mutation of +/- 0.1, applied to every cell.
        for s in new_sol.iter_mut() {
            let r: f64 = self.rng.gen();
            if r > 0.9 {
                *s = (*s + 0.1).min(1.0);
            } else if r < 0.1 {
                *s = (*s - 0.1).max(0.0);
            }
        }

        // Rebellion in nations whose average solidarity is low enough.
        let mut cells_by_nation: Vec<Vec<u32>> = vec![Vec::new(); avg.len()];
        for (i, &nat) in self.nation.iter().enumerate() {
            if nat != 0 {
                cells_by_nation[nat as usize].push(i as u32);
            }
        }
        for (id, cells) in cells_by_nation.iter().enumerate().skip(1) {
            if cells.is_empty() || avg[id] > 1.0 - self.nation_stability {
                continue;
            }
            let mut eligible = Vec::new();
            for &c in cells {
                let c = c as usize;
                if self.nation[c] != id as u32 {
                    continue;
                }
                let attempt: f64 = self.rng.gen();
                if attempt > avg[id] && attempt < new_sol[c] && self.rng.gen::<f64>() < 0.1 {
                    eligible.push(c);
                }
            }
            for c in eligible {
                let tyrant = self.nation[c];
                self.create_nation_at(c, 10.0, new_sol[c]);
                self.create_rebellion(c, 0.4, tyrant);
            }
        }

        self.solidarity = new_sol;
    }

    /// Nations take over weaker neighbouring land cells. All cells are decided
    /// against the same snapshot, then applied together.
    pub fn conquer(&mut self) {
        let (w, h) = (self.width, self.height);
        let mut new_nation = self.nation.clone();
        let mut new_sol = self.solidarity.clone();

        for i in 0..w * h {
            if !self.land_mask[i] {
                continue;
            }
            let (x, y) = (i % w, i / w);
            let neighbours = [
                (x > 0).then(|| i - 1),
                (x + 1 < w).then(|| i + 1),
                (y > 0).then(|| i - w),
                (y + 1 < h).then(|| i + w),
            ];
            let mut best: Option<usize> = None;
            for j in neighbours.into_iter().flatten() {
                let can = self.nation[j] != self.nation[i]
                    && self.nation[j] != 0
                    && self.fitness[j] > self.fitness[i] * self.conquest_difficulty;
                if can && best.is_none_or(|b| self.fitness[j] > self.fitness[b]) {
                    best = Some(j);
                }
            }
            if let Some(j) = best {
                new_nation[i] = self.nation[j];
                if self.rng.gen::<f64>() < self.conquest_assimilation_rate {
                    new_sol[i] = self.solidarity[j];
                }
            }
        }

        self.nation = new_nation;
        self.solidarity = new_sol;
    }

    /// Average solidarity per nation id (index 0 unused). Absent nations read 0.
    pub fn avg_solidarity_by_nation(&self) -> Vec<f64> {
        let mut sum = vec![0.0; self.num_nations as usize + 1];
        let mut count = vec![0u32; self.num_nations as usize + 1];
        for (&n, &s) in self.nation.iter().zip(&self.solidarity) {
            if n != 0 {
                sum[n as usize] += s;
                count[n as usize] += 1;
            }
        }
        sum.iter()
            .zip(&count)
            .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
            .collect()
    }

    /// Advance the simulation one step.
    pub fn update_world(&mut self) {
        self.calc_influence();
        self.calc_fitness();
        self.update_solidarity();
        self.conquer();
        self.land_mask = self
            .terrain
            .iter()
            .map(|&t| t as f64 >= self.sea_level)
            .collect();
    }

    /// Number of nations that currently own at least one cell.
    pub fn live_nations(&self) -> usize {
        let mut seen = vec![false; self.num_nations as usize + 1];
        for &n in &self.nation {
            seen[n as usize] = true;
        }
        seen.iter().skip(1).filter(|&&s| s).count()
    }
}
