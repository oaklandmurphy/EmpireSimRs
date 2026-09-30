//! Saving and loading worlds as JSON.
//!
//! The distance matrix is not stored; it is rebuilt from the terrain on load.

use crate::world::World;
use serde::{Deserialize, Serialize};
use std::{fs, io, path::Path};

#[derive(Serialize, Deserialize)]
struct SavedWorld {
    width: usize,
    height: usize,
    seed: u64,
    influence: Vec<f64>,
    solidarity: Vec<f64>,
    terrain: Vec<i32>,
    fitness: Vec<f64>,
    nation: Vec<u32>,
    tectonic_plates: Vec<i32>,
    tectonic_plate_drift: Vec<[f64; 2]>,
    terrain_variability: f64,
    num_nations: u32,
    nation_colors: Vec<[u8; 3]>,
    conquest_difficulty: f64,
    solidarity_spread_rate: f64,
    nation_stability: f64,
    conquest_assimilation_rate: f64,
    sea_level: f64,
    distance_sea_level: f64,
}

pub fn save_world(world: &World, path: impl AsRef<Path>) -> io::Result<()> {
    let saved = SavedWorld {
        width: world.width,
        height: world.height,
        seed: world.seed,
        influence: world.influence.clone(),
        solidarity: world.solidarity.clone(),
        terrain: world.terrain.clone(),
        fitness: world.fitness.clone(),
        nation: world.nation.clone(),
        tectonic_plates: world.tectonic_plates.clone(),
        tectonic_plate_drift: world.tectonic_plate_drift.clone(),
        terrain_variability: world.terrain_variability,
        num_nations: world.num_nations,
        nation_colors: world.nation_colors.clone(),
        conquest_difficulty: world.conquest_difficulty,
        solidarity_spread_rate: world.solidarity_spread_rate,
        nation_stability: world.nation_stability,
        conquest_assimilation_rate: world.conquest_assimilation_rate,
        sea_level: world.sea_level,
        distance_sea_level: world.distance_sea_level,
    };
    if let Some(dir) = path.as_ref().parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(path, serde_json::to_vec(&saved).map_err(io::Error::other)?)
}

pub fn load_world(path: impl AsRef<Path>) -> io::Result<World> {
    let s: SavedWorld = serde_json::from_slice(&fs::read(path)?).map_err(io::Error::other)?;
    let n = s.width * s.height;
    let consistent = [
        s.influence.len(),
        s.solidarity.len(),
        s.terrain.len(),
        s.fitness.len(),
        s.nation.len(),
    ]
    .iter()
    .all(|&l| l == n)
        && s.tectonic_plates.len() == n
        && s.tectonic_plate_drift.len() == n
        && s.nation_colors.len() == s.num_nations as usize + 1;
    if !consistent {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "saved world has inconsistent array sizes",
        ));
    }

    let mut world = World::from_parts(
        s.width,
        s.height,
        s.seed,
        s.terrain,
        s.sea_level,
        s.tectonic_plates,
        s.tectonic_plate_drift,
        s.terrain_variability,
        Some(s.distance_sea_level),
    );
    world.influence = s.influence;
    world.solidarity = s.solidarity;
    world.fitness = s.fitness;
    world.nation = s.nation;
    world.num_nations = s.num_nations;
    world.nation_colors = s.nation_colors;
    world.conquest_difficulty = s.conquest_difficulty;
    world.solidarity_spread_rate = s.solidarity_spread_rate;
    world.nation_stability = s.nation_stability;
    world.conquest_assimilation_rate = s.conquest_assimilation_rate;
    Ok(world)
}
