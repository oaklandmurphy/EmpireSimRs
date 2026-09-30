use empiresim::save::{load_world, save_world};
use empiresim::World;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn ints(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn assert_close(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-9, "{what}[{i}]: got {g}, want {w}");
    }
}

/// The fixture holds a world plus the outputs the original Python code produced
/// for it (see tests/fixtures/gen_parity.py).
fn fixture() -> (World, Value) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/parity.json");
    let json: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    (load_world(path).unwrap(), json["expected"].clone())
}

#[test]
fn influence_matches_python() {
    let (mut world, expected) = fixture();
    world.calc_influence();
    assert_close(
        "influence",
        &world.influence,
        &floats(&expected["influence"]),
    );
}

#[test]
fn fitness_matches_python() {
    let (mut world, expected) = fixture();
    world.calc_influence();
    world.calc_fitness();
    assert_close("fitness", &world.fitness, &floats(&expected["fitness"]));
}

#[test]
fn conquest_matches_python() {
    let (mut world, expected) = fixture();
    world.calc_influence();
    world.calc_fitness();
    let before = world.nation.clone();

    // Without assimilation only ownership changes.
    world.conquest_assimilation_rate = 0.0;
    world.conquer();
    assert_eq!(world.nation, ints(&expected["conquer_nation"]));
    assert_ne!(world.nation, before, "fixture should exercise conquest");

    // With guaranteed assimilation, solidarity is copied from the conqueror too.
    let (mut world, _) = fixture();
    world.calc_influence();
    world.calc_fitness();
    world.conquest_assimilation_rate = 1.0;
    world.conquer();
    assert_eq!(world.nation, ints(&expected["conquer_nation_assimilated"]));
    assert_close(
        "solidarity",
        &world.solidarity,
        &floats(&expected["conquer_solidarity_assimilated"]),
    );
}

fn populated(seed: u64) -> World {
    let mut world = World::new(40, 40, seed);
    for (x, y) in world.random_land_cells(20) {
        world.create_nation(x, y, 10.0, 0.6);
    }
    world
}

#[test]
fn same_seed_gives_identical_runs() {
    let run = |seed| {
        let mut w = populated(seed);
        for _ in 0..30 {
            w.update_world();
        }
        (w.nation, w.solidarity)
    };
    assert_eq!(run(9), run(9));
    assert_ne!(run(9).0, run(10).0);
}

#[test]
fn simulation_invariants_hold() {
    let mut world = populated(4);
    for _ in 0..100 {
        world.update_world();
        for i in 0..world.nation.len() {
            assert!((0.0..=1.0).contains(&world.solidarity[i]));
            assert!(world.fitness[i].is_finite() && world.influence[i].is_finite());
            assert!((world.nation[i] as usize) < world.nation_colors.len());
            if world.nation[i] == 0 {
                assert_eq!(world.fitness[i], 0.0);
            }
        }
    }
}

#[test]
fn distances_respect_terrain_and_water() {
    let world = World::new(30, 30, 2);
    let land: Vec<usize> = (0..900).filter(|&i| world.land_mask[i]).collect();
    let a = land[0];
    // Entering a neighbouring cell costs that cell's terrain value.
    for &i in &land {
        for j in [i + 1, i + 30] {
            if j < 900 && world.land_mask[j] && (j != i + 1 || i % 30 != 29) {
                assert!(world.distance.get(i, j).unwrap() <= world.terrain[j] as u32);
            }
        }
    }
    assert_eq!(world.distance.get(a, a), Some(0));
    if let Some(water) = (0..900).find(|&i| !world.land_mask[i]) {
        assert_eq!(world.distance.get(a, water), None);
        assert_eq!(world.distance.get(water, a), None);
    }
}

#[test]
fn save_and_load_roundtrip() {
    let mut world = populated(6);
    for _ in 0..10 {
        world.update_world();
    }
    world.sea_level = 9.5;
    let path = std::env::temp_dir().join(format!("empiresim_test_{}.json", std::process::id()));
    save_world(&world, &path).unwrap();
    let loaded = load_world(&path).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(loaded.nation, world.nation);
    assert_eq!(loaded.terrain, world.terrain);
    assert_eq!(loaded.solidarity, world.solidarity);
    assert_eq!(loaded.nation_colors, world.nation_colors);
    assert_eq!(loaded.sea_level, 9.5);
    assert_eq!(loaded.distance_sea_level, world.distance_sea_level);

    // The rebuilt distance matrix must give the same fitness as the original.
    let (mut a, mut b) = (world, loaded);
    a.calc_influence();
    a.calc_fitness();
    b.calc_influence();
    b.calc_fitness();
    assert_eq!(a.fitness, b.fitness);
}

#[test]
fn rebellion_splits_low_solidarity_nations() {
    let mut world = World::new(20, 20, 1);
    let cells = world.random_land_cells(1);
    let (x, y) = cells[0];
    world.create_nation(x, y, 10.0, 0.0);
    world.nation_stability = 0.0; // stability 0 lets every nation rebel
    world.conquest_difficulty = 1e9; // no conquest
                                     // Give the one nation the whole map.
    for i in 0..world.nation.len() {
        if world.land_mask[i] {
            world.nation[i] = 1;
        }
    }
    for _ in 0..200 {
        world.update_world();
    }
    assert!(
        world.num_nations > 1,
        "an all-one-nation map should eventually split"
    );
}
