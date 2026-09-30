"""Generate tests/fixtures/parity.json from the original Python implementation.

The Python sources were removed when the project moved to Rust; check them out of
git history first (commit 3730c5b), then install numpy, scipy and perlin-noise:

    git show 3730c5b:world.py > world.py && git show 3730c5b:terrain.py > terrain.py
    python3 gen_parity.py | tail -1 > parity.json   # terrain.py prints a line on import

The file is a world in the Rust save format plus the outputs the Python code
produced for the deterministic steps (influence, fitness, conquest).
Arrays are transposed because Python indexes [x, y] and Rust uses y * width + x.
"""
import json
import numpy as np
from world import World

N = 30
w = World(N, N, seed=3)
rng = np.random.default_rng(0)

land = w.land_mask.astype(bool)
w.nation = np.where(land, rng.integers(0, 7, size=(N, N)), 0)
w.solidarity = rng.random((N, N))
w.conquest_difficulty = 1.1

w.calc_influence()
influence = w.influence.copy()
w.calc_fitness()
fitness = w.fitness.copy()

def conquered(rate):
    c = World.__new__(World)
    c.__dict__.update(w.__dict__)
    c.nation, c.solidarity = w.nation.copy(), w.solidarity.copy()
    c.conquest_assimilation_rate = rate
    c.conquer()
    return c.nation, c.solidarity

nation_a0, _ = conquered(0.0)
nation_a1, sol_a1 = conquered(1.0)

flat = lambda a: a.T.flatten().tolist()
print(json.dumps({
    "width": N, "height": N, "seed": 3,
    "influence": flat(influence), "solidarity": flat(w.solidarity),
    "terrain": flat(w.terrain), "fitness": flat(fitness),
    "nation": flat(w.nation),
    "tectonic_plates": [0] * (N * N), "tectonic_plate_drift": [[0.0, 0.0]] * (N * N),
    "terrain_variability": 20.0, "num_nations": 6,
    "nation_colors": [[200, 200, 200]] * 7,
    "conquest_difficulty": 1.1, "solidarity_spread_rate": 0.25,
    "nation_stability": 0.75, "conquest_assimilation_rate": 0.5,
    "sea_level": float(w.sea_level), "distance_sea_level": float(w.sea_level),
    "expected": {
        "influence": flat(influence), "fitness": flat(fitness),
        "conquer_nation": flat(nation_a0),
        "conquer_nation_assimilated": flat(nation_a1),
        "conquer_solidarity_assimilated": flat(sol_a1),
    },
}))
