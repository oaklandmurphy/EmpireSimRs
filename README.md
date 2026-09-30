# EmpireSim
a project for modeling the rise and fall of power structures and societies

Written in Rust. The simulation core is a library (`src/lib.rs`); the interactive viewer is a
[macroquad](https://macroquad.rs) window on top of it.

## Run

```
cargo run --release                      # 100x100 world, 100 nations, interactive viewer
cargo run --release -- --seed 7 --width 150 --height 150 --nations 200
cargo run --release -- --headless 1000   # no window: run 1000 steps and print timings
cargo run --release -- --help
```

Always use `--release`; debug builds are an order of magnitude slower.

The viewer needs a desktop with OpenGL. To build only the simulation (no window system libraries),
use `cargo build --release --no-default-features`; that build supports `--headless` only.

### Viewer controls

* **Update** advances one step, **Continuous** steps every frame, **Save** writes `saved_maps/saved_map_<time>.json`
  (reload with `--load <file>`).
* Bottom row picks the attribute to display; hover a tile for its values.
* Drag to pan, arrow keys to pan, mouse wheel to zoom.
* The sliders on the right change conquest difficulty, nation stability, solidarity spread,
  conquest assimilation and sea level while the simulation runs.

## Layout

| File | |
| --- | --- |
| `src/terrain.rs` | plate tectonics, Perlin noise, Gaussian filter |
| `src/world.rs` | world state, distance matrix, and the per-step rules (influence, fitness, solidarity, rebellion, conquest) |
| `src/save.rs` | JSON save/load |
| `src/gui.rs`, `src/main.rs` | viewer and command line |

## Tests

`cargo test --release`. `tests/fixtures/parity.json` holds a world plus the influence, fitness and
conquest results the original Python implementation produced for it; the Rust code must reproduce them.
