use empiresim::save::load_world;
use empiresim::World;
use std::process::exit;
use std::time::Instant;

#[cfg(feature = "gui")]
mod gui;

const USAGE: &str = "\
EmpireSim

USAGE:
    empiresim [OPTIONS]

OPTIONS:
    --seed <N>          World seed (default 42)
    --width <N>         World width in cells (default 100)
    --height <N>        World height in cells (default 100)
    --nations <N>       Number of starting nations (default 100)
    --cell-size <N>     Initial cell size in pixels (default 40)
    --load <FILE>       Load a saved world instead of generating one
    --headless <STEPS>  Run STEPS simulation steps without a window and print timings
    -h, --help          Show this help
";

struct Args {
    seed: u64,
    width: usize,
    height: usize,
    nations: usize,
    cell_size: f32,
    load: Option<String>,
    headless: Option<usize>,
}

fn parse_args() -> Args {
    let mut args = Args {
        seed: 42,
        width: 100,
        height: 100,
        nations: 100,
        cell_size: 40.0,
        load: None,
        headless: None,
    };
    let mut it = std::env::args().skip(1);
    let fail = |msg: String| -> ! {
        eprintln!("error: {msg}\n\n{USAGE}");
        exit(2)
    };
    while let Some(flag) = it.next() {
        if flag == "-h" || flag == "--help" {
            print!("{USAGE}");
            exit(0);
        }
        let mut value = |name: &str| {
            it.next()
                .unwrap_or_else(|| fail(format!("{name} needs a value")))
        };
        macro_rules! num {
            ($name:expr) => {
                value($name)
                    .parse()
                    .unwrap_or_else(|_| fail(format!("invalid value for {}", $name)))
            };
        }
        match flag.as_str() {
            "--seed" => args.seed = num!("--seed"),
            "--width" => args.width = num!("--width"),
            "--height" => args.height = num!("--height"),
            "--nations" => args.nations = num!("--nations"),
            "--cell-size" => args.cell_size = num!("--cell-size"),
            "--load" => args.load = Some(value("--load")),
            "--headless" => args.headless = Some(num!("--headless")),
            other => fail(format!("unknown option {other}")),
        }
    }
    args
}

fn build_world(args: &Args) -> World {
    if let Some(path) = &args.load {
        return load_world(path).unwrap_or_else(|e| {
            eprintln!("error: could not load {path}: {e}");
            exit(1)
        });
    }
    let start = Instant::now();
    let mut world = World::new(args.width, args.height, args.seed);
    println!(
        "Generated {}x{} world in {:.2?}",
        args.width,
        args.height,
        start.elapsed()
    );

    for (x, y) in world.random_land_cells(args.nations) {
        let id = world.create_nation(x, y, 10.0, 0.6);
        println!("Created nation {id} at ({x}, {y})");
    }
    world
}

fn run_headless(mut world: World, steps: usize) {
    let start = Instant::now();
    for _ in 0..steps {
        world.update_world();
    }
    let elapsed = start.elapsed();
    let per_step = elapsed / steps.max(1) as u32;
    let owned = world.nation.iter().filter(|&&n| n != 0).count();
    println!(
        "{steps} steps in {elapsed:.2?} ({per_step:.2?}/step); {} nations alive, {owned} cells owned",
        world.live_nations()
    );
}

fn main() {
    let args = parse_args();
    let world = build_world(&args);

    if let Some(steps) = args.headless {
        run_headless(world, steps);
        return;
    }

    #[cfg(feature = "gui")]
    macroquad::Window::from_config(gui::window_conf(), gui::run(world, args.cell_size));
    #[cfg(not(feature = "gui"))]
    {
        let _ = world;
        eprintln!("built without the `gui` feature; use --headless <STEPS>");
        exit(1);
    }
}
