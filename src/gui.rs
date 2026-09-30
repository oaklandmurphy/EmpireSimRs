//! Interactive viewer: the world drawn as a single texture with sliders and buttons.

use empiresim::save::save_world;
use empiresim::World;
use macroquad::prelude::*;

#[derive(Clone, Copy, PartialEq)]
enum Attribute {
    Nation,
    Influence,
    Solidarity,
    Fitness,
    Terrain,
    TectonicPlates,
    TectonicPlateDrift,
}

impl Attribute {
    const ALL: [Attribute; 7] = [
        Attribute::Nation,
        Attribute::Influence,
        Attribute::Solidarity,
        Attribute::Fitness,
        Attribute::Terrain,
        Attribute::TectonicPlates,
        Attribute::TectonicPlateDrift,
    ];

    fn label(self) -> &'static str {
        match self {
            Attribute::Nation => "Nation",
            Attribute::Influence => "Influence",
            Attribute::Solidarity => "Solidarity",
            Attribute::Fitness => "Fitness",
            Attribute::Terrain => "Terrain",
            Attribute::TectonicPlates => "Tectonic plates",
            Attribute::TectonicPlateDrift => "Tectonic plate drift",
        }
    }
}

const SLIDER_LABELS: [&str; 5] = [
    "Conquest Difficulty",
    "Nation Stability",
    "Solidarity Spread Rate",
    "Conquest Assimilation Rate",
    "Sea Level",
];

fn slider_range(index: usize, world: &World) -> (f64, f64) {
    match index {
        0 => (1.1, 5.0),
        4 => (0.0, world.terrain_variability),
        _ => (0.0, 1.0),
    }
}

fn slider_value(index: usize, world: &mut World) -> &mut f64 {
    match index {
        0 => &mut world.conquest_difficulty,
        1 => &mut world.nation_stability,
        2 => &mut world.solidarity_spread_rate,
        3 => &mut world.conquest_assimilation_rate,
        _ => &mut world.sea_level,
    }
}

fn norm(v: f64, min: f64, max: f64) -> f64 {
    if max > min {
        ((v - min) / (max - min)).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn min_max<T: Copy + Into<f64>>(data: &[T]) -> (f64, f64) {
    data.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            let v: f64 = v.into();
            (lo.min(v), hi.max(v))
        })
}

/// RGBA pixels (row-major, `y * width + x`) for the selected attribute.
fn fill_pixels(world: &World, attr: Attribute, out: &mut [u8]) {
    let ramp = |t: f64| [(255.0 * (1.0 - t)) as u8, (255.0 * t) as u8, 40];
    let gray = |t: f64| {
        let v = (255.0 * t) as u8;
        [v, v, v]
    };
    let n = world.width * world.height;

    let (fit_min, fit_max) = min_max(&world.fitness);
    let (ter_min, ter_max) = min_max(&world.terrain);
    let plates: Vec<f64> = world.tectonic_plates.iter().map(|&p| p as f64).collect();
    let (plate_min, plate_max) = min_max(&plates);

    for i in 0..n {
        let rgb = match attr {
            Attribute::Nation => world.nation_colors[world.nation[i] as usize],
            Attribute::Influence => gray(norm(world.influence[i], 0.0, 5.0)),
            Attribute::Solidarity => ramp(norm(world.solidarity[i], 0.0, 1.0)),
            Attribute::Fitness => ramp(norm(world.fitness[i], fit_min, fit_max)),
            Attribute::Terrain => {
                let t = norm(world.terrain[i] as f64, ter_min, ter_max);
                let blue = if (world.terrain[i] as f64) < world.sea_level {
                    255
                } else {
                    0
                };
                [40, (220.0 - 120.0 * t) as u8, blue]
            }
            Attribute::TectonicPlates => gray(norm(plates[i], plate_min, plate_max)),
            Attribute::TectonicPlateDrift => {
                let [dy, dx] = world.tectonic_plate_drift[i];
                [
                    (255.0 * dy.abs().min(1.0)) as u8,
                    0,
                    (255.0 * dx.abs().min(1.0)) as u8,
                ]
            }
        };
        out[i * 4..i * 4 + 4].copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
    }
}

fn in_rect(p: (f32, f32), r: Rect) -> bool {
    r.contains(vec2(p.0, p.1))
}

fn button(rect: Rect, fill: Color, label: &str) {
    draw_rectangle(rect.x, rect.y, rect.w, rect.h, fill);
    draw_rectangle_lines(
        rect.x,
        rect.y,
        rect.w,
        rect.h,
        2.0,
        Color::from_rgba(40, 40, 40, 255),
    );
    // Shrink the text if it would overflow the button.
    let full = measure_text(label, None, 24, 1.0).width;
    let size = if full > rect.w - 8.0 {
        (24.0 * (rect.w - 8.0) / full).floor().max(8.0)
    } else {
        24.0
    };
    let dims = measure_text(label, None, size as u16, 1.0);
    draw_text(
        label,
        rect.x + (rect.w - dims.width) / 2.0,
        rect.y + (rect.h + dims.height) / 2.0,
        size,
        WHITE,
    );
}

pub fn window_conf() -> Conf {
    Conf {
        window_title: "EmpireSim".to_string(),
        window_width: 1400,
        window_height: 900,
        window_resizable: true,
        ..Default::default()
    }
}

pub async fn run(mut world: World, mut cell_size: f32) {
    const BUTTON_H: f32 = 40.0;
    const PAD: f32 = 10.0;
    const SLIDER_W: f32 = 30.0;

    let mut attribute = Attribute::Nation;
    let mut continuous = false;
    let (mut cam_x, mut cam_y) = (0.0f32, 0.0f32);
    let mut panning = false;
    let mut last_mouse = (0.0f32, 0.0f32);
    let mut dragging_slider: Option<usize> = None;

    let mut image = Image::gen_image_color(world.width as u16, world.height as u16, BLACK);
    let texture = Texture2D::from_image(&image);
    texture.set_filter(FilterMode::Nearest);

    loop {
        let (sw, sh) = (screen_width(), screen_height());
        let mouse = mouse_position();

        // Layout.
        let save_rect = Rect::new(PAD, PAD, 120.0, BUTTON_H);
        let continuous_rect = Rect::new(sw - 180.0 - PAD, PAD, 180.0, BUTTON_H);
        let update_rect = Rect::new(continuous_rect.x - PAD - 160.0, PAD, 160.0, BUTTON_H);
        let attr_w = sw / Attribute::ALL.len() as f32 - PAD;
        let attr_rects: Vec<Rect> = (0..Attribute::ALL.len())
            .map(|i| {
                Rect::new(
                    i as f32 * (attr_w + PAD) + PAD,
                    sh - BUTTON_H - PAD,
                    attr_w,
                    BUTTON_H,
                )
            })
            .collect();
        let slot = ((sh - 100.0 - 80.0) / SLIDER_LABELS.len() as f32).max(60.0);
        let slider_rects: Vec<Rect> = (0..SLIDER_LABELS.len())
            .map(|i| {
                Rect::new(
                    sw - SLIDER_W - 80.0,
                    100.0 + i as f32 * slot,
                    SLIDER_W,
                    (slot - 40.0).max(20.0),
                )
            })
            .collect();

        // Input.
        if is_mouse_button_pressed(MouseButton::Left) {
            if let Some(i) = slider_rects.iter().position(|r| in_rect(mouse, *r)) {
                dragging_slider = Some(i);
            } else if let Some(i) = attr_rects.iter().position(|r| in_rect(mouse, *r)) {
                attribute = Attribute::ALL[i];
            } else if in_rect(mouse, update_rect) {
                world.update_world();
            } else if in_rect(mouse, continuous_rect) {
                continuous = !continuous;
            } else if in_rect(mouse, save_rect) {
                let secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs());
                let path = format!("saved_maps/saved_map_{secs}.json");
                match save_world(&world, &path) {
                    Ok(()) => println!("Map saved to {path}"),
                    Err(e) => eprintln!("Failed to save map to {path}: {e}"),
                }
            } else {
                panning = true;
            }
            last_mouse = mouse;
        }
        if is_mouse_button_released(MouseButton::Left) {
            panning = false;
            dragging_slider = None;
        }
        if let Some(i) = dragging_slider {
            let r = slider_rects[i];
            let rel = ((mouse.1 - r.y) / r.h).clamp(0.0, 1.0) as f64;
            let (lo, hi) = slider_range(i, &world);
            *slider_value(i, &mut world) = hi - rel * (hi - lo);
        } else if panning {
            cam_x -= mouse.0 - last_mouse.0;
            cam_y -= mouse.1 - last_mouse.1;
            last_mouse = mouse;
        }
        if is_key_pressed(KeyCode::Left) {
            cam_x -= 40.0;
        }
        if is_key_pressed(KeyCode::Right) {
            cam_x += 40.0;
        }
        if is_key_pressed(KeyCode::Up) {
            cam_y -= 40.0;
        }
        if is_key_pressed(KeyCode::Down) {
            cam_y += 40.0;
        }
        let wheel = mouse_wheel().1;
        if wheel != 0.0 {
            // Zoom about the cursor so the tile under it stays put.
            let new_size = (cell_size + 2.0 * wheel.signum()).clamp(2.0, 200.0);
            let scale = new_size / cell_size;
            cam_x = (cam_x + mouse.0) * scale - mouse.0;
            cam_y = (cam_y + mouse.1) * scale - mouse.1;
            cell_size = new_size;
        }

        if continuous {
            world.update_world();
        }

        // Grid.
        clear_background(Color::new(0.12, 0.12, 0.12, 1.0));
        fill_pixels(&world, attribute, &mut image.bytes);
        texture.update(&image);
        draw_texture_ex(
            &texture,
            -cam_x,
            -cam_y,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(
                    world.width as f32 * cell_size,
                    world.height as f32 * cell_size,
                )),
                ..Default::default()
            },
        );

        // Sliders.
        for (i, r) in slider_rects.iter().enumerate() {
            let (lo, hi) = slider_range(i, &world);
            let value = *slider_value(i, &mut world);
            draw_rectangle(r.x, r.y, r.w, r.h, Color::from_rgba(60, 60, 60, 255));
            let handle_y = r.y + ((hi - value) / (hi - lo)) as f32 * r.h;
            draw_rectangle(
                r.x,
                handle_y - 4.0,
                r.w,
                8.0,
                Color::from_rgba(180, 180, 80, 255),
            );
            let label = SLIDER_LABELS[i];
            let dims = measure_text(label, None, 24, 1.0);
            draw_text(
                label,
                r.x - dims.width - 10.0,
                r.y + r.h / 2.0 + dims.height / 2.0,
                24.0,
                WHITE,
            );
            draw_text(
                format!("{value:.2}"),
                r.x + r.w + 10.0,
                r.y + r.h / 2.0 + 8.0,
                24.0,
                WHITE,
            );
        }

        // Buttons.
        button(save_rect, Color::from_rgba(60, 180, 60, 255), "Save");
        button(update_rect, Color::from_rgba(70, 120, 200, 255), "Update");
        let cont_fill = if continuous {
            Color::from_rgba(100, 200, 100, 255)
        } else {
            Color::from_rgba(120, 120, 120, 255)
        };
        button(continuous_rect, cont_fill, "Continuous");
        for (a, r) in Attribute::ALL.iter().zip(&attr_rects) {
            let fill = if *a == attribute {
                Color::from_rgba(100, 180, 100, 255)
            } else {
                Color::from_rgba(80, 80, 80, 255)
            };
            button(*r, fill, a.label());
        }

        // Tooltip for the hovered tile.
        if !panning && dragging_slider.is_none() {
            let (tx, ty) = ((mouse.0 + cam_x) / cell_size, (mouse.1 + cam_y) / cell_size);
            if tx >= 0.0 && ty >= 0.0 && (tx as usize) < world.width && (ty as usize) < world.height
            {
                let (x, y) = (tx as usize, ty as usize);
                let i = world.idx(x, y);
                let [dy, dx] = world.tectonic_plate_drift[i];
                let lines = [
                    format!("x={x}, y={y}"),
                    format!("influence: {:.2}", world.influence[i]),
                    format!("solidarity: {:.2}", world.solidarity[i]),
                    format!("terrain: {}", world.terrain[i]),
                    format!("fitness: {:.2}", world.fitness[i]),
                    format!("nation: {}", world.nation[i]),
                    format!("tectonic plate: {}", world.tectonic_plates[i]),
                    format!("tectonic drift: [{dy:.2} {dx:.2}]"),
                    format!("land_mask: {}", world.land_mask[i] as u8),
                ];
                let width = lines
                    .iter()
                    .map(|l| measure_text(l, None, 18, 1.0).width)
                    .fold(0.0, f32::max)
                    + 10.0;
                let height = lines.len() as f32 * 20.0 + 10.0;
                let bx = (mouse.0 + 20.0).min(sw - width - 10.0);
                let by = (mouse.1 + 20.0).min(sh - height - 10.0);
                draw_rectangle(bx, by, width, height, Color::from_rgba(30, 30, 30, 255));
                for (n, line) in lines.iter().enumerate() {
                    draw_text(line, bx + 5.0, by + 20.0 + n as f32 * 20.0, 18.0, WHITE);
                }
            }
        }

        next_frame().await;
    }
}
