//! Procedural terrain: tectonic plates, plate motion and Perlin noise.
//!
//! All grids are row-major with `idx = y * width + x`.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

/// Classic 2D gradient (Perlin) noise with a seeded permutation table.
pub struct Perlin {
    perm: [u8; 512],
}

impl Perlin {
    pub fn new(rng: &mut StdRng) -> Self {
        let mut p: Vec<u8> = (0..=255).collect();
        p.shuffle(rng);
        let mut perm = [0u8; 512];
        for i in 0..512 {
            perm[i] = p[i & 255];
        }
        Perlin { perm }
    }

    fn grad(hash: u8, x: f64, y: f64) -> f64 {
        // Eight unit-ish gradient directions.
        match hash & 7 {
            0 => x + y,
            1 => -x + y,
            2 => x - y,
            3 => -x - y,
            4 => x,
            5 => -x,
            6 => y,
            _ => -y,
        }
    }

    pub fn get(&self, x: f64, y: f64) -> f64 {
        let xf = x.floor();
        let yf = y.floor();
        let xi = (xf as i64 & 255) as usize;
        let yi = (yf as i64 & 255) as usize;
        let x = x - xf;
        let y = y - yf;
        let fade = |t: f64| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let (u, v) = (fade(x), fade(y));
        let p = &self.perm;
        let aa = p[p[xi] as usize + yi];
        let ab = p[p[xi] as usize + yi + 1];
        let ba = p[p[xi + 1] as usize + yi];
        let bb = p[p[xi + 1] as usize + yi + 1];
        let lerp = |a: f64, b: f64, t: f64| a + t * (b - a);
        let x1 = lerp(Self::grad(aa, x, y), Self::grad(ba, x - 1.0, y), u);
        let x2 = lerp(
            Self::grad(ab, x, y - 1.0),
            Self::grad(bb, x - 1.0, y - 1.0),
            u,
        );
        lerp(x1, x2, v)
    }
}

/// 1D Gaussian kernel, matching `scipy.ndimage.gaussian_filter` (truncate = 4.0).
fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let radius = (4.0 * sigma + 0.5) as i64;
    let mut k: Vec<f64> = (-radius..=radius)
        .map(|i| (-0.5 * (i * i) as f64 / (sigma * sigma)).exp())
        .collect();
    let sum: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    k
}

/// Reflect an index into `0..n` using scipy's default "reflect" boundary mode
/// (`d c b a | a b c d | d c b a`).
fn reflect(mut i: i64, n: i64) -> usize {
    let period = 2 * n;
    i = i.rem_euclid(period);
    if i >= n {
        i = period - 1 - i;
    }
    i as usize
}

/// Separable Gaussian blur of a `height x width` grid.
pub fn gaussian_filter(data: &[f64], width: usize, height: usize, sigma: f64) -> Vec<f64> {
    let kernel = gaussian_kernel(sigma);
    let radius = (kernel.len() / 2) as i64;

    let mut tmp = vec![0.0; data.len()];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0;
            for (k, w) in kernel.iter().enumerate() {
                let sx = reflect(x as i64 + k as i64 - radius, width as i64);
                acc += w * data[y * width + sx];
            }
            tmp[y * width + x] = acc;
        }
    }
    let mut out = vec![0.0; data.len()];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0;
            for (k, w) in kernel.iter().enumerate() {
                let sy = reflect(y as i64 + k as i64 - radius, height as i64);
                acc += w * tmp[sy * width + x];
            }
            out[y * width + x] = acc;
        }
    }
    out
}

pub struct PlateTectonics {
    pub width: usize,
    pub height: usize,
    pub num_plates: usize,
    pub sea_level: f64,
    pub elevation: Vec<f64>,
    pub plate_id: Vec<i32>,
    /// `(dy, dx)` drift per plate.
    pub plate_motion: Vec<[f64; 2]>,
    /// `(dy, dx)` drift per cell.
    pub plate_motion_map: Vec<[f64; 2]>,
    pub plate_rotation: Vec<f64>,
    pub plate_buoyancy: Vec<f64>,
}

impl PlateTectonics {
    pub fn new(width: usize, height: usize, num_plates: usize, seed: u64, sea_level: f64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = width * height;
        let mut pt = PlateTectonics {
            width,
            height,
            num_plates,
            sea_level,
            elevation: vec![0.0; n],
            plate_id: vec![-1; n],
            plate_motion: vec![[0.0; 2]; num_plates],
            plate_motion_map: vec![[0.0; 2]; n],
            plate_rotation: vec![0.0; num_plates],
            plate_buoyancy: vec![0.0; num_plates],
        };
        pt.generate_plates(&mut rng);
        pt.assign_plate_motion(&mut rng);
        pt.generate_motion_map(3.0);
        pt.generate_elevation(&mut rng, 0.05, 4);
        pt.apply_plate_motion_to_elevation(1.0);
        pt.normalize_elevation();
        pt
    }

    /// Grow plates from random seed cells by probabilistic flood fill, then
    /// fill any leftover holes from a random assigned neighbour.
    fn generate_plates(&mut self, rng: &mut StdRng) {
        let (w, h) = (self.width, self.height);
        let mut unassigned = w * h;
        let mut frontier: Vec<(usize, usize, i32)> = Vec::new();

        for pid in 0..self.num_plates {
            let (y, x) = (rng.gen_range(0..h), rng.gen_range(0..w));
            if self.plate_id[y * w + x] == -1 {
                unassigned -= 1;
            }
            self.plate_id[y * w + x] = pid as i32;
            frontier.push((y, x, pid as i32));
        }

        const DIRS: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        let in_bounds = |y: i64, x: i64| y >= 0 && y < h as i64 && x >= 0 && x < w as i64;

        while unassigned > 0 {
            if frontier.is_empty() {
                let holes: Vec<usize> = (0..w * h).filter(|&i| self.plate_id[i] == -1).collect();
                let i = holes[rng.gen_range(0..holes.len())];
                let (y, x) = ((i / w) as i64, (i % w) as i64);
                let neighbours: Vec<i32> = DIRS
                    .iter()
                    .filter(|(dy, dx)| in_bounds(y + dy, x + dx))
                    .map(|(dy, dx)| self.plate_id[((y + dy) as usize) * w + (x + dx) as usize])
                    .filter(|&p| p != -1)
                    .collect();
                let pid = if neighbours.is_empty() {
                    rng.gen_range(0..self.num_plates) as i32
                } else {
                    neighbours[rng.gen_range(0..neighbours.len())]
                };
                self.plate_id[i] = pid;
                unassigned -= 1;
                frontier.push((y as usize, x as usize, pid));
                continue;
            }

            let (y, x, pid) = frontier.swap_remove(rng.gen_range(0..frontier.len()));
            for (dy, dx) in DIRS {
                let (ny, nx) = (y as i64 + dy, x as i64 + dx);
                if !in_bounds(ny, nx) {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                if self.plate_id[ni] == -1 && rng.gen::<f64>() < 0.8 {
                    self.plate_id[ni] = pid;
                    unassigned -= 1;
                    frontier.push((ny as usize, nx as usize, pid));
                }
            }
        }

        // Buoyancy ~ truncated normal (loc 0.5, scale 0.45) on standard bounds [0, 1].
        for b in self.plate_buoyancy.iter_mut() {
            let z = loop {
                let z: f64 = rng.sample(StandardNormal);
                if (0.0..=1.0).contains(&z) {
                    break z;
                }
            };
            *b = 0.5 + 0.45 * z;
        }
    }

    fn assign_plate_motion(&mut self, rng: &mut StdRng) {
        for i in 0..self.num_plates {
            let angle = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
            let speed = rng.gen_range(0.1..1.0);
            self.plate_motion[i] = [angle.sin() * speed, angle.cos() * speed];
            self.plate_rotation[i] = rng.gen_range(-0.02..0.02);
        }
    }

    /// Per-cell motion from plate drift plus rotation about the plate centre,
    /// smoothed so mountains and valleys come out larger.
    pub fn generate_motion_map(&mut self, smooth_sigma: f64) {
        let (w, h) = (self.width, self.height);
        let mut sum = vec![[0.0f64; 2]; self.num_plates];
        let mut count = vec![0usize; self.num_plates];
        for y in 0..h {
            for x in 0..w {
                let p = self.plate_id[y * w + x] as usize;
                sum[p][0] += y as f64;
                sum[p][1] += x as f64;
                count[p] += 1;
            }
        }

        let mut dy = vec![0.0; w * h];
        let mut dx = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let p = self.plate_id[i] as usize;
                let c = count[p] as f64;
                let (rel_y, rel_x) = (y as f64 - sum[p][0] / c, x as f64 - sum[p][1] / c);
                let rot = self.plate_rotation[p];
                dy[i] = self.plate_motion[p][0] - rel_x * rot;
                dx[i] = self.plate_motion[p][1] + rel_y * rot;
            }
        }
        let dy = gaussian_filter(&dy, w, h, smooth_sigma);
        let dx = gaussian_filter(&dx, w, h, smooth_sigma);
        for i in 0..w * h {
            self.plate_motion_map[i] = [dy[i], dx[i]];
        }
    }

    /// Both sides of a converging boundary rise; both sides of a diverging one sink.
    pub fn apply_plate_motion_to_elevation(&mut self, strength: f64) {
        let (w, h) = (self.width, self.height);
        let m = &self.plate_motion_map;
        let mut delta = vec![0.0f64; w * h];

        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if y + 1 < h {
                    let j = i + w;
                    let (ry, rx) = (m[j][0] - m[i][0], m[j][1] - m[i][1]);
                    let d = -ry * (ry * ry + rx * rx).sqrt();
                    delta[i] += d;
                    delta[j] += d;
                }
                if x + 1 < w {
                    let j = i + 1;
                    let (ry, rx) = (m[j][0] - m[i][0], m[j][1] - m[i][1]);
                    let d = -rx * (ry * ry + rx * rx).sqrt();
                    delta[i] += d;
                    delta[j] += d;
                }
            }
        }

        let min = delta.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = delta.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - min;
        for (e, d) in self.elevation.iter_mut().zip(&delta) {
            let norm = if max > 0.0 { (d - min) / max } else { 0.0 };
            *e += strength * norm;
        }
    }

    /// Plate buoyancy plus layered Perlin noise.
    fn generate_elevation(&mut self, rng: &mut StdRng, noise_scale: f64, octaves: usize) {
        let (w, h) = (self.width, self.height);
        let perlin = Perlin::new(rng);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let (mut freq, mut amp, mut noise) = (noise_scale, 1.0, 0.0);
                for _ in 0..octaves {
                    noise += amp * perlin.get(x as f64 * freq, y as f64 * freq);
                    amp *= 0.5;
                    freq *= 2.0;
                }
                self.elevation[i] = self.plate_buoyancy[self.plate_id[i] as usize] + noise;
            }
        }
    }

    fn normalize_elevation(&mut self) {
        let min = self.elevation.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self
            .elevation
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let range = if max > min { max - min } else { 1.0 };
        self.elevation
            .iter_mut()
            .for_each(|e| *e = (*e - min) / range);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_filter_matches_scipy() {
        // Reference values from scipy.ndimage.gaussian_filter(a, sigma=3.0) on a 6x5 array.
        let input = [
            0.0,
            1.0,
            2.8284271247461903,
            5.196152422706632,
            1.0,
            4.180339887498949,
            0.6969384566990691,
            4.520259177452132,
            1.6274169979695223,
            6.0,
            3.6227766016837926,
            1.4828726939093997,
            6.569219381653056,
            4.872166581031863,
            3.3832034148351795,
            2.0947501931112527,
            1.0,
            0.09279563550022374,
            6.367532368147138,
            5.819079927272796,
            5.442719099991592,
            5.234089594072643,
            5.189146716115445,
            5.304125036192545,
            5.5755076535925525,
            6.0,
            6.5745073534124,
            0.29611541307906464,
            1.1620734196170588,
            2.1697794069006306,
        ];
        let expected = [
            3.4966652809829917,
            3.4558019097020223,
            3.38265330044744,
            3.2945215724246983,
            3.214970133542059,
            3.1677138273485834,
            3.596451801363093,
            3.5465210321297382,
            3.4579376177283683,
            3.352401280441662,
            3.258149734938932,
            3.2025595897028167,
            3.7573611778864855,
            3.6928192907128317,
            3.5793649417247977,
            3.445784430736231,
            3.3278473512937765,
            3.258829680483546,
            3.9176182624010285,
            3.8385402905428343,
            3.7003436479384715,
            3.538865370335935,
            3.3973677801465088,
            3.314994048993575,
            4.016362522055305,
            3.9283417385304564,
            3.774923881602337,
            3.5962840794352964,
            3.440293953232709,
            3.349705028386607,
        ];
        let out = gaussian_filter(&input, 6, 5, 3.0);
        for (a, b) in out.iter().zip(expected) {
            assert!((a - b).abs() < 1e-9, "{a} != {b}");
        }
    }

    #[test]
    fn plates_cover_the_grid_and_elevation_is_normalized() {
        let pt = PlateTectonics::new(40, 30, 6, 1, 0.55);
        assert!(pt.plate_id.iter().all(|&p| (0..6).contains(&p)));
        let min = pt.elevation.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = pt
            .elevation
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!((min, max), (0.0, 1.0));
    }

    #[test]
    fn perlin_is_smooth_and_bounded() {
        let p = Perlin::new(&mut StdRng::seed_from_u64(5));
        let mut prev = p.get(0.0, 0.3);
        for i in 1..2000 {
            let v = p.get(i as f64 * 0.01, 0.3);
            assert!(v.abs() <= 1.0 + 1e-9);
            assert!(
                (v - prev).abs() < 0.1,
                "noise should not jump between nearby samples"
            );
            prev = v;
        }
    }
}
