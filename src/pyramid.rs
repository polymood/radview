//! 2x2 mean pyramid over a `Raster`. 8-bit data stays u8. Other data goes to f16 (value * k).
use crate::raster::{Codec, Raster};
use half::f16;
use half::slice::HalfFloatSliceExt;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering::*};
use std::sync::{Arc, RwLock};
use std::time::Instant;

pub const TILE: usize = 512;

pub enum TileBuf {
    U8(Vec<u8>),
    F16(Vec<f16>),
}

impl TileBuf {
    pub fn bytes(&self) -> &[u8] {
        match self {
            TileBuf::U8(v) => v,
            TileBuf::F16(v) => bytemuck::cast_slice(v),
        }
    }
}

pub trait Px: Copy + Default + Send + Sync + 'static {
    fn fill(r: &Raster, raw: &[u8], out: &mut [Self], k: f32, tmp: &mut Vec<f32>);
    /// out[i] = mean of a[2i..2i+2] and b[2i..2i+2]. The odd last column uses one column.
    fn half_row(a: &[Self], b: &[Self], out: &mut [Self]);
    fn get(self) -> f32;
    fn wrap(v: Vec<Self>) -> TileBuf;
}

impl Px for u8 {
    fn fill(r: &Raster, raw: &[u8], out: &mut [u8], _: f32, _: &mut Vec<f32>) {
        r.to_u8(raw, out)
    }
    fn half_row(a: &[u8], b: &[u8], out: &mut [u8]) {
        let pairs = a.as_chunks::<2>().0.iter().zip(b.as_chunks::<2>().0);
        for (o, (a, b)) in out.iter_mut().zip(pairs) {
            *o = ((a[0] as u16 + a[1] as u16 + b[0] as u16 + b[1] as u16 + 2) >> 2) as u8;
        }
        if a.len() % 2 == 1 {
            let n = a.len() - 1;
            out[n / 2] = ((a[n] as u16 + b[n] as u16 + 1) >> 1) as u8;
        }
    }
    fn get(self) -> f32 {
        self as f32
    }
    fn wrap(v: Vec<u8>) -> TileBuf {
        TileBuf::U8(v)
    }
}

impl Px for f16 {
    fn fill(r: &Raster, raw: &[u8], out: &mut [f16], k: f32, tmp: &mut Vec<f32>) {
        tmp.resize(out.len(), 0.0);
        r.to_f32(raw, tmp);
        if k != 1.0 {
            tmp.iter_mut().for_each(|v| *v *= k);
        }
        out.convert_from_f32_slice(tmp);
    }
    fn half_row(a: &[f16], b: &[f16], out: &mut [f16]) {
        // Blocks of 64 outputs: the f16 <-> f32 slice conversions use F16C.
        let (mut fa, mut fb, mut fo) = ([0f32; 128], [0f32; 128], [0f32; 64]);
        for (o, (a, b)) in out.chunks_mut(64).zip(a.chunks(128).zip(b.chunks(128))) {
            let n = a.len();
            a.convert_to_f32_slice(&mut fa[..n]);
            b.convert_to_f32_slice(&mut fb[..n]);
            for i in 0..n / 2 {
                fo[i] = (fa[2 * i] + fa[2 * i + 1] + fb[2 * i] + fb[2 * i + 1]) * 0.25;
            }
            if n % 2 == 1 {
                fo[n / 2] = (fa[n - 1] + fb[n - 1]) * 0.5;
            }
            o.convert_from_f32_slice(&fo[..o.len()]);
        }
    }
    fn get(self) -> f32 {
        self.to_f32()
    }
    fn wrap(v: Vec<f16>) -> TileBuf {
        TileBuf::F16(v)
    }
}

pub struct Meta {
    pub w: usize,
    pub h: usize,
    pub desc: String,
    /// Sorted sample of raw values (finite, non-zero).
    pub sample: Vec<f32>,
    /// Stored value = raw value * k.
    pub k: f32,
    pub is_u8: bool,
    pub bands: Vec<String>,
    pub band: usize,
}

/// Object-safe view of a `Pyramid<P>` for the viewer.
pub trait Image: Send + Sync {
    fn meta(&self) -> &Meta;
    fn dims(&self) -> &[(usize, usize)];
    fn ready(&self, l: usize) -> bool;
    fn version(&self, l: usize) -> u32;
    fn tile(&self, l: usize, tx: usize, ty: usize) -> Option<(TileBuf, usize, usize)>;
    /// Raw value at level-0 pixel (x, y).
    fn value(&self, x: usize, y: usize) -> Option<f32>;
    fn build(&self, wake: &dyn Fn());
    fn cancel(&self);
    /// Build progress 0..1, and total build time when done.
    fn progress(&self) -> (f32, Option<f32>);
}

pub fn open(path: &str, band: usize) -> Result<Arc<dyn Image>, String> {
    let src = Raster::open(path, band)?;
    let sample = src.sample();
    if src.is_u8() {
        return Ok(Arc::new(Pyramid::<u8>::new(src, 1.0, sample)));
    }
    // Keep values far below the f16 limit (65504).
    let m = sample.first().map_or(1.0, |a| a.abs()).max(sample.get(sample.len() * 999 / 1000).map_or(1.0, |b| b.abs()));
    let k = if m > 16384.0 { 16384.0 / m } else { 1.0 };
    Ok(Arc::new(Pyramid::<f16>::new(src, k, sample)))
}

type Level<P> = RwLock<Option<Arc<Vec<P>>>>;

pub struct Pyramid<P: Px> {
    src: Raster,
    meta: Meta,
    dims: Vec<(usize, usize)>,
    levels: Vec<Level<P>>, // level 0 is stored only for compressed sources
    versions: Vec<AtomicU32>,
    cancel: AtomicBool,
    done: AtomicUsize,
    total: usize,
    build_ms: AtomicU32,
}

impl<P: Px> Pyramid<P> {
    fn new(src: Raster, k: f32, sample: Vec<f32>) -> Self {
        let mut dims = vec![(src.w, src.h)];
        while let Some(&(w, h)) = dims.last().filter(|d| d.0.max(d.1) > TILE) {
            dims.push((w.div_ceil(2), h.div_ceil(2)));
        }
        let n = dims.len();
        let decode = if src.codec == Codec::None { 0 } else { src.h };
        let meta = Meta {
            w: src.w,
            h: src.h,
            desc: src.desc.clone(),
            sample,
            k,
            is_u8: src.is_u8(),
            bands: src.bands.clone(),
            band: src.band,
        };
        Pyramid {
            meta,
            total: decode + dims[1..].iter().map(|d| d.1).sum::<usize>(),
            dims,
            levels: (0..n).map(|_| RwLock::new(None)).collect(),
            versions: (0..n).map(|_| AtomicU32::new(0)).collect(),
            cancel: AtomicBool::new(false),
            done: AtomicUsize::new(0),
            build_ms: AtomicU32::new(u32::MAX),
            src,
        }
    }

    fn level(&self, l: usize) -> Option<Arc<Vec<P>>> {
        self.levels[l].read().unwrap().clone()
    }

    fn set(&self, l: usize, d: Vec<P>) {
        *self.levels[l].write().unwrap() = Some(Arc::new(d));
        self.versions[l].fetch_add(1, Release);
    }

    fn mapped(&self) -> bool {
        self.src.codec == Codec::None
    }

    /// Level-0 pixels of row y from column x0. Only for uncompressed (mapped) sources.
    fn row0(&self, y: usize, x0: usize, out: &mut [P], tmp: &mut Vec<f32>) {
        let k = self.meta.k;
        self.src.spans(y, x0, out.len(), |s, r| match s {
            Some(s) => P::fill(&self.src, s, &mut out[r], k, tmp),
            None => out[r].fill(P::default()),
        });
    }

    fn stop(&self) -> bool {
        self.cancel.load(Relaxed)
    }

    fn decode_all(&self) -> Vec<P> {
        let (w, h) = self.dims[0];
        let (cw, ch, nx) = self.src.chunk_grid();
        let pb = self.src.px();
        let mut img = vec![P::default(); w * h];
        img.par_chunks_mut(ch * w).enumerate().for_each_init(Vec::new, |tmp, (cy, band)| {
            if self.stop() {
                return;
            }
            let rows = band.len() / w;
            for cx in 0..nx {
                let d = self.src.chunk(cy * nx + cx);
                let (x0, row) = (cx * cw, cw * pb);
                let n = cw.min(w - x0);
                for (r, o) in band.chunks_exact_mut(w).enumerate() {
                    P::fill(&self.src, &d[r * row..r * row + n * pb], &mut o[x0..x0 + n], self.meta.k, tmp);
                }
            }
            self.done.fetch_add(rows, Relaxed);
        });
        img
    }
}

impl<P: Px> Image for Pyramid<P> {
    fn meta(&self) -> &Meta {
        &self.meta
    }

    fn dims(&self) -> &[(usize, usize)] {
        &self.dims
    }

    fn ready(&self, l: usize) -> bool {
        (l == 0 && self.mapped()) || self.levels[l].read().unwrap().is_some()
    }

    fn version(&self, l: usize) -> u32 {
        self.versions[l].load(Acquire)
    }

    fn cancel(&self) {
        self.cancel.store(true, Relaxed);
    }

    fn progress(&self) -> (f32, Option<f32>) {
        let ms = self.build_ms.load(Relaxed);
        let p = self.done.load(Relaxed) as f32 / self.total.max(1) as f32;
        (p.min(1.0), (ms != u32::MAX).then(|| ms as f32 / 1000.0))
    }

    fn build(&self, wake: &dyn Fn()) {
        let t = Instant::now();
        let n = self.dims.len();
        if !self.mapped() {
            let img = self.decode_all();
            if self.stop() {
                return;
            }
            self.set(0, img);
            wake();
        } else if n > 1 {
            // Nearest-sampled preview of the top level: it shows in a few ms.
            let top = n - 1;
            let ((tw, th), (w, h), s) = (self.dims[top], self.dims[0], 1usize << top);
            let mut prev = vec![P::default(); tw * th];
            prev.par_chunks_mut(tw).enumerate().for_each_init(
                || (vec![P::default(); w], Vec::new()),
                |(row, tmp), (y, o)| {
                    self.row0((y * s).min(h - 1), 0, row, tmp);
                    o.iter_mut().enumerate().for_each(|(x, p)| *p = row[(x * s).min(w - 1)]);
                },
            );
            self.set(top, prev);
            wake();
        }
        for l in 1..n {
            let ((pw, ph), (lw, lh)) = (self.dims[l - 1], self.dims[l]);
            let mut out = vec![P::default(); lw * lh];
            let prev = self.level(l - 1);
            match &prev {
                Some(p) => out.par_chunks_mut(lw).enumerate().for_each(|(y, o)| {
                    let (a, b) = (2 * y * pw, (2 * y + 1).min(ph - 1) * pw);
                    P::half_row(&p[a..a + pw], &p[b..b + pw], o);
                    self.done.fetch_add(1, Relaxed);
                }),
                None => out.par_chunks_mut(lw).enumerate().for_each_init(
                    || (vec![P::default(); pw], vec![P::default(); pw], Vec::new()),
                    |(a, b, tmp), (y, o)| {
                        if self.stop() {
                            return;
                        }
                        self.row0(2 * y, 0, a, tmp);
                        self.row0((2 * y + 1).min(ph - 1), 0, b, tmp);
                        P::half_row(a, b, o);
                        self.done.fetch_add(1, Relaxed);
                    },
                ),
            }
            if self.stop() {
                return;
            }
            self.set(l, out);
            wake();
        }
        self.build_ms.store(t.elapsed().as_millis() as u32, Relaxed);
        wake();
    }

    fn tile(&self, l: usize, tx: usize, ty: usize) -> Option<(TileBuf, usize, usize)> {
        let (lw, lh) = self.dims[l];
        let (x0, y0) = (tx * TILE, ty * TILE);
        let (w, h) = (TILE.min(lw - x0), TILE.min(lh - y0));
        let mut v = vec![P::default(); w * h];
        match self.level(l) {
            Some(d) => v.chunks_exact_mut(w).enumerate().for_each(|(y, o)| o.copy_from_slice(&d[(y0 + y) * lw + x0..][..w])),
            None if l == 0 && self.mapped() => {
                let mut tmp = Vec::new();
                v.chunks_exact_mut(w).enumerate().for_each(|(y, o)| self.row0(y0 + y, x0, o, &mut tmp));
            }
            None => return None,
        }
        Some((P::wrap(v), w, h))
    }

    fn value(&self, x: usize, y: usize) -> Option<f32> {
        let (w, h) = self.dims[0];
        if x >= w || y >= h {
            return None;
        }
        let v = match self.level(0) {
            Some(d) => d[y * w + x],
            None if self.mapped() => {
                let mut o = [P::default()];
                self.row0(y, x, &mut o, &mut Vec::new());
                o[0]
            }
            None => return None,
        };
        Some(v.get() / self.meta.k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_row_u8_and_f16_agree() {
        let a: Vec<u8> = (0..131).map(|i| (i * 7 % 251) as u8).collect();
        let b: Vec<u8> = (0..131).map(|i| (i * 13 % 241) as u8).collect();
        let mut o8 = vec![0u8; 66];
        u8::half_row(&a, &b, &mut o8);
        let (fa, fb): (Vec<f16>, Vec<f16>) =
            (a.iter().map(|&v| f16::from_f32(v as f32)).collect(), b.iter().map(|&v| f16::from_f32(v as f32)).collect());
        let mut o16 = vec![f16::ZERO; 66];
        f16::half_row(&fa, &fb, &mut o16);
        for i in 0..66 {
            let x0 = 2 * i;
            let x1 = (2 * i + 1).min(130);
            let m = if x0 == x1 {
                (a[x0] as f32 + b[x0] as f32) / 2.0
            } else {
                (a[x0] as f32 + a[x1] as f32 + b[x0] as f32 + b[x1] as f32) / 4.0
            };
            assert!((o8[i] as f32 - m).abs() <= 0.5, "u8 {i}");
            assert!((o16[i].to_f32() - m).abs() <= 0.25, "f16 {i}");
        }
    }
}

/// Check decoded values against files made by an external writer (tifffile).
/// Set TEST_IMAGES to a tab-separated list: path, w, h, then "x y value" per point.
#[test]
fn files_match_expected() {
    let Ok(list) = std::env::var("TEST_IMAGES") else { return };
    let dir = std::path::Path::new(&list).parent().unwrap();
    for line in std::fs::read_to_string(&list).unwrap().lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (name, band) = f[0].split_once('#').map_or((f[0], 0), |(n, b)| (n, b.parse().unwrap()));
        let img = open(dir.join(name).to_str().unwrap(), band).unwrap_or_else(|e| panic!("{}: {e}", f[0]));
        let small = img.meta().w * img.meta().h < 50_000_000;
        if small || !img.ready(0) {
            img.build(&|| {});
        }
        assert_eq!((img.meta().w, img.meta().h), (f[1].parse().unwrap(), f[2].parse().unwrap()), "{}", f[0]);
        for p in &f[3..] {
            let v: Vec<f32> = p.split(' ').map(|s| s.parse().unwrap()).collect();
            let got = img.value(v[0] as usize, v[1] as usize).unwrap();
            assert!((got - v[2]).abs() <= v[2].abs() * 2e-3 + 1e-6, "{} at {p}: got {got}", f[0]);
            let top = img.dims().len() - 1;
            if small {
                let (_, w, h) = img.tile(top, 0, 0).unwrap();
                assert_eq!((w, h), img.dims()[top]);
            }
        }
        println!("ok {} {}", f[0], img.meta().desc);
    }
}
