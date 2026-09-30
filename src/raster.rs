//! Chunked raster over a memory-mapped file. NITF 2.1 / NSIF and TIFF / BigTIFF.
use memmap2::Mmap;
use std::borrow::Cow;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Ty {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    F64,
}

impl Ty {
    pub fn size(self) -> usize {
        match self {
            Ty::U8 | Ty::I8 => 1,
            Ty::U16 | Ty::I16 => 2,
            Ty::U32 | Ty::I32 | Ty::F32 => 4,
            Ty::F64 => 8,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Codec {
    None,
    Lzw,
    Deflate,
    Zstd,
    PackBits,
}

/// How to get one value from a pixel.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Cx {
    Real, // component at `coff`
    Amp,  // |c0 + i c1|
    Phase,
}

/// Image as a grid of chunks (NITF blocks, TIFF strips or tiles). Each chunk holds rows of `cw`
/// pixels. A pixel holds `stride` components of type `ty`.
pub struct Raster {
    pub w: usize,
    pub h: usize,
    pub ty: Ty,
    pub codec: Codec,
    pub desc: String,
    /// Names of the values the user can select (bands, or complex parts).
    pub bands: Vec<String>,
    pub band: usize,
    cx: Cx,
    coff: usize, // byte offset of the selected component in a pixel
    stride: usize,
    be: bool,
    predictor: u16,
    cw: usize,
    ch: usize,
    nx: usize,
    crow: Vec<usize>,            // first image row of each chunk row
    chunks: Vec<(usize, usize)>, // (offset, length). Length 0 = missing chunk (zeros).
    map: Mmap,
}

fn trim(b: &[u8]) -> &str {
    std::str::from_utf8(b).unwrap_or("").trim()
}

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl Cur<'_> {
    fn str(&mut self, n: usize) -> Result<&str, String> {
        let s = self.b.get(self.p..self.p + n).ok_or("truncated header")?;
        self.p += n;
        Ok(trim(s))
    }
    fn num(&mut self, n: usize) -> Result<usize, String> {
        let at = self.p;
        let s = self.str(n)?;
        s.parse().map_err(|_| format!("bad number {s:?} at byte {at}"))
    }
    fn skip(&mut self, n: usize) {
        self.p += n;
    }
}

/// Selectable values and, for `band`, the value kind and the component index in a pixel.
fn band_names(complex: bool, n: usize) -> Vec<String> {
    if complex {
        ["Amplitude", "Phase", "I (real)", "Q (imaginary)"].map(String::from).to_vec()
    } else {
        (1..=n).map(|i| format!("Band {i}")).collect()
    }
}

fn select(complex: bool, band: usize) -> (Cx, usize) {
    match (complex, band) {
        (true, 0) => (Cx::Amp, 0),
        (true, 1) => (Cx::Phase, 0),
        (true, b) => (Cx::Real, b - 2),
        (false, b) => (Cx::Real, b),
    }
}

/// One NITF image segment subheader.
struct Seg {
    rows: usize,
    cols: usize,
    pvtype: String,
    nbpp: usize,
    nbands: usize,
    iq: bool,
    imode: String,
    ic: String,
    bpr: usize,
    bpc: usize,
    bw: usize,
    bh: usize,
    idlvl: usize,
    ialvl: usize,
    iloc_row: usize,
    data: usize,
}

fn nitf_seg(b: &[u8], sub: usize, data: usize) -> Result<Seg, String> {
    let mut c = Cur { b, p: sub + 333 };
    let (rows, cols) = (c.num(8)?, c.num(8)?);
    let pvtype = c.str(3)?.to_string();
    c.skip(8 + 8 + 2 + 1); // IREP ICAT ABPP PJUST
    if !c.str(1)?.is_empty() {
        c.skip(60); // IGEOLO
    }
    let nicom = c.num(1)?;
    c.skip(80 * nicom);
    let ic = c.str(2)?.to_string();
    if ic != "NC" && ic != "NM" {
        return Err(format!("NITF compression {ic} not supported (NC, NM only)"));
    }
    let mut nbands = c.num(1)?;
    if nbands == 0 {
        nbands = c.num(5)?;
    }
    let mut subcat = Vec::new();
    for _ in 0..nbands {
        c.skip(2);
        subcat.push(c.str(6)?.to_string());
        c.skip(4);
        let nluts = c.num(1)?;
        if nluts > 0 {
            let nelut = c.num(5)?;
            c.skip(nluts * nelut);
        }
    }
    c.skip(1); // ISYNC
    let imode = c.str(1)?.to_string();
    let (bpr, bpc, bw, bh, nbpp) = (c.num(4)?, c.num(4)?, c.num(4)?, c.num(4)?, c.num(2)?);
    let (idlvl, ialvl) = (c.num(3)?, c.num(3)?);
    let iloc_row = c.num(5)?;
    let iq = nbands == 2 && subcat[0] == "I" && subcat[1] == "Q";
    let (bw, bh) = (if bw == 0 { cols } else { bw }, if bh == 0 { rows } else { bh });
    Ok(Seg { rows, cols, pvtype, nbpp, nbands, iq, imode, ic, bpr, bpc, bw, bh, idlvl, ialvl, iloc_row, data })
}

impl Raster {
    pub fn open(path: &str, band: usize) -> Result<Self, String> {
        let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        // SAFETY: read-only map. If another process truncates the file, access fails with SIGBUS.
        let map = unsafe { Mmap::map(&f) }.map_err(|e| e.to_string())?;
        let r = match map.get(..4) {
            Some(b"NITF") | Some(b"NSIF") => Self::nitf(map, band)?,
            Some(b"II*\0") | Some(b"MM\0*") | Some(b"II+\0") | Some(b"MM\0+") => Self::tiff(map, band)?,
            _ => return Err("unknown format (NITF or TIFF expected)".into()),
        };
        r.validate()?;
        #[cfg(unix)]
        let _ = r.map.advise(memmap2::Advice::Random);
        Ok(r)
    }

    fn validate(&self) -> Result<(), String> {
        if self.w == 0 || self.h == 0 {
            return Err("empty image".into());
        }
        let need = self.nx * self.crow.len();
        if self.chunks.len() < need {
            return Err(format!("{} chunks, {need} expected", self.chunks.len()));
        }
        if let Some(&(o, l)) = self.chunks.iter().find(|&&(o, l)| o + l > self.map.len()) {
            return Err(format!("chunk at {o} (+{l}) is after end of file"));
        }
        Ok(())
    }

    fn nitf(map: Mmap, band: usize) -> Result<Self, String> {
        let b = &map[..];
        let mut c = Cur { b, p: 354 };
        let (hl, numi) = (c.num(6)?, c.num(3)?);
        if numi == 0 {
            return Err("NITF has no image segment".into());
        }
        let mut segs = Vec::new();
        let mut off = hl;
        for _ in 0..numi {
            let (lish, li) = (c.num(6)?, c.num(10)?);
            segs.push(nitf_seg(b, off, off + lish));
            off += lish + li;
        }
        let s0 = segs.remove(0)?;
        // Large images (SICD, SIDD > 10 GB) continue in the next segments, attached below.
        let mut used = vec![s0];
        for s in segs {
            let Ok(s) = s else { break };
            let (p, f) = (used.last().unwrap(), &used[0]);
            let same = s.cols == f.cols && s.pvtype == f.pvtype && s.nbpp == f.nbpp && s.nbands == f.nbands;
            let same = same && s.imode == f.imode && s.bw == f.bw && s.bpr == f.bpr;
            let rows: usize = used.iter().map(|u| u.rows).sum();
            let below = (s.ialvl == p.idlvl && s.iloc_row == p.rows) || (s.ialvl == f.idlvl && s.iloc_row == rows);
            if !(same && below) {
                break;
            }
            used.push(s);
        }
        let f = &used[0];
        let (pv, nbands) = (f.pvtype.as_str(), f.nbands);
        let comps = if pv == "C" { 2 } else { 1 };
        let complex = pv == "C" || (f.iq && f.imode == "P");
        let comp_bits = f.nbpp / comps;
        let ty = match (pv, comp_bits) {
            ("INT", 8) => Ty::U8,
            ("INT", 16) => Ty::U16,
            ("INT", 32) => Ty::U32,
            ("SI", 8) => Ty::I8,
            ("SI", 16) => Ty::I16,
            ("SI", 32) => Ty::I32,
            ("R" | "C", 32) => Ty::F32,
            ("R", 64) => Ty::F64,
            _ => return Err(format!("NITF pixel type {pv} {}-bit not supported", f.nbpp)),
        };
        let bands = band_names(complex, if complex { 1 } else { nbands });
        let band = band.min(bands.len() - 1);
        let (cx, comp) = select(complex, band);
        let s = ty.size();
        // Component stride in a pixel, band offset in a block, block step, block index step for band.
        let (stride, coff) = match f.imode.as_str() {
            _ if nbands == 1 => (comps, comp * s),
            "P" => (nbands * comps, comp * comps * s),
            "B" | "S" if !complex => (comps, 0),
            m => return Err(format!("NITF IMODE {m} with {nbands} bands not supported")),
        };
        let (mut chunks, mut crow, mut y0) = (Vec::new(), Vec::new(), 0);
        for g in &used {
            let nb = g.bpr * g.bpc;
            let bb = g.bw * g.bh * s * comps; // one band of one block
            let (step, boff, first) = match g.imode.as_str() {
                "B" if nbands > 1 => (bb * nbands, band * bb, 0),
                "S" if nbands > 1 => (bb, 0, band * nb),
                "P" => (bb * nbands, 0, 0),
                _ => (bb, 0, 0),
            };
            let len = g.bw * g.bh * s * stride;
            if g.ic == "NM" {
                let be32 = |p: usize| b.get(p..p + 4).map(|x| u32::from_be_bytes(x.try_into().unwrap()) as usize);
                let be16 = |p: usize| b.get(p..p + 2).map(|x| u16::from_be_bytes(x.try_into().unwrap()) as usize);
                let bad = || "truncated NITF mask table".to_string();
                let imdatoff = be32(g.data).ok_or_else(bad)?;
                let bmrlnth = be16(g.data + 4).ok_or_else(bad)?;
                let bmr = g.data + 10 + be16(g.data + 8).ok_or_else(bad)?.div_ceil(8);
                for i in 0..nb {
                    let o = if bmrlnth == 4 { be32(bmr + 4 * (first + i)).ok_or_else(bad)? } else { (first + i) * step };
                    chunks.push(if o == 0xFFFF_FFFF { (0, 0) } else { (g.data + imdatoff + o + boff, len) });
                }
            } else {
                chunks.extend((0..nb).map(|i| (g.data + (first + i) * step + boff, len)));
            }
            crow.extend((0..g.bpc).map(|by| y0 + by * g.bh));
            y0 += g.rows;
        }
        let segs = if used.len() > 1 { format!(", {} segments", used.len()) } else { String::new() };
        let desc = format!("NITF {pv} {}-bit, {nbands} band(s), {}{segs}", f.nbpp, f.ic);
        Ok(Raster {
            w: f.cols,
            h: y0,
            ty,
            codec: Codec::None,
            desc,
            bands,
            band,
            cx,
            coff,
            stride,
            be: true,
            predictor: 1,
            cw: f.bw,
            ch: f.bh,
            nx: f.bpr,
            crow,
            chunks,
            map,
        })
    }

    fn tiff(map: Mmap, band: usize) -> Result<Self, String> {
        let b = &map[..];
        let le = b[0] == b'I';
        let big = b[2] == 43 || b[3] == 43;
        let rd = |p: usize, n: usize| -> Result<u64, String> {
            let s = b.get(p..p + n).ok_or("truncated TIFF")?;
            let mut v = 0u64;
            for i in 0..n {
                v |= (s[if le { i } else { n - 1 - i }] as u64) << (8 * i);
            }
            Ok(v)
        };
        let ifd = if big { rd(8, 8)? } else { rd(4, 4)? } as usize;
        let (nent, esz, hdr) = if big { (rd(ifd, 8)? as usize, 20, 8) } else { (rd(ifd, 2)? as usize, 12, 2) };
        let mut tags = std::collections::HashMap::new();
        for e in 0..nent {
            let p = ifd + hdr + e * esz;
            let (tag, typ) = (rd(p, 2)? as u16, rd(p + 2, 2)?);
            let (count, vp) = if big { (rd(p + 4, 8)? as usize, p + 12) } else { (rd(p + 4, 4)? as usize, p + 8) };
            let sz = match typ {
                1 => 1,
                3 => 2,
                4 => 4,
                16 => 8,
                _ => continue, // other types are not used here
            };
            let inline = if big { 8 } else { 4 };
            let at = if sz * count <= inline { vp } else { rd(vp, inline)? as usize };
            let v: Result<Vec<u64>, String> = (0..count).map(|i| rd(at + i * sz, sz)).collect();
            tags.insert(tag, v?);
        }
        let one = |t: u16, d: u64| tags.get(&t).and_then(|v| v.first().copied()).unwrap_or(d) as usize;
        let (w, h) = (one(256, 0), one(257, 0));
        let (bps, spp, sfmt) = (one(258, 1), one(277, 1), one(339, 1));
        let planar = one(284, 1);
        let codec = match one(259, 1) {
            1 => Codec::None,
            5 => Codec::Lzw,
            8 | 32946 => Codec::Deflate,
            50000 => Codec::Zstd,
            32773 => Codec::PackBits,
            c => return Err(format!("TIFF compression {c} not supported (none, LZW, Deflate, Zstd, PackBits)")),
        };
        let complex = sfmt == 5 || sfmt == 6;
        let ty = match (sfmt, if complex { bps / 2 } else { bps }) {
            (1, 8) => Ty::U8,
            (2, 8) => Ty::I8,
            (1, 16) => Ty::U16,
            (2 | 5, 16) => Ty::I16,
            (1, 32) => Ty::U32,
            (2 | 5, 32) => Ty::I32,
            (3 | 6, 32) => Ty::F32,
            (3 | 6, 64) => Ty::F64,
            _ => return Err(format!("TIFF sample format {sfmt} {bps}-bit not supported")),
        };
        let comps = if complex { 2 } else { 1 };
        let (cw, ch, offs, lens) = if tags.contains_key(&322) {
            (one(322, 0), one(323, 0), 324, 325)
        } else {
            (w, one(278, h as u64).min(h), 273, 279)
        };
        let (cw, ch) = (cw.max(1), ch.max(1));
        let (offs, lens) = (tags.get(&offs).ok_or("TIFF without chunk offsets")?, tags.get(&lens).ok_or("TIFF without chunk sizes")?);
        let mut chunks: Vec<(usize, usize)> = offs.iter().zip(lens).map(|(&o, &l)| (o as usize, l as usize)).collect();
        let bands = band_names(complex, if complex { 1 } else { spp });
        let band = band.min(bands.len() - 1);
        let (cx, comp) = select(complex, band);
        let (stride, coff) = if planar == 2 && spp > 1 {
            let per = chunks.len() / spp;
            chunks = chunks[if complex { 0 } else { band * per }..][..per].to_vec();
            (comps, if complex { comp * ty.size() } else { 0 })
        } else {
            (spp * comps, comp * ty.size())
        };
        let desc = format!(
            "{}TIFF {}{:?}, {spp} sample(s), {codec:?}",
            if big { "Big" } else { "" },
            if complex { "complex " } else { "" },
            ty
        );
        Ok(Raster {
            w,
            h,
            ty,
            codec,
            desc,
            bands,
            band,
            cx,
            coff,
            stride,
            be: !le,
            predictor: one(317, 1) as u16,
            cw,
            ch,
            nx: w.div_ceil(cw),
            crow: (0..h.div_ceil(ch)).map(|i| i * ch).collect(),
            chunks,
            map,
        })
    }

    fn px_bytes(&self) -> usize {
        self.ty.size() * self.stride
    }

    /// Decoded bytes of chunk `i`, always of the exact expected size.
    pub fn chunk(&self, i: usize) -> Cow<'_, [u8]> {
        let row = self.cw * self.px_bytes();
        let need = row * self.ch;
        let (o, l) = self.chunks[i];
        let src = &self.map[o..o + l];
        if self.codec == Codec::None && l >= need {
            return Cow::Borrowed(&src[..need]);
        }
        let mut out = vec![0u8; need];
        let res = match self.codec {
            Codec::None => {
                out[..l.min(need)].copy_from_slice(&src[..l.min(need)]);
                Ok(())
            }
            Codec::Deflate => libdeflater::Decompressor::new().zlib_decompress(src, &mut out).map(drop).map_err(|e| e.to_string()),
            Codec::Zstd => zstd::bulk::decompress_to_buffer(src, &mut out).map(drop).map_err(|e| e.to_string()),
            Codec::Lzw => {
                let mut d = weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
                let (mut a, mut b) = (0, 0);
                loop {
                    let r = d.decode_bytes(&src[a..], &mut out[b..]);
                    (a, b) = (a + r.consumed_in, b + r.consumed_out);
                    match r.status {
                        Ok(weezl::LzwStatus::Ok) if b < need && r.consumed_in + r.consumed_out > 0 => continue,
                        s => break s.map(drop).map_err(|e| e.to_string()),
                    }
                }
            }
            Codec::PackBits => {
                packbits(src, &mut out);
                Ok(())
            }
        };
        if let Err(e) = res {
            eprintln!("chunk {i}: {e}");
        }
        if l > 0 && self.predictor > 1 {
            for r in out.chunks_exact_mut(row) {
                self.unpredict(r);
            }
        }
        Cow::Owned(out)
    }

    fn unpredict(&self, row: &mut [u8]) {
        let (s, n) = (self.ty.size(), self.stride);
        match (self.predictor, s) {
            (2, 1) => (n..row.len()).for_each(|i| row[i] = row[i].wrapping_add(row[i - n])),
            (2, 2) => {
                let (g, p): (fn([u8; 2]) -> u16, fn(u16) -> [u8; 2]) =
                    if self.be { (u16::from_be_bytes, u16::to_be_bytes) } else { (u16::from_le_bytes, u16::to_le_bytes) };
                for i in n..row.len() / 2 {
                    let v = g([row[2 * i], row[2 * i + 1]]).wrapping_add(g([row[2 * i - 2 * n], row[2 * i - 2 * n + 1]]));
                    row[2 * i..2 * i + 2].copy_from_slice(&p(v));
                }
            }
            (2, 4) => {
                let (g, p): (fn([u8; 4]) -> u32, fn(u32) -> [u8; 4]) =
                    if self.be { (u32::from_be_bytes, u32::to_be_bytes) } else { (u32::from_le_bytes, u32::to_le_bytes) };
                for i in n..row.len() / 4 {
                    let a = g(row[4 * i..4 * i + 4].try_into().unwrap());
                    let b = g(row[4 * (i - n)..4 * (i - n) + 4].try_into().unwrap());
                    row[4 * i..4 * i + 4].copy_from_slice(&p(a.wrapping_add(b)));
                }
            }
            (3, _) => {
                // Floating point predictor: byte differences, then byte planes (MSB plane first).
                (n..row.len()).for_each(|i| row[i] = row[i].wrapping_add(row[i - n]));
                let tmp = row.to_vec();
                let wc = row.len() / s;
                for j in 0..wc {
                    for k in 0..s {
                        let plane = if self.be { k } else { s - 1 - k };
                        row[j * s + k] = tmp[plane * wc + j];
                    }
                }
            }
            _ => {}
        }
    }

    /// Convert `out.len()` raw pixels to f32 values.
    pub fn to_f32(&self, src: &[u8], out: &mut [f32]) {
        macro_rules! go {
            ($t:ty) => {{
                const N: usize = std::mem::size_of::<$t>();
                let get: fn(&[u8]) -> f32 = if self.be {
                    |p| <$t>::from_be_bytes(p[..N].try_into().unwrap()) as f32
                } else {
                    |p| <$t>::from_le_bytes(p[..N].try_into().unwrap()) as f32
                };
                let it = out.iter_mut().zip(src.chunks_exact(N * self.stride));
                match self.cx {
                    Cx::Real => {
                        let c = self.coff;
                        it.for_each(|(o, p)| *o = get(&p[c..]))
                    }
                    Cx::Amp => it.for_each(|(o, p)| {
                        let (a, b) = (get(p), get(&p[N..]));
                        *o = (a * a + b * b).sqrt()
                    }),
                    Cx::Phase => it.for_each(|(o, p)| *o = get(&p[N..]).atan2(get(p))),
                }
            }};
        }
        match self.ty {
            Ty::U8 => go!(u8),
            Ty::I8 => go!(i8),
            Ty::U16 => go!(u16),
            Ty::I16 => go!(i16),
            Ty::U32 => go!(u32),
            Ty::I32 => go!(i32),
            Ty::F32 => go!(f32),
            Ty::F64 => go!(f64),
        }
    }

    /// Copy raw 8-bit values of the selected component.
    pub fn to_u8(&self, src: &[u8], out: &mut [u8]) {
        if self.stride == 1 {
            out.copy_from_slice(&src[..out.len()]);
        } else {
            out.iter_mut().zip(src[self.coff..].iter().step_by(self.stride)).for_each(|(o, &v)| *o = v);
        }
    }

    pub fn is_u8(&self) -> bool {
        self.ty == Ty::U8 && self.cx == Cx::Real
    }

    /// Call `f(raw bytes, out range)` for each chunk span of row `y`, columns `x0..x0 + n`.
    /// Uncompressed data only. A missing chunk gives `None`.
    pub fn spans(&self, y: usize, x0: usize, n: usize, mut f: impl FnMut(Option<&[u8]>, std::ops::Range<usize>)) {
        let pb = self.px_bytes();
        let cy = self.crow.partition_point(|&s| s <= y) - 1;
        let ry = y - self.crow[cy];
        let (mut x, mut o) = (x0, 0);
        while o < n {
            let (cx, rx) = (x / self.cw, x % self.cw);
            let k = (self.cw - rx).min(n - o);
            let (off, len) = self.chunks[cy * self.nx + cx];
            let p = off + (ry * self.cw + rx) * pb;
            f((len > 0).then(|| &self.map[p..p + k * pb]), o..o + k);
            o += k;
            x += k;
        }
    }

    /// Chunk width, chunk height, chunks per row. Uniform grid (TIFF) only.
    pub fn chunk_grid(&self) -> (usize, usize, usize) {
        (self.cw, self.ch, self.nx)
    }

    pub fn px(&self) -> usize {
        self.px_bytes()
    }

    /// Sparse sample of values (finite, non-zero), sorted. Used for the stretch and the f16 range.
    pub fn sample(&self) -> Vec<f32> {
        let mut v = Vec::new();
        let pb = self.px_bytes();
        if self.codec == Codec::None {
            // Read spaced pixels only: few pages per row.
            let (rows, cols) = (256.min(self.h), 1024.min(self.w));
            let mut o = [0f32];
            for i in 0..rows {
                let y = i * self.h / rows;
                for j in 0..cols {
                    self.spans(y, j * self.w / cols, 1, |s, _| {
                        if let Some(s) = s {
                            self.to_f32(&s[..pb], &mut o);
                            v.push(o[0]);
                        }
                    });
                }
            }
        } else {
            let n = self.chunks.len();
            let k = 24.min(n);
            let row = self.cw * pb;
            let mut cbuf = vec![0f32; self.cw];
            for j in 0..k {
                let d = self.chunk(j * n / k);
                for r in d.chunks_exact(row).step_by(4) {
                    self.to_f32(r, &mut cbuf);
                    v.extend(cbuf.iter().step_by(4).copied());
                }
            }
        }
        v.retain(|x| x.is_finite() && *x != 0.0);
        v.sort_unstable_by(f32::total_cmp);
        v
    }
}

fn packbits(src: &[u8], out: &mut [u8]) {
    let (mut i, mut o) = (0, 0);
    while i < src.len() && o < out.len() {
        let n = src[i] as i8;
        i += 1;
        if n >= 0 {
            let k = (n as usize + 1).min(src.len() - i).min(out.len() - o);
            out[o..o + k].copy_from_slice(&src[i..i + k]);
            i += n as usize + 1;
            o += k;
        } else if n != -128 && i < src.len() {
            let k = (1 - n as isize) as usize;
            let k = k.min(out.len() - o);
            out[o..o + k].fill(src[i]);
            i += 1;
            o += k;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packbits_decodes() {
        let src = [0xFEu8, 0xAA, 0x02, 0x80, 0x00, 0x2A];
        let mut out = [0u8; 6];
        packbits(&src, &mut out);
        assert_eq!(out, [0xAA, 0xAA, 0xAA, 0x80, 0x00, 0x2A]);
    }
}
