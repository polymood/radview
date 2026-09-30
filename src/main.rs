#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]
mod pyramid;
mod raster;

use egui::{Color32, Key, Rect, Sense};
use pyramid::{Image, TILE};
use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc};
use winit::application::ApplicationHandler;
use winit::event::{StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

const APP: &str = "radview";

const SHADER: &str = r#"
struct U {
    center: vec2f, scale: f32, vmul: f32,
    view: vec2f, lo: f32, hi: f32,
    gamma: f32, flags: u32, p0: f32, p1: f32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var tiles: texture_2d_array<f32>;
@group(0) @binding(3) var lut: texture_2d<f32>;

struct VO {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) @interpolate(flat) uvmax: vec2f,
    @location(2) @interpolate(flat) layer: u32,
};

@vertex fn vs(@builtin(vertex_index) i: u32, @location(0) rect: vec4f, @location(1) uvl: vec4f) -> VO {
    let t = vec2f(f32(i & 1u), f32(i >> 1u));
    let ndc = (mix(rect.xy, rect.zw, t) - u.center) * u.scale / (u.view * 0.5);
    return VO(vec4f(ndc.x, -ndc.y, 0.0, 1.0), t * uvl.xy, uvl.xy - vec2f(0.5 / 512.0), u32(uvl.z));
}

@fragment fn fs(v: VO) -> @location(0) vec4f {
    let raw = textureSample(tiles, smp, min(v.uv, v.uvmax), v.layer).r * u.vmul;
    // 20 log10(x) = 6.0206 log2(x)
    let x = select(raw, 6.0206 * log2(max(abs(raw), 1e-10)), (u.flags & 1u) != 0u);
    var t = pow(clamp((x - u.lo) / (u.hi - u.lo), 0.0, 1.0), u.gamma);
    t = select(t, 1.0 - t, (u.flags & 2u) != 0u);
    let c = textureSampleLevel(lut, smp, vec2f(t * (255.0 / 256.0) + 0.5 / 256.0, 0.5), 0.0);
    if ((u.flags & 4u) != 0u && raw == 0.0) { discard; }
    return vec4f(c.rgb, 1.0);
}
"#;

const CMAPS: &[(&str, &[u32])] = &[
    ("Gray", &[0x000000, 0xFFFFFF]),
    ("Viridis", &[0x440154, 0x482878, 0x3E4A89, 0x31688E, 0x26828E, 0x1F9E89, 0x35B779, 0x6DCD59, 0xB4DE2C, 0xFDE725]),
    ("Magma", &[0x000004, 0x180F3D, 0x440F76, 0x721F81, 0x9E2F7F, 0xCD4071, 0xF1605D, 0xFD9668, 0xFECA8D, 0xFCFDBF]),
    ("Inferno", &[0x000004, 0x1B0C41, 0x4A0C6B, 0x781C6D, 0xA52C60, 0xCF4446, 0xED6925, 0xFB9B06, 0xF7D13D, 0xFCFFA4]),
    ("Plasma", &[0x0D0887, 0x46039F, 0x7201A8, 0x9C179E, 0xBD3786, 0xD8576B, 0xED7953, 0xFB9F3A, 0xFDCA26, 0xF0F921]),
    ("Cividis", &[0x00224E, 0x123570, 0x3B496C, 0x575D6D, 0x707173, 0x8A8779, 0xA69D75, 0xC4B56C, 0xE4CF5B, 0xFEE838]),
    ("Turbo", &[0x30123B, 0x4662D7, 0x36AAF9, 0x1AE4B6, 0x72FE5E, 0xC8EF34, 0xFABA39, 0xF66B19, 0xCA2A04, 0x7A0403]),
    ("Jet", &[0x00007F, 0x0000FF, 0x007FFF, 0x00FFFF, 0x7FFF7F, 0xFFFF00, 0xFF7F00, 0xFF0000, 0x7F0000]),
    ("Hot", &[0x000000, 0xE60000, 0xFFD200, 0xFFFFFF]),
    ("Sepia", &[0x000000, 0x4A2E14, 0xA67C52, 0xF5E6C8]),
];

// ponytail: fixed VRAM budget for the tile array, LRU-evicted.
const VRAM: usize = 256 << 20;
const MAX_DRAWS: usize = 8192;
const MAX_PENDING: usize = 96;
const UPLOADS_PER_FRAME: usize = 48;

type TileKey = (usize, usize, usize); // (level, tx, ty)

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    center: [f32; 2],
    scale: f32,
    vmul: f32,
    view: [f32; 2],
    lo: f32,
    hi: f32,
    gamma: f32,
    flags: u32,
    pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Inst {
    rect: [f32; 4],
    uvl: [f32; 4], // uv max x, uv max y, layer, unused
}

enum Ev {
    Wake,
    Opened(u64, Result<Arc<dyn Image>, String>),
}

struct Loaded {
    id: u64,
    key: TileKey,
    version: u32,
    buf: pyramid::TileBuf,
    w: usize,
    h: usize,
}

struct Slot {
    layer: u32,
    version: u32,
    used: u64,
}

/// GPU tile cache: one texture array, one layer per tile.
#[derive(Default)]
struct Tiles {
    map: HashMap<TileKey, Slot>,
    free: Vec<u32>,
    pending: HashSet<TileKey>,
}

impl Tiles {
    fn reset(&mut self, layers: u32) {
        self.map.clear();
        self.pending.clear();
        self.free = (0..layers).rev().collect();
    }

    /// Layer for `key`. If the array is full, evict the least recently used tile.
    fn alloc(&mut self, key: TileKey, version: u32, frame: u64) -> Option<u32> {
        if let Some(s) = self.map.get_mut(&key) {
            s.version = version;
            return Some(s.layer);
        }
        let layer = match self.free.pop() {
            Some(l) => l,
            None => {
                let (&k, s) = self.map.iter().filter(|(_, s)| s.used < frame).min_by_key(|(_, s)| s.used)?;
                let l = s.layer;
                self.map.remove(&k);
                l
            }
        };
        self.map.insert(key, Slot { layer, version, used: frame });
        Some(layer)
    }
}

struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    ubuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    sampler: wgpu::Sampler,
    lut: wgpu::Texture,
    array: Option<(wgpu::Texture, wgpu::BindGroup, u32)>, // texture, bind group, bytes per texel
    max_layers: u32,
    name: String,
    egui: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
}

struct Stretch {
    lo: f32,
    hi: f32,
    gamma: f32,
    db: bool,
    clip: f32,
    invert: bool,
    nodata: bool,
}

struct App {
    proxy: EventLoopProxy<Ev>,
    gpu: Option<Gpu>,
    ctx: egui::Context,
    img: Option<Arc<dyn Image>>,
    id: u64,
    path: String,
    opening: bool,
    error: Option<String>,
    tiles: Tiles,
    tx: mpsc::Sender<Loaded>,
    rx: mpsc::Receiver<Loaded>,
    pool: rayon::ThreadPool,
    frame: u64,
    center: [f64; 2],
    scale: f64,
    view: Rect, // image area, physical pixels
    fit: bool,
    st: Stretch,
    cmap: usize,
    stops: Vec<[u8; 3]>,
    lut_dirty: bool,
    panel: bool,
    path_edit: String,
    cursor: Option<(usize, usize, Option<f32>)>,
    dialog: bool,
    keep_view: bool,
}

fn hex(c: u32) -> [u8; 3] {
    [(c >> 16) as u8, (c >> 8) as u8, c as u8]
}

fn lut(stops: &[[u8; 3]]) -> Vec<[u8; 4]> {
    let n = stops.len() - 1;
    (0..256)
        .map(|i| {
            let t = i as f32 / 255.0 * n as f32;
            let j = (t as usize).min(n - 1);
            let f = t - j as f32;
            let (a, b) = (stops[j], stops[j + 1]);
            let m = |k: usize| (a[k] as f32 + (b[k] as f32 - a[k] as f32) * f).round() as u8;
            [m(0), m(1), m(2), 255]
        })
        .collect()
}

fn db(v: f32) -> f32 {
    20.0 * v.abs().max(1e-10).log10()
}

impl App {
    fn open(&mut self, path: String) {
        self.open_band(path, 0);
    }

    fn open_band(&mut self, path: String, band: usize) {
        self.keep_view = self.img.is_some() && path == self.path;
        if let Some(i) = self.img.take() {
            i.cancel();
        }
        self.id += 1;
        self.opening = true;
        self.error = None;
        self.cursor = None;
        self.path_edit = path.clone();
        self.path = path.clone();
        let (id, proxy) = (self.id, self.proxy.clone());
        std::thread::spawn(move || {
            let r = pyramid::open(&path, band);
            let _ = proxy.send_event(Ev::Opened(id, r));
        });
    }

    fn opened(&mut self, img: Arc<dyn Image>) {
        self.opening = false;
        let (proxy, b) = (self.proxy.clone(), img.clone());
        std::thread::spawn(move || b.build(&|| drop(proxy.send_event(Ev::Wake))));
        let gpu = self.gpu.as_mut().unwrap();
        let bpp = if img.meta().is_u8 { 1 } else { 2 };
        if gpu.array.as_ref().is_none_or(|a| a.2 != bpp) {
            let layers = gpu.max_layers.min((VRAM / (TILE * TILE * bpp as usize)) as u32);
            let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("tiles"),
                size: wgpu::Extent3d { width: TILE as u32, height: TILE as u32, depth_or_array_layers: layers },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: if bpp == 1 { wgpu::TextureFormat::R8Unorm } else { wgpu::TextureFormat::R16Float },
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let tv = tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
            let lv = gpu.lut.create_view(&Default::default());
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &gpu.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: gpu.ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&gpu.sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&tv) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&lv) },
                ],
            });
            gpu.array = Some((tex, bg, bpp));
        }
        let layers = gpu.array.as_ref().unwrap().0.depth_or_array_layers();
        self.tiles.reset(layers);
        let name = std::path::Path::new(&self.path).file_name().map_or(self.path.clone(), |n| n.to_string_lossy().into());
        gpu.window.set_title(&format!("{APP} - {name}"));
        let m = img.meta();
        if m.bands.len() == 4 && m.bands[0] == "Amplitude" {
            self.st.db = m.band == 0; // complex data: amplitude in dB, other parts linear
        }
        self.img = Some(img);
        self.auto();
        self.fit = !self.keep_view;
    }

    /// Stretch limits from the sample percentiles.
    fn auto(&mut self) {
        let Some(img) = &self.img else { return };
        let s = &img.meta().sample;
        if s.is_empty() {
            (self.st.lo, self.st.hi) = (0.0, 1.0);
            return;
        }
        let mut t;
        let s = if self.st.db {
            t = s.iter().map(|&v| db(v)).collect::<Vec<f32>>();
            t.sort_unstable_by(f32::total_cmp);
            &t
        } else {
            s
        };
        let q = |p: f32| s[((s.len() - 1) as f32 * p) as usize];
        let c = self.st.clip / 100.0;
        (self.st.lo, self.st.hi) = (q(c), q(1.0 - c));
        if self.st.hi <= self.st.lo {
            self.st.hi = self.st.lo + 1.0;
        }
    }

    fn fit_view(&mut self) {
        let Some(img) = &self.img else { return };
        let m = img.meta();
        self.center = [m.w as f64 / 2.0, m.h as f64 / 2.0];
        self.scale = (self.view.width() as f64 / m.w as f64).min(self.view.height() as f64 / m.h as f64);
    }

    fn set_cmap(&mut self, i: usize) {
        self.cmap = i;
        self.stops = CMAPS[i].1.iter().map(|&c| hex(c)).collect();
        self.lut_dirty = true;
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        if let Some(p) = ctx.input(|i| i.raw.dropped_files.first().map(|f| f.path().to_path_buf())) {
            self.open(p.to_string_lossy().into());
        }
        if !ctx.egui_wants_keyboard_input() {
            let (o, f, one, h, i, c) = ctx.input(|i| {
                let k = |key| i.key_pressed(key);
                (i.modifiers.command && k(Key::O), k(Key::F), k(Key::Num1), k(Key::H), k(Key::I), k(Key::C))
            });
            self.dialog |= o;
            self.fit |= f;
            self.panel ^= h;
            self.st.invert ^= i;
            if one {
                self.scale = 1.0;
            }
            if c {
                self.set_cmap((self.cmap + 1) % CMAPS.len());
            }
        }
        if self.panel {
            egui::Panel::left("side").resizable(true).default_size(270.0).show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.side(ui));
            });
        }
        egui::CentralPanel::no_frame().show(ui, |ui| self.canvas(ui));
    }

    fn side(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.heading(APP);
            if ui.button("Open...").on_hover_text("Ctrl+O").clicked() {
                self.dialog = true;
            }
        });
        ui.horizontal(|ui| {
            let r = ui.add(egui::TextEdit::singleline(&mut self.path_edit).hint_text("file path").desired_width(190.0));
            if ui.button("Go").clicked() || (r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter))) {
                let p = self.path_edit.trim().trim_matches('"').to_string();
                self.open(p);
            }
        });
        if let Some(e) = &self.error {
            ui.colored_label(Color32::from_rgb(255, 110, 110), e);
        }
        if self.opening {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Opening...");
            });
        }
        if let Some(img) = self.img.clone() {
            let m = img.meta();
            ui.label(format!("{} x {}  |  {}", m.w, m.h, m.desc));
            if m.bands.len() > 1 {
                let mut b = m.band;
                egui::ComboBox::from_label("Band").selected_text(&m.bands[b]).show_ui(ui, |ui| {
                    for (i, n) in m.bands.iter().enumerate() {
                        ui.selectable_value(&mut b, i, n);
                    }
                });
                if b != m.band {
                    self.open_band(self.path.clone(), b);
                }
            }
            match img.progress() {
                (_, Some(t)) => ui.label(format!("Pyramid ready in {t:.2} s")),
                (p, None) => ui.add(egui::ProgressBar::new(p).text(format!("Pyramid {:.0} %", p * 100.0))),
            };
        }

        ui.separator();
        ui.strong("Color map");
        let before = self.cmap;
        egui::ComboBox::from_id_salt("cmap").selected_text(CMAPS[self.cmap].0).show_ui(ui, |ui| {
            for (i, (n, _)) in CMAPS.iter().enumerate() {
                ui.selectable_value(&mut self.cmap, i, *n);
            }
        });
        if self.cmap != before {
            self.set_cmap(self.cmap);
        }
        let (r, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 18.0), Sense::hover());
        let l = lut(&self.stops);
        let n = 128;
        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            let j = ((if self.st.invert { 1.0 - t } else { t }) * 255.0) as usize;
            let x0 = r.left() + r.width() * i as f32 / n as f32;
            let c = Color32::from_rgb(l[j][0], l[j][1], l[j][2]);
            ui.painter().rect_filled(Rect::from_x_y_ranges(x0..=x0 + r.width() / n as f32 + 0.5, r.y_range()), 0.0, c);
        }
        ui.horizontal_wrapped(|ui| {
            for s in self.stops.iter_mut() {
                self.lut_dirty |= ui.color_edit_button_srgb(s).changed();
            }
            if ui.small_button("+").on_hover_text("Add a stop").clicked() {
                self.stops.push(*self.stops.last().unwrap());
                self.lut_dirty = true;
            }
            if self.stops.len() > 2 && ui.small_button("-").on_hover_text("Remove the last stop").clicked() {
                self.stops.pop();
                self.lut_dirty = true;
            }
        });
        ui.checkbox(&mut self.st.invert, "Invert (I)");

        ui.separator();
        ui.strong("Stretch");
        ui.horizontal(|ui| {
            let d = self.st.db;
            ui.radio_value(&mut self.st.db, false, "Linear");
            ui.radio_value(&mut self.st.db, true, "dB (20 log10)");
            if d != self.st.db {
                self.auto();
            }
        });
        let speed = ((self.st.hi - self.st.lo).abs() / 300.0).max(1e-6) as f64;
        egui::Grid::new("st").num_columns(2).show(ui, |ui| {
            ui.label("Min");
            ui.add(egui::DragValue::new(&mut self.st.lo).speed(speed).max_decimals(4));
            ui.end_row();
            ui.label("Max");
            ui.add(egui::DragValue::new(&mut self.st.hi).speed(speed).max_decimals(4));
            ui.end_row();
            ui.label("Gamma");
            ui.add(egui::Slider::new(&mut self.st.gamma, 0.1..=5.0).logarithmic(true));
            ui.end_row();
            ui.label("Clip %");
            if ui.add(egui::Slider::new(&mut self.st.clip, 0.0..=10.0)).changed() {
                self.auto();
            }
            ui.end_row();
        });
        ui.horizontal(|ui| {
            if ui.button("Auto").clicked() {
                self.auto();
            }
            if ui.button("Reset gamma").clicked() {
                self.st.gamma = 1.0;
            }
        });
        ui.checkbox(&mut self.st.nodata, "Value 0 = no data");

        ui.separator();
        if let Some((x, y, v)) = self.cursor {
            let u8 = self.img.as_ref().is_some_and(|i| i.meta().is_u8);
            let v = v.map_or("-".into(), |v| if u8 { format!("{v}") } else { format!("{v:.5}") });
            ui.monospace(format!("x {x}  y {y}\nvalue {v}"));
        }
        ui.monospace(format!("zoom {:.4}  tiles {}", self.scale, self.tiles.map.len()));
        if let Some(g) = &self.gpu {
            ui.small(&g.name);
        }
        ui.separator();
        ui.small("Wheel: zoom. Drag: pan. Double-click or F: fit.\n1: 1:1. C: next color map. I: invert.\nH: hide panel. Ctrl+O: open. Drop a file to open it.");
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let ppp = ui.ctx().pixels_per_point();
        self.view = Rect::from_min_max((rect.min.to_vec2() * ppp).to_pos2(), (rect.max.to_vec2() * ppp).to_pos2());
        if self.img.is_none() {
            let msg = if self.opening { "Opening..." } else { "Drop a NITF or TIFF file here, or press Ctrl+O" };
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, msg, egui::FontId::proportional(18.0), Color32::GRAY);
            return;
        }
        if self.fit || resp.double_clicked() {
            self.fit = false;
            self.fit_view();
        }
        if resp.dragged() {
            let d = resp.drag_delta() * ppp;
            self.center[0] -= d.x as f64 / self.scale;
            self.center[1] -= d.y as f64 / self.scale;
        }
        self.cursor = None;
        if let Some(p) = resp.hover_pos() {
            let p = [(p.x * ppp - self.view.min.x) as f64, (p.y * ppp - self.view.min.y) as f64];
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let f = pinch as f64 * 2f64.powf(scroll as f64 / 200.0);
            if f != 1.0 {
                let before = self.to_image(p);
                self.scale = (self.scale * f).clamp(1e-5, 256.0);
                let after = self.to_image(p);
                self.center[0] += before[0] - after[0];
                self.center[1] += before[1] - after[1];
            }
            let [x, y] = self.to_image(p);
            let img = self.img.as_ref().unwrap();
            if x >= 0.0 && y >= 0.0 && (x as usize) < img.meta().w && (y as usize) < img.meta().h {
                let (x, y) = (x as usize, y as usize);
                self.cursor = Some((x, y, img.value(x, y)));
            }
        }
    }

    fn to_image(&self, p: [f64; 2]) -> [f64; 2] {
        let (w, h) = (self.view.width() as f64, self.view.height() as f64);
        [self.center[0] + (p[0] - w / 2.0) / self.scale, self.center[1] + (p[1] - h / 2.0) / self.scale]
    }

    fn request(&mut self, key: TileKey) {
        if self.tiles.pending.len() >= MAX_PENDING || !self.tiles.pending.insert(key) {
            return;
        }
        let (img, tx, proxy, id) = (self.img.clone().unwrap(), self.tx.clone(), self.proxy.clone(), self.id);
        self.pool.spawn(move || {
            let version = img.version(key.0);
            if let Some((buf, w, h)) = img.tile(key.0, key.1, key.2) {
                let _ = tx.send(Loaded { id, key, version, buf, w, h });
                let _ = proxy.send_event(Ev::Wake);
            }
        });
    }

    /// Upload loaded tiles. Return true if more tiles wait.
    fn upload(&mut self) -> bool {
        let gpu = self.gpu.as_ref().unwrap();
        for _ in 0..UPLOADS_PER_FRAME {
            let Ok(t) = self.rx.try_recv() else { return false };
            if t.id != self.id {
                continue;
            }
            self.tiles.pending.remove(&t.key);
            let Some((tex, _, bpp)) = &gpu.array else { continue };
            let Some(layer) = self.tiles.alloc(t.key, t.version, self.frame) else { continue };
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                    aspect: wgpu::TextureAspect::All,
                },
                t.buf.bytes(),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(t.w as u32 * bpp), rows_per_image: None },
                wgpu::Extent3d { width: t.w as u32, height: t.h as u32, depth_or_array_layers: 1 },
            );
        }
        true
    }

    /// Visible tiles, coarse to fine. Resident coarse tiles fill the gaps while fine tiles load.
    fn draws(&mut self) -> Vec<Inst> {
        let Some(img) = self.img.clone() else { return Vec::new() };
        let dims = img.dims();
        let n = dims.len();
        let target = ((1.0 / self.scale).log2().floor().max(0.0) as usize).min(n - 1);
        let (a, b) = (self.to_image([0.0, 0.0]), self.to_image([self.view.width() as f64, self.view.height() as f64]));
        let (mut out, mut want) = (Vec::new(), Vec::new());
        for l in (target..n).rev() {
            let s = (TILE << l) as f64;
            let (lw, lh) = dims[l];
            let (nx, ny) = (lw.div_ceil(TILE), lh.div_ceil(TILE));
            let r = |v: f64, m: usize| ((v / s).max(0.0) as usize).min(m);
            let (tx0, ty0, tx1, ty1) = (r(a[0], nx), r(a[1], ny), r(b[0], nx - 1) + 1, r(b[1], ny - 1) + 1);
            let (version, ready) = (img.version(l), img.ready(l));
            for ty in ty0..ty1 {
                for tx in tx0..tx1 {
                    let key = (l, tx, ty);
                    let fresh = match self.tiles.map.get_mut(&key) {
                        Some(t) => {
                            t.used = self.frame;
                            let (w, h) = (TILE.min(lw - tx * TILE), TILE.min(lh - ty * TILE));
                            let (x0, y0, k) = (((tx * TILE) << l) as f32, ((ty * TILE) << l) as f32, (1usize << l) as f32);
                            let (u, v) = (w as f32 / TILE as f32, h as f32 / TILE as f32);
                            out.push(Inst { rect: [x0, y0, x0 + w as f32 * k, y0 + h as f32 * k], uvl: [u, v, t.layer as f32, 0.0] });
                            t.version == version
                        }
                        None => false,
                    };
                    if !fresh && ready && (l == target || l == n - 1) {
                        want.push(key);
                    }
                }
            }
        }
        let c = self.center;
        want.sort_by_key(|&(l, tx, ty)| {
            let s = (TILE << l) as f64;
            (((tx as f64 + 0.5) * s - c[0]).powi(2) + ((ty as f64 + 0.5) * s - c[1]).powi(2)) as u64
        });
        want.into_iter().for_each(|k| self.request(k));
        out.truncate(MAX_DRAWS);
        out
    }

    fn render(&mut self, el: &ActiveEventLoop) {
        self.frame += 1;
        let more = self.upload();
        let raw = {
            let g = self.gpu.as_mut().unwrap();
            g.egui_state.take_egui_input(&g.window)
        };
        let ctx = self.ctx.clone();
        let out = ctx.run_ui(raw, |ui| self.ui(ui));
        if std::mem::take(&mut self.dialog) {
            let f = rfd::FileDialog::new()
                .add_filter("SAR images", &["nitf", "ntf", "nsf", "tif", "tiff", "gtiff"])
                .add_filter("All files", &["*"])
                .pick_file();
            if let Some(p) = f {
                self.open(p.to_string_lossy().into());
            }
        }
        let insts = self.draws();
        let g = self.gpu.as_mut().unwrap();
        g.egui_state.handle_platform_output(&g.window, out.platform_output);
        let prims = ctx.tessellate(out.shapes, out.pixels_per_point);
        if std::mem::take(&mut self.lut_dirty) {
            g.queue.write_texture(
                g.lut.as_image_copy(),
                bytemuck::cast_slice(&lut(&self.stops)),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256 * 4), rows_per_image: None },
                wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
            );
        }
        let vmul = self.img.as_ref().map_or(1.0, |i| if i.meta().is_u8 { 255.0 } else { 1.0 / i.meta().k });
        let st = &self.st;
        let u = Uniforms {
            center: [self.center[0] as f32, self.center[1] as f32],
            scale: self.scale as f32,
            vmul,
            view: [self.view.width(), self.view.height()],
            lo: st.lo,
            hi: if st.hi == st.lo { st.lo + 1e-6 } else { st.hi },
            gamma: st.gamma,
            flags: st.db as u32 | (st.invert as u32) << 1 | (st.nodata as u32) << 2,
            pad: [0.0; 2],
        };
        g.queue.write_buffer(&g.ubuf, 0, bytemuck::bytes_of(&u));
        if !insts.is_empty() {
            g.queue.write_buffer(&g.ibuf, 0, bytemuck::cast_slice(&insts));
        }
        for (id, d) in &out.textures_delta.set {
            d.iter().for_each(|d| g.egui.update_texture(&g.device, &g.queue, *id, d));
        }

        let frame = match g.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                g.surface.configure(&g.device, &g.config);
                g.window.request_redraw();
                return;
            }
            _ => return,
        };
        let (sw, sh) = (g.config.width, g.config.height);
        let sd = egui_wgpu::ScreenDescriptor { size_in_pixels: [sw, sh], pixels_per_point: out.pixels_per_point };
        let mut enc = g.device.create_command_encoder(&Default::default());
        let cmds = g.egui.update_buffers(&g.device, &g.queue, &mut enc, &prims, &sd);
        let view = frame.texture.create_view(&Default::default());
        {
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.06, g: 0.06, b: 0.07, a: 1.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            let v = self.view.intersect(Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(sw as f32, sh as f32)));
            if let Some((_, bg, _)) = &g.array
                && !insts.is_empty()
                && v.width() >= 1.0
                && v.height() >= 1.0
            {
                pass.set_viewport(self.view.min.x, self.view.min.y, self.view.width(), self.view.height(), 0.0, 1.0);
                pass.set_scissor_rect(v.min.x as u32, v.min.y as u32, v.width() as u32, v.height() as u32);
                pass.set_pipeline(&g.pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.set_vertex_buffer(0, g.ibuf.slice(..));
                pass.draw(0..4, 0..insts.len() as u32);
                pass.set_viewport(0.0, 0.0, sw as f32, sh as f32, 0.0, 1.0);
                pass.set_scissor_rect(0, 0, sw, sh);
            }
            g.egui.render(&mut pass, &prims, &sd);
        }
        g.queue.submit(cmds.into_iter().chain([enc.finish()]));
        g.window.pre_present_notify();
        g.queue.present(frame);
        for id in &out.textures_delta.free {
            g.egui.free_texture(id);
        }

        let delay = out.viewport_output.get(&egui::ViewportId::ROOT).map_or(std::time::Duration::MAX, |v| v.repaint_delay);
        if more || delay.is_zero() {
            g.window.request_redraw();
        } else if let Some(t) = std::time::Instant::now().checked_add(delay) {
            el.set_control_flow(ControlFlow::WaitUntil(t));
        }
    }
}

fn init_gpu(el: &ActiveEventLoop, ctx: &egui::Context) -> Gpu {
    let attrs = Window::default_attributes().with_title(APP).with_inner_size(winit::dpi::LogicalSize::new(1500, 950));
    let window = Arc::new(el.create_window(attrs).unwrap());
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(el.owned_display_handle())));
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        ..Default::default()
    }))
    .expect("no GPU adapter");
    let info = adapter.get_info();
    let name = format!("{} ({:?})", info.name, info.backend);
    let limits = adapter.limits();
    let desc = wgpu::DeviceDescriptor { required_limits: limits.clone(), ..Default::default() };
    let (device, queue) = pollster::block_on(adapter.request_device(&desc)).unwrap();

    let size = window.inner_size();
    let mut config = surface.get_default_config(&adapter, size.width.max(1), size.height.max(1)).unwrap();
    // Non-sRGB target: colors go to the screen as written. egui also expects this.
    let caps = surface.get_capabilities(&adapter);
    if let Some(f) = caps.formats.iter().find(|f| !f.is_srgb()) {
        config.format = *f;
    }
    config.present_mode = wgpu::PresentMode::AutoVsync;
    config.desired_maximum_frame_latency = 1;
    surface.configure(&device, &config);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
    let tex = |binding, dim| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: dim,
            multisampled: false,
        },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            tex(2, wgpu::TextureViewDimension::D2Array),
            tex(3, wgpu::TextureViewDimension::D2),
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Inst>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4],
            })],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(config.format.into())],
        }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let buf = |size, usage| device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false });
    let ubuf = buf(std::mem::size_of::<Uniforms>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
    let ibuf = buf((MAX_DRAWS * std::mem::size_of::<Inst>()) as u64, wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST);
    // Nearest when magnified (show real pixels). Linear when minified.
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let lut = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("lut"),
        size: wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let egui = egui_wgpu::Renderer::new(&device, config.format, egui_wgpu::RendererOptions::default());
    let egui_state = egui_winit::State::new(
        ctx.clone(),
        egui::ViewportId::ROOT,
        &window,
        Some(window.scale_factor() as f32),
        None,
        Some(limits.max_texture_dimension_2d as usize),
    );
    Gpu {
        window,
        surface,
        device,
        queue,
        config,
        pipeline,
        bgl,
        ubuf,
        ibuf,
        sampler,
        lut,
        array: None,
        max_layers: limits.max_texture_array_layers,
        name,
        egui,
        egui_state,
    }
}

impl ApplicationHandler<Ev> for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        self.gpu = Some(init_gpu(el, &self.ctx));
        if let Some(p) = std::env::args().nth(1) {
            self.open(p);
        }
    }

    fn new_events(&mut self, _: &ActiveEventLoop, cause: StartCause) {
        if let (StartCause::ResumeTimeReached { .. }, Some(g)) = (cause, &self.gpu) {
            g.window.request_redraw();
        }
    }

    fn user_event(&mut self, _: &ActiveEventLoop, ev: Ev) {
        match ev {
            Ev::Wake => {}
            Ev::Opened(id, _) if id != self.id => return,
            Ev::Opened(_, Ok(img)) => self.opened(img),
            Ev::Opened(_, Err(e)) => {
                self.opening = false;
                self.error = Some(e);
            }
        }
        if let Some(g) = &self.gpu {
            g.window.request_redraw();
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, ev: WindowEvent) {
        let Some(g) = &mut self.gpu else { return };
        let resp = g.egui_state.on_window_event(&g.window, &ev);
        match ev {
            WindowEvent::CloseRequested => {
                if let Some(i) = &self.img {
                    i.cancel();
                }
                el.exit();
            }
            WindowEvent::Resized(s) => {
                g.config.width = s.width.max(1);
                g.config.height = s.height.max(1);
                g.surface.configure(&g.device, &g.config);
                g.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                el.set_control_flow(ControlFlow::Wait);
                self.render(el);
            }
            _ if resp.repaint => g.window.request_redraw(),
            _ => {}
        }
    }
}

fn main() {
    // WSLg: Vulkan uses the CPU (lavapipe) and the Wayland socket is not stable. Mesa d3d12 GL over X11 uses the GPU.
    #[cfg(target_os = "linux")]
    let wsl = std::env::var_os("WSL_DISTRO_NAME").is_some();
    #[cfg(target_os = "linux")]
    if wsl && std::env::var_os("GALLIUM_DRIVER").is_none() {
        // SAFETY: no other thread exists yet.
        unsafe { std::env::set_var("GALLIUM_DRIVER", "d3d12") };
    }
    #[allow(unused_mut)]
    let mut builder = EventLoop::<Ev>::with_user_event();
    #[cfg(target_os = "linux")]
    if wsl {
        use winit::platform::x11::EventLoopBuilderExtX11;
        builder.with_x11();
    }
    let el = builder.build().unwrap();
    let (tx, rx) = mpsc::channel();
    let threads = std::thread::available_parallelism().map_or(4, |n| (n.get() / 2).clamp(2, 8));
    let mut app = App {
        proxy: el.create_proxy(),
        gpu: None,
        ctx: egui::Context::default(),
        img: None,
        id: 0,
        path: String::new(),
        opening: false,
        error: None,
        tiles: Tiles::default(),
        tx,
        rx,
        pool: rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap(),
        frame: 0,
        center: [0.0; 2],
        scale: 1.0,
        view: Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1.0, 1.0)),
        fit: false,
        st: Stretch { lo: 0.0, hi: 1.0, gamma: 1.0, db: false, clip: 0.5, invert: false, nodata: true },
        cmap: 0,
        stops: CMAPS[0].1.iter().map(|&c| hex(c)).collect(),
        lut_dirty: true,
        panel: true,
        path_edit: String::new(),
        cursor: None,
        dialog: false,
        keep_view: false,
    };
    el.run_app(&mut app).unwrap();
}
