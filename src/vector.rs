//! Analytical Signed-Area 2D Vector Rasterizer (`Path2D`).
//! Flattens quadratic and cubic Béziers non-recursively via Wang's closed-form formula,
//! constructs seamless round-cap / round-join offset contours (`stroke_polyline` / `stroke_polygon`),
//! and computes exact anti-aliased trapezoidal coverage in a row-bucketed active-edge scanline sweep.
//! Uploads crisp vector icons, sparklines, and custom shapes into `GlyphAtlas` with zero GPU compute overhead!

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillRule {
    NonZero,
    EvenOdd,
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

#[derive(Clone, Copy, Debug)]
struct PreparedEdge {
    dir: f32,
    x_a: f32,
    y_a: f32,
    x_b: f32,
    y_b: f32,
    end_row: usize,
    next_in_bucket: u32,
}

#[derive(Clone, Copy, Debug)]
struct PolySeg {
    p0: [f32; 2],
    p1: [f32; 2],
    t: [f32; 2],
    n: [f32; 2],
    len: f32,
}

impl PolySeg {
    #[inline]
    fn reversed(self) -> Self {
        Self {
            p0: self.p1,
            p1: self.p0,
            t: [-self.t[0], -self.t[1]],
            n: [-self.n[0], -self.n[1]],
            len: self.len,
        }
    }
}

const BUCKET_NONE: u32 = u32::MAX;

#[inline]
fn is_valid_pt(x: f32, y: f32) -> bool {
    x.is_finite() && y.is_finite()
}

/// Reusable scratch buffers for zero-allocation steady-state `Path2D::rasterize_into` sweeps.
#[derive(Clone, Debug, Default)]
pub struct RasterScratch {
    row_heads: Vec<u32>,
    prepared: Vec<PreparedEdge>,
    active: Vec<u32>,
    area: Vec<f32>,
}

impl RasterScratch {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(edge_capacity: usize, width: u16, height: u16) -> Self {
        Self {
            row_heads: vec![BUCKET_NONE; height as usize],
            prepared: Vec::with_capacity(edge_capacity),
            active: Vec::with_capacity(32),
            area: vec![0.0; (width as usize) + 1],
        }
    }
}

/// Ergonomic 2D vector path builder supporting lines, quadratic/cubic Béziers, circles, and seamless round-join strokes.
#[derive(Clone, Debug, Default)]
pub struct Path2D {
    edges: Vec<Edge>,
    seg_scratch: Vec<PolySeg>,
    subpath_start: [f32; 2],
    cursor: [f32; 2],
    has_open_subpath: bool,
}

impl Path2D {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(edge_capacity: usize) -> Self {
        Self {
            edges: Vec::with_capacity(edge_capacity),
            seg_scratch: Vec::new(),
            subpath_start: [0.0, 0.0],
            cursor: [0.0, 0.0],
            has_open_subpath: false,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty() && !self.has_open_subpath
    }

    /// Reset accumulated path subpaths while retaining allocated `edges` and stroke segment capacities (`0 B` steady-state heap).
    #[inline]
    pub fn clear(&mut self) {
        self.edges.clear();
        self.seg_scratch.clear();
        self.subpath_start = [0.0, 0.0];
        self.cursor = [0.0, 0.0];
        self.has_open_subpath = false;
    }

    /// Uniformly or non-uniformly scale all accumulated path edges (used for HiDPI physical rasterization).
    pub fn scale(&mut self, sx: f32, sy: f32) -> &mut Self {
        if !is_valid_pt(sx, sy) {
            return self;
        }
        for e in &mut self.edges {
            e.x0 *= sx;
            e.y0 *= sy;
            e.x1 *= sx;
            e.y1 *= sy;
        }
        self.subpath_start[0] *= sx;
        self.subpath_start[1] *= sy;
        self.cursor[0] *= sx;
        self.cursor[1] *= sy;
        self
    }

    pub fn move_to(&mut self, x: f32, y: f32) -> &mut Self {
        if !is_valid_pt(x, y) {
            return self;
        }
        if self.has_open_subpath {
            self.close();
        }
        self.subpath_start = [x, y];
        self.cursor = [x, y];
        self.has_open_subpath = true;
        self
    }

    pub fn line_to(&mut self, x: f32, y: f32) -> &mut Self {
        if !is_valid_pt(x, y) {
            return self;
        }
        if !self.has_open_subpath {
            return self.move_to(x, y);
        }
        let [x0, y0] = self.cursor;
        if (y - y0).abs() > 1e-6 || (x - x0).abs() > 1e-6 {
            self.edges.push(Edge { x0, y0, x1: x, y1: y });
            self.cursor = [x, y];
        }
        self
    }

    /// Non-recursive quadratic Bézier flattening via Wang's closed-form segment bound.
    pub fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) -> &mut Self {
        if !is_valid_pt(cx, cy) || !is_valid_pt(x, y) {
            return self;
        }
        if !self.has_open_subpath {
            let [sx, sy] = self.cursor;
            self.move_to(sx, sy);
        }
        let [x0, y0] = self.cursor;
        let dx = (x0 - 2.0 * cx + x).abs();
        let dy = (y0 - 2.0 * cy + y).abs();
        let tol = 0.10_f32;
        let raw_n = (dx.hypot(dy) / (4.0 * tol)).sqrt().ceil();
        let n = if raw_n.is_finite() {
            (raw_n as usize).clamp(1, 2048)
        } else {
            1
        };
        let inv_n = 1.0 / (n as f32);
        for i in 1..n {
            let t = (i as f32) * inv_n;
            let mt = 1.0 - t;
            let px = mt * mt * x0 + 2.0 * mt * t * cx + t * t * x;
            let py = mt * mt * y0 + 2.0 * mt * t * cy + t * t * y;
            self.line_to(px, py);
        }
        self.line_to(x, y)
    }

    /// Non-recursive cubic Bézier flattening via Wang's theorem:
    /// `N = ceil(sqrt((3 / (4 * tol)) * max(|P0 - 2P1 + P2|, |P1 - 2P2 + P3|)))`.
    pub fn cubic_to(
        &mut self,
        c1x: f32,
        c1y: f32,
        c2x: f32,
        c2y: f32,
        x: f32,
        y: f32,
    ) -> &mut Self {
        if !is_valid_pt(c1x, c1y) || !is_valid_pt(c2x, c2y) || !is_valid_pt(x, y) {
            return self;
        }
        if !self.has_open_subpath {
            let [sx, sy] = self.cursor;
            self.move_to(sx, sy);
        }
        let [x0, y0] = self.cursor;
        let d1 = (x0 - 2.0 * c1x + c2x).hypot(y0 - 2.0 * c1y + c2y);
        let d2 = (c1x - 2.0 * c2x + x).hypot(c1y - 2.0 * c2y + y);
        let tol = 0.10_f32;
        let raw_n = (0.75 * d1.max(d2) / tol).sqrt().ceil();
        let n = if raw_n.is_finite() {
            (raw_n as usize).clamp(1, 2048)
        } else {
            1
        };
        let inv_n = 1.0 / (n as f32);
        for i in 1..n {
            let t = (i as f32) * inv_n;
            let mt = 1.0 - t;
            let px = mt * mt * mt * x0
                + 3.0 * mt * mt * t * c1x
                + 3.0 * mt * t * t * c2x
                + t * t * t * x;
            let py = mt * mt * mt * y0
                + 3.0 * mt * mt * t * c1y
                + 3.0 * mt * t * t * c2y
                + t * t * t * y;
            self.line_to(px, py);
        }
        self.line_to(x, y)
    }

    pub fn close(&mut self) -> &mut Self {
        if self.has_open_subpath {
            let [sx, sy] = self.subpath_start;
            let [cx, cy] = self.cursor;
            if (sx - cx).abs() > 1e-6 || (sy - cy).abs() > 1e-6 {
                self.edges.push(Edge {
                    x0: cx,
                    y0: cy,
                    x1: sx,
                    y1: sy,
                });
            }
            self.cursor = [sx, sy];
            self.has_open_subpath = false;
        }
        self
    }

    /// Add a closed circle subpath centered at `(cx, cy)` with positive winding (`NonZero` & `EvenOdd` compatible).
    pub fn circle(&mut self, cx: f32, cy: f32, radius: f32) -> &mut Self {
        if !is_valid_pt(cx, cy) || !radius.is_finite() || radius <= 0.0 {
            return self;
        }
        if self.has_open_subpath {
            self.close();
        }
        let steps = ((core::f32::consts::TAU * (radius / 0.04).sqrt()).ceil() as usize).clamp(16, 512);
        let d_theta = -core::f32::consts::TAU / (steps as f32);
        self.move_to(cx + radius, cy);
        for i in 1..steps {
            let ang = (i as f32) * d_theta;
            let (s, c) = ang.sin_cos();
            self.line_to(cx + radius * c, cy + radius * s);
        }
        self.close()
    }

    /// Add a butt-capped stroked segment of width `stroke_w` with positive winding.
    pub fn add_stroke_segment(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, stroke_w: f32) -> &mut Self {
        if !is_valid_pt(x0, y0) || !is_valid_pt(x1, y1) || !stroke_w.is_finite() || stroke_w <= 0.0 {
            return self;
        }
        let dx = x1 - x0;
        let dy = y1 - y0;
        let len = dx.hypot(dy);
        if len <= 1e-5 {
            return self;
        }
        let hw = stroke_w * 0.5;
        let nx = -dy / len * hw;
        let ny = dx / len * hw;
        // Emit in consistent positive (`+1` winding) order matching `stroke_polyline`:
        self.move_to(x0 + nx, y0 + ny)
            .line_to(x1 + nx, y1 + ny)
            .line_to(x1 - nx, y1 - ny)
            .line_to(x0 - nx, y0 - ny)
            .close()
    }

    /// Add a single capsule-stroked segment with seamless semicircular round endcaps.
    #[inline]
    pub fn add_round_stroke_segment(
        &mut self,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        width: f32,
    ) -> &mut Self {
        self.stroke_polyline(&[[x0, y0], [x1, y1]], width)
    }

    /// Stroke an open polyline `pts` of thickness `width` with exact round endcaps and round outer joins
    /// as a single unified offset contour (zero joint gaps, zero double-coverage edge seams, and zero steady-state heap allocations!).
    pub fn stroke_polyline(&mut self, pts: &[[f32; 2]], width: f32) -> &mut Self {
        if !width.is_finite() || width <= 0.0 || pts.is_empty() {
            return self;
        }
        let r = width * 0.5;

        // Single-pass finite filter + consecutive duplicate elimination directly into reusable `self.seg_scratch`:
        self.seg_scratch.clear();
        let mut first_pt: Option<[f32; 2]> = None;
        let mut prev_pt = [0.0_f32; 2];
        for &p in pts {
            if !is_valid_pt(p[0], p[1]) {
                continue;
            }
            if first_pt.is_none() {
                first_pt = Some(p);
                prev_pt = p;
                continue;
            }
            let dx = p[0] - prev_pt[0];
            let dy = p[1] - prev_pt[1];
            let len = dx.hypot(dy);
            if len <= 1e-5 {
                continue;
            }
            let tx = dx / len;
            let ty = dy / len;
            self.seg_scratch.push(PolySeg {
                p0: prev_pt,
                p1: p,
                t: [tx, ty],
                n: [-ty, tx],
                len,
            });
            prev_pt = p;
        }

        let Some(p_first) = first_pt else {
            return self;
        };
        let seg_count = self.seg_scratch.len();
        if seg_count == 0 {
            return self.circle(p_first[0], p_first[1], r);
        }

        if self.has_open_subpath {
            self.close();
        }

        // 1. Start at `P_0 + r * n_0` and walk forward along the `+n` side:
        let first = self.seg_scratch[0];
        self.move_to(first.p0[0] + r * first.n[0], first.p0[1] + r * first.n[1]);
        self.emit_offset_side_fwd(seg_count, r);

        // 2. Semicircular round cap around `P_{K-1}` from `+n_{S-1}` to `-n_{S-1}`:
        let last = self.seg_scratch[seg_count - 1];
        self.emit_arc(
            last.p1,
            r,
            last.n,
            [-last.n[0], -last.n[1]],
            -core::f32::consts::PI,
        );

        // 3. Walk backward from `P_{K-1}` to `P_0` computing reversed segments in registers (`0 B` rev allocation!):
        self.emit_offset_side_rev(seg_count, r);

        // 4. Semicircular round cap around `P_0` from `-n_0` back to `+n_0`, then close:
        self.emit_arc(
            first.p0,
            r,
            [-first.n[0], -first.n[1]],
            first.n,
            -core::f32::consts::PI,
        );
        self.seg_scratch.clear();
        self.close()
    }

    /// Stroke a closed polygon loop `pts` of thickness `width` with seamless round joins on both contours
    /// and zero steady-state heap allocations.
    pub fn stroke_polygon(&mut self, pts: &[[f32; 2]], width: f32) -> &mut Self {
        if !width.is_finite() || width <= 0.0 || pts.is_empty() {
            return self;
        }
        self.seg_scratch.clear();
        let mut first_pt: Option<[f32; 2]> = None;
        let mut prev_pt = [0.0_f32; 2];
        for &p in pts {
            if !is_valid_pt(p[0], p[1]) {
                continue;
            }
            if first_pt.is_none() {
                first_pt = Some(p);
                prev_pt = p;
                continue;
            }
            let dx = p[0] - prev_pt[0];
            let dy = p[1] - prev_pt[1];
            let len = dx.hypot(dy);
            if len <= 1e-5 {
                continue;
            }
            let tx = dx / len;
            let ty = dy / len;
            self.seg_scratch.push(PolySeg {
                p0: prev_pt,
                p1: p,
                t: [tx, ty],
                n: [-ty, tx],
                len,
            });
            prev_pt = p;
        }

        let Some(p_first) = first_pt else {
            return self;
        };
        if !self.seg_scratch.is_empty()
            && (prev_pt[0] - p_first[0]).hypot(prev_pt[1] - p_first[1]) <= 1e-5
        {
            let popped = self.seg_scratch.pop().unwrap();
            prev_pt = popped.p0;
        }
        // Distinct vertex count is `1 + self.seg_scratch.len()`:
        if self.seg_scratch.len() < 2 {
            let p_second = self.seg_scratch.first().map(|s| s.p1);
            self.seg_scratch.clear();
            return match p_second {
                Some(p1) => self.stroke_polyline(&[p_first, p1], width),
                None => self.stroke_polyline(&[p_first], width),
            };
        }

        let close_dx = p_first[0] - prev_pt[0];
        let close_dy = p_first[1] - prev_pt[1];
        let close_len = close_dx.hypot(close_dy);
        let close_tx = close_dx / close_len;
        let close_ty = close_dy / close_len;
        self.seg_scratch.push(PolySeg {
            p0: prev_pt,
            p1: p_first,
            t: [close_tx, close_ty],
            n: [-close_ty, close_tx],
            len: close_len,
        });

        let r = width * 0.5;
        let n = self.seg_scratch.len();
        self.emit_closed_offset_ring_fwd(n, r);
        self.emit_closed_offset_ring_rev(n, r);
        self.seg_scratch.clear();
        self
    }

    fn emit_offset_side_fwd(&mut self, count: usize, r: f32) {
        for i in 0..count {
            let s0 = self.seg_scratch[i];
            if i + 1 < count {
                let s1 = self.seg_scratch[i + 1];
                self.emit_join(s0, s1, r);
            } else {
                self.line_to(s0.p1[0] + r * s0.n[0], s0.p1[1] + r * s0.n[1]);
            }
        }
    }

    fn emit_offset_side_rev(&mut self, count: usize, r: f32) {
        for k in (0..count).rev() {
            let s0 = self.seg_scratch[k].reversed();
            if k > 0 {
                let s1 = self.seg_scratch[k - 1].reversed();
                self.emit_join(s0, s1, r);
            } else {
                self.line_to(s0.p1[0] + r * s0.n[0], s0.p1[1] + r * s0.n[1]);
            }
        }
    }

    fn emit_closed_offset_ring_fwd(&mut self, n: usize, r: f32) {
        if self.has_open_subpath {
            self.close();
        }
        // Start at the midpoint of segment 0's offset edge, strictly away from vertex joins:
        let s0 = self.seg_scratch[0];
        let mx = (s0.p0[0] + s0.p1[0]) * 0.5 + r * s0.n[0];
        let my = (s0.p0[1] + s0.p1[1]) * 0.5 + r * s0.n[1];
        self.move_to(mx, my);
        for i in 0..n {
            let sa = self.seg_scratch[i];
            let sb = self.seg_scratch[(i + 1) % n];
            self.emit_join(sa, sb, r);
        }
        self.line_to(mx, my);
        self.close();
    }

    fn emit_closed_offset_ring_rev(&mut self, n: usize, r: f32) {
        if self.has_open_subpath {
            self.close();
        }
        let s0 = self.seg_scratch[n - 1].reversed();
        let mx = (s0.p0[0] + s0.p1[0]) * 0.5 + r * s0.n[0];
        let my = (s0.p0[1] + s0.p1[1]) * 0.5 + r * s0.n[1];
        self.move_to(mx, my);
        for step in 0..n {
            let idx_a = n - 1 - step;
            let idx_b = if idx_a == 0 { n - 1 } else { idx_a - 1 };
            let sa = self.seg_scratch[idx_a].reversed();
            let sb = self.seg_scratch[idx_b].reversed();
            self.emit_join(sa, sb, r);
        }
        self.line_to(mx, my);
        self.close();
    }

    #[inline]
    fn emit_join(&mut self, s0: PolySeg, s1: PolySeg, r: f32) {
        let p = s0.p1;
        let cross = s0.t[0] * s1.t[1] - s0.t[1] * s1.t[0];
        let dot = (s0.t[0] * s1.t[0] + s0.t[1] * s1.t[1]).clamp(-1.0, 1.0);

        if cross < -1e-4 || (cross.abs() <= 1e-4 && dot < 0.0) {
            // Outer convex bend (or 180-degree outer turn): exact circular round join!
            let start_pt = [p[0] + r * s0.n[0], p[1] + r * s0.n[1]];
            let end_n = s1.n;
            self.line_to(start_pt[0], start_pt[1]);
            let sweep = if dot < -0.999_999 && cross.abs() <= 1e-4 {
                -core::f32::consts::PI
            } else {
                cross.atan2(dot) // in `(-PI, 0)`
            };
            self.emit_arc(p, r, s0.n, end_n, sweep);
        } else if cross > 1e-4 {
            // Inner concave bend: compute exact inner offset intersection `I` if inside half-segment lengths:
            let denom = 1.0 + dot;
            let max_d = 0.5 * s0.len.min(s1.len);
            if denom > 1e-4 {
                let d = r * (cross / denom);
                if d <= max_d {
                    let scale = r / denom;
                    let ix = p[0] + (s0.n[0] + s1.n[0]) * scale;
                    let iy = p[1] + (s0.n[1] + s1.n[1]) * scale;
                    self.line_to(ix, iy);
                    return;
                }
            }
            // Acute hairpin fallback: positive-winding inner fold through vertex `p`
            self.line_to(p[0] + r * s0.n[0], p[1] + r * s0.n[1]);
            self.line_to(p[0], p[1]);
            self.line_to(p[0] + r * s1.n[0], p[1] + r * s1.n[1]);
        } else {
            // Collinear (`dot > 0`):
            self.line_to(p[0] + r * s1.n[0], p[1] + r * s1.n[1]);
        }
    }

    #[inline]
    fn emit_arc(
        &mut self,
        center: [f32; 2],
        r: f32,
        start_n: [f32; 2],
        end_n: [f32; 2],
        sweep_rad: f32,
    ) {
        let abs_sweep = sweep_rad.abs();
        let steps = ((abs_sweep * (r / 0.12).sqrt()).ceil() as usize).clamp(2, 192);
        let theta0 = start_n[1].atan2(start_n[0]);
        let inv_steps = 1.0 / (steps as f32);
        for k in 1..steps {
            let ang = theta0 + sweep_rad * ((k as f32) * inv_steps);
            let (s, c) = ang.sin_cos();
            self.line_to(center[0] + r * c, center[1] + r * s);
        }
        self.line_to(center[0] + r * end_n[0], center[1] + r * end_n[1]);
    }

    /// Zero-allocation steady-state analytical trapezoidal signed-area rasterizer into `out_mask` (`width x height`).
    /// Tracks tight per-row `[min_row..max_row]` and `[min_col..max_col]` active spans and clears `scratch.area`
    /// inline as it sweeps so untouched rows and pixel columns are never scanned or zeroed redundantly!
    pub fn rasterize_into(
        &self,
        width: u16,
        height: u16,
        rule: FillRule,
        scratch: &mut RasterScratch,
        out_mask: &mut Vec<u8>,
    ) {
        let w = width as usize;
        let h = height as usize;
        let total = w * h;
        out_mask.clear();
        out_mask.resize(total, 0);
        if total == 0 {
            return;
        }

        let implicit_close = if self.has_open_subpath {
            let [sx, sy] = self.subpath_start;
            let [cx, cy] = self.cursor;
            if (sx - cx).abs() > 1e-6 || (sy - cy).abs() > 1e-6 {
                Some(Edge {
                    x0: cx,
                    y0: cy,
                    x1: sx,
                    y1: sy,
                })
            } else {
                None
            }
        } else {
            None
        };

        if self.edges.is_empty() && implicit_close.is_none() {
            return;
        }

        let wf = w as f32;
        let hf = h as f32;

        if scratch.row_heads.len() < h {
            scratch.row_heads.resize(h, BUCKET_NONE);
        }
        if scratch.area.len() < w + 1 {
            scratch.area.resize(w + 1, 0.0);
        }
        if scratch.active.capacity() < 64 {
            scratch.active.clear();
            scratch.active.reserve(64);
        } else {
            scratch.active.clear();
        }
        scratch.prepared.clear();

        let mut min_row = h;
        let mut max_row = 0usize;

        for e in self.edges.iter().copied().chain(implicit_close) {
            if !is_valid_pt(e.x0, e.y0) || !is_valid_pt(e.x1, e.y1) {
                continue;
            }
            let (dir, x_a, y_a, x_b, y_b) = if e.y0 < e.y1 {
                (1.0_f32, e.x0, e.y0, e.x1, e.y1)
            } else if e.y1 < e.y0 {
                (-1.0_f32, e.x1, e.y1, e.x0, e.y0)
            } else {
                continue; // Horizontal edges contribute 0 vertical winding
            };

            if y_b <= 0.0 || y_a >= hf {
                continue;
            }
            // Edges completely beyond the right edge `x >= width` contribute 0 winding to `0..w`:
            if x_a >= wf && x_b >= wf {
                continue;
            }

            let start_row = (y_a.max(0.0).floor() as usize).min(h - 1);
            let end_row = (y_b.min(hf).ceil() as usize).clamp(start_row + 1, h);
            min_row = min_row.min(start_row);
            max_row = max_row.max(end_row);

            let e_idx = scratch.prepared.len() as u32;
            scratch.prepared.push(PreparedEdge {
                dir,
                x_a,
                y_a,
                x_b,
                y_b,
                end_row,
                next_in_bucket: scratch.row_heads[start_row],
            });
            scratch.row_heads[start_row] = e_idx;
        }

        if scratch.prepared.is_empty() {
            return;
        }

        let area_buf = &mut scratch.area[..=w];

        for row in min_row..max_row {
            let mut cur = scratch.row_heads[row];
            scratch.row_heads[row] = BUCKET_NONE;
            while cur != BUCKET_NONE {
                scratch.active.push(cur);
                cur = scratch.prepared[cur as usize].next_in_bucket;
            }

            if scratch.active.is_empty() {
                continue;
            }

            let y_top = row as f32;
            let y_bot = y_top + 1.0;
            let mut min_col = w;
            let mut max_col = 0usize;

            let mut a_i = 0usize;
            while a_i < scratch.active.len() {
                let pe = scratch.prepared[scratch.active[a_i] as usize];
                let dy_full = pe.y_b - pe.y_a;

                // Overflow-proof normalized `t` clipping to `[y_top, y_bot]`:
                let (mut x0, y0) = if pe.y_a < y_top {
                    let t = ((y_top - pe.y_a) / dy_full).clamp(0.0, 1.0);
                    (pe.x_a + (pe.x_b - pe.x_a) * t, y_top)
                } else {
                    (pe.x_a, pe.y_a)
                };
                let (mut x1, y1) = if pe.y_b > y_bot {
                    let t = ((y_bot - pe.y_a) / dy_full).clamp(0.0, 1.0);
                    (pe.x_a + (pe.x_b - pe.x_a) * t, y_bot)
                } else {
                    (pe.x_b, pe.y_b)
                };

                let dy_row = y1 - y0;
                if dy_row > 0.0 {
                    // Orient `(x_min, x_max)` horizontally while preserving total signed height `dy_signed`:
                    if x0 > x1 {
                        core::mem::swap(&mut x0, &mut x1);
                    }
                    let dy_signed = dy_row * pe.dir;

                    if x1 <= 0.0 {
                        // Wholly left of `x = 0`: full signed winding enters column 0
                        area_buf[0] += dy_signed;
                        min_col = 0;
                        max_col = max_col.max(1);
                    } else if x0 < wf {
                        // Exact X-clipping against `[0, wf]` before trapezoidal column distribution:
                        let dx_full = x1 - x0;
                        let mut dy_vis = dy_signed;
                        if x0 < 0.0 && dx_full > 0.0 {
                            let left_frac = ((-x0) / dx_full).clamp(0.0, 1.0);
                            area_buf[0] += dy_vis * left_frac;
                            min_col = 0;
                            max_col = max_col.max(1);
                            dy_vis *= 1.0 - left_frac;
                            x0 = 0.0;
                        }
                        if x1 > wf && (x1 - x0) > 0.0 {
                            let keep_frac = ((wf - x0) / (x1 - x0)).clamp(0.0, 1.0);
                            dy_vis *= keep_frac;
                            x1 = wf;
                        }

                        if dy_vis.abs() > 0.0 {
                            let col_start = (x0.floor() as usize).min(w - 1);
                            let col_end = (x1.ceil() as usize).clamp(col_start + 1, w);
                            min_col = min_col.min(col_start);
                            max_col = max_col.max(col_end);

                            if col_start + 1 == col_end {
                                let avg_x = ((x0 + x1) * 0.5 - col_start as f32).clamp(0.0, 1.0);
                                area_buf[col_start] += dy_vis * (1.0 - avg_x);
                                area_buf[col_start + 1] += dy_vis * avg_x;
                            } else {
                                let slope_y = dy_vis / (x1 - x0);
                                for col in col_start..col_end {
                                    let px0 = col as f32;
                                    let px1 = px0 + 1.0;
                                    let seg_x0 = x0.max(px0);
                                    let seg_x1 = x1.min(px1);
                                    if seg_x1 > seg_x0 {
                                        let frac_h = slope_y * (seg_x1 - seg_x0);
                                        let mid_in_px =
                                            ((seg_x0 + seg_x1) * 0.5 - px0).clamp(0.0, 1.0);
                                        area_buf[col] += frac_h * (1.0 - mid_in_px);
                                        area_buf[col + 1] += frac_h * mid_in_px;
                                    }
                                }
                            }
                        }
                    }
                }

                if row + 1 >= pe.end_row {
                    scratch.active.swap_remove(a_i);
                } else {
                    a_i += 1;
                }
            }

            if min_col < max_col {
                let mut acc = 0.0_f32;
                let row_slice = &mut out_mask[row * w..(row + 1) * w];
                match rule {
                    FillRule::NonZero => {
                        for col in min_col..max_col {
                            let d = area_buf[col];
                            area_buf[col] = 0.0;
                            acc += d;
                            let cov = acc.abs().min(1.0);
                            row_slice[col] = (cov * 255.0 + 0.5) as u8;
                        }
                        let tail_d = area_buf[max_col];
                        area_buf[max_col] = 0.0;
                        acc += tail_d;
                        if max_col < w && acc.abs() >= (0.5 / 255.0) {
                            let tail_u8 = (acc.abs().min(1.0) * 255.0 + 0.5) as u8;
                            if tail_u8 != 0 {
                                row_slice[max_col..w].fill(tail_u8);
                            }
                        }
                    }
                    FillRule::EvenOdd => {
                        for col in min_col..max_col {
                            let d = area_buf[col];
                            area_buf[col] = 0.0;
                            acc += d;
                            let m = acc.abs() % 2.0;
                            let cov = if m > 1.0 { 2.0 - m } else { m };
                            row_slice[col] = (cov * 255.0 + 0.5) as u8;
                        }
                        let tail_d = area_buf[max_col];
                        area_buf[max_col] = 0.0;
                        acc += tail_d;
                        if max_col < w && acc.abs() >= (0.5 / 255.0) {
                            let m = acc.abs() % 2.0;
                            let cov = if m > 1.0 { 2.0 - m } else { m };
                            let tail_u8 = (cov * 255.0 + 0.5) as u8;
                            if tail_u8 != 0 {
                                row_slice[max_col..w].fill(tail_u8);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Exact analytical trapezoidal signed-area rasterizer into an `R8` alpha mask `(width x height)`.
    /// Convenience allocating wrapper around [`Path2D::rasterize_into`].
    pub fn rasterize_mask(&mut self, width: u16, height: u16, rule: FillRule) -> Vec<u8> {
        if self.has_open_subpath {
            self.close();
        }
        let mut scratch = RasterScratch::default();
        let mut out = Vec::new();
        self.rasterize_into(width, height, rule, &mut scratch, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytical_coverage_exactness_and_out_of_bounds_clipping() {
        // 1. Half-pixel shifted box `[0.5, 0.5]..[3.5, 3.5]` on `4x4`:
        // Corners = 0.25 (64), edges = 0.50 (128), interior = 1.00 (255)!
        let mut p = Path2D::new();
        p.move_to(0.5, 0.5)
            .line_to(3.5, 0.5)
            .line_to(3.5, 3.5)
            .line_to(0.5, 3.5)
            .close();
        let m = p.rasterize_mask(4, 4, FillRule::NonZero);
        assert_eq!(m[0], 64);
        assert_eq!(m[1], 128);
        assert_eq!(m[3], 64);
        assert_eq!(m[5], 255);
        assert_eq!(m[10], 255);

        // 2. Diagonal edge crossing from `x = -1.5` to `x = 0.5` across row 0 (regression check for X<0 clip!):
        // Polygon `(-1.5, 0) -> (4.0, 0) -> (4.0, 1) -> (0.5, 1)`:
        // Left edge goes downward `(-1.5, 0) -> (0.5, 1)`. Inside pixel 0 (`[0,1]x[0,1]`),
        // `x <= 0` for `y in [0, 0.75]` (area 0.75) + trapezoid for `y in [0.75, 1.0]` (area 0.1875) = 0.9375 -> 239!
        let mut clip_p = Path2D::new();
        clip_p
            .move_to(-1.5, 0.0)
            .line_to(6.0, -2.0)
            .line_to(6.0, 3.0)
            .line_to(0.5, 1.0)
            .close();
        let cm = clip_p.rasterize_mask(4, 1, FillRule::NonZero);
        assert!(
            cm[0].abs_diff(239) <= 1,
            "Expected exact X<0 clipped coverage ~239, got {}",
            cm[0]
        );
        assert_eq!(cm[1], 255);
        assert_eq!(cm[2], 255);
        assert_eq!(cm[3], 255);

        // 3. Zero-size, zero-length, subnormal near-horizontal, and NaN/Inf resilience:
        let mut deg = Path2D::new();
        deg.move_to(f32::NAN, f32::INFINITY)
            .move_to(2.0, 2.0)
            .line_to(2.0, 2.0) // zero-length
            .line_to(100.0, 2.0 + 1e-38) // subnormal dy
            .line_to(100.0, 4.0)
            .line_to(-50.0, 4.0)
            .close();
        assert!(deg.rasterize_mask(0, 0, FillRule::NonZero).is_empty());
        let dm = deg.rasterize_mask(8, 8, FillRule::NonZero);
        assert_eq!(dm.len(), 64);
        assert_eq!(dm[2 * 8 + 4], 255);
        assert_eq!(dm[3 * 8 + 4], 255);
    }

    #[test]
    fn wangs_bezier_and_circle_area_accuracy() {
        let mut p = Path2D::new();
        p.circle(16.0, 16.0, 10.0);
        let mask = p.rasterize_mask(32, 32, FillRule::EvenOdd);
        let measured_area: f32 = mask.iter().map(|&a| (a as f32) / 255.0).sum();
        let exact_area = core::f32::consts::PI * 100.0;
        assert!(
            (measured_area - exact_area).abs() < 1.0,
            "Circle area {measured_area} deviated from pi*r^2 = {exact_area}"
        );

        // Cubic S-curve stroke / fill stability
        let mut b = Path2D::new();
        b.move_to(2.0, 2.0)
            .cubic_to(14.0, 2.0, 2.0, 14.0, 14.0, 14.0)
            .quad_to(8.0, 16.0, 2.0, 14.0)
            .close();
        let bm = b.rasterize_mask(16, 16, FillRule::NonZero);
        assert!(bm.iter().any(|&v| v > 200));
    }

    #[test]
    fn stroke_polyline_seamless_round_joins_and_caps_under_nonzero_and_evenodd() {
        // Right-angle L-joint at `(16.0, 16.0)` with thick width `8.0` (`r = 4.0`):
        // With butt-only `add_stroke_segment`, the outer corner triangle `(16..20, 16..20)` has a huge gap!
        // With `stroke_polyline`, `(18, 18)` (distance `sqrt(8) = 2.83 < 4.0` from `(16, 16)`) must be 100% solid (255)
        // under BOTH `FillRule::NonZero` AND `FillRule::EvenOdd`!
        let pts = [[4.0, 16.0], [16.0, 16.0], [16.0, 4.0]];
        for rule in [FillRule::NonZero, FillRule::EvenOdd] {
            let mut p = Path2D::new();
            p.stroke_polyline(&pts, 8.0);
            let m = p.rasterize_mask(32, 32, rule);
            // Vertex center `(16, 16)` and outer round-join diagonal `(18, 18)`:
            assert_eq!(m[16 * 32 + 16], 255, "Joint center hole under {rule:?}");
            assert_eq!(m[17 * 32 + 17], 255, "Outer round join gap under {rule:?}");
            // Inner concave corner `(13, 13)` inside stroke vs `(10, 10)` outside stroke:
            assert_eq!(m[13 * 32 + 13], 255, "Inner join hole under {rule:?}");
            assert_eq!(m[10 * 32 + 10], 0, "Unexpected bleed outside stroke under {rule:?}");
            // Start round cap at `(4, 16)` extends left to `x = 0`: pixel `(1, 16)` is inside cap!
            assert_eq!(m[16 * 32 + 1], 255, "Round start cap missing under {rule:?}");
        }

        // Dense waveform polyline (4px spacing, 2.4px width) + acute hairpin + 1-point dot:
        let mut wave = Vec::new();
        for i in 0..=40 {
            let x = 4.0 + (i as f32) * 2.0;
            let y = 16.0 + ((i as f32) * 0.35).sin() * 8.0;
            wave.push([x, y]);
        }
        let mut wp = Path2D::new();
        wp.stroke_polyline(&wave, 3.0)
            .stroke_polyline(&[[4.0, 28.0], [28.0, 29.0], [4.0, 30.0]], 3.0) // acute hairpin
            .stroke_polyline(&[[90.0, 16.0]], 6.0); // single-point round dot
        let wm = wp.rasterize_mask(100, 32, FillRule::NonZero);
        assert_eq!(wm[16 * 100 + 90], 255, "Single-point polyline dot must be solid");
        assert!(wm.iter().filter(|&&a| a > 0).count() > 250);

        // Closed `stroke_polygon` ring under both `NonZero` and `EvenOdd` (hollow interior, solid rounded corners):
        let box_pts = [[8.0, 8.0], [24.0, 8.0], [24.0, 24.0], [8.0, 24.0]];
        for rule in [FillRule::NonZero, FillRule::EvenOdd] {
            let mut ring = Path2D::new();
            ring.stroke_polygon(&box_pts, 4.0);
            let rm = ring.rasterize_mask(32, 32, rule);
            assert_eq!(rm[16 * 32 + 16], 0, "Polygon ring interior must stay hollow under {rule:?}");
            assert_eq!(rm[8 * 32 + 16], 255, "Top stroke edge must be solid under {rule:?}");
            assert_eq!(rm[8 * 32 + 8], 255, "Polygon join corner must be solid under {rule:?}");
        }

        // Viewport-enclosing rect `[-20, -20]..[50, 50]` on `16x16` must fill all pixels to 255:
        let mut full = Path2D::new();
        full.move_to(-20.0, -20.0)
            .line_to(50.0, -20.0)
            .line_to(50.0, 50.0)
            .line_to(-20.0, 50.0); // intentionally unclosed to verify auto-close
        let mut scratch = RasterScratch::new();
        let mut fm_into = Vec::new();
        // Verify `&self` unclosed `rasterize_into` matches `&mut self` `rasterize_mask` bit-for-bit:
        full.rasterize_into(16, 16, FillRule::NonZero, &mut scratch, &mut fm_into);
        let fm = full.rasterize_mask(16, 16, FillRule::NonZero);
        assert_eq!(fm_into, fm);
        assert!(fm.iter().all(|&px| px == 255));
    }

    #[test]
    fn rasterize_into_zero_reallocation_and_bit_identical_parity_with_rasterize_mask() {
        let mut path = Path2D::new();
        let mut scratch = RasterScratch::new();
        let mut out_mask = Vec::new();
        let mut pts = [[0.0_f32; 2]; 105];

        // Frame 0: warm up capacities on `520x62` waveform
        for (step, pt) in pts.iter_mut().enumerate() {
            let x = (step as f32) * 5.0;
            let y = 31.0 + (x * 0.025).sin() * 22.0;
            *pt = [x, y];
        }
        path.stroke_polyline(&pts, 2.4);
        path.rasterize_into(520, 62, FillRule::NonZero, &mut scratch, &mut out_mask);

        let edges_cap = path.edges.capacity();
        let seg_cap = path.seg_scratch.capacity();
        let rh_cap = scratch.row_heads.capacity();
        let prep_cap = scratch.prepared.capacity();
        let act_cap = scratch.active.capacity();
        let area_cap = scratch.area.capacity();
        let out_ptr = out_mask.as_ptr();
        let out_cap = out_mask.capacity();

        // Steady-state animated frames across both NonZero & EvenOdd must keep exact buffer pointers/capacities
        // (`0` buffer reallocations) and produce bit-identical `R8` masks to `rasterize_mask`!
        for frame_i in 1..=16 {
            let phase = (frame_i as f32) * 0.37;
            path.clear();
            for (step, pt) in pts.iter_mut().enumerate() {
                let x = (step as f32) * 5.0;
                let y = 31.0 + (x * 0.025 + phase).sin() * 21.5;
                *pt = [x, y];
            }
            path.stroke_polyline(&pts, 2.4);
            let rule = if frame_i % 2 == 0 {
                FillRule::EvenOdd
            } else {
                FillRule::NonZero
            };
            path.rasterize_into(520, 62, rule, &mut scratch, &mut out_mask);

            assert_eq!(out_mask.as_ptr(), out_ptr, "out_mask reallocated on frame {frame_i}");
            assert_eq!(out_mask.capacity(), out_cap);
            assert_eq!(path.edges.capacity(), edges_cap);
            assert_eq!(path.seg_scratch.capacity(), seg_cap);
            assert_eq!(scratch.row_heads.capacity(), rh_cap);
            assert_eq!(scratch.prepared.capacity(), prep_cap);
            assert_eq!(scratch.active.capacity(), act_cap);
            assert_eq!(scratch.area.capacity(), area_cap);

            // Verify internal scratch invariants (`row_heads == BUCKET_NONE`, `area == 0.0`, `active.is_empty()`):
            assert!(scratch.active.is_empty());
            assert!(scratch.row_heads.iter().all(|&h| h == BUCKET_NONE));
            assert!(scratch.area.iter().all(|&a| a == 0.0));

            let mut ref_path = path.clone();
            let ref_mask = ref_path.rasterize_mask(520, 62, rule);
            assert_eq!(out_mask, ref_mask, "Bit mismatch vs rasterize_mask on frame {frame_i}");
        }
    }

    #[test]
    fn fractional_hidpi_scale_area_conservation_and_seamless_stroke_continuity() {
        let scales: &[f32] = &[
            1.0,
            1.125,
            1.25,
            160.0 / 120.0,
            1.5,
            1.625,
            200.0 / 120.0,
            1.75,
            2.25,
        ];
        let mut scratch = RasterScratch::new();
        let mut mask = Vec::new();

        for &sf in scales {
            // 1. Subpixel-shifted rectangle `[3.25, 2.75]..[19.75, 14.25]` (`16.5 x 11.5 = 189.75` logical area)
            //    must conserve exact `189.75 * sf^2` integrated coverage across every fractional scale!
            let mut rect = Path2D::new();
            rect.move_to(3.25, 2.75)
                .line_to(19.75, 2.75)
                .line_to(19.75, 14.25)
                .line_to(3.25, 14.25)
                .close()
                .scale(sf, sf);
            let pw = (24.0 * sf).ceil() as u16;
            let ph = (18.0 * sf).ceil() as u16;
            rect.rasterize_into(pw, ph, FillRule::NonZero, &mut scratch, &mut mask);
            let measured_area: f32 = mask.iter().map(|&a| (a as f32) / 255.0).sum();
            let exact_phys_area = 16.5 * 11.5 * sf * sf;
            assert!(
                (measured_area - exact_phys_area).abs() < 0.35 * sf,
                "Fractional scale {sf} rect area {measured_area} deviated from {exact_phys_area}"
            );

            // 2. Circle + stroked L-joint scaled to fractional physical pixels must keep solid joint center (`255`):
            let mut icon = Path2D::new();
            icon.stroke_polyline(&[[4.0, 12.0], [12.0, 12.0], [12.0, 4.0]], 4.0)
                .scale(sf, sf);
            let iw = (20.0 * sf).ceil() as u16;
            let ih = (20.0 * sf).ceil() as u16;
            icon.rasterize_into(iw, ih, FillRule::NonZero, &mut scratch, &mut mask);
            let jx = (12.0 * sf).floor() as usize;
            let jy = (12.0 * sf).floor() as usize;
            assert_eq!(
                mask[jy * (iw as usize) + jx],
                255,
                "Scaled polyline joint center must remain 100% solid at sf={sf}"
            );
        }
    }
}
