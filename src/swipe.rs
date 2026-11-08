//! Continuous touchpad swipe between workspaces while the overview is open.
//!
//! Positions are in "pages": workspace indices on the output, as `f32`, so
//! `1.5` is halfway between the second and third workspace. The strip
//! follows the fingers 1:1 (in millimetres on the touchpad), then a
//! critically damped spring carries the release velocity into the nearest
//! workspace.
//!
//! `zwp_pointer_gesture_swipe_v1` deltas are *accelerated* with the pointer
//! acceleration curve, which ranges ~16x between a slow and a fast swipe.
//! That makes a fixed "delta per page" feel dead when slow and jumpy when
//! fast, so `Deaccel` undoes libinput's touchpad curve to get back the
//! distance the fingers really travelled.

use cosmic::iced::advanced::layout::flex::Axis;
use cosmic_comp_config::{
    input::{AccelConfig, AccelProfile},
    workspace::WorkspaceLayout,
};
use std::{collections::VecDeque, sync::OnceLock, time::Instant};

use crate::backend::ExtWorkspaceHandleV1;
use cosmic::cctk::wayland_client::protocol::wl_output;

/// Finger travel, in mm, for one full workspace. Override with
/// `COSMIC_WORKSPACES_SWIPE_MM` to tune without rebuilding.
const DEFAULT_PAGE_MM: f64 = 60.0;
/// Velocity samples older than this are ignored (cosmic-comp `HISTORY_LIMIT`).
const HISTORY_MS: u32 = 100;
/// How far a fling coasts after release: `velocity * PROJECTION_S` pages.
/// A fast or long swipe can carry over several workspaces.
const PROJECTION_S: f32 = 0.3;
/// No fling if the fingers were still for this long before lifting.
const STOPPED_MS: u32 = 50;
/// Release velocity cap, pages/s.
const MAX_RELEASE_VELOCITY: f32 = 15.0;
/// Stiffness of the settle spring, rad/s. ~0.55 s to come to rest from
/// a standstill; stiffer for fast flings so they don't overshoot.
const SPRING_OMEGA: f32 = 9.0;
const MAX_SPRING_OMEGA: f32 = 30.0;
/// Rubber band past the first/last workspace.
const RUBBER_LIMIT: f32 = 0.15;
const RUBBER_STIFFNESS: f32 = 0.5;
/// Finger travel (mm) along either axis before the gesture picks a direction.
const AXIS_LOCK_MM: f64 = 1.5;
/// Window used to estimate finger speed for the acceleration inverse.
const SPEED_WINDOW_MS: u32 = 40;

fn page_mm() -> f64 {
    static PAGE_MM: OnceLock<f64> = OnceLock::new();
    *PAGE_MM.get_or_init(|| {
        std::env::var("COSMIC_WORKSPACES_SWIPE_MM")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| *v > 1.0)
            .unwrap_or(DEFAULT_PAGE_MM)
    })
}

/// Extra margin past the output edge (logical px), for the strip's gap and
/// padding, so off-screen pages stay within the surface.
pub const OVERFLOW_SLACK: i32 = 64;

pub fn layout_axis(layout: WorkspaceLayout) -> Axis {
    match layout {
        WorkspaceLayout::Horizontal => Axis::Horizontal,
        WorkspaceLayout::Vertical => Axis::Vertical,
    }
}

/// libinput's touchpad acceleration (`src/filter-touchpad.c`, 1.25), used
/// backwards: accelerated deltas (1000 dpi units) -> finger travel in mm.
#[derive(Clone, Copy, Debug)]
pub struct Deaccel {
    adaptive: bool,
    /// `speed_factor()` of the configured pointer speed.
    speed_factor: f64,
}

impl Default for Deaccel {
    fn default() -> Self {
        Self::new(None)
    }
}

const TP_MAGIC_SLOWDOWN: f64 = 0.2968;
const MM_PER_UNIT: f64 = 25.4 / 1000.0;

impl Deaccel {
    pub fn new(config: Option<&AccelConfig>) -> Self {
        let speed = config.map_or(0.0, |c| c.speed).clamp(-1.0, 1.0);
        let adaptive = !matches!(config.and_then(|c| c.profile), Some(AccelProfile::Flat));
        Self {
            adaptive,
            speed_factor: (speed + 1.0).powf(2.38) * 0.95 + 0.05,
        }
    }

    /// Acceleration factor for a finger speed in mm/s
    /// (`touchpad_accel_profile_linear`).
    fn factor(&self, v: f64) -> f64 {
        const BASELINE: f64 = 0.9;
        const THRESHOLD: f64 = 130.0;
        let f = if !self.adaptive {
            BASELINE
        } else if v < 7.0 {
            (0.1 * v + 0.3).min(BASELINE)
        } else if v < THRESHOLD {
            BASELINE
        } else {
            let v = v.min(THRESHOLD * 4.0);
            0.0025 * (v / THRESHOLD) * (v - THRESHOLD) + BASELINE
        };
        f * self.speed_factor * TP_MAGIC_SLOWDOWN
    }

    /// Finger speed (mm/s) that produces pointer speed `out` (mm/s).
    /// `v * factor(v)` is monotonic, so bisect.
    fn finger_speed(&self, out: f64) -> f64 {
        let (mut lo, mut hi) = (0.0, 10_000.0);
        for _ in 0..40 {
            let mid = (lo + hi) / 2.0;
            if mid * self.factor(mid) < out {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) / 2.0
    }

    /// Divisor turning accelerated mm into finger mm, given the accelerated
    /// pointer speed in mm/s (`None` if unknown: assume the 1:1 plateau).
    fn divisor(&self, out_speed: Option<f64>) -> f64 {
        match out_speed {
            Some(out) if out > 0.0 => self.factor(self.finger_speed(out)),
            _ => self.factor(30.0),
        }
    }
}

#[derive(Debug)]
pub struct Gesture {
    pub output: wl_output::WlOutput,
    axis: Axis,
    /// Touchpad natural scrolling, as cosmic-comp uses it for its own
    /// workspace swipe.
    natural: bool,
    deaccel: Deaccel,
    /// Position when the gesture began (may be fractional if it interrupted
    /// a settle animation).
    start_pos: f32,
    /// Number of workspaces on the output.
    count: usize,
    /// cosmic-comp's `workspace_wraparound`: past the last workspace comes
    /// the first one again (and the other way round) instead of a rubber band.
    wrap: bool,
    /// `None` until the fingers moved far enough to pick an axis; then
    /// whether that axis is the workspace axis.
    locked: Option<bool>,
    pending: (f64, f64),
    /// Accumulated movement, in pages, positive towards later workspaces.
    moved: f64,
    history: VecDeque<(u32, f64)>,
    /// Recent accelerated pointer travel (time, mm), for the speed estimate.
    recent: VecDeque<(u32, f64)>,
}

impl Gesture {
    pub fn new(
        output: wl_output::WlOutput,
        layout: WorkspaceLayout,
        natural: bool,
        deaccel: Deaccel,
        start_pos: f32,
        count: usize,
        wrap: bool,
    ) -> Self {
        Self {
            output,
            axis: layout_axis(layout),
            natural,
            deaccel,
            start_pos,
            count,
            wrap: wraps(wrap, count),
            locked: None,
            pending: (0.0, 0.0),
            moved: 0.0,
            history: VecDeque::new(),
            recent: VecDeque::new(),
        }
    }

    /// `false` once the gesture went along the other axis; it is then
    /// ignored until the fingers lift.
    pub fn is_active(&self) -> bool {
        self.locked != Some(false)
    }

    /// `dx`/`dy`: accelerated deltas from the protocol (1000 dpi units).
    pub fn update(&mut self, time: u32, dx: f64, dy: f64) {
        let (dx, dy) = self.finger_mm(time, dx, dy);
        if self.locked.is_none() {
            self.pending.0 += dx;
            self.pending.1 += dy;
            let (px, py) = self.pending;
            if px.abs().max(py.abs()) < AXIS_LOCK_MM {
                return;
            }
            let horizontal = px.abs() > py.abs();
            self.locked = Some(horizontal == matches!(self.axis, Axis::Horizontal));
            // Apply the movement collected before the lock.
            let (dx, dy) = std::mem::take(&mut self.pending);
            return self.push(time, dx, dy);
        }
        self.push(time, dx, dy);
    }

    /// Undo pointer acceleration: accelerated deltas -> finger travel in mm.
    fn finger_mm(&mut self, time: u32, dx: f64, dy: f64) -> (f64, f64) {
        let dist = dx.hypot(dy) * MM_PER_UNIT;
        self.recent.push_back((time, dist));
        while let Some(&(t, _)) = self.recent.front() {
            if time.wrapping_sub(t) <= SPEED_WINDOW_MS {
                break;
            }
            self.recent.pop_front();
        }
        // Speed over the window: travel after the oldest sample / time since it.
        let out_speed = self.recent.front().and_then(|&(t0, _)| {
            let dt = time.wrapping_sub(t0);
            (dt > 0).then(|| {
                let travel: f64 = self.recent.iter().skip(1).map(|(_, d)| d).sum();
                travel / (dt as f64 / 1000.0)
            })
        });
        let k = MM_PER_UNIT / self.deaccel.divisor(out_speed);
        (dx * k, dy * k)
    }

    fn push(&mut self, time: u32, dx_mm: f64, dy_mm: f64) {
        if !self.is_active() {
            return;
        }
        let along = match self.axis {
            Axis::Horizontal => dx_mm,
            Axis::Vertical => dy_mm,
        };
        // Same direction as cosmic-comp's workspace swipe outside the
        // overview: with natural scrolling the strip follows the fingers
        // (left/up brings in the next workspace), otherwise it's reversed.
        // Devices that emit synthetic swipes (e.g. mousetouch) adapt to that
        // convention, so they behave the same in and out of the overview.
        let pages = if self.natural { -along } else { along } / page_mm();
        self.moved += pages;
        self.history.push_back((time, pages));
        while let Some(&(t, _)) = self.history.front() {
            if time.wrapping_sub(t) <= HISTORY_MS {
                break;
            }
            self.history.pop_front();
        }
    }

    fn raw_pos(&self) -> f32 {
        self.start_pos + self.moved as f32
    }

    /// Position to draw, with rubber banding past both ends. With
    /// wraparound it is unbounded: `-1.0` is the last workspace, reached
    /// from the first one (see [`page_index`]).
    pub fn pos(&self) -> f32 {
        let last = self.count.saturating_sub(1) as f32;
        let raw = self.raw_pos();
        if self.wrap {
            raw
        } else if raw < 0.0 {
            -rubber_band(-raw)
        } else if raw > last {
            last + rubber_band(raw - last)
        } else {
            raw
        }
    }

    /// Pages per second over the recent history, as of `now` (ms): if the
    /// fingers stopped before lifting, there's no fling.
    pub fn velocity(&self, now: u32) -> f32 {
        let (Some(first), Some(last)) = (self.history.front(), self.history.back()) else {
            return 0.0;
        };
        if now.wrapping_sub(last.0) > STOPPED_MS {
            return 0.0;
        }
        let dt = last.0.wrapping_sub(first.0) as f32 / 1000.0;
        if dt <= 0.0 {
            return 0.0;
        }
        let total: f64 = self.history.iter().map(|(_, d)| d).sum();
        (total as f32 / dt).clamp(-MAX_RELEASE_VELOCITY, MAX_RELEASE_VELOCITY)
    }

    /// Page to settle on after release at `now` (ms), in the same space
    /// as [`Gesture::pos`]: always a valid index without wraparound, any
    /// whole number with it (map it with [`page_index`]).
    ///
    /// libinput marks the end `cancelled` whenever the finger count changes
    /// first (e.g. lifting fingers one at a time), so like cosmic-comp this
    /// ignores it and lands where the fingers left the strip.
    pub fn target(&self, now: u32) -> f32 {
        let last = self.count.saturating_sub(1) as f32;
        let pos = self.pos();
        let projected = (pos + self.velocity(now) * PROJECTION_S).round();
        if self.wrap {
            projected
        } else {
            projected.clamp(0.0, last)
        }
    }
}

/// Wraparound only means something with two or more workspaces.
pub fn wraps(wraparound: bool, count: usize) -> bool {
    wraparound && count > 1
}

/// Workspace index shown at whole page `page`, or `None` past the ends
/// without wraparound.
pub fn page_index(page: f32, count: usize, wraparound: bool) -> Option<usize> {
    if count == 0 {
        return None;
    }
    if wraps(wraparound, count) {
        Some((page as i64).rem_euclid(count as i64) as usize)
    } else {
        (page >= 0.0 && page <= (count - 1) as f32).then_some(page as usize)
    }
}

fn rubber_band(x: f32) -> f32 {
    (1.0 - 1.0 / (x * RUBBER_STIFFNESS / RUBBER_LIMIT + 1.0)) * RUBBER_LIMIT
}

/// Critically damped spring towards 0: x(t) = (x0 + (v0 + ω x0) t) e^(-ω t).
/// Positions in pages (offset from the target), velocities in pages/s.
#[derive(Clone, Copy, Debug)]
struct Spring {
    x0: f32,
    v0: f32,
    omega: f32,
}

impl Spring {
    fn new(x0: f32, velocity: f32) -> Self {
        // Heading away from the target (a short flick back): start from rest
        // rather than coasting out and snapping back.
        let v0 = if velocity * x0 > 0.0 { 0.0 } else { velocity };
        // It overshoots when |v0| > ω|x0|; stiffen it just enough to land on
        // the target while keeping the release velocity.
        let omega = if x0.abs() > 1e-3 {
            (v0.abs() / x0.abs()).clamp(SPRING_OMEGA, MAX_SPRING_OMEGA)
        } else {
            SPRING_OMEGA
        };
        // Past the stiffness cap, give up some speed rather than overshoot.
        let v0 = v0.clamp(-omega * x0.abs(), omega * x0.abs());
        Self { x0, v0, omega }
    }

    fn at(&self, t: f32) -> (f32, f32) {
        let w = self.omega;
        let b = self.v0 + w * self.x0;
        let e = (-w * t).exp();
        let x = (self.x0 + b * t) * e;
        let v = (b - w * (self.x0 + b * t)) * e;
        (x, v)
    }
}

/// Animation from the release point to the target workspace.
#[derive(Debug)]
pub struct Settle {
    pub output: wl_output::WlOutput,
    pub target: ExtWorkspaceHandleV1,
    /// Target page minus the target's index: a multiple of the workspace
    /// count when the swipe wrapped around, else 0.
    lap: f32,
    spring: Spring,
    start: Instant,
}

impl Settle {
    pub fn new(
        output: wl_output::WlOutput,
        target: ExtWorkspaceHandleV1,
        target_idx: usize,
        target_page: f32,
        pos: f32,
        velocity: f32,
    ) -> Self {
        Self {
            output,
            target,
            lap: target_page - target_idx as f32,
            spring: Spring::new(pos - target_page, velocity),
            start: Instant::now(),
        }
    }

    fn offset_and_velocity(&self, now: Instant) -> (f32, f32) {
        self.spring
            .at(now.saturating_duration_since(self.start).as_secs_f32())
    }

    /// Page the animation ends on, given where the target workspace is now.
    pub fn target_page(&self, target_idx: usize) -> f32 {
        target_idx as f32 + self.lap
    }

    /// Current position, given where the target workspace is now.
    pub fn pos(&self, target_idx: usize, now: Instant) -> f32 {
        self.target_page(target_idx) + self.offset_and_velocity(now).0
    }

    pub fn is_done(&self, now: Instant) -> bool {
        let (x, v) = self.offset_and_velocity(now);
        x.abs() < 0.002 && v.abs() < 0.05
    }
}

/// Indices of the (at most two) workspaces visible at `pos`, with the offset
/// of each page from the viewport in pages.
pub fn visible_pages(pos: f32, count: usize, wraparound: bool) -> Vec<(usize, f32)> {
    let lo = pos.floor();
    let mut pages = Vec::with_capacity(2);
    for page in [lo, lo + 1.0] {
        let offset = page - pos;
        if offset.abs() >= 1.0 {
            continue;
        }
        if let Some(idx) = page_index(page, count, wraparound) {
            pages.push((idx, offset));
        }
    }
    pages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_pages_at_rest_and_between() {
        assert_eq!(visible_pages(1.0, 3, false), vec![(1, 0.0)]);
        assert_eq!(visible_pages(1.25, 3, false), vec![(1, -0.25), (2, 0.75)]);
        // Rubber band past the ends only shows the end page.
        assert_eq!(visible_pages(-0.1, 3, false).len(), 1);
        assert_eq!(visible_pages(2.1, 3, false).len(), 1);
    }

    #[test]
    fn visible_pages_wrap_around() {
        // Between the last and the first workspace, both ways.
        assert_eq!(visible_pages(-0.25, 3, true), vec![(2, -0.75), (0, 0.25)]);
        assert_eq!(visible_pages(2.5, 3, true), vec![(2, -0.5), (0, 0.5)]);
        // Several laps away.
        assert_eq!(visible_pages(-4.0, 3, true), vec![(2, 0.0)]);
        // A single workspace never wraps.
        assert_eq!(visible_pages(-0.1, 1, true), vec![(0, 0.1)]);
    }

    #[test]
    fn page_index_clamps_or_wraps() {
        assert_eq!(page_index(-1.0, 4, false), None);
        assert_eq!(page_index(4.0, 4, false), None);
        assert_eq!(page_index(-1.0, 4, true), Some(3));
        assert_eq!(page_index(4.0, 4, true), Some(0));
        assert_eq!(page_index(0.0, 0, true), None);
    }

    #[test]
    fn deaccel_inverts_libinput_curve() {
        let d = Deaccel::new(None);
        // Round trip: finger speed -> pointer speed -> finger speed.
        for v in [2.0, 6.5, 50.0, 200.0, 600.0] {
            let out = v * d.factor(v);
            assert!((d.finger_speed(out) - v).abs() < 0.01, "{v}");
        }
        // A slow swipe is decelerated much more than a fast one.
        assert!(d.factor(600.0) / d.factor(0.5) > 10.0);
    }

    #[test]
    fn settle_never_overshoots_or_reverses() {
        // (offset from target, release velocity): slow, fast, multi-page
        // fling, and a flick away from the target.
        for (x0, v0) in [
            (-0.5f32, 2.0f32),
            (-0.5, 12.0),
            (-2.4, 15.0),
            (0.3, 4.0),
            (-0.05, 15.0),
        ] {
            let spring = Spring::new(x0, v0);
            let mut t = 0.0;
            while t < 2.0 {
                let (x, _) = spring.at(t);
                assert!(x * x0 >= -1e-4, "overshoot x0={x0} v0={v0} t={t} x={x}");
                assert!(x.abs() <= x0.abs() + 1e-4, "moved away x0={x0} v0={v0}");
                t += 0.005;
            }
        }
        // Release velocity is kept when heading to the target.
        let (_, v) = Spring::new(-0.5, 3.0).at(0.0);
        assert!((v - 3.0).abs() < 1e-4);
    }

    #[test]
    fn rubber_band_is_bounded() {
        assert!(rubber_band(0.0).abs() < 1e-6);
        assert!(rubber_band(100.0) < RUBBER_LIMIT);
        assert!(rubber_band(0.1) < 0.1);
    }
}
