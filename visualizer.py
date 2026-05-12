"""
interactive_visualizer.py
─────────────────────────
Dynamic, interactive matplotlib visualiser for 2-D parametric polynomial
interpolation.

Features
--------
- Curve manager sidebar: create / delete multiple curves
- Per-curve: add, delete, move control points on the canvas
- Per-curve: edit x, y, and parameter (t) values via sliders + text boxes
- Per-curve: auto-parametrize (uniform / centripetal / chordal / manual)
- Colour-mode toggle: parameter vs speed
- Sample-density slider
- All sliders also accept raw typed values

Usage
-----
    from interactive_visualizer import InteractiveVisualizer
    vis = InteractiveVisualizer()
    vis.show()

Or run this file directly for a demo with one pre-loaded curve.
"""

# todo: add extrapolation parameter slider  #  tells how much beyond the endpoints to sample

# todo: set minimum axis range to 2x2, so dont get "stuck" in a small window when have few points

# todo: make each curve have a colormap like Purples, Oranges, Blues, etc instead of always being mpl default
#   then allow to hide each curve os change its colormap (as desmos)
#   also allow to hide the control polygon separately from the curve

# todo: generalize color_modes to position, speed, acceleration, etc  # generalize variation operator already used to get speed (delta(any)/delta(t))

import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider, TextBox, RadioButtons
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import warnings

# ── project modules ───────────────────────────────────────────────────────────
import vandermond
import sample_polynomials as sp_mod
from parametize import parametize

# ── palette ───────────────────────────────────────────────────────────────────
CURVE_COLORS = [
    "#4e8ef7", "#e05c5c", "#3dc98f", "#f5a623",
    "#9b59b6", "#1abc9c", "#e67e22", "#e91e63",
]
BG_DARK = "#0f0f0f"
BG_PANEL = "#1a1a1a"
BG_WIDGET = "#252525"
FG_TEXT = "#e0e0e0"
FG_DIM = "#666666"
ACCENT = "#4e8ef7"
DANGER = "#e05c5c"

COLOR_MODES = ["parameter", "speed"]

POINT_RADIUS = 8  # px hit-test radius
MIN_POINTS_FIT = 2  # minimum points to attempt interpolation


# ─────────────────────────────────────────────────────────────────────────────
# Curve data model
# ─────────────────────────────────────────────────────────────────────────────
class Curve:
    _counter = 0

    def __init__(self):
        Curve._counter += 1
        self.name: str = f"Curve {Curve._counter}"
        self.color: str = CURVE_COLORS[(Curve._counter - 1) % len(CURVE_COLORS)]
        # points stored as list of [t, x, y]
        self.points: list[list[float]] = []
        self.param_exponent: float = 0
        self.color_mode: str = "parameter"
        self.samples: int = 15
        self.visible: bool = True

    # ── derived ──────────────────────────────────────────────────────────────
    def get_array(self) -> npt.NDArray:
        """Return (N, 3) array [t, x, y]."""
        return np.array(self.points, dtype=float) if self.points else np.empty((0, 3))

    def apply_parametrization(self):
        """Re-compute t values using param_exponent (in-place)."""
        arr = self.get_array()
        if len(arr) < 2:
            return
        reparametrized = parametize(arr[:, 1:], exponent=self.param_exponent)
        for i, row in enumerate(reparametrized):
            self.points[i][0] = float(row[0])

    def interpolate(self, draft=False):
        """
        Return (resampled_points, success: bool).
        resampled_points is (M, 3) [t, x, y] or None on failure.
        """
        arr = self.get_array()
        if len(arr) < MIN_POINTS_FIT:
            return None, False
        # need unique, sorted t values
        ts = arr[:, 0]
        if len(np.unique(ts)) < len(ts):
            return None, False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cx = vandermond.coefficients(arr[:, [0, 1]])
                cy = vandermond.coefficients(arr[:, [0, 2]])
            segments = len(arr) - 1
            rate = self.samples if not draft else min(4, self.samples)
            resampled = sp_mod.sample_polynomials(
                parameters=ts,
                polynomials=np.array([cx, cy]),
                relative_sample_rate=max(1, rate),
            )
            return resampled, True
        except Exception:
            return None, False


# ─────────────────────────────────────────────────────────────────────────────
# Main visualiser
# ─────────────────────────────────────────────────────────────────────────────
class InteractiveVisualizer:

    def __init__(self):
        self.curves: list[Curve] = []
        self.active_idx: int = -1  # index into self.curves
        self.interaction_mode: str = "add"  # "add" | "move" | "delete"
        self._drag_pt_idx: int = -1
        self._drag_curve_idx: int = -1

        self._build_figure()
        self._connect_events()

        # start with one empty curve
        self._add_curve()

    # ══════════════════════════════════════════════════════════════════════════
    # Figure layout
    # ══════════════════════════════════════════════════════════════════════════
    def _build_figure(self):
        mpl.rcParams.update({
            "figure.facecolor": BG_DARK,
            "axes.facecolor": BG_DARK,
            "axes.edgecolor": FG_DIM,
            "axes.labelcolor": FG_TEXT,
            "xtick.color": FG_DIM,
            "ytick.color": FG_DIM,
            "text.color": FG_TEXT,
            "grid.color": "#2a2a2a",
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
        })

        self.fig = plt.figure(figsize=(15, 9), facecolor=BG_DARK)
        self.fig.canvas.manager.set_window_title("Parametric Polynomial Interpolator")

        # ── outer split: canvas (left) | sidebar (right) ─────────────────────
        outer = gridspec.GridSpec(
            1, 2,
            figure=self.fig,
            left=0.01, right=0.99,
            top=0.97, bottom=0.03,
            wspace=0.02,
            width_ratios=[3, 1.4],
        )

        # canvas axes
        self.ax_canvas = self.fig.add_subplot(outer[0])
        self.ax_canvas.set_facecolor(BG_DARK)
        self.ax_canvas.set_aspect("equal", adjustable="box")
        self.ax_canvas.grid(True)
        self.ax_canvas.set_title(
            "Left-click: add/move/delete  |  Right-click+drag: pan",
            color=FG_DIM, fontsize=8, pad=4,
        )

        # ── sidebar: tall GridSpec inside outer[1] ───────────────────────────
        sb = gridspec.GridSpecFromSubplotSpec(
            32, 1,
            subplot_spec=outer[1],
            hspace=0.4,
        )
        self._sb = sb  # keep ref for later

        row = 0

        # ── section: Interaction mode ─────────────────────────────────────────
        ax_mode_lbl = self.fig.add_subplot(sb[row]);
        row += 1
        ax_mode_lbl.axis("off")
        ax_mode_lbl.text(0.04, 0.5, "INTERACTION MODE",
                         va="center", fontsize=7, color=FG_DIM,
                         transform=ax_mode_lbl.transAxes)

        ax_mode = self.fig.add_subplot(sb[row:row + 2]);
        row += 2
        self.radio_mode = RadioButtons(
            ax_mode, ("add", "move", "delete"),
            activecolor=ACCENT,
        )
        ax_mode.set_facecolor(BG_PANEL)
        for lbl in self.radio_mode.labels:
            lbl.set_color(FG_TEXT);
            lbl.set_fontsize(9)
        self.radio_mode.on_clicked(self._on_mode_changed)

        # ── section: Curves list ──────────────────────────────────────────────
        ax_cl_lbl = self.fig.add_subplot(sb[row]);
        row += 1
        ax_cl_lbl.axis("off")
        ax_cl_lbl.text(0.04, 0.5, "CURVES",
                       va="center", fontsize=7, color=FG_DIM,
                       transform=ax_cl_lbl.transAxes)

        # We support up to 5 curve slots in the sidebar; overflow still works,
        # only the bottom controls won't update — acceptable for a demo.
        self._curve_btn_axes = []
        self._curve_btns = []
        MAX_CURVE_BTNS = 5
        for _ in range(MAX_CURVE_BTNS):
            ax = self.fig.add_subplot(sb[row]);
            row += 1
            ax.set_facecolor(BG_PANEL)
            b = Button(ax, "", color=BG_PANEL, hovercolor=BG_WIDGET)
            b.label.set_fontsize(8)
            b.label.set_color(FG_TEXT)
            self._curve_btn_axes.append(ax)
            self._curve_btns.append(b)

        ax_add = self.fig.add_subplot(sb[row]);
        row += 1
        self.btn_add_curve = Button(ax_add, "+ New Curve", color=BG_WIDGET, hovercolor=BG_PANEL)
        self.btn_add_curve.label.set_fontsize(8)
        self.btn_add_curve.label.set_color(ACCENT)
        self.btn_add_curve.on_clicked(lambda e: self._add_curve())

        ax_del = self.fig.add_subplot(sb[row]);
        row += 1
        self.btn_del_curve = Button(ax_del, "Delete Active Curve", color=BG_WIDGET, hovercolor=BG_PANEL)
        self.btn_del_curve.label.set_fontsize(8)
        self.btn_del_curve.label.set_color(DANGER)
        self.btn_del_curve.on_clicked(lambda e: self._delete_active_curve())

        # ── section: Active curve settings ───────────────────────────────────
        ax_as_lbl = self.fig.add_subplot(sb[row]);
        row += 1
        ax_as_lbl.axis("off")
        self._active_settings_label = ax_as_lbl.text(
            0.04, 0.5, "ACTIVE CURVE SETTINGS",
            va="center", fontsize=7, color=FG_DIM,
            transform=ax_as_lbl.transAxes,
        )

        # Param exponent label
        ax_exp_lbl = self.fig.add_subplot(sb[row]);
        row += 1
        ax_exp_lbl.axis("off")
        ax_exp_lbl.text(0.04, 0.5, "Param exponent  (0=uniform  0.5=centripetal  1=chordal)",
                        va="center", fontsize=7, color=FG_DIM,
                        transform=ax_exp_lbl.transAxes)

        # Param exponent slider
        ax_exp_sl = self.fig.add_subplot(sb[row]);
        row += 1
        self.slider_exp = Slider(
            ax_exp_sl, "", 0.0, 2.0,
            valinit=0.5, valstep=0.01,
            color=ACCENT,
        )
        ax_exp_sl.set_facecolor(BG_WIDGET)
        self.slider_exp.label.set_color(FG_DIM)
        self.slider_exp.valtext.set_color(FG_TEXT)
        self.slider_exp.on_changed(self._on_exp_slider)

        # Param exponent textbox
        ax_exp_tb = self.fig.add_subplot(sb[row]);
        row += 1
        self.tb_exp = TextBox(ax_exp_tb, "", initial="0.50",
                              color=BG_WIDGET, hovercolor=BG_PANEL)
        self.tb_exp.label.set_color(FG_DIM)
        self.tb_exp.text_disp.set_color(FG_TEXT)
        self.tb_exp.text_disp.set_fontsize(8)
        self.tb_exp.on_submit(self._on_exp_text)

        # Re-parametrize button
        ax_reparam = self.fig.add_subplot(sb[row]);
        row += 1
        self.btn_reparam = Button(ax_reparam, "Apply Parametrization",
                                  color=BG_WIDGET, hovercolor=BG_PANEL)
        self.btn_reparam.label.set_fontsize(8)
        self.btn_reparam.label.set_color(ACCENT)
        self.btn_reparam.on_clicked(lambda e: self._apply_parametrization())

        # Color mode radio
        ax_cm_lbl = self.fig.add_subplot(sb[row]);
        row += 1
        ax_cm_lbl.axis("off")
        ax_cm_lbl.text(0.04, 0.5, "Colour mode",
                       va="center", fontsize=7, color=FG_DIM,
                       transform=ax_cm_lbl.transAxes)

        ax_cm = self.fig.add_subplot(sb[row:row + 2]);
        row += 2
        self.radio_colormode = RadioButtons(
            ax_cm, tuple(COLOR_MODES),
            activecolor=ACCENT,
        )
        ax_cm.set_facecolor(BG_PANEL)
        for lbl in self.radio_colormode.labels:
            lbl.set_color(FG_TEXT);
            lbl.set_fontsize(8)
        self.radio_colormode.on_clicked(self._on_colormode_changed)

        # Samples slider + textbox
        ax_smp_lbl = self.fig.add_subplot(sb[row]);
        row += 1
        ax_smp_lbl.axis("off")
        ax_smp_lbl.text(0.04, 0.5, "Samples per segment",
                        va="center", fontsize=7, color=FG_DIM,
                        transform=ax_smp_lbl.transAxes)

        ax_smp_sl = self.fig.add_subplot(sb[row]);
        row += 1
        self.slider_samples = Slider(
            ax_smp_sl, "", 1, 200,
            valinit=40, valstep=1,
            color=ACCENT,
        )
        ax_smp_sl.set_facecolor(BG_WIDGET)
        self.slider_samples.label.set_color(FG_DIM)
        self.slider_samples.valtext.set_color(FG_TEXT)
        self.slider_samples.on_changed(self._on_samples_slider)

        ax_smp_tb = self.fig.add_subplot(sb[row]);
        row += 1
        self.tb_samples = TextBox(ax_smp_tb, "", initial="40",
                                  color=BG_WIDGET, hovercolor=BG_PANEL)
        self.tb_samples.label.set_color(FG_DIM)
        self.tb_samples.text_disp.set_color(FG_TEXT)
        self.tb_samples.on_submit(self._on_samples_text)

        # ── section: Selected point editor ────────────────────────────────────
        ax_pe_lbl = self.fig.add_subplot(sb[row]);
        row += 1
        ax_pe_lbl.axis("off")
        ax_pe_lbl.text(0.04, 0.5, "SELECTED POINT",
                       va="center", fontsize=7, color=FG_DIM,
                       transform=ax_pe_lbl.transAxes)

        self._selected_pt: int = -1  # index within active curve

        def _make_coord_widgets(label, sb_row):
            ax_sl = self.fig.add_subplot(sb[sb_row])
            sl = Slider(ax_sl, label, -20, 20, valinit=0, color=ACCENT)
            ax_sl.set_facecolor(BG_WIDGET)
            sl.label.set_color(FG_DIM);
            sl.label.set_fontsize(8)
            sl.valtext.set_color(FG_TEXT)
            ax_tb = self.fig.add_subplot(sb[sb_row + 1])
            tb = TextBox(ax_tb, "", initial="0.00",
                         color=BG_WIDGET, hovercolor=BG_PANEL)
            tb.label.set_color(FG_DIM)
            tb.text_disp.set_color(FG_TEXT)
            tb.text_disp.set_fontsize(8)
            return sl, tb

        self.sl_px, self.tb_px = _make_coord_widgets("x", row);
        row += 2
        self.sl_py, self.tb_py = _make_coord_widgets("y", row);
        row += 2
        self.sl_pt, self.tb_pt = _make_coord_widgets("t", row);
        row += 2

        self.sl_pt.ax.set_facecolor(BG_WIDGET)
        # t slider range will be updated dynamically

        self.sl_px.on_changed(lambda v: self._on_coord_slider("x", v))
        self.sl_py.on_changed(lambda v: self._on_coord_slider("y", v))
        self.sl_pt.on_changed(lambda v: self._on_coord_slider("t", v))
        self.tb_px.on_submit(lambda s: self._on_coord_text("x", s))
        self.tb_py.on_submit(lambda s: self._on_coord_text("y", s))
        self.tb_pt.on_submit(lambda s: self._on_coord_text("t", s))

        ax_del_pt = self.fig.add_subplot(sb[row]);
        row += 1
        self.btn_del_pt = Button(ax_del_pt, "Delete Selected Point",
                                 color=BG_WIDGET, hovercolor=BG_PANEL)
        self.btn_del_pt.label.set_fontsize(8)
        self.btn_del_pt.label.set_color(DANGER)
        self.btn_del_pt.on_clicked(lambda e: self._delete_selected_point())

        # keep all widget axes in a list so we can toggle visibility
        self._sidebar_point_axes = [
            self.sl_px.ax, self.tb_px.ax,
            self.sl_py.ax, self.tb_py.ax,
            self.sl_pt.ax, self.tb_pt.ax,
            ax_del_pt,
        ]

        self._update_sidebar_curve_buttons()

    # ══════════════════════════════════════════════════════════════════════════
    # Event wiring
    # ══════════════════════════════════════════════════════════════════════════
    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

    # ══════════════════════════════════════════════════════════════════════════
    # Curve management
    # ══════════════════════════════════════════════════════════════════════════
    def _add_curve(self):
        c = Curve()
        self.curves.append(c)
        self.active_idx = len(self.curves) - 1
        self._selected_pt = -1
        self._update_sidebar_curve_buttons()
        self._sync_sidebar_to_active()
        self._redraw()

    def _delete_active_curve(self):
        if not self.curves:
            return
        self.curves.pop(self.active_idx)
        self.active_idx = max(0, self.active_idx - 1) if self.curves else -1
        self._selected_pt = -1
        self._update_sidebar_curve_buttons()
        self._sync_sidebar_to_active()
        self._redraw()

    def _set_active_curve(self, idx: int):
        if 0 <= idx < len(self.curves):
            self.active_idx = idx
            self._selected_pt = -1
            self._sync_sidebar_to_active()
            self._redraw()

    # ── sidebar curve buttons ─────────────────────────────────────────────────
    def _update_sidebar_curve_buttons(self):
        for i, (ax, btn) in enumerate(zip(self._curve_btn_axes, self._curve_btns)):
            if i < len(self.curves):
                c = self.curves[i]
                label = f"● {c.name}"
                btn.label.set_text(label)
                btn.label.set_color(c.color if i == self.active_idx else FG_TEXT)
                ax.set_facecolor(BG_WIDGET if i == self.active_idx else BG_PANEL)
                ax.set_visible(True)
                # rebind — capture i in closure
                btn.on_clicked(lambda e, idx=i: self._set_active_curve(idx))
            else:
                ax.set_visible(False)
        self.fig.canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════════════
    # Sidebar ↔ model synchronisation
    # ══════════════════════════════════════════════════════════════════════════
    def _sync_sidebar_to_active(self):
        """Push active-curve state into sidebar widgets."""
        if self.active_idx < 0 or not self.curves:
            return
        c = self.curves[self.active_idx]

        # param exponent
        self.slider_exp.set_val(np.clip(c.param_exponent, 0.0, 2.0))
        self.tb_exp.set_val(f"{c.param_exponent:.2f}")

        # color mode radio
        self.radio_colormode.set_active(COLOR_MODES.index(c.color_mode))

        # samples
        self.slider_samples.set_val(c.samples)
        self.tb_samples.set_val(str(c.samples))

        # point editor
        self._sync_point_editor()

    def _sync_point_editor(self):
        """Push selected-point coords into the x/y/t sliders and textboxes."""
        c = self._active_curve()
        has_pt = (c is not None and 0 <= self._selected_pt < len(c.points))

        for ax in self._sidebar_point_axes:
            ax.set_visible(has_pt)

        if not has_pt:
            self.fig.canvas.draw_idle()
            return

        pt = c.points[self._selected_pt]  # [t, x, y]
        t_vals = [p[0] for p in c.points]
        t_min, t_max = min(t_vals) - 1, max(t_vals) + 1

        def _set_sl(sl, val, lo, hi):
            sl.valmin = lo;
            sl.valmax = hi
            sl.ax.set_xlim(lo, hi)
            sl.set_val(np.clip(val, lo, hi))

        _set_sl(self.sl_px, pt[1], pt[1] - 20, pt[1] + 20)
        _set_sl(self.sl_py, pt[2], pt[2] - 20, pt[2] + 20)
        _set_sl(self.sl_pt, pt[0], t_min, t_max)

        self.tb_px.set_val(f"{pt[1]:.3f}")
        self.tb_py.set_val(f"{pt[2]:.3f}")
        self.tb_pt.set_val(f"{pt[0]:.3f}")

        self.fig.canvas.draw_idle()

    def _active_curve(self) -> Curve | None:
        if 0 <= self.active_idx < len(self.curves):
            return self.curves[self.active_idx]
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # Widget callbacks
    # ══════════════════════════════════════════════════════════════════════════
    def _on_mode_changed(self, label):
        self.interaction_mode = label

    def _on_exp_slider(self, val):
        c = self._active_curve()
        if c:
            c.param_exponent = round(float(val), 4)
            self.tb_exp.set_val(f"{c.param_exponent:.2f}")

    def _on_exp_text(self, text):
        try:
            val = float(text)
        except ValueError:
            return
        c = self._active_curve()
        if c:
            c.param_exponent = val
        self.slider_exp.set_val(np.clip(val, 0.0, 2.0))

    def _apply_parametrization(self):
        c = self._active_curve()
        if c:
            c.apply_parametrization()
            self._sync_point_editor()
            self._redraw()

    def _on_colormode_changed(self, label):
        c = self._active_curve()
        if c:
            c.color_mode = label
            self._redraw()

    def _on_samples_slider(self, val):
        c = self._active_curve()
        if c:
            c.samples = int(val)
            self.tb_samples.set_val(str(int(val)))
            self._redraw()

    def _on_samples_text(self, text):
        try:
            v = int(float(text))
            v = np.clip(v, 1, 10000)
            c = self._active_curve()
            if c:
                c.samples = v
            self.slider_samples.set_val(np.clip(v, 1, 200))
            self._redraw()
        except ValueError:
            pass

    # ── point coord callbacks ─────────────────────────────────────────────────
    def _set_point_coord(self, axis: str, val: float):
        """Write one coordinate of the selected point, re-sorting by t if needed."""
        c = self._active_curve()
        if c is None or self._selected_pt < 0:
            return
        pt = c.points[self._selected_pt]
        pt[{"x": 1, "y": 2, "t": 0}[axis]] = val
        if axis == "t":
            # re-sort and update selected index by object identity
            c.points.sort(key=lambda p: p[0])
            self._selected_pt = next(
                i for i, p in enumerate(c.points) if p is pt
            )

    def _on_coord_slider(self, axis: str, val: float):
        c = self._active_curve()
        if c is None or self._selected_pt < 0:
            return
        self._set_point_coord(axis, val)
        tb = {"x": self.tb_px, "y": self.tb_py, "t": self.tb_pt}[axis]
        tb.set_val(f"{val:.3f}")
        self._redraw()

    def _on_coord_text(self, axis: str, text: str):
        try:
            val = float(text)
        except ValueError:
            return
        c = self._active_curve()
        if c is None or self._selected_pt < 0:
            return
        self._set_point_coord(axis, val)
        sl = {"x": self.sl_px, "y": self.sl_py, "t": self.sl_pt}[axis]
        # widen slider range if value is outside current bounds
        if val < sl.valmin:
            sl.valmin = val - 1;
            sl.ax.set_xlim(sl.valmin, sl.valmax)
        if val > sl.valmax:
            sl.valmax = val + 1;
            sl.ax.set_xlim(sl.valmin, sl.valmax)
        sl.set_val(val)
        self._redraw()

    def _delete_selected_point(self):
        c = self._active_curve()
        if c is None or self._selected_pt < 0:
            return
        if 0 <= self._selected_pt < len(c.points):
            c.points.pop(self._selected_pt)
        self._selected_pt = -1
        self._sync_point_editor()
        self._redraw()

    # ══════════════════════════════════════════════════════════════════════════
    # Canvas mouse events
    # ══════════════════════════════════════════════════════════════════════════
    def _on_click(self, event):
        if event.inaxes is not self.ax_canvas:
            return
        if event.button == 3:  # right-click: pan handled by matplotlib toolbar
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        hit_curve, hit_pt = self._hit_test(x, y)

        if self.interaction_mode == "add":
            c = self._active_curve()
            if c is None:
                return
            if hit_curve == self.active_idx and hit_pt >= 0:
                # click on existing point → select it
                self._selected_pt = hit_pt
            else:
                # add new point; insert sorted by current parameter if possible
                new_t = self._guess_t_for_new_point(c, x, y)
                new_pt = [new_t, x, y]
                c.points.append(new_pt)
                # sort by t so interpolation stays valid when manual mode
                c.points.sort(key=lambda p: p[0])
                # use object identity — immune to float tolerance issues
                self._selected_pt = next(
                    i for i, p in enumerate(c.points) if p is new_pt
                )
                c.apply_parametrization()

        elif self.interaction_mode == "move":
            if hit_curve >= 0 and hit_pt >= 0:
                self._drag_curve_idx = hit_curve
                self._drag_pt_idx = hit_pt
                self._set_active_curve(hit_curve)
                self._selected_pt = hit_pt

        elif self.interaction_mode == "delete":
            if hit_curve >= 0 and hit_pt >= 0:
                self._set_active_curve(hit_curve)
                self._selected_pt = hit_pt
                self._delete_selected_point()
                return

        self._sync_point_editor()
        self._redraw()

    def _on_release(self, event):
        if self._drag_pt_idx >= 0:
            c = self._active_curve()
            if c:
                c.apply_parametrization()
            self._sync_point_editor()
            self._redraw()
        self._drag_pt_idx = -1
        self._drag_curve_idx = -1

    def _on_motion(self, event):
        if event.inaxes is not self.ax_canvas:
            return
        if self._drag_pt_idx < 0:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        c = self.curves[self._drag_curve_idx]
        if 0 <= self._drag_pt_idx < len(c.points):
            c.points[self._drag_pt_idx][1] = x
            c.points[self._drag_pt_idx][2] = y
            self._redraw()

    # ── hit testing ──────────────────────────────────────────────────────────
    def _hit_test(self, x: float, y: float) -> tuple[int, int]:
        """Return (curve_idx, point_idx) of the closest point within threshold, else (-1,-1)."""
        best_dist = float("inf")
        best_ci, best_pi = -1, -1
        ax = self.ax_canvas
        # Convert RADIUS px to data units
        px_per_data_x = ax.get_window_extent().width / (ax.get_xlim()[1] - ax.get_xlim()[0] + 1e-12)
        px_per_data_y = ax.get_window_extent().height / (ax.get_ylim()[1] - ax.get_ylim()[0] + 1e-12)
        rx = POINT_RADIUS / px_per_data_x
        ry = POINT_RADIUS / px_per_data_y

        for ci, c in enumerate(self.curves):
            for pi, pt in enumerate(c.points):
                dx = (x - pt[1]) / (rx + 1e-12)
                dy = (y - pt[2]) / (ry + 1e-12)
                d = np.hypot(dx, dy)
                if d < 1.0 and d < best_dist:
                    best_dist = d
                    best_ci, best_pi = ci, pi
        return best_ci, best_pi

    def _guess_t_for_new_point(self, c: Curve, x: float, y: float) -> float:
        """Assign t so the new point sorts immediately after the selected point."""
        if not c.points:
            return 0.0
        # Determine the reference point: selected if valid, else last
        if 0 <= self._selected_pt < len(c.points):
            ref_idx = self._selected_pt
        else:
            ref_idx = len(c.points) - 1
        ref = c.points[ref_idx]
        # t of the point that currently follows the reference (if any)
        if ref_idx + 1 < len(c.points):
            t_next = c.points[ref_idx + 1][0]
        else:
            t_next = None
        d = np.hypot(x - ref[1], y - ref[2])
        # midpoint t puts new point right after ref in sorted order;
        # apply_parametrization will recompute all t values anyway
        if t_next is not None:
            return (ref[0] + t_next) / 2.0
        return ref[0] + max(d, 1e-3)

    # ══════════════════════════════════════════════════════════════════════════
    # Drawing
    # ══════════════════════════════════════════════════════════════════════════
    def _redraw(self):
        ax = self.ax_canvas
        ax.cla()
        ax.set_facecolor(BG_DARK)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_title(
            "Left-click: add/move/delete  |  Right-click+drag: pan",
            color=FG_DIM, fontsize=8, pad=4,
        )

        for ci, c in enumerate(self.curves):
            if not c.visible or not c.points:
                continue

            arr = c.get_array()
            pts_x = arr[:, 1]
            pts_y = arr[:, 2]

            col = c.color
            is_active = (ci == self.active_idx)

            # ── interpolated curve ────────────────────────────────────────────
            resampled, ok = c.interpolate(draft=self._drag_pt_idx >= 0)
            if ok and resampled is not None and len(resampled) > 1:
                self._draw_curve_colored(ax, resampled, c, alpha=1.0 if is_active else 0.45)

            # ── control polygon ───────────────────────────────────────────────
            if len(arr) >= 2:
                ax.plot(pts_x, pts_y, "--",
                        color=col, lw=0.7, alpha=0.35, zorder=2)

            # ── control points ────────────────────────────────────────────────
            for pi, pt in enumerate(c.points):
                is_sel = (is_active and pi == self._selected_pt)
                ms = 10 if is_sel else 6
                mk = "D" if is_sel else "o"
                ec = "white" if is_sel else col
                ax.plot(pt[1], pt[2], mk,
                        ms=ms, color=col,
                        mec=ec, mew=1.5 if is_sel else 0.8,
                        zorder=5, alpha=1.0 if is_active else 0.6)

                # t label
                t_lbl = f"t={pt[0]:.2f}"
                ax.annotate(
                    t_lbl, xy=(pt[1], pt[2]),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=6.5, color=col,
                    alpha=0.9 if is_active else 0.5,
                    zorder=6,
                )

            # ── start / end markers ───────────────────────────────────────────
            ax.plot(*pts_x[:1], *pts_y[:1], "o",
                    ms=9, color=col, mec="white", mew=1.2, zorder=6,
                    alpha=1.0 if is_active else 0.5)
            ax.plot(*pts_x[-1:], *pts_y[-1:], "x",
                    ms=9, color=col, mew=2, zorder=6,
                    alpha=1.0 if is_active else 0.5)

            # curve name label near first point
            ax.text(
                pts_x[0], pts_y[0],
                f"  {c.name}",
                fontsize=7.5, color=col,
                va="center", alpha=0.85 if is_active else 0.4,
                zorder=7,
            )

        # auto-fit view — skip during drag to avoid bbox recalc every pixel
        if self._drag_pt_idx < 0:
            ax.margins(0.12)
            ax.autoscale_view()

        self.fig.canvas.draw_idle()

    def _draw_curve_colored(self, ax, resampled: npt.NDArray, c: Curve, alpha=1.0):
        """Draw interpolated curve as a single LineCollection — one draw call."""
        from matplotlib.collections import LineCollection
        cmap_name = "plasma" if c.color_mode == "parameter" else "inferno"
        cmap = plt.get_cmap(cmap_name)

        p0 = resampled[:-1]
        p1 = resampled[1:]

        if c.color_mode == "speed":
            values = np.linalg.norm(p1[:, 1:] - p0[:, 1:], axis=1)
        else:
            values = (p0[:, 0] + p1[:, 0]) / 2.0

        vmin, vmax = values.min(), values.max()
        if vmin == vmax:
            vmax = vmin + 1e-12
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        # shape: (N, 2, 2) — N segments, each with 2 points of (x, y)
        segments = np.stack([p0[:, 1:3], p1[:, 1:3]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm,
                            linewidth=2.0, alpha=alpha, zorder=3,
                            capstyle="round")
        lc.set_array(values)
        ax.add_collection(lc)

    # ══════════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════════
    def show(self):
        plt.show()

    def load_points(self, points: npt.NDArray, param_exponent: float = 0):
        """
        Load a (N,2) or (N,3) array into a new curve.
        If (N,3): columns are [t, x, y].
        If (N,2): columns are [x, y]; parametrization applied automatically.
        """
        c = self._active_curve()
        if c is None:
            self._add_curve()
            c = self._active_curve()
        c.param_exponent = param_exponent
        if points.shape[1] == 2:
            tmp = parametize(points, exponent=param_exponent)
        else:
            tmp = points.copy()
        c.points = [[row[0], row[1], row[2]] for row in tmp]
        self._sync_sidebar_to_active()
        self._redraw()


# ─────────────────────────────────────────────────────────────────────────────
# Demo entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vis = InteractiveVisualizer()

    demo_pts = np.array([
        [0.0, 0.0],
        [2.0, 1.0],
    ])
    vis.load_points(demo_pts, param_exponent=0)

    vis.show()
