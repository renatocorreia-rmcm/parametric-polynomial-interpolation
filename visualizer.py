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

# todo: add x(t) and y(t) coefficients tracker

import numpy as np
import numpy.typing as npt
import matplotlib as mpl
mpl.use("qtagg")  # ou PyQt6
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider, TextBox, RadioButtons
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import warnings

import sampling
# ── project modules ───────────────────────────────────────────────────────────
import vandermond
import sampling as sp_mod
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

# Per-curve monotonic colormaps (cycle in order of curve creation)
CURVE_COLORMAPS = [
    "Blues_r", "Oranges_r", "Purples_r", "Greens_r",
    "Reds_r", "YlOrBr_r", "PuBu_r", "RdPu_r",
]

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
        self.colormap: str = CURVE_COLORMAPS[(Curve._counter - 1) % len(CURVE_COLORMAPS)]
        self.points: list[list[float]] = []  # points stored as list of [t, x, y]
        self.param_exponent: float = 0
        self.color_mode: str = "parameter"
        self.samples: int = 15
        self.extrapolation: float = 0.0
        self.visible: bool = True
        self.show_polygon: bool = True
        self.show_labels: bool = True
        self._coeff_cache: np.ndarray | None = None
        self._cache_key: tuple | None = None
        self._cmap_cache: tuple | None = None  # (colormap_name, cmap_object)

    # ── derived ──────────────────────────────────────────────────────────────
    def get_array(self) -> npt.NDArray:  # curve points
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

    def _make_cache_key(self) -> bytes:
        """A hashable key that changes iff the control points change."""
        return np.array(self.points, dtype=float).tobytes()

    def interpolate(self, draft=False):
        """
        Return resampled_points.
        resampled_points is (M, 3) [t, x, y].
        Polynomial coefficients are cached and reused as long as control
        points have not changed; only polyval is re-run when sample
        count or extrapolation changes.
        """

        arr = self.get_array()
        if len(arr) < MIN_POINTS_FIT:
            return None, False

        # ── coefficient cache ─────────────────────────────────────────────
        key = self._make_cache_key()
        if key != self._cache_key or self._coeff_cache is None:
            self._coeff_cache = vandermond.coefficients(arr)
            self._cache_key = key
        polynomials = self._coeff_cache

        # set up samples
        ts = arr[:, 0]
        relative_sample_rate = self.samples if not draft else min(5, self.samples)
        relative_extrapolation_rate = self.extrapolation
        parameter_samples = sampling.generate_samples(
            ts=ts,
            relative_sample_rate=relative_sample_rate,
            relative_extrapolation_rate=relative_extrapolation_rate
        )

        # evaluate polynomials at samples
        resampled = sampling.sample_polynomials(
            parameter_samples=parameter_samples,
            polynomials=polynomials
        )

        return resampled, True


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
        self._drag_xlim: tuple = (-2, 2)
        self._drag_ylim: tuple = (-2, 2)

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
            49, 1,
            subplot_spec=outer[1],
            hspace=0.4,
        )
        self._sb = sb  # keep ref for later

        row = 0

        # ── section: Interaction mode ─────────────────────────────────────────
        ax_mode_lbl = self.fig.add_subplot(sb[row])
        row += 1
        ax_mode_lbl.axis("off")
        ax_mode_lbl.text(0.04, 0.5, "INTERACTION MODE",
                         va="center", fontsize=7, color=FG_DIM,
                         transform=ax_mode_lbl.transAxes)

        ax_mode = self.fig.add_subplot(sb[row:row + 3])
        row += 3
        self.radio_mode = RadioButtons(
            ax_mode, ("add", "move", "delete"),
            activecolor=ACCENT,
            radio_props={"s": [36]},
            label_props={"fontsize": [9, 9, 9], "color": [FG_TEXT]*3},
        )
        ax_mode.set_facecolor(BG_PANEL)
        ax_mode.axis("off")
        self.radio_mode.on_clicked(self._on_mode_changed)

        # ── section: Curves list ──────────────────────────────────────────────
        ax_cl_lbl = self.fig.add_subplot(sb[row])
        row += 1
        ax_cl_lbl.axis("off")
        ax_cl_lbl.text(0.04, 0.5, "CURVES",
                       va="center", fontsize=7, color=FG_DIM,
                       transform=ax_cl_lbl.transAxes)

        # Each curve gets 2 sidebar rows:
        #   Row A — one select button spanning the full width
        #   Row B — three pre-built toggle buttons side-by-side
        # All axes are built once here; _update_sidebar_curve_buttons only
        # mutates labels/colours, never creates new axes.
        self._curve_btn_axes = []
        self._curve_btns     = []
        self._curve_vis_btns = []
        self._curve_pol_btns = []
        self._curve_cmp_btns = []
        self._curve_lbl_btns = []
        MAX_CURVE_SLOTS = 5
        for slot in range(MAX_CURVE_SLOTS):
            # Row A — full-width select button
            ax_a = self.fig.add_subplot(sb[row])
            row += 1
            ax_a.set_facecolor(BG_PANEL)
            b_sel = Button(ax_a, "", color=BG_PANEL, hovercolor=BG_WIDGET)
            b_sel.label.set_fontsize(8)
            b_sel.label.set_color(FG_TEXT)
            self._curve_btn_axes.append(ax_a)
            self._curve_btns.append(b_sel)
            # Row B — 3 toggle buttons using a nested 1x3 GridSpec
            sub_gs = gridspec.GridSpecFromSubplotSpec(
                1, 4, subplot_spec=sb[row], wspace=0.04
            )
            row += 1
            ax_vis = self.fig.add_subplot(sub_gs[0, 0])
            ax_pol = self.fig.add_subplot(sub_gs[0, 1])
            ax_cmp = self.fig.add_subplot(sub_gs[0, 2])
            ax_lbl = self.fig.add_subplot(sub_gs[0, 3])
            for ax_t in (ax_vis, ax_pol, ax_cmp, ax_lbl):
                ax_t.set_facecolor(BG_PANEL)
                ax_t.set_visible(False)
            b_vis = Button(ax_vis, "", color=BG_PANEL, hovercolor=BG_WIDGET)
            b_pol = Button(ax_pol, "", color=BG_PANEL, hovercolor=BG_WIDGET)
            b_cmp = Button(ax_cmp, "", color=BG_PANEL, hovercolor=BG_WIDGET)
            b_lbl = Button(ax_lbl, "", color=BG_PANEL, hovercolor=BG_WIDGET)
            for b in (b_vis, b_pol, b_cmp, b_lbl):
                b.label.set_fontsize(7)
                b.label.set_color(FG_DIM)
            b_vis.on_clicked(lambda e, idx=slot: self._toggle_curve_visible(idx))
            b_pol.on_clicked(lambda e, idx=slot: self._toggle_curve_polygon(idx))
            b_cmp.on_clicked(lambda e, idx=slot: self._cycle_curve_colormap(idx))
            b_lbl.on_clicked(lambda e, idx=slot: self._toggle_curve_labels(idx))
            self._curve_vis_btns.append(b_vis)
            self._curve_pol_btns.append(b_pol)
            self._curve_cmp_btns.append(b_cmp)
            self._curve_lbl_btns.append(b_lbl)

        ax_add = self.fig.add_subplot(sb[row])
        row += 1
        self.btn_add_curve = Button(ax_add, "+ New Curve", color=BG_WIDGET, hovercolor=BG_PANEL)
        self.btn_add_curve.label.set_fontsize(8)
        self.btn_add_curve.label.set_color(ACCENT)
        self.btn_add_curve.on_clicked(lambda e: self._add_curve())

        ax_del = self.fig.add_subplot(sb[row])
        row += 1
        self.btn_del_curve = Button(ax_del, "Delete Active Curve", color=BG_WIDGET, hovercolor=BG_PANEL)
        self.btn_del_curve.label.set_fontsize(8)
        self.btn_del_curve.label.set_color(DANGER)
        self.btn_del_curve.on_clicked(lambda e: self._delete_active_curve())

        ax_save = self.fig.add_subplot(sb[row])
        row += 1
        self.btn_save = Button(ax_save, "Save Image", color=BG_WIDGET, hovercolor=BG_PANEL)
        self.btn_save.label.set_fontsize(8)
        self.btn_save.label.set_color(ACCENT)
        self.btn_save.on_clicked(lambda e: self._save_image())

        # ── section: Active curve settings ───────────────────────────────────
        ax_as_lbl = self.fig.add_subplot(sb[row])
        row += 1
        ax_as_lbl.axis("off")
        self._active_settings_label = ax_as_lbl.text(
            0.04, 0.5, "ACTIVE CURVE SETTINGS",
            va="center", fontsize=7, color=FG_DIM,
            transform=ax_as_lbl.transAxes,
        )

        # Param exponent label
        ax_exp_lbl = self.fig.add_subplot(sb[row])
        row += 1
        ax_exp_lbl.axis("off")
        ax_exp_lbl.text(0.04, 0.5, "Param exponent  (0=uniform  0.5=centripetal  1=chordal)",
                        va="center", fontsize=7, color=FG_DIM,
                        transform=ax_exp_lbl.transAxes)

        # Param exponent slider
        ax_exp_sl = self.fig.add_subplot(sb[row])
        row += 1
        self.slider_exp = Slider(
            ax_exp_sl, "", 0.0, 2.0,
            valinit=0.5, valstep=0.05,
            color=ACCENT,
        )
        ax_exp_sl.set_facecolor(BG_WIDGET)
        self.slider_exp.label.set_color(FG_DIM)
        self.slider_exp.valtext.set_color(FG_TEXT)
        self.slider_exp.on_changed(self._on_exp_slider)

        # Param exponent textbox
        ax_exp_tb = self.fig.add_subplot(sb[row])
        row += 1
        self.tb_exp = TextBox(ax_exp_tb, "", initial="0.50",
                              color=BG_WIDGET, hovercolor=BG_PANEL)
        self.tb_exp.label.set_color(FG_DIM)
        self.tb_exp.text_disp.set_color(FG_TEXT)
        self.tb_exp.text_disp.set_fontsize(8)
        self.tb_exp.on_submit(self._on_exp_text)

        # Color mode radio
        ax_cm_lbl = self.fig.add_subplot(sb[row])
        row += 1
        ax_cm_lbl.axis("off")
        ax_cm_lbl.text(0.04, 0.5, "Colour mode",
                       va="center", fontsize=7, color=FG_DIM,
                       transform=ax_cm_lbl.transAxes)

        ax_cm = self.fig.add_subplot(sb[row:row + 3])
        row += 3
        n_cm = len(COLOR_MODES)
        self.radio_colormode = RadioButtons(
            ax_cm, tuple(COLOR_MODES),
            activecolor=ACCENT,
            radio_props={"s": [36]},
            label_props={"fontsize": [8]*n_cm, "color": [FG_TEXT]*n_cm},
        )
        ax_cm.set_facecolor(BG_PANEL)
        ax_cm.axis("off")
        self.radio_colormode.on_clicked(self._on_colormode_changed)

        # Samples slider + textbox
        ax_smp_lbl = self.fig.add_subplot(sb[row])
        row += 1
        ax_smp_lbl.axis("off")
        ax_smp_lbl.text(0.04, 0.5, "Samples per segment",
                        va="center", fontsize=7, color=FG_DIM,
                        transform=ax_smp_lbl.transAxes)

        ax_smp_sl = self.fig.add_subplot(sb[row])
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

        ax_smp_tb = self.fig.add_subplot(sb[row])
        row += 1
        self.tb_samples = TextBox(ax_smp_tb, "", initial="40",
                                  color=BG_WIDGET, hovercolor=BG_PANEL)
        self.tb_samples.label.set_color(FG_DIM)
        self.tb_samples.text_disp.set_color(FG_TEXT)
        self.tb_samples.on_submit(self._on_samples_text)

        # Extrapolation slider + textbox
        ax_ext_lbl = self.fig.add_subplot(sb[row])
        row += 1
        ax_ext_lbl.axis("off")
        ax_ext_lbl.text(0.04, 0.5, "Extrapolation (fraction of t-span per side)",
                        va="center", fontsize=7, color=FG_DIM,
                        transform=ax_ext_lbl.transAxes)

        ax_ext_sl = self.fig.add_subplot(sb[row])
        row += 1
        self.slider_ext = Slider(
            ax_ext_sl, "", 0.0, 0.5,
            valinit=0.0,
            color=ACCENT,
        )
        ax_ext_sl.set_facecolor(BG_WIDGET)
        self.slider_ext.label.set_color(FG_DIM)
        self.slider_ext.valtext.set_color(FG_TEXT)
        self.slider_ext.on_changed(self._on_ext_slider)

        ax_ext_tb = self.fig.add_subplot(sb[row])
        row += 1
        self.tb_ext = TextBox(ax_ext_tb, "", initial="0.00",
                              color=BG_WIDGET, hovercolor=BG_PANEL)
        self.tb_ext.label.set_color(FG_DIM)
        self.tb_ext.text_disp.set_color(FG_TEXT)
        self.tb_ext.text_disp.set_fontsize(8)
        self.tb_ext.on_submit(self._on_ext_text)

        # ── section: Selected point editor ────────────────────────────────────
        ax_pe_lbl = self.fig.add_subplot(sb[row])
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
            sl.label.set_color(FG_DIM)
            sl.label.set_fontsize(8)
            sl.valtext.set_color(FG_TEXT)
            ax_tb = self.fig.add_subplot(sb[sb_row + 1])
            tb = TextBox(ax_tb, "", initial="0.00",
                         color=BG_WIDGET, hovercolor=BG_PANEL)
            tb.label.set_color(FG_DIM)
            tb.text_disp.set_color(FG_TEXT)
            tb.text_disp.set_fontsize(8)
            return sl, tb

        self.sl_px, self.tb_px = _make_coord_widgets("x", row)
        row += 2
        self.sl_py, self.tb_py = _make_coord_widgets("y", row)
        row += 2
        self.sl_pt, self.tb_pt = _make_coord_widgets("t", row)
        row += 2

        self.sl_pt.ax.set_facecolor(BG_WIDGET)
        # t slider range will be updated dynamically

        self.sl_px.on_changed(lambda v: self._on_coord_slider("x", v))
        self.sl_py.on_changed(lambda v: self._on_coord_slider("y", v))
        self.sl_pt.on_changed(lambda v: self._on_coord_slider("t", v))
        self.tb_px.on_submit(lambda s: self._on_coord_text("x", s))
        self.tb_py.on_submit(lambda s: self._on_coord_text("y", s))
        self.tb_pt.on_submit(lambda s: self._on_coord_text("t", s))

        ax_del_pt = self.fig.add_subplot(sb[row])
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
            self._update_sidebar_curve_buttons()
            self._sync_sidebar_to_active()
            self._redraw()

    def _toggle_curve_visible(self, idx: int):
        if 0 <= idx < len(self.curves):
            self.curves[idx].visible = not self.curves[idx].visible
            self._update_sidebar_curve_buttons()
            self._redraw()

    def _toggle_curve_polygon(self, idx: int):
        if 0 <= idx < len(self.curves):
            self.curves[idx].show_polygon = not self.curves[idx].show_polygon
            self._update_sidebar_curve_buttons()
            self._redraw()

    def _cycle_curve_colormap(self, idx: int):
        if 0 <= idx < len(self.curves):
            c = self.curves[idx]
            cur = CURVE_COLORMAPS.index(c.colormap) if c.colormap in CURVE_COLORMAPS else 0
            c.colormap = CURVE_COLORMAPS[(cur + 1) % len(CURVE_COLORMAPS)]
            self._update_sidebar_curve_buttons()
            self._redraw()

    def _toggle_curve_labels(self, idx: int):
        if 0 <= idx < len(self.curves):
            self.curves[idx].show_labels = not self.curves[idx].show_labels
            self._update_sidebar_curve_buttons()
            self._redraw()

    # ── sidebar curve buttons ─────────────────────────────────────────────────
    def _update_sidebar_curve_buttons(self):
        for i in range(len(self._curve_btns)):
            ax_a  = self._curve_btn_axes[i]
            btn   = self._curve_btns[i]
            b_vis = self._curve_vis_btns[i]
            b_pol = self._curve_pol_btns[i]
            b_cmp = self._curve_cmp_btns[i]
            b_lbl = self._curve_lbl_btns[i]

            if i < len(self.curves):
                c = self.curves[i]
                is_active = (i == self.active_idx)
                bg = BG_WIDGET if is_active else BG_PANEL

                # ── Row A: select button ──────────────────────────────────
                active_marker = "▶ " if is_active else "   "
                btn.label.set_text(f"{active_marker}● {c.name}")
                btn.label.set_color(c.color)
                btn.label.set_fontweight("bold" if is_active else "normal")
                btn.ax.set_facecolor(bg)
                btn.color = bg          # keep Button internal state in sync
                ax_a.set_visible(True)
                btn.on_clicked(lambda e, idx=i: self._set_active_curve(idx))

                # ── Row B: toggle buttons (pre-built, just update) ────────
                cmap_short = c.colormap.replace('_r', '')
                b_vis.label.set_text("show" if c.visible      else "hide")
                b_pol.label.set_text("poly" if c.show_polygon  else "poly off")
                b_cmp.label.set_text(cmap_short[:7])
                b_lbl.label.set_text("t=" if c.show_labels    else "t= off")
                b_vis.label.set_color(c.color if c.visible      else FG_DIM)
                b_pol.label.set_color(c.color if c.show_polygon  else FG_DIM)
                b_cmp.label.set_color(c.color)
                b_lbl.label.set_color(c.color if c.show_labels  else FG_DIM)
                for b in (b_vis, b_pol, b_cmp, b_lbl):
                    b.ax.set_facecolor(bg)
                    b.color = bg
                    b.ax.set_visible(True)
            else:
                ax_a.set_visible(False)
                for b in (b_vis, b_pol, b_cmp, b_lbl):
                    b.ax.set_visible(False)

        self.fig.canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════════════
    # Sidebar ↔ model synchronisation
    # ══════════════════════════════════════════════════════════════════════════
    def _sync_sidebar_to_active(self):
        """Push active-curve state into sidebar widgets."""
        if self.active_idx < 0 or not self.curves:
            return
        c = self.curves[self.active_idx]

        # active curve label
        self._active_settings_label.set_text(f"CURVE: {c.name}")

        # param exponent
        self.slider_exp.set_val(np.clip(c.param_exponent, 0.0, 2.0))
        self.tb_exp.set_val(f"{c.param_exponent:.2f}")

        # color mode radio
        self.radio_colormode.set_active(COLOR_MODES.index(c.color_mode))

        # samples
        self.slider_samples.set_val(c.samples)
        self.tb_samples.set_val(str(c.samples))

        # extrapolation
        self.slider_ext.set_val(np.clip(c.extrapolation, 0.0, 1.0))
        self.tb_ext.set_val(f"{c.extrapolation:.2f}")

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
            sl.valmin = lo
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
            c.apply_parametrization()
            self._sync_point_editor()
        self._redraw()

    def _on_exp_text(self, text):
        try:
            val = float(text)
        except ValueError:
            return
        c = self._active_curve()
        if c:
            c.param_exponent = val
            c.apply_parametrization()
            self._sync_point_editor()
        self.slider_exp.set_val(np.clip(val, 0.0, 2.0))
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

    def _on_ext_slider(self, val):
        c = self._active_curve()
        if c:
            c.extrapolation = round(float(val), 4)
            self.tb_ext.set_val(f"{c.extrapolation:.2f}")
        self._redraw()

    def _on_ext_text(self, text):
        try:
            val = max(0.0, float(text))
        except ValueError:
            return
        c = self._active_curve()
        if c:
            c.extrapolation = val
        self.slider_ext.set_val(np.clip(val, 0.0, 0.5))
        self._redraw()

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
        self._redraw(draft=True)

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
            sl.valmin = val - 1
            sl.ax.set_xlim(sl.valmin, sl.valmax)
        if val > sl.valmax:
            sl.valmax = val + 1
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
                # snapshot current view so we can freeze it during the drag
                self._drag_xlim = self.ax_canvas.get_xlim()
                self._drag_ylim = self.ax_canvas.get_ylim()
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
    def _redraw(self, draft: bool = False):
        ax = self.ax_canvas
        ax.cla()
        ax.set_facecolor(BG_DARK)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_title(
            "Left-click: add/move/delete  |  Right-click+drag: pan",
            color=FG_DIM, fontsize=8, pad=4,
        )

        # bbox accumulators — filled inside the draw loop to avoid a second pass
        bbox_x: list[npt.NDArray] = []
        bbox_y: list[npt.NDArray] = []

        for ci, c in enumerate(self.curves):
            if not c.visible or not c.points:
                continue

            arr = c.get_array()
            pts_x = arr[:, 1]
            pts_y = arr[:, 2]

            col = c.color
            is_active = (ci == self.active_idx)

            # ── interpolated curve ────────────────────────────────────────────
            resampled, ok = c.interpolate(draft=self._drag_pt_idx >= 0 or draft)
            if ok and resampled is not None and len(resampled) > 1:
                self._draw_curve_colored(ax, resampled, c, alpha=1.0)

            # ── control polygon ───────────────────────────────────────────────
            if len(arr) >= 2 and c.show_polygon:
                ax.plot(pts_x, pts_y, "--",
                        color=col, lw=0.7, alpha=0.35, zorder=2)

            # ── control points (batched) ──────────────────────────────────────
            sel = self._selected_pt if is_active else -1

            normal_idx = [pi for pi in range(len(c.points)) if pi != sel]
            if normal_idx:
                nx = [c.points[pi][1] for pi in normal_idx]
                ny = [c.points[pi][2] for pi in normal_idx]
                ax.scatter(nx, ny, s=36, color=col,
                           edgecolors=col, linewidths=0.8,
                           marker="o", zorder=5)

            if 0 <= sel < len(c.points):
                sp = c.points[sel]
                ax.scatter([sp[1]], [sp[2]], s=100, color=col,
                           edgecolors="white", linewidths=1.5,
                           marker="D", zorder=5)

            # t labels — one annotate per point (text artists can't be batched)
            if c.show_labels:
                for pt in c.points:
                    ax.annotate(
                        f"t={pt[0]:.2f}", xy=(pt[1], pt[2]),
                        xytext=(4, 6), textcoords="offset points",
                        fontsize=6.5, color=col, alpha=0.9, zorder=6,
                    )

            # ── start / end markers ───────────────────────────────────────────
            ax.plot(*pts_x[:1], *pts_y[:1], "o",
                    ms=9, color=col, mec="white", mew=1.2, zorder=6)
            ax.plot(*pts_x[-1:], *pts_y[-1:], "x",
                    ms=9, color=col, mew=2, zorder=6)

            # ── accumulate bbox data (reuse arr already in hand) ──────────────
            bbox_x.append(pts_x)
            bbox_y.append(pts_y)
            if resampled is not None:
                bbox_x.append(resampled[:, 1])
                bbox_y.append(resampled[:, 2])

        # always-square view
        if bbox_x:
            all_x = np.concatenate(bbox_x)
            all_y = np.concatenate(bbox_y)
            xlo, xhi = all_x.min(), all_x.max()
            ylo, yhi = all_y.min(), all_y.max()
            xmid = (xlo + xhi) / 2
            ymid = (ylo + yhi) / 2
            half = max((xhi - xlo) / 2, (yhi - ylo) / 2, 1.0) * 1.12
            if self._drag_pt_idx >= 0:
                # union with the saved drag-start bounds so we only expand
                new_xlo = min(xmid - half, self._drag_xlim[0])
                new_xhi = max(xmid + half, self._drag_xlim[1])
                new_ylo = min(ymid - half, self._drag_ylim[0])
                new_yhi = max(ymid + half, self._drag_ylim[1])
                xmid = (new_xlo + new_xhi) / 2
                ymid = (new_ylo + new_yhi) / 2
                half = max((new_xhi - new_xlo) / 2, (new_yhi - new_ylo) / 2)
            ax.set_xlim(xmid - half, xmid + half)
            ax.set_ylim(ymid - half, ymid + half)
        else:
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)

        self.fig.canvas.draw_idle()

    def _draw_curve_colored(self, ax, resampled: npt.NDArray, c: Curve, alpha=1.0):
        """Draw interpolated curve as a single LineCollection — one draw call."""
        # cache cmap object — colormaps are immutable, no need to re-fetch every frame
        if c._cmap_cache is None or c._cmap_cache[0] != c.colormap:
            c._cmap_cache = (c.colormap, plt.get_cmap(c.colormap))
        cmap = c._cmap_cache[1]

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
        # build as a view: xy is (M, 2), sliding window of pairs along axis 0
        xy = resampled[:, 1:3]  # view, no copy
        segments = np.lib.stride_tricks.sliding_window_view(xy, (2, 2)).reshape(-1, 2, 2)
        lc = LineCollection(segments, cmap=cmap, norm=norm,
                            linewidth=2.0, alpha=alpha, zorder=3,
                            capstyle="round")
        lc.set_array(values)
        ax.add_collection(lc)

    def _save_image(self):
        """Render a clean publication-style image of all visible curves and save it."""
        import os

        os.makedirs("output", exist_ok=True)

        # count existing files to avoid overwriting
        existing = [f for f in os.listdir("output") if f.startswith("curves_") and f.endswith(".svg")]
        save_path = f"output/curves_{len(existing):03d}.svg"

        fig_out, ax_out = plt.subplots(figsize=(8, 8))
        fig_out.patch.set_facecolor(BG_DARK)
        ax_out.set_facecolor(BG_DARK)
        ax_out.set_aspect("equal", adjustable="box")
        ax_out.grid(True, color="#2a2a2a", linestyle="--", alpha=0.6)
        ax_out.tick_params(colors=FG_DIM)
        for spine in ax_out.spines.values():
            spine.set_edgecolor(FG_DIM)

        all_x, all_y = [], []

        for c in self.curves:
            if not c.visible or not c.points:
                continue
            arr = c.get_array()
            pts_x, pts_y = arr[:, 1], arr[:, 2]
            all_x.extend(pts_x); all_y.extend(pts_y)

            # interpolated curve
            resampled, ok = c.interpolate()
            if ok and resampled is not None and len(resampled) > 1:
                self._draw_curve_colored(ax_out, resampled, c, alpha=1.0)
                all_x.extend(resampled[:, 1]); all_y.extend(resampled[:, 2])

            # control polygon
            if len(arr) >= 2:
                ax_out.plot(pts_x, pts_y, "--", color=c.color, lw=0.7, alpha=0.35, zorder=2)

            # control points
            ax_out.scatter(pts_x[1:-1], pts_y[1:-1], color=c.color, s=18, zorder=5)
            ax_out.plot(pts_x[0],  pts_y[0],  "o", ms=7, color=c.color, mec="white", mew=1, zorder=6)
            ax_out.plot(pts_x[-1], pts_y[-1], "x", ms=7, color=c.color, mew=1.8, zorder=6)

        # square bbox matching the interactive view
        if all_x:
            xlo, xhi = min(all_x), max(all_x)
            ylo, yhi = min(all_y), max(all_y)
            xmid, ymid = (xlo + xhi) / 2, (ylo + yhi) / 2
            half = max((xhi - xlo) / 2, (yhi - ylo) / 2, 1.0) * 1.12
            ax_out.set_xlim(xmid - half, xmid + half)
            ax_out.set_ylim(ymid - half, ymid + half)

        fig_out.tight_layout()
        fig_out.savefig(save_path, dpi=300, facecolor=BG_DARK)
        plt.close(fig_out)
        print(f"Saved: {save_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════════
    def show(self):

        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()

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
    vis.load_points(demo_pts)

    vis.show()