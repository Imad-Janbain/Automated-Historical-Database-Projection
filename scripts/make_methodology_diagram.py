"""Multi-panel methodology figure for the README and for slide decks.

The figure is a **schematic**, not a data figure. Every visual element
describes algorithm structure; nothing inside it claims to be a result.
The waveforms in Panel A are deterministic synthetic illustrations
(stylised sin/cos sums) used the way a textbook diagram uses placeholder
shapes — to help the reader picture the structure of the data without
representing any specific measurement.

Panels
------
A — *The reconstruction problem.*
    Channel-bands view. Horizontal axis is time, each row is a channel.
    A vertical boundary separates the historical window (where target
    channels are missing) from the modern window (where every channel
    is present). The visual encoding is: blue waveforms = continuous,
    red hatching = missing.

B — *Phase 1: Build the trial database.*
    The trial space ``T × F × M`` rendered as an actual grid (rows are
    feature combinations, columns are model architectures). The grid
    feeds into a model-cohort glyph, then into a queryable database.
    "For each target T" sits above the grid to make the outer loop
    explicit without redrawing it.

C — *Phase 2: Progressive reconstruction.*
    Three sequential step cards showing how the available-features set
    grows over time. Each card visualises which channels exist at that
    moment, which model is selected from the database, and which target
    gets unlocked. A loop arrow above the cards conveys iteration; a
    decision diamond + green output box terminate the flow.

Outputs
-------
* ``images/methodology_workflow.svg`` — vector
* ``images/methodology_workflow.png`` — 200 DPI raster
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

# ============================================================================
# Style tokens
# ============================================================================
FIG_W = 14.0
FIG_H = 12.0
DPI = 200

NAVY = "#1f3a5f"
BLUE = "#4a72a8"
LIGHT_BLUE = "#cfdef0"
PALE_BLUE = "#eaf1f9"
SAND = "#efd9a8"
WARM = "#b8782e"
PALE_SAND = "#faf3e3"
RED = "#a83228"
PALE_RED = "#f7e8e6"
GREEN = "#2d7d3f"
PALE_GREEN = "#e3f0e6"
INK = "#1a1a1a"
GREY = "#7a7a7a"
LIGHT_GREY = "#d8d8d8"
PALE_GREY = "#eeeeee"
WHITE = "#ffffff"

PANEL_A_BG = "#f4f7fb"
PANEL_B_BG = "#faf5ec"
PANEL_C_BG = "#f7f1ef"


# ============================================================================
# Drawing primitives
# ============================================================================
def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 10,
            "text.color": INK,
            "axes.edgecolor": INK,
            "savefig.facecolor": WHITE,
        },
    )


def stylized_signal(n: int, freq: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """Deterministic stylised waveform — pure schematic, no data."""
    t = np.linspace(0, 4 * np.pi, n)
    return (
        np.sin(freq * t + phase)
        + 0.35 * np.sin(2.7 * freq * t + phase * 1.3)
        + 0.20 * np.cos(5.1 * freq * t + phase * 0.7)
    )


def panel_bg(ax, x, y, w, h, fill, edge=NAVY) -> None:
    panel = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.18,rounding_size=0.30",
        facecolor=fill, edgecolor=edge, linewidth=0.9, zorder=0,
    )
    ax.add_patch(panel)


def panel_letter(ax, x, y, letter, color) -> None:
    ax.add_patch(Circle((x, y), 0.30, facecolor=color, edgecolor=INK,
                        linewidth=1.4, zorder=10))
    ax.text(x, y, letter, ha="center", va="center",
            fontsize=15, fontweight="bold", color=WHITE, zorder=11)


def panel_heading(ax, x, y, title, subtitle, accent=NAVY) -> None:
    ax.text(x, y, title, fontsize=14, fontweight="bold",
            color=accent, va="center", ha="left")
    ax.text(x, y - 0.32, subtitle, fontsize=9.5, color=GREY,
            style="italic", va="center", ha="left")


def database_glyph(ax, cx, cy, w, h, label, label_color=WHITE) -> None:
    ellipse_h = h * 0.18
    body_top = cy + h / 2 - ellipse_h / 2
    body_bottom = cy - h / 2 + ellipse_h / 2
    ax.add_patch(Rectangle(
        (cx - w / 2, body_bottom), w, body_top - body_bottom,
        facecolor=NAVY, edgecolor=NAVY, linewidth=1.3,
    ))
    ax.add_patch(Ellipse((cx, body_top), w, ellipse_h,
                         facecolor=BLUE, edgecolor=NAVY, linewidth=1.3))
    ax.add_patch(Ellipse((cx, body_bottom), w, ellipse_h,
                         facecolor="none", edgecolor=NAVY, linewidth=1.3))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=label_color)


def channel_dot(ax, cx, cy, r, color, edge=INK) -> None:
    ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor=edge, linewidth=1.0))


# ============================================================================
# Panel A — The reconstruction problem
# ============================================================================
def draw_panel_a(ax) -> None:
    panel_bg(ax, 0.4, 7.65, 13.2, 3.55, PANEL_A_BG)

    panel_letter(ax, 0.95, 10.78, "A", NAVY)
    panel_heading(
        ax, 1.45, 10.78,
        "The reconstruction problem",
        "Predictor channels are continuous over the full record. Target "
        "channels exist only in the recent modern window.",
    )

    # Visual area: 5 horizontal channel bands
    x_start = 2.95
    x_modern = 9.40
    x_end = 12.95

    # Window labels above the bands
    ax.text((x_start + x_modern) / 2, 9.95,
            "HISTORICAL WINDOW",
            fontsize=9.5, fontweight="bold", color=NAVY, ha="center")
    ax.text((x_start + x_modern) / 2, 9.75,
            "target channels missing — to reconstruct",
            fontsize=8, color=GREY, ha="center", style="italic")

    ax.text((x_modern + x_end) / 2, 9.95,
            "MODERN WINDOW",
            fontsize=9.5, fontweight="bold", color=NAVY, ha="center")
    ax.text((x_modern + x_end) / 2, 9.75,
            "all channels present — used for training",
            fontsize=8, color=GREY, ha="center", style="italic")

    bands = [
        ("Predictor 1", 9.55, BLUE,  x_start),
        ("Predictor 2", 9.20, BLUE,  x_start),
        ("Target 1",    8.85, GREEN, x_modern),
        ("Target 2",    8.50, GREEN, x_modern),
        ("Target 3",    8.15, GREEN, x_modern),
    ]

    band_h = 0.30
    for label, y, colour, available_from in bands:
        # Background strip showing the full historical extent (pale grey)
        ax.add_patch(Rectangle(
            (x_start, y - band_h / 2), x_end - x_start, band_h,
            facecolor=PALE_GREY, edgecolor="none", zorder=1,
        ))
        # Available region: solid coloured block (data present)
        ax.add_patch(Rectangle(
            (available_from, y - band_h / 2),
            x_end - available_from, band_h,
            facecolor=colour, edgecolor="none", alpha=0.85, zorder=2,
        ))
        # Missing region: red wash + diagonal hatching
        if available_from > x_start:
            ax.add_patch(Rectangle(
                (x_start, y - band_h / 2),
                available_from - x_start, band_h,
                facecolor=PALE_RED, edgecolor="none", zorder=1.5,
            ))
            ax.add_patch(Rectangle(
                (x_start, y - band_h / 2),
                available_from - x_start, band_h,
                facecolor="none", edgecolor=RED, alpha=0.65,
                hatch="////", linewidth=0, zorder=1.7,
            ))
        # Hairline border around the band
        ax.add_patch(Rectangle(
            (x_start, y - band_h / 2), x_end - x_start, band_h,
            facecolor="none", edgecolor=INK, linewidth=0.6, zorder=3,
        ))
        # Channel label on the left
        ax.text(x_start - 0.12, y, label,
                ha="right", va="center", fontsize=9, color=INK)

    # Vertical dashed boundary
    ax.plot([x_modern, x_modern], [7.90, 9.75],
            color=INK, linestyle=(0, (4, 3)), linewidth=1.2, zorder=4)

    # Compact legend strip beneath the bands
    legend_y = 7.82
    legend_items = [
        ("predictor present", BLUE, NAVY, None),
        ("target present", GREEN, NAVY, None),
        ("missing", PALE_RED, RED, "////"),
    ]
    legend_x = x_start + 0.10
    for label, fill_c, edge_c, hatch in legend_items:
        ax.add_patch(Rectangle(
            (legend_x, legend_y - 0.07), 0.28, 0.14,
            facecolor=fill_c, edgecolor=edge_c, linewidth=0.7,
            hatch=hatch, alpha=0.85,
        ))
        ax.text(legend_x + 0.36, legend_y, label,
                ha="left", va="center", fontsize=8, color=INK)
        legend_x += 1.95

    # Time arrow on the right
    ax.add_patch(FancyArrowPatch(
        (x_end - 0.05, legend_y), (x_end + 0.20, legend_y),
        arrowstyle="->", mutation_scale=12, linewidth=1.0, color=GREY,
    ))
    ax.text(x_end + 0.30, legend_y, "time",
            ha="left", va="center", fontsize=8.5, color=GREY, style="italic")


# ============================================================================
# Panel B — Phase 1: Build the trial database
# ============================================================================
def draw_panel_b(ax) -> None:
    panel_bg(ax, 0.4, 4.20, 13.2, 3.20, PANEL_B_BG, edge=WARM)

    panel_letter(ax, 0.95, 7.00, "B", WARM)
    panel_heading(
        ax, 1.45, 7.00,
        "Phase 1 — Build the trial database",
        "Sweep every (target × feature combination × architecture) "
        "trial. Persist trained models and held-out scores.",
        accent=WARM,
    )

    # ----- Trial grid (left) -----
    grid_x0, grid_y0 = 2.65, 4.55
    cell_w, cell_h = 0.55, 0.35
    n_rows, n_cols = 5, 4
    grid_w = cell_w * n_cols
    grid_h = cell_h * n_rows

    # Subtle "× T targets" stacked-grid shadows behind the front grid
    for offset in (0.07, 0.14):
        ax.add_patch(Rectangle(
            (grid_x0 + offset, grid_y0 - offset),
            grid_w, grid_h,
            facecolor=PALE_SAND, edgecolor=WARM, linewidth=0.6,
            linestyle=(0, (2, 2)), zorder=-1,
        ))

    # Front grid cells (every cell uniform — colour does NOT encode performance)
    for i in range(n_rows):
        for j in range(n_cols):
            ax.add_patch(Rectangle(
                (grid_x0 + j * cell_w, grid_y0 + i * cell_h),
                cell_w * 0.92, cell_h * 0.85,
                facecolor=LIGHT_BLUE, edgecolor=NAVY, linewidth=0.6,
            ))

    # Row label (feature combinations)
    ax.text(grid_x0 - 0.15, grid_y0 + grid_h / 2,
            "feature\ncombinations\n(F)",
            ha="right", va="center", fontsize=8, color=GREY)
    # Column label (architectures)
    ax.text(grid_x0 + grid_w / 2, grid_y0 - 0.20,
            "architectures (M)",
            ha="center", va="top", fontsize=8, color=GREY)

    # ----- "Train + evaluate" arrow -----
    arrow_y = grid_y0 + grid_h / 2
    ax.add_patch(FancyArrowPatch(
        (grid_x0 + grid_w + 0.40, arrow_y),
        (8.05, arrow_y),
        arrowstyle="-|>,head_length=10,head_width=6",
        linewidth=1.6, color=WARM,
    ))
    ax.text((grid_x0 + grid_w + 0.40 + 8.05) / 2, arrow_y + 0.18,
            "train + evaluate",
            ha="center", va="bottom", fontsize=9, color=WARM, fontweight="bold")
    ax.text((grid_x0 + grid_w + 0.40 + 8.05) / 2, arrow_y - 0.20,
            "test-set R²,  RMSE,  MAE,\nNSE,  KGE  per trial",
            ha="center", va="top", fontsize=7.5, color=GREY, fontstyle="italic")

    # ----- Model cohort glyph (3 stacked model boxes) -----
    cohort_x = 8.65
    for i in range(3):
        ax.add_patch(FancyBboxPatch(
            (cohort_x + i * 0.07, arrow_y - 0.45 + i * 0.06),
            1.30, 0.90,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            facecolor=WHITE, edgecolor=NAVY, linewidth=1.0,
        ))
    ax.text(cohort_x + 0.65 + 0.07, arrow_y + 0.04,
            "trained\nmodels",
            ha="center", va="center", fontsize=9, color=NAVY, fontweight="bold")

    # Arrow into the database
    ax.add_patch(FancyArrowPatch(
        (cohort_x + 1.55, arrow_y),
        (10.95, arrow_y),
        arrowstyle="-|>,head_length=10,head_width=6",
        linewidth=1.6, color=WARM,
    ))
    ax.text((cohort_x + 1.55 + 10.95) / 2, arrow_y + 0.18,
            "persist",
            ha="center", va="bottom", fontsize=9, color=WARM, fontweight="bold")

    # ----- Database glyph -----
    database_glyph(ax, 11.85, arrow_y, 1.55, 1.40, "Trial\ndatabase")


# ============================================================================
# Panel C — Phase 2: Progressive reconstruction
# ============================================================================
def draw_panel_c(ax) -> None:
    panel_bg(ax, 0.4, 0.55, 13.2, 3.40, PANEL_C_BG, edge=RED)

    panel_letter(ax, 0.95, 3.65, "C", RED)
    panel_heading(
        ax, 1.45, 3.65,
        "Phase 2 — Progressive reconstruction",
        "At each step, query the database for the highest-scoring model "
        "whose required inputs are currently available. Unlock the next target.",
        accent=RED,
    )

    # ----- Three step cards -----
    card_y_top = 2.95
    card_y_bot = 0.80
    card_h = card_y_top - card_y_bot
    card_w = 2.95
    card_xs = [1.10, 4.45, 7.80]   # left edge of each card

    step_data = [
        # (step number, available features as list of (label, colour),
        #  unlocked target label)
        (
            1,
            [("P1", BLUE), ("P2", BLUE)],
            "T1",
        ),
        (
            2,
            [("P1", BLUE), ("P2", BLUE), ("T1", GREEN)],
            "T2",
        ),
        (
            3,
            [("P1", BLUE), ("P2", BLUE), ("T1", GREEN), ("T2", GREEN)],
            "T3",
        ),
    ]

    for (step, available, unlocked), card_x in zip(step_data, card_xs):
        # Card background
        ax.add_patch(FancyBboxPatch(
            (card_x, card_y_bot), card_w, card_h,
            boxstyle="round,pad=0.06,rounding_size=0.18",
            facecolor=WHITE, edgecolor=RED, linewidth=1.2,
        ))
        # Step header
        ax.text(card_x + 0.20, card_y_top - 0.22,
                f"Step {step}",
                fontsize=10, fontweight="bold", color=RED, ha="left", va="center")

        # "Available features" label + dots
        feat_y = card_y_top - 0.62
        ax.text(card_x + 0.20, feat_y,
                "available features",
                fontsize=8, color=GREY, ha="left", va="center", fontstyle="italic")
        dot_y = feat_y - 0.30
        dot_spacing = 0.35
        dot_start_x = card_x + 0.30
        for i, (lbl, col) in enumerate(available):
            cx = dot_start_x + i * dot_spacing
            channel_dot(ax, cx, dot_y, 0.13, col)
            ax.text(cx, dot_y - 0.27, lbl,
                    ha="center", va="center", fontsize=7.5, color=INK)

        # Selected model box
        model_y = dot_y - 0.65
        ax.add_patch(FancyBboxPatch(
            (card_x + 0.30, model_y - 0.18),
            card_w - 0.60, 0.36,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=PALE_BLUE, edgecolor=NAVY, linewidth=0.9,
        ))
        ax.text(card_x + card_w / 2, model_y,
                "best feasible model from DB",
                ha="center", va="center", fontsize=8, color=NAVY, fontweight="bold")

        # Down arrow → unlocked target
        unlock_y_top = model_y - 0.25
        unlock_y_bot = card_y_bot + 0.30
        ax.add_patch(FancyArrowPatch(
            (card_x + card_w / 2, unlock_y_top),
            (card_x + card_w / 2, unlock_y_bot + 0.12),
            arrowstyle="-|>,head_length=8,head_width=5",
            linewidth=1.4, color=RED,
        ))
        # Unlocked target glyph (filled green dot + label)
        channel_dot(ax, card_x + card_w / 2, unlock_y_bot, 0.15, GREEN)
        ax.text(card_x + card_w / 2 + 0.30, unlock_y_bot,
                f"unlock {unlocked}",
                ha="left", va="center", fontsize=9, color=GREEN, fontweight="bold")

    # ----- Forward arrows between cards -----
    for i in range(len(card_xs) - 1):
        x_from = card_xs[i] + card_w
        x_to = card_xs[i + 1]
        ax.add_patch(FancyArrowPatch(
            (x_from + 0.05, (card_y_top + card_y_bot) / 2),
            (x_to - 0.05, (card_y_top + card_y_bot) / 2),
            arrowstyle="-|>,head_length=10,head_width=6",
            linewidth=1.6, color=RED,
        ))

    # ----- Loop back arrow above the cards (dashed = iteration) -----
    loop_y = card_y_top + 0.18
    last_card_right = card_xs[-1] + card_w
    # Up from card 3 right edge
    ax.add_patch(FancyArrowPatch(
        (last_card_right - card_w / 2, card_y_top),
        (last_card_right - card_w / 2, loop_y),
        arrowstyle="-", linewidth=1.2, color=RED, linestyle=(0, (3, 2)),
    ))
    # Sweep left
    ax.add_patch(FancyArrowPatch(
        (last_card_right - card_w / 2, loop_y),
        (card_xs[0] + card_w / 2, loop_y),
        arrowstyle="-", linewidth=1.2, color=RED, linestyle=(0, (3, 2)),
    ))
    # Drop into card 1
    ax.add_patch(FancyArrowPatch(
        (card_xs[0] + card_w / 2, loop_y),
        (card_xs[0] + card_w / 2, card_y_top + 0.02),
        arrowstyle="-|>,head_length=8,head_width=5",
        linewidth=1.2, color=RED, linestyle=(0, (3, 2)),
    ))

    # ----- Output box on the right -----
    out_x = card_xs[-1] + card_w + 0.45
    out_w = 13.0 - out_x
    out_y = card_y_bot + 0.55
    out_h = card_h - 1.10
    ax.add_patch(FancyBboxPatch(
        (out_x, out_y), out_w, out_h,
        boxstyle="round,pad=0.08,rounding_size=0.14",
        facecolor=PALE_GREEN, edgecolor=GREEN, linewidth=1.4,
    ))
    ax.text(out_x + out_w / 2, out_y + out_h / 2 + 0.10,
            "Reconstructed\nhistorical record",
            ha="center", va="center", fontsize=10, fontweight="bold", color=GREEN)
    ax.text(out_x + out_w / 2, out_y + out_h / 2 - 0.30,
            "+ ordered audit trail",
            ha="center", va="center", fontsize=8.5, color=GREEN, fontstyle="italic")
    # Forward arrow from card 3 to output
    ax.add_patch(FancyArrowPatch(
        (last_card_right + 0.05, (card_y_top + card_y_bot) / 2),
        (out_x - 0.05, out_y + out_h / 2),
        arrowstyle="-|>,head_length=10,head_width=6",
        linewidth=1.6, color=GREEN,
    ))


# ============================================================================
# Inter-panel database arrow (B → C)
# ============================================================================
def draw_db_query_arrow(ax) -> None:
    """A subtle dashed arrow from the Phase 1 database glyph down into
    the Phase 2 panel, annotated 'queryable at every step'."""
    src_x, src_y = 11.85, 4.85   # bottom of database glyph (approx)
    dst_x, dst_y = 11.85, 3.95   # top of Phase 2 panel
    ax.add_patch(FancyArrowPatch(
        (src_x, src_y), (dst_x, dst_y),
        arrowstyle="-|>,head_length=8,head_width=5",
        linewidth=1.2, color=NAVY, linestyle=(0, (3, 2)),
    ))
    ax.text(src_x + 0.15, (src_y + dst_y) / 2,
            "queryable\nat every step",
            ha="left", va="center",
            fontsize=8, color=NAVY, fontstyle="italic")


# ============================================================================
# Composition
# ============================================================================
def build_figure() -> plt.Figure:
    setup_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_axis_off()

    # ---------- Title block ----------
    fig.text(
        0.5, 0.965,
        "Two-phase progressive historical reconstruction",
        ha="center", va="center",
        fontsize=18, fontweight="bold", color=NAVY,
    )
    fig.text(
        0.5, 0.937,
        "A general framework for multivariate time series with long-term sensor gaps",
        ha="center", va="center",
        fontsize=11, color=GREY, style="italic",
    )

    draw_panel_a(ax)
    draw_panel_b(ax)
    draw_db_query_arrow(ax)
    draw_panel_c(ax)

    # ---------- Footer ----------
    fig.text(
        0.5, 0.022,
        "Conceptual workflow re-rendered from Janbain et al. 2023 (Water, "
        "doi:10.3390/w15091773, §2.5, Figure 6).  "
        "Schematic only — contains no data.",
        ha="center", va="center",
        fontsize=8.5, color=GREY, fontstyle="italic",
    )

    return fig


def main() -> None:
    out_dir = Path("images")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    svg_path = out_dir / "methodology_workflow.svg"
    png_path = out_dir / "methodology_workflow.png"
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, format="png", dpi=DPI)
    plt.close(fig)

    print(f"SVG: {svg_path}  ({svg_path.stat().st_size // 1024} KB)")
    print(f"PNG: {png_path}  ({png_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
