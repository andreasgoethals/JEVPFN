"""THE shared visual style. One place, every figure, every notebook.

Import and call `apply()` once at the top of a notebook, then never choose a font, a
size, a grid or — above all — a colour again. Two things this buys:

1. Figures from different notebooks sit together in one paper without looking like they
   came from different projects.
2. **A name means the same colour everywhere.** A reader learns the legend once. That is
   what `register_series()` and `color()` are for: the project declares its series names
   ONCE, here or at import time, and every figure asks this module which colour a name
   gets. A notebook never picks a colour, so no two notebooks can disagree.

WHY the colour assignment is by name and not by plot order: if a figure drops one series,
matplotlib's cycler shifts every colour after it, and the same model is blue in one
figure and orange in the next. Colour has to follow the entity, never its rank.

THE PALETTE IS VALIDATED, NOT CHOSEN BY EYE. The categorical order below clears, on a
white surface: the OKLCH lightness band, a chroma floor, colour-vision-deficiency
separation of every ADJACENT pair (worst 9.2 deutan, OKLab ΔE x100, target >= 8), and a
normal-vision floor on adjacent pairs (worst 20.8, floor 15). Three slots sit below 3:1
contrast against white (aqua 2.8, magenta 2.7, yellow 2.2) — which is why a legend is
always present and marks are directly labelled where it matters, so identity never rests
on colour alone.

The ORDER is the safety mechanism, not decoration. Reordering the slots changes which
pairs are adjacent and can break the CVD gate. If you must change it, re-validate the
whole order — do not swap two entries because one looks nicer.

ONE MODE, deliberately: a figure destined for a paper always renders on white. There is
no viewer theme to follow, so there is no dark variant to keep in sync.

SCATTER AND SMALL MULTIPLES CAP AT FOUR SERIES. In those forms every pair of colours is
on screen at once, not just neighbours, and only the first four slots clear the floors
all-pairs (worst 9.2 CVD, 16.3 normal vision). Past four, fold the tail into "other",
facet, or add a second channel (marker shape). The fifth slot beside the second measures
12.9 to normal vision — genuinely hard to tell apart, colourblind or not.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Ink and chrome. Data is the only thing allowed to be loud.
# ---------------------------------------------------------------------------

INK = {
    "primary": "#0b0b0b",     # titles, values — near-black, not pure black
    "secondary": "#52514e",   # subtitles, annotations
    "muted": "#898781",       # axis labels, tick labels
    "grid": "#e1e0d9",        # hairline gridlines
    "axis": "#c3c2b7",        # baseline and spines
    "surface": "#ffffff",     # the paper
}

# ---------------------------------------------------------------------------
# SEMANTIC roles — the colours that carry a MEANING rather than an identity.
#
# These are the anchors of the shared legend. "baseline" is grey in every figure in the
# project; "proposed" is blue in every figure. A reader who has seen one figure can read
# the next without the legend.
#
# `baseline` and `secondary_baseline` are deliberately achromatic, so they sit OUTSIDE
# the categorical palette: a reference condition should recede, and greying it is the
# cheapest way to say "this is the thing being improved on" without spending a hue.
# ---------------------------------------------------------------------------

COLORS = {
    "baseline": "#52514e",          # the control / unmodified / reference condition
    "secondary_baseline": "#898781",  # a second reference, when there are two
    "proposed": "#2a78d6",          # this project's method
    "observed": "#eb6834",          # measured from real data / ground truth
    "alternative": "#1baf7a",       # a competing method being compared against
    "highlight": "#4a3aa7",         # the one thing the figure is about
    "annotation": "#52514e",        # reference lines, thresholds, arrows
    "surface": "#ffffff",
}

# Status, fixed and never reused as a series colour: a red bar must not be able to mean
# "model 8" in one figure and "failed" in the next. Always paired with a label, never
# carrying the meaning alone.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# ---------------------------------------------------------------------------
# CATEGORICAL palette — identity. Fixed order; see the module docstring.
# ---------------------------------------------------------------------------

SERIES: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#4a3aa7",  # 4 violet     <- first four are safe when ALL pairs are visible at once
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#eda100",  # 7 yellow
    "#e34948",  # 8 red
)

#: Beyond this many series, colour alone stops working. Fold the tail into "other", use
#: small multiples, or add a marker-shape channel. Enforced by `series_colors`.
MAX_SERIES = len(SERIES)
#: The cap for forms where every pair is simultaneously visible (scatter, bubble, small
#: multiples, choropleth) rather than only neighbours (bars, stacks, lines).
MAX_SERIES_ALL_PAIRS = 4

# ---------------------------------------------------------------------------
# SEQUENTIAL and DIVERGING — magnitude and polarity. Never a rainbow.
# ---------------------------------------------------------------------------

#: One hue, light to dark. For continuous magnitude: heatmaps, correlation-magnitude,
#: density. The light end is allowed to recede into the paper because it means "near zero".
_SEQUENTIAL_STEPS = (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
)

#: Discrete ordered categories (tiers, stages, ordinal bins) need visible gaps between
#: steps AND a light end that still clears the paper — a barely-visible first bar is not
#: "low", it is missing. These five steps are validated for that: monotone lightness, all
#: adjacent lightness gaps >= 0.06, light end 2.1:1 against white.
ORDINAL = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")

#: Two hues plus a NEUTRAL midpoint, for signed quantities (differences, correlations,
#: residuals). Blue against red because they read as opposites; the midpoint is grey, not
#: a third hue, so zero reads as "nothing" rather than as its own category.
_DIVERGING_STEPS = (
    "#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a6a5", "#e34948", "#a32120",
)


def sequential_cmap(name: str = "seq") -> LinearSegmentedColormap:
    """The one-hue light-to-dark map. Use for unsigned magnitude."""
    return LinearSegmentedColormap.from_list(name, _SEQUENTIAL_STEPS)


def diverging_cmap(name: str = "div") -> LinearSegmentedColormap:
    """The two-hue map with a neutral midpoint. Use for signed quantities.

    Always centre it on zero (`vmin=-v, vmax=+v`, or a `TwoSlopeNorm`). An off-centre
    diverging map puts the neutral colour at a value that is not neutral, which is a
    figure that lies.
    """
    return LinearSegmentedColormap.from_list(name, _DIVERGING_STEPS)


# ---------------------------------------------------------------------------
# Figure sizes, in inches.
#
# Journals specify a column width, and a figure scaled after the fact has the wrong font
# size — 8pt text in a figure squeezed to 70% arrives as 5.6pt. Draw at final width
# instead, so the point sizes below are the point sizes on the printed page.
# ---------------------------------------------------------------------------

WIDTH_SINGLE = 3.5   # a single column in a two-column paper
WIDTH_DOUBLE = 7.16  # the full text width
GOLDEN = 0.618       # height = width * GOLDEN, unless the data wants otherwise


def figsize(width: float = WIDTH_DOUBLE, ratio: float = GOLDEN) -> tuple[float, float]:
    """(width, height) in inches. `ratio` is height/width."""
    return (width, width * ratio)


# ---------------------------------------------------------------------------
# The style itself.
# ---------------------------------------------------------------------------

#: Font stack, most preferred first. DejaVu Sans is LAST and is the one that matters: it
#: ships with matplotlib, so it is the only face guaranteed present on both the Windows
#: dev machine and a Linux compute node. When a face is missing matplotlib silently falls
#: back and text metrics change, which moves every label and makes a figure regenerated
#: on the cluster differ from the one drawn locally — for no visible reason.
_FONT_STACK = ["Source Sans 3", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"]

_RC = {
    "figure.figsize": figsize(),
    "figure.dpi": 110,
    # constrained_layout, and NOT savefig.bbox="tight". Tight-bbox crops to the drawn
    # content, so two figures declared at the same width come out at different widths and
    # the paper's font sizes stop matching between them. constrained_layout fits the
    # content INSIDE the declared size instead. `None` is matplotlib's spelling for "use
    # the declared figure size" — it is the only other accepted value besides "tight".
    "figure.constrained_layout.use": True,
    "savefig.bbox": None,
    "savefig.facecolor": INK["surface"],
    "savefig.transparent": False,
    # Embed TrueType rather than the default Type 3. Type 3 fonts are rejected by several
    # journal submission systems and cannot be searched or copied out of the PDF.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "font.family": "sans-serif",
    "font.sans-serif": _FONT_STACK,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,

    "text.color": INK["primary"],
    "axes.titlecolor": INK["primary"],
    "axes.labelcolor": INK["secondary"],
    "xtick.color": INK["muted"],
    "ytick.color": INK["muted"],
    "axes.edgecolor": INK["axis"],
    "axes.facecolor": INK["surface"],
    "figure.facecolor": INK["surface"],

    # Recessive chrome: a hairline y-grid only, behind the data, and no box around the
    # plot. Vertical gridlines are off because they compete with categorical bars; turn
    # them on per-axes when the x axis is genuinely continuous and being read off.
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": INK["grid"],
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,

    # Thin marks. 2pt lines and >=8pt markers read at print size without turning the
    # figure into a poster.
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,
    "lines.markeredgewidth": 0.0,
    "patch.linewidth": 0.0,
    "hatch.linewidth": 0.6,
    "scatter.edgecolors": "none",

    # A legend is a label list, not a panel. No frame, no shadow.
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.3,

    "image.cmap": "viridis",  # replaced below by the project's sequential map
    "errorbar.capsize": 2.0,
    "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
}


def apply() -> None:
    """Install the style. Call once, at the top of every notebook and plotting script.

    Registers the project colour maps under fixed names so `cmap="seq"` and `cmap="div"`
    work anywhere, and makes `seq` the default for `imshow`/`pcolormesh` — otherwise the
    default viridis quietly appears in one heatmap and breaks the shared look.
    """
    for cmap in (sequential_cmap(), diverging_cmap()):
        # Unregister first rather than passing `force=True`. Both make re-running a
        # notebook's setup cell idempotent, but `force=True` emits a UserWarning every
        # time — and a warning that always fires is a warning nobody reads.
        if cmap.name in mpl.colormaps:
            mpl.colormaps.unregister(cmap.name)
        mpl.colormaps.register(cmap, name=cmap.name)
    rc = dict(_RC)
    rc["image.cmap"] = "seq"
    mpl.rcParams.update(rc)


# ---------------------------------------------------------------------------
# Name -> colour. The mechanism that makes a legend mean one thing project-wide.
# ---------------------------------------------------------------------------

#: Declared series names, in slot order. A PROJECT FILLS THIS IN — once, here — with the
#: things it plots repeatedly: model names, dataset groups, experiment arms. Order is what
#: assigns the colours, so append rather than insert: putting a new name in the middle
#: repaints every figure after it and silently invalidates every figure already in the
#: paper.
REGISTERED: list[str] = []


def register_series(*names: str) -> None:
    """Declare series names in slot order. Idempotent, so a notebook can call it too.

    Names already registered keep their slot; new ones take the next free slots. That is
    what lets a notebook register only the series it uses without disturbing the others.
    """
    for name in names:
        if name not in REGISTERED:
            REGISTERED.append(name)


def color(name: str) -> str:
    """The colour for a series name, or a semantic role, or a status.

    Resolution order, first match wins:
      1. a semantic role in `COLORS`      — "baseline", "proposed", ...
      2. a status in `STATUS`             — "good", "critical", ...
      3. a registered series name         — its fixed categorical slot
      4. an unregistered name             — registers it, then returns its slot

    Falling through to (4) rather than raising is deliberate: a missing
    `register_series()` call should not stop a figure from being drawn while it is being
    iterated on. It costs order-independence, though — two notebooks that first meet the
    same name in different orders will disagree. Register the project's names once, in
    this module, and (4) never fires.
    """
    if name in COLORS:
        return COLORS[name]
    if name in STATUS:
        return STATUS[name]
    if name not in REGISTERED:
        register_series(name)
    idx = REGISTERED.index(name)
    if idx >= MAX_SERIES:
        raise ValueError(
            f"{len(REGISTERED)} series registered but only {MAX_SERIES} colours are "
            f"distinguishable ({name!r} is number {idx + 1}). Fold the tail into one "
            f"'other' series, use small multiples, or add a marker-shape channel — do "
            f"not generate a ninth hue."
        )
    return SERIES[idx]


def series_colors(names: list[str] | tuple[str, ...], *, all_pairs: bool = False) -> list[str]:
    """Colours for several series at once, stable under a changing subset.

    `all_pairs=True` for forms where every pair is visible simultaneously — scatter,
    bubble, small multiples, choropleth. Those cap at `MAX_SERIES_ALL_PAIRS`; the
    neighbour-only forms (bars, stacks, lines) get the full eight.
    """
    cap = MAX_SERIES_ALL_PAIRS if all_pairs else MAX_SERIES
    if len(names) > cap:
        form = "all pairs visible at once" if all_pairs else "adjacent pairs only"
        raise ValueError(
            f"{len(names)} series requested but the cap for this form ({form}) is {cap}. "
            f"Fold the tail into 'other' or facet the figure."
        )
    return [color(n) for n in names]


# ---------------------------------------------------------------------------
# Small shared helpers. Anything used in more than one notebook belongs here.
# ---------------------------------------------------------------------------


def despine(ax: plt.Axes, *, left: bool = False, bottom: bool = False) -> plt.Axes:
    """Drop spines. Top and right are already off; this removes the other two."""
    if left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
    if bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)
    return ax


def legend(ax: plt.Axes, *, ncol: int = 1, **kwargs) -> None:
    """The project's legend. Present whenever a figure has two or more series.

    Not optional: three of the eight categorical colours sit below 3:1 against white, so
    a figure with no legend and no direct labels asks the reader to identify a series by
    a colour they may not be able to see.
    """
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) < 2:
        return  # one series is named by the title; a one-entry legend is noise
    ax.legend(handles, labels, ncol=ncol, **kwargs)


def annotate_reference(ax: plt.Axes, value: float, label: str = "", *, axis: str = "y") -> None:
    """A dashed reference line (a threshold, a chance level, a published number).

    Always the annotation colour and always dashed, so a reference line can never be
    mistaken for data in any figure in the project.
    """
    draw = ax.axhline if axis == "y" else ax.axvline
    draw(value, color=COLORS["annotation"], linestyle="--", linewidth=0.9, zorder=1)
    if not label:
        return
    # An opaque, borderless backing box. A reference line crosses the data by definition, so
    # its label lands on top of something sooner or later; without this it becomes
    # unreadable in exactly the figures where the reference matters most.
    backing = {"facecolor": INK["surface"], "edgecolor": "none", "pad": 1.0}
    if axis == "y":
        ax.annotate(label, (0.995, value), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7, color=COLORS["annotation"],
                    bbox=backing, zorder=5)
    else:
        # Bottom, not top: a distribution's mass sits high in the panel, so the strip just
        # above the x axis is the one place a vertical line's label is usually clear.
        ax.annotate(label, (value, 0.02), xycoords=("data", "axes fraction"),
                    ha="left", va="bottom", fontsize=7, color=COLORS["annotation"],
                    bbox=backing, zorder=5)


def palette_table() -> str:
    """The palette as text, for a notebook's printed summary.

    A figure of swatches is not readable by an agent or a diff; a table is.
    """
    lines = ["role/name              colour", "-" * 32]
    for k, v in COLORS.items():
        lines.append(f"{k:<22} {v}")
    for k, v in STATUS.items():
        lines.append(f"{'status:' + k:<22} {v}")
    for i, name in enumerate(REGISTERED):
        lines.append(f"{'series:' + name:<22} {SERIES[i]}")
    return "\n".join(lines)
