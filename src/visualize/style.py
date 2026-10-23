"""Shared JEVPFN figure style and named canvases.

Standard figures use final A4 print dimensions and embedded TrueType fonts.
Dense diagnostics are paginated at final print dimensions. Every notebook calls apply() and takes its visual
settings from this module.
"""

from __future__ import annotations

import matplotlib as mpl

# ---------------------------------------------------------------------------
# DEFAULT FIGURE SIZES â€” final A4 print dimensions.
#
# A4 is 210 x 297 mm. With 25 mm margins that leaves a 160 x 247 mm text block, and the numbers
# below are that block in inches.
#
# DRAW AT FINAL WIDTH, and never rescale a figure in the document. Rescaling carries the text with
# it: 9pt squeezed to 70% arrives as 6.3pt, under the ~7pt floor where small print stops being
# legible on paper. So the point sizes in `_RC` are the point sizes ON THE PRINTED PAGE.
# ---------------------------------------------------------------------------

WIDTH_FULL = 6.30  # 160 mm â€” the full A4 text width
WIDTH_HALF = 3.05  # two side by side, with a ~5 mm gutter
WIDTH_THIRD = 1.95  # three side by side. Label sparingly at this width.

#: The A4 text block is 247 mm tall, but a figure taking all of it leaves no room for its caption
#: and pushes every surrounding paragraph onto another page. Half a page is the practical ceiling,
#: and `figsize` clamps to it rather than letting a tall panel grid silently overflow.
MAX_HEIGHT = 4.80  # 122 mm

GOLDEN = 0.618  # height = width * GOLDEN, unless the data wants otherwise


def figsize(width: float = WIDTH_FULL, ratio: float = GOLDEN) -> tuple[float, float]:
    """(width, height) in inches, clamped to `MAX_HEIGHT`. `ratio` is height/width.

    Pass `WIDTH_FULL`, `WIDTH_HALF` or `WIDTH_THIRD` â€” never a number of your own, because the
    whole point is that the figure arrives on the page at exactly the width it was drawn at.
    """
    return (width, min(width * ratio, MAX_HEIGHT))


# ---------------------------------------------------------------------------
# What A4 output requires. Everything here is about the figure being correct on paper, so it is
# the same in every project.
# ---------------------------------------------------------------------------

#: Most preferred first. DejaVu Sans is LAST and is the one that matters: it ships with matplotlib,
#: so it is the only face guaranteed present both locally and on a compute node. A missing face
#: makes matplotlib fall back silently, which changes text metrics â€” moving every label and making
#: a cluster-drawn figure differ from the local one for no visible reason.
_FONT_STACK = ["Source Sans 3", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"]

_RC = {
    # A4 full text width by default, so a figure saved without thinking about it is already the
    # right size for the page.
    "figure.figsize": figsize(),
    "figure.dpi": 110,
    # constrained_layout, and NOT savefig.bbox="tight". Tight-bbox crops to the drawn content, so
    # two figures declared at the same width come out at different widths and the paper's font
    # sizes stop matching between them. constrained_layout fits the content INSIDE the declared
    # size instead. `None` is matplotlib's spelling for "use the declared figure size".
    "figure.constrained_layout.use": True,
    "savefig.bbox": None,
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    # TrueType, not the default Type 3: Type 3 is rejected by several journal submission systems
    # and cannot be searched or copied out of the PDF.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Point sizes ON THE PRINTED A4 PAGE, since the figure is drawn at final width. 9pt sits just
    # under a paper's own 10-11pt, which reads as "part of the document" rather than shrunken;
    # 7pt is the floor below which small print stops being legible on paper.
    "font.family": "sans-serif",
    "font.sans-serif": _FONT_STACK,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
}

# ---------------------------------------------------------------------------
# JEVPFN visual conventions.
#
# Colours, grid, spines, line widths, marker shapes, colormaps: whatever this project's figures
# need. Put them here rather than in a notebook, so every figure changes together and a reader
# never has to re-learn a legend between two figures of the same paper.
#
# Two things worth deciding up front:
#   * Give a colour a MEANING and keep it. If a name means one thing in one figure and another
#     somewhere else, the legend has to be read twice.
#   * Check the palette is readable when printed, and in greyscale â€” a paper gets photocopied.
# ---------------------------------------------------------------------------

_PROJECT_RC: dict = {}

# Audit plots: the same quantity always has the same visual meaning.
AUDIT_RATIO = 0.76
MEDIAN_COLOR = "#276A87"
P95_COLOR = "#C37929"
TOTAL_COLOR = "#54765A"
AUDIT_MARKER_SIZE = 18

# Named final-print canvases; paginate dense diagnostics instead of shrinking text.
DASHBOARD_SIZE = (WIDTH_FULL, MAX_HEIGHT)
COMPACT_FLOW_SIZE = (WIDTH_FULL, 3.4)
LONG_TEXT_SIZE = (WIDTH_FULL, MAX_HEIGHT)
LONG_TEXT_PAGE_SIZE = (WIDTH_FULL, MAX_HEIGHT)
REGRESSION_RUBRIC_SIZE = (WIDTH_FULL, MAX_HEIGHT)
RUBRIC_FONT = 9
LONG_TEXT_LABEL_FONT = 8
LONG_TEXT_PAGE_FONT = 7
OVERVIEW_FONT = 9
OVERVIEW_CMAP = "Blues"
GRID_COLOR = "#D9E2E8"
TEXT_COLOR = "black"
BACKGROUND_COLOR = "white"
MISSING_COLOR = "#846C9C"
REMOVED_COLOR = "#707070"
FAINT_COLOR = "#F1F4F6"
EVIDENCE_COLORS = {
    "observed": "#D9EBE3",
    "review": "#F6E4BE",
    "failed": "#F2D6D6",
    "unknown": "#F1F4F6",
    "not_applicable": "white",
}
METHOD_COLORS = (MEDIAN_COLOR, P95_COLOR, TOTAL_COLOR)
EMBEDDING_COLORS = {"e5_small_v2": MEDIAN_COLOR, "e5_large_v2": P95_COLOR,
                    "qwen3_embedding_0_6b": TOTAL_COLOR}
STUDY_MAP_SIZE = (WIDTH_FULL, MAX_HEIGHT)
METHOD_MAP_SIZE = (WIDTH_FULL, 3.7)
PROMPT_SENSITIVITY_SIZE = (WIDTH_FULL, MAX_HEIGHT)
REGRESSION_CASE_SIZE = (WIDTH_FULL, 3.6)
EVIDENCE_MATRIX_SIZE = (WIDTH_FULL, MAX_HEIGHT)
COMPACT_TABLE_SIZE = (WIDTH_FULL, 3.3)
TASK_COLORS = {
    "binary_classification": MEDIAN_COLOR,
    "multiclass_classification": P95_COLOR,
    "regression": TOTAL_COLOR,
}

# Saved decision-feature comparisons: consistent provider colors across all diagnostics.
DECISION_COLORS = {
    "jev": "#276A87",
    "liquid": "#C37929",
    "perplexity": "#54765A",
    "cloudflare": "#846C9C",
    "intern": "#8C6D31",
    "strands": "#AF4C59",
    "h2o": "#2D8C7C",
    "sii": "#855C75",
    "kev": "#647B9C",
    "decision3": "#B56845",
    "nimble": "#797D44",
}
COMPARISON_MATRIX_SIZE = (WIDTH_FULL, MAX_HEIGHT)
COMPARISON_COVERAGE_SIZE = (WIDTH_FULL, MAX_HEIGHT)
COMPARISON_PENDING_SIZE = (WIDTH_FULL, 3.6)
COMPARISON_SCATTER_FIGURE_SIZE = (WIDTH_FULL, 2.5)
COMPARISON_SCATTER_ALPHA = 0.3
COMPARISON_SCATTER_SIZE = 6
COMPARISON_MATRIX_FONT = 7
AGREEMENT_CMAP = "Blues"
DISTANCE_CMAP = "YlOrRd"


def apply() -> None:
    """Install the style. Call once, at the top of every notebook and plotting script."""
    mpl.rcParams.update(_RC)
    mpl.rcParams.update(_PROJECT_RC)
