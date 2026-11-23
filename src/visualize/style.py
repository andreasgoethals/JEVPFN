"""Shared JEVPFN figure style and named canvases.

Standard figures use final A4 print dimensions and embedded TrueType fonts.
Dense diagnostics are paginated at final print dimensions. Every notebook calls apply() and takes its visual
settings from this module.
"""

from __future__ import annotations

import matplotlib as mpl

# ---------------------------------------------------------------------------
# DEFAULT FIGURE SIZES — final A4 print dimensions.
#
# A4 is 210 x 297 mm. With 25 mm margins that leaves a 160 x 247 mm text block, and the numbers
# below are that block in inches.
#
# DRAW AT FINAL WIDTH, and never rescale a figure in the document. Rescaling carries the text with
# it: 9pt squeezed to 70% arrives as 6.3pt, under the ~7pt floor where small print stops being
# legible on paper. So the point sizes in `_RC` are the point sizes ON THE PRINTED PAGE.
# ---------------------------------------------------------------------------

WIDTH_FULL = 6.30  # 160 mm — the full A4 text width
WIDTH_HALF = 3.05  # two side by side, with a ~5 mm gutter
WIDTH_THIRD = 1.95  # three side by side. Label sparingly at this width.

#: The A4 text block is 247 mm tall, but a figure taking all of it leaves no room for its caption
#: and pushes every surrounding paragraph onto another page. Half a page is the practical ceiling,
#: and `figsize` clamps to it rather than letting a tall panel grid silently overflow.
MAX_HEIGHT = 4.80  # 122 mm

GOLDEN = 0.618  # height = width * GOLDEN, unless the data wants otherwise


def figsize(width: float = WIDTH_FULL, ratio: float = GOLDEN) -> tuple[float, float]:
    """(width, height) in inches, clamped to `MAX_HEIGHT`. `ratio` is height/width.

    Pass `WIDTH_FULL`, `WIDTH_HALF` or `WIDTH_THIRD` — never a number of your own, because the
    whole point is that the figure arrives on the page at exactly the width it was drawn at.
    """
    return (width, min(width * ratio, MAX_HEIGHT))


# ---------------------------------------------------------------------------
# What A4 output requires. Everything here is about the figure being correct on paper, so it is
# the same in every project.
# ---------------------------------------------------------------------------

#: Most preferred first. DejaVu Sans is LAST and is the one that matters: it ships with matplotlib,
#: so it is the only face guaranteed present both locally and on a compute node. A missing face
#: makes matplotlib fall back silently, which changes text metrics — moving every label and making
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
#   * Check the palette is readable when printed, and in greyscale — a paper gets photocopied.
# ---------------------------------------------------------------------------

_PROJECT_RC: dict = {}

# Audit plots: the same quantity always has the same visual meaning.
AUDIT_RATIO = 0.76
# Neutral ink for quantitative summaries. Category palettes below are deliberately
# separate: a task, encoder or extractor never borrows a quantile's color.
MEDIAN_COLOR = "#3D4248"
P95_COLOR = "#7C858E"
TOTAL_COLOR = "#59636F"
QUANTITY_COLOR = "#59636F"
MODE_COLORS = {"per_column": "#2166AC", "joint": "#B35806", "both": "#1B7837"}
DIRECTION_COLORS = {"lower": "#3274A1", "higher": "#AC5353"}
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
OVERVIEW_TINT = "#4E79A7"  # Darkest shade of per-column ranks in tabular overview figures.
GRID_COLOR = "#D9E2E8"
WINDOW_COLOR = "#9C4A3C"  # Context-window reference lines; never a data series.
TEXT_COLOR = "black"
BACKGROUND_COLOR = "white"
MISSING_COLOR = "#B9BDC5"
REMOVED_COLOR = "#707070"
FAINT_COLOR = "#F1F4F6"
EVIDENCE_COLORS = {
    "observed": "#D9EBE3",
    "review": "#F6E4BE",
    "failed": "#F2D6D6",
    "unknown": "#F1F4F6",
    "not_applicable": "white",
}
METHOD_COLORS = ("#5A6E8C", "#987448", "#4F8075")
MAIN_PREDICTORS = ("catboost", "tabpfn_3_5", "tabicl_v2", "limix_2", "kumo_tabular", "linear",
                   "lightgbm")
PREDICTOR_LABELS = {"catboost": "CatBoost", "tabpfn_3_5": "TabPFN-3.5", "tabicl_v2": "TabICL v2",
                    "limix_2": "LimiX-2", "kumo_tabular": "Kumo", "linear": "Linear/logistic", "lightgbm": "LightGBM",
                    "tabstar": "TabSTAR v1.1", "contexttab": "ConTextTab", "ag_mm": "AG-MM"}
NATIVE_COLORS = {"tabstar": "#473B78", "contexttab": "#26736C", "ag_mm": "#9B6034"}
NATIVE_MARKERS = {"tabstar": "o", "contexttab": "s", "ag_mm": "^"}
FAMILY_COLORS = dict(zip(("experiment_1", "experiment_2", "experiment_4"), METHOD_COLORS))
FAMILY_COLORS["experiment_3"] = "#9467BD"
EMBEDDING_COLORS = {"e5_small_v2_tar": "#9467BD", "e5_small_v2": "#A64B72", "e5_large_v2": "#4E79A7",
                    "qwen3_embedding_0_6b": "#558B2F"}
EFFECT_CMAP = "BrBG"  # Positive effect means improvement in the explicitly named contrast.
# Predictors are distinguished by MARKER SHAPE in one ink; colour is reserved for the
# representation (decision view, encoder, label-trained route, native system).
PREDICTOR_MARKERS = dict(zip(MAIN_PREDICTORS, ("o", "s", "D", "^", "v", "P", "X")))
PREDICTOR_INK = "#3D4248"
PROBE_COLOR = "#D6604D"  # The label-trained classifier on E5-small (B2b, B4).
TAR_COLOR = EMBEDDING_COLORS["e5_small_v2_tar"]
NON_TEXT_COLOR = "#8A8F96"  # The same predictor without text: the reference of every gain.
DECISION_POOL_COLOR = "#B4BAC1"  # One grey dot per decision model; never a provider colour.
# Group C: Strands arms as a lightness ramp of the Strands provider colour (zero-shot darkest).
STRANDS_COLORS = {"zero_shot": "#AF4C59", "hidden": "#C97A84", "head": "#7E2F3A",
                  "lora": "#D9A0A8", "lora_head": "#5A1F28"}
WTL_COLORS = {"win": "#1B7837", "tie": "#C9CDD2", "loss": "#B2182B"}
# Raw gains are never averaged across task families: AUC (binary and multiclass) and R^2.
TASK_FAMILIES = ("classification", "regression")
TASK_FAMILY_LABELS = {"classification": "Classification (ΔAUC, 9 datasets)",
                      "regression": "Regression (ΔR², 9 datasets)"}
EXCLUDED_HATCH = "////"  # Cells excluded by design (not missing results).
# Group B: decision both + E5-small PCA32 (B2a), and label-free text-column field types (B3).
DECISION_EMBEDDING_COLOR = "#0B4F2A"
FIELD_TYPE_COLORS = {"numeric_text": "#8C6D31", "categorical": "#2166AC",
                     "identifier_or_code": "#7C858E", "short_text": "#C1447E",
                     "long_text": "#1B7837", "empty": "#D9E2E8"}
COHORT_COLORS = {"seen_joint": "#72518A", "unseen_joint": "#27878E",
                 "all_columns_unseen": "#BA7B28", "empty_text": "#777777"}
COHORT_MARKERS = {"seen_joint": "o", "unseen_joint": "s", "all_columns_unseen": "^", "empty_text": "x"}
STUDY_MAP_SIZE = (WIDTH_FULL, MAX_HEIGHT)
METHOD_MAP_SIZE = (WIDTH_FULL, 3.7)
PROMPT_SENSITIVITY_SIZE = (WIDTH_FULL, MAX_HEIGHT)
REGRESSION_CASE_SIZE = (WIDTH_FULL, 3.6)
EVIDENCE_MATRIX_SIZE = (WIDTH_FULL, MAX_HEIGHT)
COMPACT_TABLE_SIZE = (WIDTH_FULL, 3.3)
TASK_COLORS = {
    "binary_classification": "#6A51A3",
    "multiclass_classification": "#008B8B",
    "regression": "#B8860B",
}

# Saved decision-feature comparisons: consistent provider colors across all diagnostics.
DECISION_COLORS = {
    "jev": "#276A87",
    "liquid": "#C37929",
    "liquid_lfm3": "#885200",
    "perplexity": "#54765A",
    "cloudflare": "#846C9C",
    "intern": "#8C6D31",
    "strands": "#AF4C59",
    "h2o": "#2D8C7C",
    "sii": "#855C75",
    "kev": "#647B9C",
    "decision3": "#B56845",
    "nimble": "#797D44",
    "glide": "#386F93",
    "mammoth": "#654A91",
    "drex_max": "#995A33",
    "knowline": "#605D89", "sieve": "#386D51",
}
# One label per decision model in every figure: family plus size where the name has none.
DECISION_SHORT = {
    "jev": "Jev 1.14", "liquid": "Liquid d1-3B", "liquid_lfm3": "Liquid d1-2B",
    "perplexity": "Perplexity 27B", "cloudflare": "Clef-flash", "intern": "Intern 4B",
    "strands": "Strands 2B", "h2o": "H2O 4B", "sii": "SII 27B", "kev": "Kev 27B",
    "decision3": "D3-turbo", "nimble": "Nimble 9B", "glide": "GLiDE 27B", "mammoth": "Mammoth 27B", "drex_max": "Drex Max 27B",
    "knowline": "KnowLine 31B", "sieve": "Sieve 31B v3",
}
EVIDENCE_MARKS = {"observed": "#2E7D4F", "failed": "#B03A2E", "unknown": "#FFFFFF",
                  "not_applicable": "#E6E9EC", "review": "#D8A23A"}
AGREEMENT_DIVERGING = "RdBu"  # Rank agreement in [-1, 1]; 0 is white.
# Exp 8 row-level views: benefit is centred at zero (EFFECT_CMAP); sparse cells are masked.
DATA_MAP_MIN_ROWS = 5
DATA_MAP_PANELS = (3, 3)  # Datasets per page (rows, columns) at full width.
POST_TREATMENT_COLOR = "#8A8F96"  # Prediction-shift cohorts: descriptive, post-treatment.
ARCHITECTURE_MARKERS = {"decoder_pointer_head": "o", "decoder_lm_head": "s",
                        "bidirectional_encoder": "D", "other": "^", "undisclosed": "x"}
COMPARISON_MATRIX_SIZE = (WIDTH_FULL, MAX_HEIGHT)
COMPARISON_COVERAGE_SIZE = (WIDTH_FULL, MAX_HEIGHT)
COMPARISON_PENDING_SIZE = (WIDTH_FULL, 3.6)
COMPARISON_SCATTER_FIGURE_SIZE = (WIDTH_FULL, 2.5)
COMPARISON_SCATTER_ALPHA = 0.3
COMPARISON_SCATTER_SIZE = 6
COMPARISON_MATRIX_FONT = 7
AGREEMENT_CMAP = "Blues"
DISTANCE_CMAP = "YlOrRd"


# Critical-difference diagrams (src/visualize/critical_difference.py): mean-rank axis, the
# descriptive Nemenyi CD bar, and the confirmatory Holm-adjusted Wilcoxon annotations.
CD_INK = MEDIAN_COLOR  # Axis, learner leader lines and labels.
CD_BAR_COLOR = "#59636F"  # Nemenyi critical difference (descriptive).
CD_GROUP_COLOR = "#2166AC"  # Bars joining learners without a Holm-significant difference.
CD_SIGNIFICANT_COLOR = "#B2182B"  # Holm-significant versus the control learner.
CD_FONT = 7.5
CD_NOTE_FONT = 7
CD_ROW_STEP = 0.17  # Inches between learner label rows.
CD_GROUP_STEP = 0.075  # Inches between stacked no-difference bars.


def cd_diagram_size(rows: int, groups: int = 0, *, title: bool = False) -> tuple[float, float]:
    """Full A4 width; height grows with the label rows per side and no-difference bars."""
    return WIDTH_FULL, min(MAX_HEIGHT, 1.15 + 0.22 * title + CD_ROW_STEP * rows + CD_GROUP_STEP * groups)


def comparison_size(rows: int, *, rotated_labels: bool = False) -> tuple[float, float]:
    """Keep sparse last pages compact without changing final-print label sizes."""
    margin = 1.95 if rotated_labels else 1.35
    return WIDTH_FULL, min(MAX_HEIGHT, margin + 0.23 * rows)


def native_reference_size(rows: int) -> tuple[float, float]:
    """Reserve title/legend space without stretching sparse native-system rows."""
    return WIDTH_FULL, min(MAX_HEIGHT, 2.1 + 0.25 * rows)


def apply() -> None:
    """Install the style. Call once, at the top of every notebook and plotting script."""
    mpl.rcParams.update(_RC)
    mpl.rcParams.update(_PROJECT_RC)
