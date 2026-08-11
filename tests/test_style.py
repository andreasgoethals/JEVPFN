"""`src/visualize/style.py` — the shared style.

The tests that matter are the ones about COLOUR STABILITY. A palette that quietly reassigns
colours when a figure drops a series produces a paper where the same model is blue in one
figure and orange in the next, and no test failure ever pointed at it.
"""

from __future__ import annotations

import matplotlib as mpl
import pytest

from src.visualize import style


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts from an empty series registry — order is global state here."""
    saved = list(style.REGISTERED)
    style.REGISTERED.clear()
    yield
    style.REGISTERED[:] = saved


def test_apply_sets_the_shared_look_and_the_colour_cycle() -> None:
    style.apply()
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == list(style.SERIES)
    assert mpl.rcParams["font.family"] == ["sans-serif"]
    # TrueType, not the default Type 3: several journals reject Type 3 outright, and it
    # cannot be searched or copied out of the PDF.
    assert mpl.rcParams["pdf.fonttype"] == 42
    # constrained_layout instead of a tight bbox, so two figures declared the same width
    # come out the same width. `None` is matplotlib's "use the declared size".
    assert mpl.rcParams["figure.constrained_layout.use"] is True
    assert mpl.rcParams["savefig.bbox"] is None


def test_apply_registers_the_project_colormaps() -> None:
    style.apply()
    assert "seq" in mpl.colormaps and "div" in mpl.colormaps
    assert mpl.rcParams["image.cmap"] == "seq"


def test_apply_is_idempotent() -> None:
    """A notebook re-runs its setup cell; a second registration must not raise."""
    style.apply()
    style.apply()


def test_a_name_keeps_its_colour_when_other_series_disappear() -> None:
    """THE rule: colour follows the entity, never its rank in this particular figure."""
    style.register_series("alpha", "beta", "gamma")
    full = style.series_colors(["alpha", "beta", "gamma"])
    subset = style.series_colors(["alpha", "gamma"])
    assert subset == [full[0], full[2]]


def test_registration_order_assigns_the_slots_and_appending_is_safe() -> None:
    style.register_series("alpha", "beta")
    before = style.color("beta")
    style.register_series("gamma")          # appended
    assert style.color("beta") == before    # ...so nothing already drawn is repainted
    assert style.color("gamma") == style.SERIES[2]


def test_register_series_is_idempotent() -> None:
    style.register_series("alpha")
    style.register_series("alpha", "beta")
    assert style.REGISTERED == ["alpha", "beta"]


def test_semantic_roles_win_over_the_categorical_slots() -> None:
    """"baseline" means grey in every figure — it must never be handed a series slot."""
    assert style.color("baseline") == style.COLORS["baseline"]
    assert style.color("proposed") == style.COLORS["proposed"]
    assert style.color("critical") == style.STATUS["critical"]
    assert style.color("baseline") not in style.SERIES


def test_more_series_than_distinguishable_colours_is_an_error() -> None:
    """Past eight, colour has stopped working. Generating a ninth hue hides that."""
    style.register_series(*[f"s{i}" for i in range(style.MAX_SERIES)])
    with pytest.raises(ValueError, match="distinguishable"):
        style.color("one_too_many")


def test_scatter_forms_cap_lower_than_bar_forms() -> None:
    """In a scatter every pair of colours is on screen at once, not just neighbours, and
    only the first four slots clear the separation floors under that condition."""
    names = [f"s{i}" for i in range(5)]
    style.series_colors(names)  # fine for bars and lines
    with pytest.raises(ValueError, match="cap for this form"):
        style.series_colors(names, all_pairs=True)


def test_the_palette_has_no_duplicate_colours() -> None:
    assert len(set(style.SERIES)) == len(style.SERIES)
    assert len(set(style.STATUS.values())) == len(style.STATUS)


def test_status_colours_are_never_series_colours() -> None:
    """A red bar must not mean "model 8" in one figure and "failed" in the next."""
    assert set(style.STATUS.values()).isdisjoint(style.SERIES)


def test_every_declared_colour_is_a_full_hex_string() -> None:
    for name, value in {**style.COLORS, **style.STATUS, **style.INK}.items():
        assert isinstance(value, str) and value.startswith("#") and len(value) == 7, name
    for value in style.SERIES + style.ORDINAL:
        assert value.startswith("#") and len(value) == 7


def test_the_diverging_map_has_a_neutral_midpoint() -> None:
    """Zero must read as "nothing", so the middle is grey — near-equal R, G and B — and not
    a third hue."""
    r, g, b, _ = style.diverging_cmap()(0.5)
    assert max(r, g, b) - min(r, g, b) < 0.06


def test_the_sequential_map_gets_monotonically_darker() -> None:
    """Magnitude has to read off the ramp without a legend."""
    lum = [sum(style.sequential_cmap()(x)[:3]) for x in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert lum == sorted(lum, reverse=True)


def test_figsize_uses_the_journal_column_widths() -> None:
    """Draw at final width: a figure scaled after the fact has the wrong font size."""
    assert style.figsize(style.WIDTH_SINGLE)[0] == style.WIDTH_SINGLE
    w, h = style.figsize()
    assert h < w


def test_helpers_run_on_a_real_axes() -> None:
    import matplotlib.pyplot as plt

    style.apply()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="alpha", color=style.color("alpha"))
    ax.plot([0, 1], [1, 0], label="beta", color=style.color("beta"))
    style.legend(ax)
    style.annotate_reference(ax, 0.5, "chance")
    style.despine(ax, left=True)
    assert ax.get_legend() is not None
    plt.close(fig)


def test_legend_is_suppressed_for_a_single_series() -> None:
    """One series is named by the title; a one-entry legend is noise."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="only")
    style.legend(ax)
    assert ax.get_legend() is None
    plt.close(fig)


def test_palette_table_is_readable_text() -> None:
    """A swatch figure is invisible to a diff and to an agent; a table is not."""
    style.register_series("alpha")
    table = style.palette_table()
    assert "baseline" in table and "series:alpha" in table
