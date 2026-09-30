"""Reports must preserve complete displayed content and survive partial reruns."""

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import notebook_report as report
from src.utils import paths, run_notebooks
from src.utils.notebook_display import display_json, display_local
from src.visualize.figures import FigureSaver


def test_complete_tables_json_and_plot_values_are_printed_and_saved(
    isolated_output, monkeypatch, capsys
):
    monkeypatch.setattr(run_notebooks, "discover", lambda: ("sample",))
    report.begin_report()
    report.report_section("All observations")
    long_text = "exact long value " * 200
    display_local(pd.DataFrame({"row": range(100), "value": [long_text] * 100}))
    display_json("Preview", {"text": long_text, "missing": None})
    fig, ax = plt.subplots()
    ax.plot([1, 2], [123.456, 789.012])
    save = FigureSaver("sample")
    save(fig, "line", caption="Two plotted observations.")
    plt.close(fig)
    report.finish_report("sample", "Final summary")
    printed = capsys.readouterr().out
    saved = paths.all_results_path().read_text(encoding="utf-8")
    assert long_text in printed and long_text in saved
    assert "99" in saved and "123.456" in saved and "789.012" in saved
    assert "Two plotted observations." in saved
    assert "Final summary" in saved
    assert "Two plotted observations." in paths.captions_path().read_text(encoding="utf-8")


def test_rebuild_keeps_other_notebook_summaries_and_phase_files(isolated_output):
    names = ("01_data_exploration", "03_feature_creation")
    report.atomic_text(paths.reports_dir(names[0]), "Complete exploration detail")
    report.atomic_text(paths.reports_dir(names[1]), "Complete feature detail")
    run_notebooks.write_all_results(names)
    report.atomic_text(paths.reports_dir(names[1]), "Updated feature detail")
    rebuilt = run_notebooks.write_all_results(names).read_text(encoding="utf-8")
    assert "Complete exploration detail" in rebuilt
    assert "Updated feature detail" in rebuilt and "Complete feature detail" not in rebuilt
    assert "Complete exploration detail" in paths.all_results_path("exploration").read_text(
        encoding="utf-8"
    )
    assert "Updated feature detail" in paths.all_results_path("feature_creation").read_text(
        encoding="utf-8"
    )
