"""Reports must preserve complete displayed content and survive partial reruns."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.utils import files, paths, run_notebooks
from src.utils import notebook_report as report
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


def test_transient_windows_report_sharing_error_is_retried(tmp_path, monkeypatch):
    path = tmp_path / "Captions.md"
    path.write_text("old", encoding="utf-8")
    replace = files.os.replace
    calls = []

    def sharing_violation_then_success(source, target):
        calls.append(target)
        if len(calls) < 3:
            raise PermissionError("[WinError 5] Access is denied")
        replace(source, target)

    monkeypatch.setattr(files.os, "replace", sharing_violation_then_success)
    monkeypatch.setattr(files.time, "sleep", lambda _: None)
    report.atomic_text(path, "complete new captions")
    assert path.read_text() == "complete new captions"
    assert len(calls) == 3
    assert not list(tmp_path.glob("*.pending"))


def test_persistent_report_error_keeps_old_file_and_recoverable_new_content(tmp_path, monkeypatch):
    path = tmp_path / "All Results.md"
    path.write_text("previous complete report", encoding="utf-8")

    def denied(*_):
        raise PermissionError("blocked")

    monkeypatch.setattr(files.os, "replace", denied)
    monkeypatch.setattr(files.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError, match="new content is saved"):
        report.atomic_text(path, "new complete report")
    assert path.read_text() == "previous complete report"
    pending = list(tmp_path.glob("*.pending"))
    assert len(pending) == 1 and pending[0].read_text() == "new complete report"


def test_unchanged_report_does_not_replace_a_potentially_open_file(tmp_path, monkeypatch):
    path = tmp_path / "Captions.md"
    path.write_text("unchanged", encoding="utf-8")

    def forbidden(*_):
        pytest.fail("unchanged report should not be replaced")

    monkeypatch.setattr(files.os, "replace", forbidden)
    report.atomic_text(path, "unchanged")


def test_shared_audit_csv_retries_and_reuses_unchanged_content(tmp_path, monkeypatch):
    from src.data.audit import _atomic_csv

    path = tmp_path / "text.csv"
    frame = pd.DataFrame({"feature": ["description"], "count": [7]})
    replace = files.os.replace
    attempts = []

    def temporarily_locked(source, target):
        attempts.append(target)
        if len(attempts) == 1:
            raise PermissionError("[WinError 5] Access is denied")
        replace(source, target)

    monkeypatch.setattr(files.os, "replace", temporarily_locked)
    monkeypatch.setattr(files.time, "sleep", lambda _: None)
    _atomic_csv(frame, path)
    _atomic_csv(frame, path)
    pd.testing.assert_frame_equal(pd.read_csv(path), frame)
    assert len(attempts) == 2
