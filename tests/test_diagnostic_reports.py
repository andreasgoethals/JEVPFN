import json

import pytest

from src.evaluation.diagnostic_reports import analyze


def test_exp0_uses_explicit_real_reports_and_measures_aligned_changes(tmp_path):
    report = {
        "provider": "intern",
        "status": "gpu_contracts_passed_requires_review",
        "real_prompt_robustness": {
            "probes": [
                {"probe": "real_fixture__base", "features": {"p_a": 0.1, "p_b": 0.7, "p_c": 0.2}},
                {
                    "probe": "real_fixture__option_order",
                    "features": {"p_c": 0.2, "p_b": 0.7, "p_a": 0.1},
                    "canonical_probability_order": [2, 1, 0],
                },
                {
                    "probe": "real_fixture__paraphrase",
                    "features": {"p_a": 0.3, "p_b": 0.5, "p_c": 0.2},
                },
            ]
        },
    }
    file = tmp_path / "gpu.json"
    file.write_text(json.dumps(report), encoding="utf-8")
    result = analyze([file])
    assert result.total_variation_from_base.tolist() == pytest.approx([0, 0, 0.2])
    assert result.maximum_probability_change.tolist() == pytest.approx([0, 0, 0.2])
    assert result.same_top_option.all() and set(result.provider) == {"intern"}
    with pytest.raises(ValueError, match="twice"):
        analyze([file, file])
    report["status"] = "running"
    file.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="completed"):
        analyze([file])


def test_exp0_accepts_local_real_jev_diagnostics(tmp_path):
    report = {
        "status": "jev_dataset_robustness_requires_review",
        "attempts": [
            {"probe": "real_binary__base", "features": {"p_positive": 0.2}},
            {"probe": "real_binary__embedded_instruction", "features": {"p_positive": 0.7}},
        ],
    }
    file = tmp_path / "jev.json"
    file.write_text(json.dumps(report), encoding="utf-8")
    result = analyze([file])
    assert result.total_variation_from_base.iloc[1] == pytest.approx(0.5)
    assert not result.same_top_option.iloc[1]
    assert set(result.provider) == {"jev"}
