"""Per-dataset learner scores for the statistical framework (`config/synthesis/analysis.yaml`).

Turns the compact study exports (A1 `comparison.zip`, A2 `robustness.zip`, B1-B5 Group B and
C1-C2 Group C ZIPs) into the ``(dataset_id, learner, score)`` frames that
`inference.compare_learners` expects, one frame per comparison group of each
``framework.application`` entry. `run_application` runs the comparison only when
``framework.status == "approved"``; until then it returns an explicit awaiting-approval
record and reads no export.

Units and scores (fixed before results):

* Datasets are the unit; folds only form each dataset's five-fold mean.
* B and C families: headroom-normalised gain of an arm over the SAME predictor's non-text
  baseline, ``(augmented - baseline) / (1 - baseline)`` on the five-fold means of the fixed
  headline metric (binary ROC-AUC, multiclass macro OvR ROC-AUC, regression R2). All arms of a
  dataset and predictor share one denominator, so per-dataset ranks equal raw-metric ranks and
  differences between arms stay commensurable across the 18 datasets. A baseline >= 1 (zero
  headroom) leaves the gain UNDEFINED: no floor is applied here (unlike the descriptive
  ``headroom_share`` of `group_b`), and inference stops.
* Medians over decision models use the configured roster (never the results present); a
  dataset lacking any roster model leaves the median undefined.
* A2: `study_reports.POOLED_SKILL` per dataset (log loss or RPS skill of the averaged over the
  reference presentation); a ``reference`` learner with skill 0 is the control.
* Expected datasets and learners come from the configuration (`benchmark_plan`), never from
  the results. A missing, incomplete (fewer than five finite folds) or undefined cell stops
  that group's inference with the cells named; nothing is dropped. Arms the plan marks
  ``unsupported_width`` are excluded by design before results, and listed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation import group_b
from src.evaluation.endpoints import HEADLINES

APPROVED = "approved"
AWAITING = "awaiting_owner_approval"
NORMALISED_GAIN = "headroom_normalised_gain"
HEADLINE = "headline_metric"
REGRET = "regret"
BASELINE = "baseline"
REFERENCE = "reference"
MODES = ("per_column", "joint", "both")
SCORE_COLUMNS = ["group", "dataset_id", "task_type", "learner", "score", "status"]
#: Cell statuses; only "ok" enters inference.
OK, MISSING, INCOMPLETE, ZERO_HEADROOM, UNDEFINED = (
    "ok", "missing", "incomplete_folds", "zero_headroom", "undefined")


@dataclass(frozen=True)
class Design:
    """What the plan expects, read from configuration only (no results)."""

    datasets: dict  # dataset_id -> task_type, every selected dataset
    roster: tuple  # decision models every median must contain
    plan: pd.DataFrame  # planned arms: phase, dataset_id, model_id, representation, width_status, ...
    lodo_input_sets: tuple = ()
    lodo_threshold: float = 0.0
    #: A2 factor -> eligible datasets; None means "not declared before results" (stop).
    a2_datasets: dict = field(default_factory=dict)


def planned_design() -> Design:
    """The configured design: 18 datasets, 17 decision models, every planned arm."""
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries
    from src.evaluation.campaign import benchmark_plan
    from src.utils.config import load_yaml

    main = load_yaml("experiment_1/main")
    semantics = load_yaml("dataset_semantics")["datasets"]
    datasets = {e["id"]: semantics[e["id"]]["task_type"] for e in selected_entries(load_catalog(), main)}
    plan = pd.DataFrame(benchmark_plan())
    plan["model_id"] = plan["model"]
    plan["representation"] = [group_b.representation(r) for r in plan.to_dict("records")]
    rule = load_yaml("rq2/when_text_helps")["predict_in_advance"]
    multiclass = [d for d, t in datasets.items() if t == "multiclass_classification"]
    return Design(datasets=datasets, roster=tuple(main["candidate_extractors"]), plan=plan,
                  lodo_input_sets=tuple(rule["input_sets"]), lodo_threshold=float(rule["use_threshold"]),
                  a2_datasets={"wording": list(datasets), "option_order": multiclass, "field_order": None})


# ------------------------------------------------------------------ scores from fold tables


def normalised_gain(augmented, baseline):
    """(augmented - baseline) / (1 - baseline); NaN (undefined) where the headroom is <= 0."""
    augmented, baseline = np.asarray(augmented, float), np.asarray(baseline, float)
    headroom = 1 - baseline
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(headroom > 0, (augmented - baseline) / headroom, np.nan)


def arm_means(folds: pd.DataFrame, extra=()) -> pd.DataFrame:
    """Five-fold mean headline metric per dataset, predictor and arm (plus `extra` keys)."""
    keys = ["dataset_id", "task_type", "model_id", "representation", *extra]
    head = group_b.headline(folds)
    if head.empty:
        return pd.DataFrame(columns=[*keys, "mean", "split_id", "complete"])
    if head.duplicated([*keys, "outer_fold"]).any():
        raise ValueError("Several fold records of one arm; select one campaign")
    identity = head.groupby(keys, dropna=False)[["split_id", "checkpoint_id"]].nunique(dropna=False)
    if identity.gt(1).any(axis=None):
        raise ValueError("A dataset mean must use five folds of one split and checkpoint")
    out = head.groupby(keys, dropna=False, as_index=False).agg(
        mean=("value", "mean"), split_id=("split_id", "first"), folds=("outer_fold", "nunique"),
        finite=("value", lambda v: int(np.isfinite(v.to_numpy(float)).sum())))
    out["complete"] = out.folds.eq(5) & out.finite.eq(5)
    return out.drop(columns=["folds", "finite"])


def gains_over_baseline(means: pd.DataFrame, extra=()) -> pd.DataFrame:
    """Normalised gain of every arm over its predictor's baseline, with a cell status."""
    pair = ["dataset_id", "task_type", "model_id", *extra]
    base = means[means.representation.eq(BASELINE)][[*pair, "mean", "split_id", "complete"]]
    base = base.rename(columns={"mean": "baseline", "split_id": "baseline_split", "complete": "base_ok"})
    data = means.merge(base, on=pair, how="left")
    if (data.base_ok.notna() & data.split_id.ne(data.baseline_split)).any():
        raise ValueError("An arm and its baseline use different splits")
    data["score"] = normalised_gain(data["mean"], data["baseline"])
    data["status"] = np.select(
        [data.baseline.isna(), ~data.complete.eq(True) | ~data.base_ok.eq(True),
         data.baseline >= 1, ~np.isfinite(data.score)],
        [MISSING, INCOMPLETE, ZERO_HEADROOM, UNDEFINED], OK)
    data.loc[data.status.ne(OK), "score"] = np.nan
    return data[[*pair, "representation", "score", "status"]]


def _worst(statuses) -> str:
    for status in (MISSING, INCOMPLETE, ZERO_HEADROOM, UNDEFINED):
        if status in set(statuses):
            return status
    return OK


def decision_medians(gains: pd.DataFrame, roster, *, prefixes=("decision",), extra=()) -> pd.DataFrame:
    """Median over the full roster per dataset, predictor and mode; undefined if any is absent."""
    rows = []
    keys = ["dataset_id", "task_type", "model_id", *extra]
    for prefix in prefixes:
        for mode in MODES:
            wanted = [f"{prefix}/{mode}/{p}" for p in roster]
            part = gains[gains.representation.isin(wanted)]
            for key, group in part.groupby(keys, dropna=False):
                status = MISSING if group.representation.nunique() < len(wanted) else _worst(group.status)
                rows.append(dict(zip(keys, key)) | {
                    "representation": f"{prefix}/{mode}/median",
                    "score": float(group.score.median()) if status == OK else np.nan, "status": status})
    return pd.DataFrame(rows, columns=[*keys, "representation", "score", "status"])


def _excluded(design: Design, model, learners, roster, **match) -> list[str]:
    """Learners whose planned arm is `unsupported_width` for this predictor on any dataset."""
    plan = design.plan[design.plan.model_id.eq(model)]
    for name, value in match.items():
        plan = plan[plan[name].eq(value)]
    unsupported = set(plan.loc[plan.width_status.eq("unsupported_width"), "representation"])
    out = []
    for learner in learners:
        parts = learner.split("/")
        underlying = ({"/".join([*parts[:2], p]) for p in roster} if parts[-1] == "median" else {learner})
        if underlying & unsupported:
            out.append(learner)
    return out


def _grid(frame: pd.DataFrame, group: str, learners, datasets: dict) -> pd.DataFrame:
    """Every expected (dataset, learner) cell; absent cells are MISSING, never dropped."""
    index = pd.MultiIndex.from_product([list(datasets), list(learners)], names=["dataset_id", "learner"])
    found = frame.rename(columns={"representation": "learner"}).set_index(["dataset_id", "learner"])
    if found.index.duplicated().any():
        raise ValueError(f"{group}: several scores for one dataset and learner")
    out = found.reindex(index)[["score", "status"]].reset_index()
    out["status"] = out.status.fillna(MISSING)
    out["task_type"] = out.dataset_id.map(datasets)
    out["group"] = group
    return out[SCORE_COLUMNS]


@dataclass
class Group:
    group: str
    learners: list
    control: str | None
    scores: pd.DataFrame
    excluded_by_design: list = field(default_factory=list)
    #: Set when the group cannot be scored at all (declared before results), e.g. undeclared datasets.
    stop_reason: str | None = None


def _main_models(design: Design) -> list[str]:
    return list(dict.fromkeys(design.plan.loc[design.plan.phase.eq("experiment_1"), "model_id"]))


def _planned(design: Design, phase, prefix=None) -> list[str]:
    reps = design.plan.loc[design.plan.phase.eq(phase), "representation"]
    reps = list(dict.fromkeys(r for r in reps if prefix is None or r.split("/", 1)[0] in prefix))
    return reps


def _per_predictor(design, gains, learners, control, *, groups_of=None, datasets=None, match=None):
    out = []
    for model in _main_models(design):
        excluded = _excluded(design, model, learners, design.roster, **(match or {}))
        kept = [learner for learner in learners if learner not in excluded]
        frame = gains[gains.model_id.eq(model)]
        name = model if groups_of is None else f"{model}/{groups_of}"
        out.append(Group(name, kept, control, _grid(frame, name, kept, datasets or design.datasets), excluded))
    return out


def b1_groups(frames, design: Design) -> list[Group]:
    gains = gains_over_baseline(arm_means(frames["folds"]))
    gains = pd.concat([gains, decision_medians(gains, design.roster)], ignore_index=True)
    learners = [BASELINE, *(f"decision/{m}/median" for m in MODES)]
    return _per_predictor(design, gains, learners, BASELINE)


def b2_groups(frames, design: Design) -> list[Group]:
    gains = gains_over_baseline(arm_means(frames["folds"]))
    gains = pd.concat([gains, decision_medians(gains, design.roster, prefixes=("decision", "decision+frozen"))],
                      ignore_index=True)
    others = [*_planned(design, "experiment_2", ("frozen",)), *_planned(design, "experiment_3", ("tar", "probe"))]
    learners = ["decision/both/median", "decision+frozen/both/median", *others]
    return _per_predictor(design, gains, learners, "decision/both/median")


def b4_groups(frames, design: Design) -> list[Group]:
    """Per predictor and training size: gain per draw (five-fold means), then mean over draws."""
    means = arm_means(frames["folds"], extra=("training_size", "draw"))
    means = means[means.training_size.notna()]
    per_draw = gains_over_baseline(means, extra=("training_size", "draw"))
    plan = design.plan[design.plan.phase.eq("data_efficiency")]
    out = []
    for size in sorted(plan.training_size.dropna().unique()):
        at = plan[plan.training_size.eq(size)]
        draws = sorted(at.draw.dropna().unique())
        datasets = {d: design.datasets[d] for d in design.datasets if d in set(at.dataset_id)}
        rows = []
        part = per_draw[per_draw.training_size.eq(size)]
        for key, g in part.groupby(["dataset_id", "task_type", "model_id", "representation"]):
            status = MISSING if sorted(g.draw.unique()) != draws else _worst(g.status)
            rows.append(dict(zip(["dataset_id", "task_type", "model_id", "representation"], key))
                        | {"score": float(g.score.mean()) if status == OK else np.nan, "status": status})
        gains = pd.DataFrame(rows, columns=["dataset_id", "task_type", "model_id", "representation", "score",
                                            "status"])
        learners = list(dict.fromkeys(at.representation))
        out += _per_predictor(design, gains, learners, BASELINE, groups_of=f"n{int(size)}",
                              datasets=datasets, match={"phase": "data_efficiency", "training_size": size})
    return out


def b5_groups(frames, design: Design) -> list[Group]:
    """Native systems: five-fold mean headline metric (descriptive; ranks per dataset)."""
    means = arm_means(frames["folds"])
    means = means.assign(score=means["mean"].where(means.complete),
                         status=np.where(means.complete, OK, INCOMPLETE))
    learners = _planned(design, "experiment_4")
    return [Group("native_systems", learners, None, _grid(means, "native_systems", learners, design.datasets))]


def _c_groups(frames, design, phase, extra_learners, control):
    gains = gains_over_baseline(arm_means(frames["folds"]))
    learners = [*extra_learners, *_planned(design, phase)]
    out = []
    for model in _main_models(design):
        excluded = _excluded(design, model, learners, design.roster)
        kept = [learner for learner in learners if learner not in excluded]
        out.append(Group(model, kept, control, _grid(gains[gains.model_id.eq(model)], model, kept,
                                                     design.datasets), excluded))
    return out


def c1_groups(frames, design: Design) -> list[Group]:
    return _c_groups(frames, design, "representations", [BASELINE], BASELINE)


def c2_groups(frames, design: Design) -> list[Group]:
    from src.evaluation.group_c import PROVIDER

    return _c_groups(frames, design, "rq3", [f"decision/joint/{PROVIDER}"], None)


def b3_groups(frames, design: Design) -> list[Group]:
    """Per-dataset regret of the three LODO rules and of always/never using the features.

    Regret = max(actual, 0) - realised, realised = actual gain if the rule uses the features
    (predicted > threshold) else 0, exactly as `group_b.lodo_evaluation` averages it.
    """
    table = frames["lodo_predictions"]
    learners = [*(f"rule/{s}" for s in design.lodo_input_sets), "always", "never"]
    out = []
    for model in _main_models(design):
        part = table[table.model_id.eq(model)] if not table.empty else table
        rows = []
        for dataset, g in part.groupby("dataset_id"):
            actual = g.actual.to_numpy(float)
            if not np.allclose(actual, actual[0]):
                raise ValueError(f"{model}/{dataset}: input sets disagree on the actual gain")
            oracle = max(actual[0], 0.0)
            for name, row in g.set_index("input_set").iterrows():
                use = row.predicted > design.lodo_threshold
                rows.append({"dataset_id": dataset, "representation": f"rule/{name}",
                             "score": oracle - (row.actual if use else 0.0)})
            rows += [{"dataset_id": dataset, "representation": "always", "score": oracle - actual[0]},
                     {"dataset_id": dataset, "representation": "never", "score": oracle}]
        frame = pd.DataFrame(rows, columns=["dataset_id", "representation", "score"])
        frame["status"] = np.where(np.isfinite(frame.score.to_numpy(float)), OK, UNDEFINED)
        out.append(Group(model, learners, None, _grid(frame, model, learners, design.datasets)))
    return out


def a1_groups(frames, design: Design) -> list[Group]:
    """Stand-alone log loss of every decision model on the joint requests (classification)."""
    quality = frames["quality"]
    part = quality[quality.metric.eq("log_loss")][["dataset_id", "provider", "value"]]
    part = part.rename(columns={"provider": "representation", "value": "score"})
    part["status"] = np.where(np.isfinite(part.score.to_numpy(float)), OK, UNDEFINED)
    datasets = {d: t for d, t in design.datasets.items() if t != "regression"}
    learners = list(design.roster)
    return [Group("joint_requests", learners, None, _grid(part, "joint_requests", learners, datasets))]


def a2_skill(table: pd.DataFrame) -> pd.DataFrame:
    """Per dataset, model, factor and contrast: `proper_score_skill` (NaN kept, never dropped)."""
    from src.evaluation.robustness_export import CONTRASTS
    from src.evaluation.study_reports import _pooled_skill

    columns = ["dataset", "provider", "factor", "contrast", "skill"]
    if table.empty:
        return pd.DataFrame(columns=columns)
    keys = ["dataset", "provider", "factor", "metric", "presentation"]
    if table.duplicated(keys).any():
        raise ValueError("Several A2 quality rows for one dataset, model, factor and presentation")
    wide = table.set_index(keys)["value"].unstack("presentation").reset_index()  # NaN values kept
    out = []
    for contrast, presentation, reference in CONTRASTS:
        if not {presentation, reference} <= set(wide.columns):
            continue
        part = wide[["dataset", "provider", "factor", "metric"]].copy()
        part["reference"], part["average"] = wide[reference], wide[presentation]
        skill = _pooled_skill(part)
        out.append(skill.assign(contrast=contrast).rename(columns={"average": "skill"})[columns])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=columns)


def a2_groups(frames, design: Design) -> list[Group]:
    """Per factor and contrast: every decision model's skill against a zero-skill reference."""
    from src.evaluation.robustness_export import CONTRASTS

    skill = a2_skill(frames["averaging_quality"])
    learners = [REFERENCE, *design.roster]
    out = []
    for factor, eligible in design.a2_datasets.items():
        for contrast, *_ in CONTRASTS:
            name = f"{factor}/{contrast}"
            if eligible is None:
                out.append(Group(name, learners, REFERENCE, pd.DataFrame(columns=SCORE_COLUMNS),
                                 stop_reason=f"eligible {factor} datasets not declared before results"))
                continue
            part = skill[skill.factor.eq(factor) & skill.contrast.eq(contrast)]
            frame = part.rename(columns={"dataset": "dataset_id", "provider": "representation",
                                         "skill": "score"})[["dataset_id", "representation", "score"]]
            frame["status"] = np.where(np.isfinite(frame.score.to_numpy(float)), OK, UNDEFINED)
            reference = pd.DataFrame({"dataset_id": list(eligible), "representation": REFERENCE,
                                      "score": 0.0, "status": OK})
            datasets = {d: design.datasets[d] for d in eligible}
            out.append(Group(name, learners, REFERENCE,
                             _grid(pd.concat([frame, reference], ignore_index=True), name, learners, datasets)))
    return out


# ------------------------------------------------------------------ applications


@dataclass(frozen=True)
class Application:
    source: str  # export read by `read_frames`
    tables: tuple
    builder: object
    score_type: str
    higher_is_better: bool


#: One reader per `framework.application` entry (A3 is descriptive only and has none).
APPLICATIONS = {
    "A1_outputs": Application("A1", ("quality",), a1_groups, "log_loss", False),
    "A2_stability": Application("A2", ("averaging_quality",), a2_groups, "proper_score_skill", True),
    "B1_decision_features": Application("B1", ("folds",), b1_groups, NORMALISED_GAIN, True),
    "B2_representations": Application("B2", ("folds",), b2_groups, NORMALISED_GAIN, True),
    "B3_when_text_helps": Application("B3", ("lodo_predictions",), b3_groups, REGRET, False),
    "B4_data_efficiency": Application("B4", ("folds",), b4_groups, NORMALISED_GAIN, True),
    "B5_native_text_systems": Application("B5", ("folds",), b5_groups, HEADLINE, True),
    "C1_strands_states": Application("C1", ("folds",), c1_groups, NORMALISED_GAIN, True),
    "C2_strands_adaptation": Application("C2", ("folds",), c2_groups, NORMALISED_GAIN, True),
}


def read_frames(source: str):
    """(frames, receipt) of one verified export; empty frames while it is awaited."""
    from src.evaluation import benchmark_export, group_c, robustness_export, study_export

    if source == "A1":
        from src.evaluation import comparison_report

        return study_export.read_export(comparison_report.report_path(), schema=comparison_report.SCHEMA,
                                        tables=comparison_report.TABLES)
    if source == "A2":
        return robustness_export.read_export()
    if source in group_c.STUDIES:
        return group_c.read(source)
    return benchmark_export.read(source)


def application_groups(name: str, frames: dict, design: Design | None = None) -> list[Group]:
    """The comparison groups (scores with cell statuses) of one application."""
    spec = APPLICATIONS[name]
    missing = [t for t in spec.tables if t not in frames]
    if missing:
        raise ValueError(f"{name}: export lacks tables {missing}")
    return spec.builder(frames, design or planned_design())


def application_scores(name: str, frames: dict, design: Design | None = None) -> pd.DataFrame:
    """All groups' cells as one long frame (columns `SCORE_COLUMNS`)."""
    groups = application_groups(name, frames, design)
    return pd.concat([g.scores for g in groups], ignore_index=True) if groups else pd.DataFrame(
        columns=SCORE_COLUMNS)


def rope_for(score_type: str, framework: dict):
    """The prespecified ROPE; headline-metric and regret scores need one value shared by the
    three headline metrics. None when the plan declares none (inference then stops)."""
    rope = framework.get("rope") or {}
    if score_type in (HEADLINE, REGRET):
        values = {rope.get(m) for m in HEADLINES.values()}
        return values.pop() if len(values) == 1 and None not in values else None
    return rope.get(score_type)


def framework_state(analysis: dict | None = None) -> dict:
    from src.utils.config import load_yaml

    analysis = analysis or load_yaml("synthesis/analysis")
    framework = analysis.get("framework") or {}
    return {"framework_status": framework.get("status"), "approved": framework.get("status") == APPROVED,
            "applications": sorted(framework.get("application") or {})}


def run_application(name: str, frames: dict | None = None, *, analysis: dict | None = None,
                    design: Design | None = None, alpha: float = 0.05) -> dict:
    """Compare the learners of one application, only when the framework is approved.

    Returns a record: ``awaiting_owner_approval`` (nothing read), ``not_applicable`` (no
    learner comparison planned), ``awaiting_export`` or ``compared``; within ``compared``
    each group is ``compared`` (omnibus + pairwise table) or ``stopped`` (cells named).
    """
    from src.evaluation.inference import compare_learners
    from src.utils.config import load_yaml

    analysis = analysis or load_yaml("synthesis/analysis")
    framework = analysis.get("framework") or {}
    entry = (framework.get("application") or {}).get(name)
    if entry is None:
        raise KeyError(f"No framework.application entry {name}")
    base = {"application": name, "framework_status": framework.get("status"), "procedure": entry}
    if framework.get("status") != APPROVED:
        return base | {"status": AWAITING, "note": "framework.status is not 'approved'; no export was read "
                       "and no test was run."}
    if name not in APPLICATIONS:
        return base | {"status": "not_applicable", "note": f"inference: {entry.get('inference')}"}
    spec = APPLICATIONS[name]
    receipt = {}
    if frames is None:
        frames, receipt = read_frames(spec.source)
        if not frames:
            return base | {"status": "awaiting_export", "receipt": receipt}
    rope = rope_for(spec.score_type, framework)
    out = base | {"status": "compared", "score_type": spec.score_type, "rope": rope,
                  "higher_is_better": spec.higher_is_better, "receipt": receipt, "groups": []}
    for group in application_groups(name, frames, design):
        record = {"group": group.group, "learners": group.learners, "control": group.control,
                  "excluded_by_design": group.excluded_by_design}
        bad = group.scores[group.scores.status.ne(OK)]
        reason = (group.stop_reason or ("no expected cells" if group.scores.empty else None)
                  or ("incomplete, missing or undefined cells" if not bad.empty else None)
                  or (None if rope is not None else f"no prespecified ROPE for {spec.score_type}"))
        if reason:
            out["groups"].append(record | {"status": "stopped", "reason": reason,
                                           "cells": bad[["dataset_id", "learner", "status"]].to_dict("records")})
            continue
        omnibus, table = compare_learners(
            group.scores[["dataset_id", "learner", "score"]], higher_is_better=spec.higher_is_better,
            control=group.control, rope=rope, alpha=alpha)
        out["groups"].append(record | {"status": "compared", "omnibus": omnibus, "table": table})
    return out


def summary_record(result: dict) -> dict:
    """JSON-ready summary for notebooks (pairwise tables stay in `result`)."""
    keep = {k: v for k, v in result.items() if k not in {"groups", "receipt"}}
    if "groups" in result:
        keep["groups"] = [
            {k: v for k, v in g.items() if k not in {"table", "omnibus", "cells"}}
            | ({"stopped_cells": len(g["cells"])} if "cells" in g else {})
            | ({"iman_davenport_p": g["omnibus"]["iman_davenport_p"],
                "holm_significant": int((g["table"].p_holm < g["omnibus"]["alpha"]).sum())}
               if "omnibus" in g else {})
            for g in result["groups"]]
    return keep


def applications_table(analysis: dict | None = None) -> pd.DataFrame:
    """Every framework application with its planned procedure and current gate."""
    from src.utils.config import load_yaml

    analysis = analysis or load_yaml("synthesis/analysis")
    framework = analysis.get("framework") or {}
    rows = []
    for name, entry in (framework.get("application") or {}).items():
        spec = APPLICATIONS.get(name)
        rows.append({"application": name, "comparison": entry.get("comparison"),
                     "procedure": entry.get("procedure"), "inference": entry.get("inference"),
                     "score": spec.score_type if spec else None,
                     "export": spec.source if spec else None,
                     "gate": "approved" if framework.get("status") == APPROVED else framework.get("status")})
    return pd.DataFrame(rows)


def pooled_scores(frames_by_application: dict, design: Design | None = None) -> pd.DataFrame:
    """Long frame for `inference.pooled_confirmatory_tests` (columns `POOLED_SCORE_COLUMNS`)."""
    from src.evaluation.inference import POOLED_CONTRASTS, POOLED_SCORE_COLUMNS

    design = design or planned_design()
    parts = []
    for name, frames in frames_by_application.items():
        score_type = APPLICATIONS[name].score_type
        if score_type not in POOLED_CONTRASTS.values():
            raise ValueError(f"{name} scores ({score_type}) are not a pooled confirmatory score")
        parts.append(application_scores(name, frames, design).assign(application=name, score_type=score_type))
    if not parts:
        return pd.DataFrame(columns=list(POOLED_SCORE_COLUMNS))
    return pd.concat(parts, ignore_index=True)[list(POOLED_SCORE_COLUMNS)]


def confirmatory_family_state(analysis: dict | None = None) -> dict:
    """What the declared confirmatory family contains and whether it may run (no tests run)."""
    from src.evaluation.inference import POOLED_CONTRASTS
    from src.utils.config import load_yaml

    analysis = analysis or load_yaml("synthesis/analysis")
    contrasts = analysis.get("contrasts") or []
    pooled = [c for c in contrasts if c.get("type") in POOLED_CONTRASTS]
    ready = (analysis.get("approved_before_results") is True and bool(analysis.get("reviewed_by"))
             and bool(contrasts) and (analysis.get("framework") or {}).get("status") == APPROVED)
    return {"approved_before_results": analysis.get("approved_before_results"),
            "reviewed_by": analysis.get("reviewed_by"),
            "framework_status": (analysis.get("framework") or {}).get("status"),
            "proposal_status": (analysis.get("proposal") or {}).get("status"),
            "declared_contrasts": len(contrasts), "pooled_contrasts": len(pooled),
            "per_task_contrasts": len(contrasts) - len(pooled),
            "pooled_types": sorted(POOLED_CONTRASTS),
            "status": "ready_to_run" if ready else AWAITING}
