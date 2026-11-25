"""Per-dataset learner scores for the statistical framework (`config/05_synthesis/analysis.yaml`).

Turns the compact study exports (A1 `comparison.zip`, A2 `robustness.zip`, B1-B5 Group B and
C1-C2 Group C ZIPs) into the ``(dataset_id, learner, score)`` frames that
`inference.compare_learners` expects, one frame per comparison group of each
``framework.application`` entry. `run_application` runs the comparison only when
``framework.status == "approved"`` (owner approval of 24-11-2026); otherwise it returns an
explicit awaiting-approval record and reads no export. `run_confirmatory_family` runs the
declared pooled contrasts of ``contrasts`` (Holm over all of them) once their exports exist.

Units and scores (fixed before results):

* Datasets are the unit; folds only form each dataset's five-fold mean.
* B and C families: headroom-normalised gain of an arm over the SAME predictor's non-text
  baseline, ``(augmented - baseline) / (1 - baseline)`` on the five-fold means of the fixed
  headline metric (binary ROC-AUC, multiclass macro OvR ROC-AUC, regression R2). All arms of a
  dataset and predictor share one denominator, so per-dataset ranks equal raw-metric ranks and
  differences between arms stay commensurable across the 18 datasets. A baseline >= 1 (zero
  headroom) leaves the gain UNDEFINED: the descriptive and confirmatory
  calculations both leave the value missing.
* Medians over decision models use the configured roster (never the results present); a
  dataset lacking any roster model leaves the median undefined.
* A2: `study_reports.POOLED_SKILL` per dataset (log loss or RPS skill of the averaged over the
  reference presentation); a ``reference`` learner with skill 0 is the control.
* Expected datasets and learners come from the configuration (`benchmark_plan`), never from
  the results. A missing, incomplete (fewer than five finite folds) or undefined cell stops
  that group's inference with the cells named; nothing is dropped. Arms the plan marks
  ``unsupported_width`` are excluded by design before results, and listed.
* Fold level (descriptive, `fold_wins`): for applications with fold tables (B1, B2, B4, B5, C1,
  C2) each group also carries its (dataset, outer fold, learner) headline metric. Within one
  (dataset, fold) cell all arms of a predictor share the baseline, so ordering the raw headline
  metric equals ordering the fold-level normalised gain. Roster medians are taken per fold; B4
  averages a fold over its planned draws. Win rates (vs the group's control, or all pairs) and
  PAMA use dataset-cluster bootstrap intervals; A1, A2 and B3 have no fold-level scores.
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
FOLD_SCORE_COLUMNS = ["group", "dataset_id", "outer_fold", "learner", "score"]
OUTER_FOLDS = tuple(range(5))
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


def planned_design(cohort=None) -> Design:
    """The configured datasets, complete selected decision-model cohort, and planned arms."""
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries
    from src.evaluation.benchmark_export import comparison_plan, decision_roster
    from src.utils.config import load_yaml

    main = load_yaml("experiment_1/main")
    semantics = load_yaml("dataset_semantics")["datasets"]
    entries = selected_entries(load_catalog(), main)
    datasets = {e["id"]: semantics[e["id"]]["task_type"] for e in entries}
    # A2 field order (behavior.yaml B3_field_order) needs joint requests with at least two
    # named text fields; the catalog's text-column count fixes the eligible datasets before
    # results (16 of 18: Jigsaw Toxicity and Product Sentiment have one text column).
    multi_field = [e["id"] for e in entries if e["paper_summary"]["text_columns"] >= 2]
    plan = pd.DataFrame(comparison_plan(cohort))
    plan["model_id"] = plan["model"]
    plan["representation"] = [group_b.representation(r) for r in plan.to_dict("records")]
    rule = load_yaml("rq2/when_text_helps")["predict_in_advance"]
    multiclass = [d for d, t in datasets.items() if t == "multiclass_classification"]
    return Design(datasets=datasets, roster=tuple(decision_roster(cohort)), plan=plan,
                  lodo_input_sets=tuple(rule["input_sets"]), lodo_threshold=float(rule["use_threshold"]),
                  a2_datasets={"wording": list(datasets), "option_order": multiclass, "field_order": multi_field})


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
        return pd.DataFrame(columns=[*keys, "mean", "split_id", "checkpoint_id", "complete"])
    if head.duplicated([*keys, "outer_fold"]).any():
        raise ValueError("Several fold records of one arm; select one campaign")
    group_b.validate_fold_identities(head, extra=extra)
    identity = head.groupby(keys, dropna=False)[["split_id", "checkpoint_id"]].nunique(dropna=False)
    if identity.gt(1).any(axis=None):
        raise ValueError("A dataset mean must use five folds of one split and checkpoint")
    out = head.groupby(keys, dropna=False, as_index=False).agg(
        mean=("value", "mean"), split_id=("split_id", "first"), checkpoint_id=("checkpoint_id", "first"),
        folds=("outer_fold", "nunique"),
        finite=("value", lambda v: int(np.isfinite(v.to_numpy(float)).sum())))
    out["complete"] = out.folds.eq(5) & out.finite.eq(5)
    return out.drop(columns=["folds", "finite"])


def gains_over_baseline(means: pd.DataFrame, extra=()) -> pd.DataFrame:
    """Normalised gain of every arm over its predictor's baseline, with a cell status."""
    pair = ["dataset_id", "task_type", "model_id", *extra]
    base = means[means.representation.eq(BASELINE)][[*pair, "mean", "split_id", "checkpoint_id", "complete"]]
    base = base.rename(columns={"mean": "baseline", "split_id": "baseline_split",
                                "checkpoint_id": "baseline_checkpoint", "complete": "base_ok"})
    data = means.merge(base, on=pair, how="left")
    if (data.base_ok.notna() & data.split_id.ne(data.baseline_split)).any():
        raise ValueError("An arm and its baseline use different splits")
    if (data.base_ok.notna() & data.checkpoint_id.ne(data.baseline_checkpoint)).any():
        raise ValueError("An arm and its baseline use different predictor checkpoints")
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


def fold_values(folds: pd.DataFrame, roster=(), *, prefixes=(), extra=()) -> pd.DataFrame:
    """Fold-level headline metric per dataset, predictor, arm and fold (plus roster medians).

    A median over the roster is NaN on a fold where any roster model is absent or non-finite.
    """
    keys = ["dataset_id", "model_id", "representation", *extra, "outer_fold"]
    head = group_b.headline(folds)
    if head.empty:
        return pd.DataFrame(columns=[*keys, "score"])
    group_b.validate_fold_identities(head, extra=extra)
    if head.duplicated(keys).any():
        raise ValueError("Several fold records of one arm; select one campaign")
    out = head[keys].assign(score=head["value"].astype(float))
    rows = []
    cell = [k for k in keys if k != "representation"]
    for prefix in prefixes:
        for mode in MODES:
            wanted = [f"{prefix}/{mode}/{p}" for p in roster]
            for key, group in out[out.representation.isin(wanted)].groupby(cell, dropna=False):
                values = group.score.to_numpy(float)
                full = group.representation.nunique() == len(wanted) and np.isfinite(values).all()
                rows.append(dict(zip(cell, key)) | {"representation": f"{prefix}/{mode}/median",
                                                     "score": float(np.median(values)) if full else np.nan})
    if not rows:
        return out
    return pd.concat([out, pd.DataFrame(rows)[out.columns]], ignore_index=True)


def _fold_grid(frame: pd.DataFrame, group: str, learners, datasets: dict) -> pd.DataFrame:
    """Every expected (dataset, fold, learner) row; absent scores are NaN, never dropped."""
    names = ["dataset_id", "outer_fold", "learner"]
    index = pd.MultiIndex.from_product([list(datasets), list(OUTER_FOLDS), list(learners)], names=names)
    frame = frame[frame.representation.isin(list(learners))]
    found = frame.rename(columns={"representation": "learner"}).set_index(names)
    if found.index.duplicated().any():
        raise ValueError(f"{group}: several fold scores for one dataset, fold and learner")
    out = found.reindex(index)[["score"]].reset_index()
    out["group"] = group
    return out[FOLD_SCORE_COLUMNS]


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
    #: Fold-level headline metric (columns `FOLD_SCORE_COLUMNS`); None when the export has no folds.
    folds: pd.DataFrame | None = None


def _main_models(design: Design) -> list[str]:
    return list(dict.fromkeys(design.plan.loc[design.plan.phase.eq("experiment_1"), "model_id"]))


def _planned(design: Design, phase, prefix=None) -> list[str]:
    reps = design.plan.loc[design.plan.phase.eq(phase), "representation"]
    reps = list(dict.fromkeys(r for r in reps if prefix is None or r.split("/", 1)[0] in prefix))
    return reps


def _per_predictor(design, gains, learners, control, *, groups_of=None, datasets=None, match=None,
                   fold_scores=None):
    out = []
    datasets = datasets or design.datasets
    for model in _main_models(design):
        excluded = _excluded(design, model, learners, design.roster, **(match or {}))
        kept = [learner for learner in learners if learner not in excluded]
        frame = gains[gains.model_id.eq(model)]
        name = model if groups_of is None else f"{model}/{groups_of}"
        folds = (None if fold_scores is None else
                 _fold_grid(fold_scores[fold_scores.model_id.eq(model)], name, kept, datasets))
        out.append(Group(name, kept, control, _grid(frame, name, kept, datasets), excluded, folds=folds))
    return out


def b1_groups(frames, design: Design) -> list[Group]:
    gains = gains_over_baseline(arm_means(frames["folds"]))
    gains = pd.concat([gains, decision_medians(gains, design.roster)], ignore_index=True)
    learners = [BASELINE, *(f"decision/{m}/median" for m in MODES)]
    folds = fold_values(frames["folds"], design.roster, prefixes=("decision",))
    return _per_predictor(design, gains, learners, BASELINE, fold_scores=folds)


def b2_groups(frames, design: Design) -> list[Group]:
    gains = gains_over_baseline(arm_means(frames["folds"]))
    gains = pd.concat([gains, decision_medians(gains, design.roster, prefixes=("decision", "decision+frozen"))],
                      ignore_index=True)
    others = [*_planned(design, "experiment_2", ("frozen",)), *_planned(design, "experiment_3", ("tar", "probe"))]
    learners = ["decision/both/median", "decision+frozen/both/median", *others]
    folds = fold_values(frames["folds"], design.roster, prefixes=("decision", "decision+frozen"))
    return _per_predictor(design, gains, learners, "decision/both/median", fold_scores=folds)


def b4_groups(frames, design: Design) -> list[Group]:
    """Per predictor and training size: gain per draw (five-fold means), then mean over draws."""
    means = arm_means(frames["folds"], extra=("training_size", "draw"))
    means = means[means.training_size.notna()]
    per_draw = gains_over_baseline(means, extra=("training_size", "draw"))
    fold_draws = fold_values(frames["folds"], extra=("training_size", "draw"))
    fold_draws = fold_draws[fold_draws.training_size.notna()]
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
        # Fold level: mean over the planned draws of each fold; NaN unless every draw is finite.
        keys = ["dataset_id", "model_id", "representation", "outer_fold"]
        fold_rows = [dict(zip(keys, key)) | {"score": float(g.score.mean()) if sorted(g.draw.unique()) == draws
                                              and np.isfinite(g.score.to_numpy(float)).all() else np.nan}
                     for key, g in fold_draws[fold_draws.training_size.eq(size)].groupby(keys)]
        out += _per_predictor(design, gains, learners, BASELINE, groups_of=f"n{int(size)}",
                              datasets=datasets, match={"phase": "data_efficiency", "training_size": size},
                              fold_scores=pd.DataFrame(fold_rows, columns=[*keys, "score"]))
    return out


def b5_groups(frames, design: Design) -> list[Group]:
    """Native systems: five-fold mean headline metric (descriptive; ranks per dataset)."""
    # The B5 export also carries structured baselines for its separate reference plots.
    folds = frames["folds"]
    native = folds if folds.empty else folds[folds.phase.eq("experiment_4")]
    means = arm_means(native)
    means = means.assign(score=means["mean"].where(means.complete),
                         status=np.where(means.complete, OK, INCOMPLETE))
    learners = _planned(design, "experiment_4")
    folds = _fold_grid(fold_values(native), "native_systems", learners, design.datasets)
    return [Group("native_systems", learners, None, _grid(means, "native_systems", learners, design.datasets),
                  folds=folds)]


def _c_groups(frames, design, phase, extra_learners, control):
    gains = gains_over_baseline(arm_means(frames["folds"]))
    learners = [*extra_learners, *_planned(design, phase)]
    return _per_predictor(design, gains, learners, control, fold_scores=fold_values(frames["folds"]))


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
    if table.empty:
        table = pd.DataFrame(columns=["model_id", "dataset_id", "input_set", "actual", "predicted"])
    elif "outcome" not in table or not table.outcome.eq("headroom_share").fillna(False).all():
        raise ValueError("The all-dataset B3 regret comparison requires headroom-share outcomes")
    learners = [*(f"rule/{s}" for s in design.lodo_input_sets), "always", "never"]
    out = []
    for model in _main_models(design):
        part = table[table.model_id.eq(model)]
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
            coverage = frames.get("coverage")
            if factor == "field_order" and coverage is not None and not coverage.empty:
                fields = coverage[coverage.variant.eq("field_order")]
                canonical = fields[fields.status.eq("canonicalized_by_runtime") & fields["count"].gt(0)]
                other = fields[fields.status.ne("canonicalized_by_runtime") & fields["count"].gt(0)]
                unavailable = set(zip(canonical.dataset, canonical.provider)) - set(zip(other.dataset, other.provider))
                existing = set(zip(frame.dataset_id, frame.representation))
                absent = [{"dataset_id": d, "representation": p, "score": np.nan, "status": "canonicalized_by_runtime"}
                          for d, p in sorted(unavailable - existing)]
                if absent:
                    frame = pd.concat([frame, pd.DataFrame(absent)], ignore_index=True)
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


def read_frames(source: str, *, cohort=None):
    """(frames, receipt) of one verified export; empty frames while it is awaited."""
    from src.evaluation import benchmark_export, group_c, robustness_export

    if source == "A1":
        from src.evaluation import comparison_report

        return comparison_report.read_tables(comparison_report.report_config(cohort), cohort=cohort)
    if source == "A2":
        return robustness_export.read_export(cohort=cohort)
    if source in group_c.STUDIES:
        return group_c.read(source)
    return benchmark_export.read(source, cohort=cohort)


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
    """Look up the prespecified ROPE in the units of the score."""
    rope = framework.get("rope") or {}
    if score_type == REGRET:
        return rope.get(NORMALISED_GAIN)
    if score_type == HEADLINE:
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
                    design: Design | None = None, alpha: float = 0.05, cohort=None) -> dict:
    """Compare the learners of one application, only when the framework is approved.

    Returns a record: ``awaiting_owner_approval`` (nothing read), ``not_applicable`` (no
    learner comparison planned), ``awaiting_export`` or ``compared``; within ``compared``
    each group is ``compared`` (omnibus + pairwise table) or ``stopped`` (cells named).
    Groups with fold tables also carry ``fold_level`` (`fold_level_tables`): descriptive, reported
    for stopped groups too, with their excluded incomplete (dataset, fold) cells counted.
    """
    from src.evaluation.inference import compare_learners
    from src.utils.config import load_yaml

    analysis = analysis or load_yaml("synthesis/analysis")
    framework = analysis.get("framework") or {}
    entry = (framework.get("application") or {}).get(name)
    if entry is None:
        raise KeyError(f"No framework.application entry {name}")
    scope = entry.get("holm_family", "application_group")
    if entry.get("inference") in {"descriptive", "descriptive_only"}:
        scope = None
    base = {"application": name, "framework_status": framework.get("status"), "procedure": entry,
            "multiplicity_scope": scope}
    if framework.get("status") != APPROVED:
        return base | {"status": AWAITING, "note": "framework.status is not 'approved'; no export was read "
                       "and no test was run."}
    if name not in APPLICATIONS:
        return base | {"status": "not_applicable", "note": f"inference: {entry.get('inference')}"}
    from src.evaluation import benchmark_export

    base.update(benchmark_export.cohort_record(cohort) if design is None else {
        "decision_cohort": cohort or "supplied_design", "decision_models": len(design.roster),
        "decision_roster": list(design.roster)})
    spec = APPLICATIONS[name]
    receipt = {}
    if frames is None:
        frames, receipt = read_frames(spec.source, cohort=cohort) if cohort is not None else read_frames(spec.source)
        if not frames:
            return base | {"status": "awaiting_export", "receipt": receipt}
    rope = rope_for(spec.score_type, framework)
    out = base | {"status": "compared", "score_type": spec.score_type, "rope": rope,
                  "higher_is_better": spec.higher_is_better, "receipt": receipt, "groups": []}
    try:
        groups = application_groups(name, frames, design or planned_design(cohort))
    except ValueError as error:
        return out | {"status": "stopped", "reason": str(error)}
    for group in groups:
        descriptive = entry.get("inference") == "descriptive" or group.group.endswith("/average_vs_mean_single")
        record = {"group": group.group, "learners": group.learners, "control": group.control,
                  "inference": "descriptive" if descriptive else entry.get("inference"),
                  "excluded_by_design": group.excluded_by_design}
        if group.folds is not None:
            record["fold_level"] = fold_level_tables(group, higher_is_better=spec.higher_is_better)
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
            control=group.control, rope=rope, alpha=alpha,
            pairwise=not descriptive)
        table["holm_family"] = f"{name}/{group.group}"
        table["holm_family_size"] = len(table)
        if name == "C2_strands_adaptation":
            from src.evaluation.group_c import c2_contrasts

            named = {frozenset((c["a"], c["b"])) for c in c2_contrasts() if "context" not in c}
            table["named_recipe_contrast"] = [frozenset((a, b)) in named for a, b in zip(table.learner, table.versus)]
        out["groups"].append(record | {"status": "compared", "omnibus": omnibus, "table": table})
    return out


def fold_level_tables(group: Group, *, higher_is_better: bool, resamples: int = 10_000, seed: int = 0) -> dict:
    """Fold-level win rates (vs the control, else all pairs) and PAMA of one group.

    Descriptive; the intervals resample datasets (all five folds of each), never folds. Status
    ``reported`` or ``not_available`` (with the reason); missing cells are counted, not raised.
    """
    from itertools import combinations

    from src.evaluation import fold_wins

    learners = list(group.learners)
    pairs = ([(learner, group.control) for learner in learners if learner != group.control]
             if group.control is not None else list(combinations(learners, 2)))
    frame = group.folds[["dataset_id", "outer_fold", "learner", "score"]]
    options = {"higher_is_better": higher_is_better, "resamples": resamples, "seed": seed}
    out = {"unit": "dataset_x_outer_fold", "bootstrap_unit": fold_wins.BOOTSTRAP_UNIT,
           "expected_cells": int(len(frame[["dataset_id", "outer_fold"]].drop_duplicates()))}
    try:
        return out | {"status": "reported", "win_rates": fold_wins.fold_win_rates(frame, pairs, **options),
                      "pama": fold_wins.pama(frame, learners=learners, **options)}
    except ValueError as error:
        return out | {"status": "not_available", "reason": str(error)}


def summary_record(result: dict) -> dict:
    """JSON-ready summary for notebooks (pairwise tables stay in `result`)."""
    keep = {k: v for k, v in result.items() if k not in {"groups", "receipt"}}
    if "groups" in result:
        keep["groups"] = [
            {k: v for k, v in g.items() if k not in {"table", "omnibus", "cells", "fold_level"}}
            | ({"stopped_cells": len(g["cells"])} if "cells" in g else {})
            | ({"fold_level": g["fold_level"]["status"],
                "pama_excluded_incomplete_cells": int(g["fold_level"]["pama"].excluded_incomplete_cells.iloc[0])}
               if g.get("fold_level", {}).get("status") == "reported" else {})
            | ({"omnibus_status": g["omnibus"]["status"], "iman_davenport_p": g["omnibus"]["iman_davenport_p"],
                "holm_tests": len(g["table"]),
                "holm_significant": int((g["table"].p_holm < g["omnibus"]["alpha"]).sum()) if len(g["table"]) else None}
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
                     "holm_family": entry.get("holm_family"),
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


#: Short names of the confirmatory learners (figure and table labels only).
CONTRAST_LABELS = {"baseline": "no text", "decision/both/median": "decision",
                   "frozen/e5_small_v2/per_column32": "frozen E5-small", "tar/e5_small_v2_tar/per_column32":
                   "TAR E5-small"}


def contrast_label(contrast: dict) -> str:
    learner, versus = (CONTRAST_LABELS.get(contrast[k], contrast[k]) for k in ("learner", "versus"))
    return f"{contrast['group']}: {learner} vs {versus}"


def confirmatory_forest_table(pooled: pd.DataFrame, contrasts, results: pd.DataFrame, *,
                              n_boot: int = group_b.N_BOOT, seed: int = group_b.SEED) -> pd.DataFrame:
    """One row per declared pooled contrast for `visualize.overview.confirmatory_forest`: the
    median oriented per-dataset difference with a 95% percentile interval from resampling
    datasets, the Holm p over the whole family and the Bayesian signed-rank posteriors."""
    from src.evaluation.inference import POOLED_CONTRASTS, _pooled_values

    rows = []
    rng = np.random.default_rng(seed)
    for contrast, result in zip(contrasts, results.to_dict("records")):
        values = _pooled_values(pooled, contrast)
        medians = np.median(values[rng.integers(0, len(values), (n_boot, len(values)))], axis=1)
        rows.append({"contrast": contrast_label(contrast), "estimate": float(np.median(values)),
                     "low": float(np.quantile(medians, 0.025)), "high": float(np.quantile(medians, 0.975)),
                     "p_holm": result["p_holm"], "p_left": result.get("bayes_p_worse", np.nan),
                     "p_rope": result.get("bayes_p_equivalent", np.nan),
                     "p_right": result.get("bayes_p_better", np.nan), "datasets": len(values),
                     "score_type": POOLED_CONTRASTS[contrast["type"]], "rope": result.get("bayes_rope", np.nan)})
    return pd.DataFrame(rows)


def run_confirmatory_family(frames_by_application: dict | None = None, *, analysis: dict | None = None,
                            design: Design | None = None, n_boot: int = group_b.N_BOOT, cohort=None) -> dict:
    """Run the declared confirmatory family (`config/05_synthesis/analysis.yaml` ``contrasts``).

    Returns a record with ``status``: ``awaiting_owner_approval`` (plan not approved; nothing
    read), ``awaiting_export`` (``missing`` lists the applications whose export is absent; no
    test is run on part of the family), ``stopped`` (``reason``: incomplete or undefined cells,
    or per-task contrasts without their dataset differences) or ``compared`` (``results`` from
    `inference.pooled_confirmatory_tests`, Holm over every declared contrast, and ``forest`` for
    `visualize.overview.confirmatory_forest`).
    """
    from src.evaluation.design import incremental_cohort
    from src.evaluation.inference import pooled_confirmatory_tests
    from src.utils.config import load_yaml

    analysis = analysis or load_yaml("synthesis/analysis")
    state = confirmatory_family_state(analysis)
    from src.evaluation import benchmark_export

    cohort_info = (benchmark_export.cohort_record(cohort) if design is None else {
        "decision_cohort": cohort or "supplied_design", "decision_models": len(design.roster),
        "decision_roster": list(design.roster)})
    out = {"family": state, **cohort_info}
    if state["status"] != "ready_to_run":
        return out | {"status": AWAITING}
    if incremental_cohort(cohort_info["decision_cohort"]):
        return out | {"status": "not_applicable", "reason":
                      "This cohort selects incremental fits. Use the full named edition for the expanded primary release."}
    contrasts = analysis["contrasts"]
    if any("type" not in c for c in contrasts):
        return out | {"status": "stopped", "reason": "per-task contrasts are declared; pass their dataset "
                      "differences to inference.pooled_confirmatory_tests"}
    frames_by_application = dict(frames_by_application or {})
    names = list(dict.fromkeys(c["application"] for c in contrasts))
    missing = []
    out["export_receipts"] = {}
    for name in names:
        if name not in frames_by_application:
            frames, receipt = (read_frames(APPLICATIONS[name].source, cohort=cohort) if cohort is not None
                         else read_frames(APPLICATIONS[name].source))
            if not frames:
                missing.append(name)
                continue
            if design is None and any(receipt.get(k) != cohort_info[k] for k in
                                      ("decision_cohort", "decision_roster", "decision_model_versions", "roster_sha256")):
                return out | {"status": "stopped", "reason": f"{name} export has different decision-model versions or cohort"}
            out["export_receipts"][name] = {k: receipt.get(k) for k in
                                            ("sha256", "receipt_sha256", "config_sha256", "roster_sha256")}
            frames_by_application[name] = frames
    if missing:
        return out | {"status": "awaiting_export", "missing": missing}
    try:
        from src.evaluation.group_b_views import merged_heads

        # Shared baseline and decision measurements in separately exported studies must agree.
        merged_heads(*(frames_by_application[n].get("folds") for n in names))
        pooled = pooled_scores({n: frames_by_application[n] for n in names}, design or planned_design(cohort))
        results = pooled_confirmatory_tests(pooled, analysis)
    except ValueError as error:
        return out | {"status": "stopped", "reason": str(error)}
    forest = confirmatory_forest_table(pooled, contrasts, results, n_boot=n_boot)
    for table in (results, forest):
        table["decision_cohort"] = cohort_info["decision_cohort"]
        table["decision_models"] = cohort_info["decision_models"]
        table["decision_roster"] = ",".join(cohort_info["decision_roster"])
        if "roster_sha256" in cohort_info:
            table["roster_sha256"] = cohort_info["roster_sha256"]
    return out | {"status": "compared", "results": results, "forest": forest}


def publish_confirmatory(result, *, target=None, analysis=None):
    """Archive a full primary family and its plot data under the exact decision roster."""
    from src.evaluation import study_export
    from src.evaluation.design import decision_providers, incremental_cohort
    from src.utils import paths
    from src.utils.config import load_yaml
    from src.utils.provenance import code_identity
    from src.utils.serialization import digest

    if result["status"] != "compared":
        raise ValueError("Only a completed confirmatory family can be published")
    roster = list(result["decision_roster"])
    cohort = result["decision_cohort"]
    if incremental_cohort(cohort):
        raise ValueError("Use the full named edition to publish the expanded primary release")
    versions = result.get("decision_model_versions")
    if (not isinstance(versions, list) or not versions or result.get("decision_models") != len(roster)
            or any(not isinstance(v, dict) or any(not isinstance(v.get(k), str) or not v[k].strip()
                                                   for k in ("provider", "model", "version")) for v in versions)
            or [v.get("provider") for v in versions] != roster or digest(versions) != result.get("roster_sha256")
            or roster != list(decision_providers(cohort))):
        raise ValueError("A primary archive requires the recorded ordered cohort, exact model versions and their digest")
    signature = result["roster_sha256"]
    if target is None:
        target = (paths.notebook_artifacts("general/5_synthesis", kind="results") / "confirmatory"
                  / f"{cohort}-{signature[:16]}.zip")
    plan = analysis or load_yaml("synthesis/analysis")
    contrasts = plan["contrasts"]
    if len(result["results"]) != len(contrasts) or len(result["forest"]) != len(contrasts):
        raise ValueError("The archived tables must cover every declared contrast")
    for table in (result["results"], result["forest"]):
        for column, expected in (("roster_sha256", signature), ("decision_roster", ",".join(roster)),
                                 ("decision_cohort", cohort), ("decision_models", len(roster))):
            if column in table and not table[column].eq(expected).fillna(False).all():
                raise ValueError(f"Primary table {column} differs from the recorded analysis cohort")
    code = code_identity(["src/evaluation/learner_scores.py", "src/evaluation/group_b.py",
                          "src/evaluation/inference.py", "src/evaluation/group_b_views.py"])
    return study_export.write_export(target, {"results": result["results"], "forest": result["forest"]},
                                     schema="jevpfn.confirmatory_family.v1",
                                     identity=digest({"contrasts": plan["contrasts"], "framework": plan["framework"],
                                                      "decision_cohort": cohort, "decision_model_versions": versions, "code": code}),
                                     origin="pooled_confirmatory_tests",
                                     extra={"decision_cohort": cohort, "decision_roster": roster,
                                            "decision_models": len(roster), "decision_model_versions": versions,
                                            "roster_sha256": signature,
                                            "contrasts": len(plan["contrasts"]),
                                            "source_exports": result.get("export_receipts", {})})
