"""B4 field attribution: grouped permutation importance on outer-test rows, no extra fits.

After the final outer-training fit and the regular outer-test prediction, each feature
GROUP (one decision block per provider/mode/text field, the text-availability flags, the
non-text columns, or one raw embedded text column) is permuted jointly across the outer-test
rows. The fitted model predicts again; the recorded value is the mean per-row loss increase
over the unpermuted prediction.

Permutation scheme (``permutation`` setting, recorded per fold and per group):

* ``conditional_on_availability`` (default since 19 Nov 2026): each decision block and each
  embedding group is permuted only among rows that share its text-availability flag pattern
  (the flag of its text field; all flags for a joint block). Missing text then never swaps
  with present text, so the group is credited for its content given availability, not for
  re-encoding missingness that the availability flags already carry. Embedding groups keep
  their own flag in place. The availability-flag group and the non-text group are permuted
  unconditionally, as before. A group without any flag column falls back to unconditional.
* ``unconditional``: the original v1 scheme, every group permuted across all rows.

Losses:

* classification: log loss of the observed class (``metrics.observed_log_losses``);
* regression: squared error divided by the outer-training target variance.

Primary predictions, metrics, thresholds and records are never changed. Results live in a
separate ``field_attribution`` table of the evaluation store, outside record and fold
identities, so resume and pairing are unaffected. A fold reused on resume has no fitted
model and therefore receives no attribution; ``attribution_status`` reports such gaps.
Attribution describes what one fitted predictor relies on; it is not a causal effect.
"""

from __future__ import annotations

import hashlib
import re
import warnings

import numpy as np
import pandas as pd

from src.evaluation.features import missing_column
from src.utils.serialization import canonical_json, digest, scalar

VERSION = "grouped-permutation-attribution-v2"
CONDITIONAL = "conditional_on_availability"
UNCONDITIONAL = "unconditional"
PERMUTATIONS = (CONDITIONAL, UNCONDITIONAL)
#: Group kinds whose permutation is stratified by text availability under CONDITIONAL.
STRATIFIED_KINDS = ("decision", "embedding")
STRUCTURED_PHASES = ("B1", "B2", "B3")
NATIVE_SKIP_REASON = (
    "native_text_system: original text enters the system's own preprocessing, so no "
    "fixed decision/embedding feature groups exist to permute"
)
_DECISION = re.compile(
    r"^(?:(?P<namespace>[^_](?:(?!__).)*)__)?(?P<provider>[^_](?:(?!__).)*)__"
    r"(?P<mode>per_column|joint)__text_only__(?P<field>text_\d{3}|all_text)__(?P<key>.+)$"
)
_SUBSET_STREAM = 0x5EB5E7


def settings_for_phase(config, phase):
    """Resolve the protocol switch for one benchmark phase; native text is skipped by design."""
    config = dict(config or {})
    unknown = set(config) - {"enabled", "repeats", "max_test_rows", "permutation"}
    if unknown:
        raise ValueError(f"Unknown field-attribution settings: {sorted(unknown)}")
    settings = {
        "version": VERSION,
        "enabled": config.get("enabled", False),
        "repeats": config.get("repeats", 3),
        "max_test_rows": config.get("max_test_rows"),
        "permutation": config.get("permutation", CONDITIONAL),
        "skip_reason": None,
    }
    validate_settings(settings)
    if phase == "B6":
        settings.update(enabled=False, skip_reason=NATIVE_SKIP_REASON)
    elif phase not in STRUCTURED_PHASES:
        raise ValueError("Field attribution applies to benchmark phases only")
    elif settings["enabled"] is not True:
        settings["skip_reason"] = "disabled_in_protocol_config"
    return settings


def validate_settings(settings):
    if settings.get("enabled") not in (True, False):
        raise ValueError("field_attribution.enabled must be a boolean")
    repeats, cap = settings.get("repeats"), settings.get("max_test_rows")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("field_attribution.repeats must be a positive integer")
    if cap is not None and (type(cap) is not int or cap < 2):
        raise ValueError("field_attribution.max_test_rows must be null or an integer >= 2")
    if settings.get("permutation", CONDITIONAL) not in PERMUTATIONS:
        raise ValueError(f"field_attribution.permutation must be one of {PERMUTATIONS}")
    return settings


def feature_groups(features: pd.DataFrame, metadata) -> list[dict]:
    """Assign every supplied column to exactly one group; unknown columns fail loudly.

    Raw frozen embeddings are grouped per text column together with that column's
    availability flag: the embedding wrapper verifies that the flag equals the vector's
    missingness, so the two carry one piece of information and must move together.
    """
    columns = list(features.columns)
    if len(columns) != len(set(columns)):
        raise ValueError("Feature column names must be unique")
    text_columns = [str(c) for c in metadata.text_columns]
    non_text = {str(c) for c in metadata.non_text_columns}
    assigned, groups = {}, {}

    def add(column, name, **info):
        if column in assigned:
            raise ValueError(f"Column {column!r} matches more than one attribution group")
        assigned[column] = name
        groups.setdefault(name, {"group": name, **info, "columns": []})["columns"].append(column)

    for i, group in enumerate(features.attrs.get("embedding_groups") or []):
        if i >= len(text_columns):
            raise ValueError("More embedding groups than text columns")
        for column in [*group, *([missing_column(i)] if missing_column(i) in features else [])]:
            add(column, f"embedding__text_{i:03d}", kind="embedding", provider=None, mode=None,
                text_index=i, text_field=text_columns[i])
    for column in columns:
        if column in assigned:
            continue
        name = str(column)
        match = _DECISION.match(name)
        if name.startswith("text_presence__"):
            if name in non_text or match:
                raise ValueError(f"Ambiguous availability column {name!r}")
            add(column, "text_availability", kind="text_availability", provider=None,
                mode=None, text_index=None, text_field=None)
        elif match:
            if name in non_text:
                raise ValueError(f"Non-text column {name!r} looks like a decision feature")
            provider = match["namespace"] or match["provider"]
            field = match["field"]
            index = None if field == "all_text" else int(field[5:])
            if index is not None and index >= len(text_columns):
                raise ValueError(f"Decision column {name!r} names an unknown text field")
            add(column, f"decision__{provider}__{match['mode']}__{field}", kind="decision",
                provider=provider, mode=match["mode"], text_index=index,
                text_field="all_text" if index is None else text_columns[index])
        elif name in non_text:
            add(column, "non_text", kind="non_text", provider=None, mode=None,
                text_index=None, text_field=None)
        else:
            raise ValueError(f"Column {name!r} cannot be assigned to an attribution group")
    return list(groups.values())


def group_hash(name: str) -> int:
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)


def group_rng(seed: int, fold: int, name: str) -> np.random.Generator:
    """Deterministic and independent of group order: run seed, outer fold, group name."""
    return np.random.default_rng([int(seed), int(fold), group_hash(name)])


def strata_columns(group: dict, frame: pd.DataFrame, permutation: str) -> list[str]:
    """Availability flags that define the group's permutation strata (empty: unconditional)."""
    if permutation == UNCONDITIONAL or group.get("kind") not in STRATIFIED_KINDS:
        return []
    flags = sorted(c for c in frame.columns if str(c).startswith("text_presence__"))
    index = group.get("text_index")
    if index is None:  # A joint block depends on every text field's availability.
        return flags
    flag = missing_column(index)
    return [flag] if flag in frame.columns else []


def stratified_order(rng: np.random.Generator, strata) -> np.ndarray:
    """A permutation that moves rows only within equal strata keys (rows x flags array).

    Strata are visited in sorted key order, so the draw depends only on the seed and the
    stratum membership. Without strata this equals ``rng.permutation(n)``.
    """
    strata = np.asarray(strata)
    n = len(strata)
    if strata.ndim == 1:
        strata = strata[:, None]
    if strata.shape[1] == 0:
        return rng.permutation(n)
    keys, codes = np.unique(strata, axis=0, return_inverse=True)
    codes = np.asarray(codes).reshape(-1)
    order = np.arange(n)
    for code in range(len(keys)):
        members = np.flatnonzero(codes == code)
        order[members] = members[rng.permutation(len(members))]
    return order


def permute_columns(frame: pd.DataFrame, columns, order) -> pd.DataFrame:
    """Move the group's values jointly across rows; row IDs and other columns stay put."""
    result = frame.copy()
    block = frame[list(columns)].iloc[np.asarray(order)].set_axis(frame.index)
    for column in columns:
        result[column] = block[column]
    return result


def row_losses(y_true, prediction, classes, variance):
    """Per-row loss with the B4 definitions used by ``row_benefit``."""
    prediction = np.asarray(prediction, dtype=float)
    if not len(classes):
        y = np.asarray(y_true, dtype=float)
        residual = y - prediction
        if not np.isfinite(residual).all() or variance is None or not np.isfinite(variance) or variance < 0:
            raise ValueError("Invalid regression prediction or training variance")
        return residual**2 / variance if variance else np.full(len(y), np.nan)
    from src.evaluation.metrics import observed_log_losses

    indices = {canonical_json(scalar(c)): i for i, c in enumerate(classes)}
    true = [indices[canonical_json(scalar(v))] for v in y_true]
    return observed_log_losses(prediction, true)


def expanded_inputs(model, x):
    """Raw inputs a wrapper builds internally (TAR vectors), and the fitted inner predictor."""
    hook = getattr(model, "attribution_inputs", None)
    if hook is None:
        return x, [], model
    return hook(x)


def fold_attribution(model, x, y_true, prediction, groups, *, predict, classes, variance,
                     seed, fold, settings):
    """Mean per-row loss increase for each group; ``predict(predictor, frame)`` -> array."""
    frame, extra, predictor = expanded_inputs(model, x)
    groups = [*groups, *extra]
    covered = [c for g in groups for c in g["columns"]]
    if len(covered) != len(set(covered)) or set(covered) != set(frame.columns):
        raise ValueError("Attribution groups must cover each predictor input exactly once")
    if not frame.index.equals(x.index):
        raise ValueError("Expanded attribution inputs changed the outer-test rows")
    y_true = np.asarray(y_true)
    rows = len(frame)
    cap = settings.get("max_test_rows")
    subset = None
    if cap is not None and rows > cap:
        stream = np.random.default_rng([int(seed), int(fold), _SUBSET_STREAM])
        subset = np.sort(stream.choice(rows, cap, replace=False))
        frame, y_true = frame.iloc[subset], y_true[subset]
        # A subset baseline is predicted again so batch-dependent predictors stay paired.
        prediction = predict(predictor, frame)
    base = row_losses(y_true, prediction, classes, variance)
    control = row_losses(y_true, predict(predictor, frame), classes, variance) - base
    permutation = settings.get("permutation", CONDITIONAL)
    if permutation not in PERMUTATIONS:
        raise ValueError(f"Unknown permutation scheme {permutation!r}")
    results = []
    for group in groups:
        rng = group_rng(seed, fold, group["group"])
        flags = strata_columns(group, frame, permutation)
        strata = frame[flags].to_numpy() if flags else np.empty((len(frame), 0))
        values = []
        for _ in range(settings["repeats"]):
            order = stratified_order(rng, strata)
            loss = row_losses(y_true, predict(predictor, permute_columns(frame, group["columns"], order)),
                              classes, variance)
            values.append(float(np.mean(loss - base)) if np.isfinite(loss - base).all() else None)
        defined = [v for v in values if v is not None]
        results.append({
            **{k: v for k, v in group.items() if k != "columns"},
            "n_columns": len(group["columns"]),
            "columns_sha256": digest(group["columns"]),
            "permutation": CONDITIONAL if flags else UNCONDITIONAL,
            "strata_columns": flags,
            "n_strata": int(len(np.unique(strata, axis=0))) if flags else 1,
            "mean_loss_increase": float(np.mean(defined)) if len(defined) == len(values) else None,
            "repeat_loss_increases": values,
        })
    return {
        "permutation": permutation,
        "status": "complete" if np.isfinite(base).all() else "undefined_loss",
        "reason": None if np.isfinite(base).all() else "zero_variance_training_target",
        "loss": "log_loss" if len(classes) else "squared_error_over_training_variance",
        "training_target_variance": variance,
        "outer_test_rows": rows,
        "rows_used": len(frame),
        "row_subset": None if subset is None else {
            "max_test_rows": cap, "row_ids_sha256": digest(frame.index.tolist()),
            "baseline": "repredicted_unpermuted_subset",
        },
        "baseline_mean_loss": float(np.mean(base)) if np.isfinite(base).all() else None,
        # Deterministic predictors give zero; anything else is predict-time noise.
        "unpermuted_repredict_mean_loss_change": float(np.mean(control))
        if np.isfinite(control).all() else None,
        "groups": results,
    }


def run_fold(model, x, y_true, prediction, groups, *, predict, classes, variance, seed, fold,
             settings, device="cpu", synchronize=None):
    """Timed in its own PhaseTimer and warning capture, so primary records are unaffected."""
    from src.evaluation.records import PhaseTimer

    timer = PhaseTimer(device=device, synchronize=synchronize)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with timer.measure("field_attribution"):
            body = fold_attribution(model, x, y_true, prediction, groups, predict=predict,
                                    classes=classes, variance=variance, seed=seed, fold=fold,
                                    settings=settings)
    body["timing"] = timer.to_dict()
    body["warnings"] = [
        {"category": w.category.__name__, "message": str(w.message)[:2000]} for w in captured
    ]
    return body


def record_body(identity, settings, result=None):
    """Store body: fold identity, resolved settings, code identity and the outcome."""
    from src.utils.provenance import code_identity

    outcome = dict(result) if result else {
        "status": "skipped", "reason": settings.get("skip_reason") or "disabled_in_protocol_config"
    }
    return {
        "schema": "jevpfn.field_attribution.v1",
        **identity,
        "settings": settings,
        "code_sha256": code_identity(["src/evaluation/attribution.py"]),
        **outcome,
    }


# ---------------------------------------------------------------- collection and summaries

def _representation(recipe, feature_set, phase):
    from src.evaluation.group_b import family, representation

    recipe = recipe or {}
    if feature_set == "non_text":
        return "decision", "non_text"
    embedding = recipe.get("embedding") or {}
    label = representation({
        "feature_set": feature_set, "provider": recipe.get("provider"),
        "encoder": embedding.get("encoder_id"), "pca_scope": embedding.get("pca_scope"),
        "pca_components": embedding.get("pca_components_per_column"),
        "probe_view": (recipe.get("probe") or {}).get("view"),
        "dropped_text_column": recipe.get("dropped_text_column"),
    })
    # Attribution has historically named pure frozen embeddings "embedding".
    label = label.replace("frozen/", "embedding/", 1) if label.startswith("frozen/") else label
    return family(label), label


def attribution_status(phases=STRUCTURED_PHASES + ("B6",)):
    """Completed folds versus stored attribution: complete, skipped, undefined or missing."""
    from src.evaluation.store import all_recipes, attribution_bodies

    rows = []
    for phase in phases:
        bodies = {}
        for body in attribution_bodies(phase):
            bodies.setdefault(body["fold_recipe_id"], []).append(body)
        recipes = {digest(r): r for r in all_recipes(phase)}
        for key, recipe in recipes.items():
            found = bodies.get(key, [])
            rows.append({
                "phase": phase, "dataset_id": recipe["dataset_id"], "model_id": recipe["model_id"],
                "feature_set": recipe["feature_set"], "outer_fold": recipe["outer_fold"],
                "fold_recipe_id": key,
                "status": "missing" if not found else "+".join(sorted({b["status"] for b in found})),
            })
        for key in set(bodies) - set(recipes):
            rows.append({"phase": phase, "dataset_id": bodies[key][0]["dataset_id"],
                         "model_id": bodies[key][0].get("model_id"),
                         "feature_set": bodies[key][0].get("feature_set"),
                         "outer_fold": bodies[key][0].get("outer_fold"), "fold_recipe_id": key,
                         "status": "orphan_without_completed_fold"})
    table = pd.DataFrame(rows, columns=["phase", "dataset_id", "model_id", "feature_set",
                                        "outer_fold", "fold_recipe_id", "status"])
    counts = (table.groupby(["phase", "status"]).size().rename("folds").reset_index()
              if not table.empty else pd.DataFrame(columns=["phase", "status", "folds"]))
    return table, counts


def attribution_folds(phases=STRUCTURED_PHASES):
    """One row per completed fold and group, only for folds whose recipe is completed."""
    from src.evaluation.store import all_recipes, attribution_bodies

    rows = []
    for phase in phases:
        completed = {digest(r) for r in all_recipes(phase)}
        for body in attribution_bodies(phase):
            if body["status"] != "complete" or body["fold_recipe_id"] not in completed:
                continue
            family, representation = _representation(body.get("representation_recipe"),
                                                     body["feature_set"], phase)
            # v1 bodies predate the setting and were unconditional.
            scheme = (body.get("settings") or {}).get("permutation", UNCONDITIONAL)
            for group in body["groups"]:
                rows.append({
                    "phase": phase, "dataset_id": body["dataset_id"],
                    "task_type": body.get("task_type"), "model_id": body["model_id"],
                    "checkpoint_id": body["checkpoint_id"], "feature_set": body["feature_set"],
                    "family": family, "representation": representation, "permutation": scheme,
                    "outer_fold": body["outer_fold"],
                    **{k: group.get(k) for k in ("group", "kind", "provider", "mode",
                                                 "text_index", "text_field", "n_columns")},
                    "mean_loss_increase": group["mean_loss_increase"],
                })
    return pd.DataFrame(rows)


_GROUP_KEYS = ["phase", "dataset_id", "task_type", "model_id", "checkpoint_id", "feature_set",
               "family", "representation", "permutation", "group", "kind", "provider", "mode",
               "text_index", "text_field"]


def attribution_summary(phases=STRUCTURED_PHASES, *, field_types=None, folds=None):
    """Per-group means over folds, then equal-dataset field-type summaries.

    Returns ``(by_group, by_field_type, status_counts)``. Groups are typed by the label-free
    taxonomy in ``field_types``: per-column decision and embedding groups take their text
    field's type; joint blocks, availability flags and non-text columns keep their own type.
    Each dataset first averages its groups of one type, then datasets carry equal weight.
    The permutation scheme is a key throughout, so conditional and unconditional results
    never mix.
    """
    counts = None
    if folds is None:
        folds = attribution_folds(phases)
        _, counts = attribution_status(phases)
    if folds.empty:
        return pd.DataFrame(), pd.DataFrame(), counts
    if "permutation" not in folds:
        folds = folds.assign(permutation=UNCONDITIONAL)
    by_group = folds.groupby(_GROUP_KEYS, dropna=False).agg(
        mean_loss_increase=("mean_loss_increase", "mean"),
        fold_sd=("mean_loss_increase", "std"),
        defined_folds=("mean_loss_increase", "count"),
        folds=("outer_fold", "nunique"),
    ).reset_index()
    if field_types is None:
        from src.evaluation.field_types import field_type_table

        field_types = field_type_table()
    lookup = {(r.dataset_id, str(r.field)): r.field_type for r in field_types.itertuples()}

    def field_type(row):
        if row.kind in ("non_text", "text_availability"):
            return row.kind
        if row.text_field == "all_text":
            return "joint_all_text"
        key = (row.dataset_id, str(row.text_field))
        if key not in lookup:
            raise ValueError(f"Field type unknown for {key}")
        return lookup[key]

    by_group["field_type"] = [field_type(r) for r in by_group.itertuples()]
    complete = by_group[by_group.defined_folds.eq(by_group.folds) & by_group.folds.eq(5)]
    keys = ["phase", "model_id", "checkpoint_id", "family", "representation", "permutation",
            "kind", "field_type"]
    per_dataset = (complete.groupby([*keys, "dataset_id"], dropna=False)
                   .mean_loss_increase.mean().reset_index())
    by_type = (per_dataset.groupby(keys, dropna=False)
               .mean_loss_increase.agg(mean_loss_increase="mean", n_datasets="count")
               .reset_index())
    by_type["weighting"] = "equal_dataset_after_within_dataset_group_mean"
    return by_group, by_type, counts
