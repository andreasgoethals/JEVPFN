"""Exercise the real 0B gate without allocating models or using synthetic evidence as approval."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from src.cluster.predictor_contracts import CONTRACT_VERSION, required_fixture_cases
from src.evaluation import qualification as q
from src.evaluation.features import assemble_feature_set, missing_column
from src.evaluation.models import MAIN_MODELS, TEXT_MODELS
from src.utils.serialization import file_sha256, write_json


@pytest.fixture
def reports(monkeypatch):
    from src.evaluation import design

    load = design.load_yaml
    catalog = load("extractors")
    for provider in ("provider", "new_model"):
        catalog["candidates"][provider] = {"delivery": "hosted", "model": provider, "version": "v1"}
    monkeypatch.setattr(design, "load_yaml", lambda name: catalog if name == "extractors" else load(name))
    model = MAIN_MODELS[0]
    arm = {
        "model": model,
        "phase": "experiment_1",
        "dataset_id": "data",
        "provider": "provider",
        "feature_set": "non_text_plus_both",
        "predictor_columns": 4,
    }
    monkeypatch.setattr(q, "evaluation_identity", lambda: "current")
    monkeypatch.setattr(q, "profile_plan", lambda **kwargs: [arm])
    monkeypatch.setattr(q, "load_catalog", lambda: {"datasets": [{"id": "data"}]})
    monkeypatch.setattr(
        q,
        "load_yaml",
        lambda name: {"candidate_extractors": ["provider"], "encoders": ["encoder"]},
    )
    preflight = {
        "phase": "experiment_0_preflight",
        "stage": "after_extraction",
        "execution_site": "vsc_compute_node",
        "status": "preflight_passed",
        "contract_tests": {"status": "passed"},
        "runtime_parity": {"matches_control": True},
        "datasets": [
            {
                "dataset_id": "data",
                "status": "passed",
                "task_type": "regression",
                "data_identity": "raw",
                "text_columns": 1,
                "split_id": "split",
                "rows": 10,
                "outer_folds": 5,
            }
        ],
        "decision_artifacts": [
            {
                "dataset_id": "data",
                "provider": "provider",
                "status": "passed",
                "artifact_ids": {"joint": "joint", "per_column": "columns"},
                "artifact_checksums": {"joint": "j", "columns": "c"},
            }
        ],
        "embedding_artifacts": [
            {
                "dataset_id": "data",
                "encoder_id": "encoder",
                "status": "passed",
                "artifact_id": "vectors",
                "prepared_pca_keys": 10,
                "prepared_probe_keys": [f"probe{i}" for i in range(10)],
            }
        ],
    }
    preflight["tar_artifacts"] = [{"dataset_id": "data", "encoder_id": "e5_small_v2_tar", "status": "passed", "artifact_ids": [f"tar{i}" for i in range(5)], "prepared_pca_keys": [f"p{i}" for i in range(10)]}]
    runtime = {"container_id": "seal"}
    predictors = [
        {
            "model": m,
            "status": "predictor_contracts_passed_requires_full_fold_profile",
            "contract_version": CONTRACT_VERSION,
            "runtime_identity": runtime,
            "fixtures": [
                {
                    "task_type": task,
                    "classes": n,
                    "feature_case": case,
                    "resume_without_refitting": True,
                    "partial_fold_resume_verified": task == "binary_classification" and case in {"non_text", "original_text"},
                    "interrupted_record_id": "incomplete" if task == "binary_classification" else None,
                    "records": [f"store#{m}_{task}_{n}_{case}_{i}"
                                for i in range(10 if task == "binary_classification" else 5)],
                }
                for task, n, case in sorted(required_fixture_cases(m))
            ],
        }
        for m in MAIN_MODELS + TEXT_MODELS
    ]
    profile = {
        "arm": arm,
        "status": "full_fold_profile_passed",
        "cluster": "mindwell",
        "runtime_identity": runtime,
        "data_identity": "raw",
        "split_id": "split",
        "parent_split_id": "split", "outer_fold": 0,
        "resources": {"device": "cpu", "peak_rss_bytes": 1024000, "device_workers": 1, "fold_wall_seconds": 100.0},
        "feature_artifact_ids": ["joint", "columns"],
        "decision_artifact_checksums": {"joint": "j", "columns": "c"},
        "predictor_columns": 4,
        "training_rows": 8,
        "roles": [{"role": "outer_test", "rows": 2, "record_id": "test", "prediction_sha256": "sha"}],
    }
    return arm, [preflight, *predictors, profile]


def write_approval(tmp_path, bodies, **scope):
    evidence = []
    for i, body in enumerate(bodies):
        path = tmp_path / f"report_{i}.json"
        write_json(path, {**body, "evaluation_identity": "current", "job_id": str(i + 1)})
        evidence.append({"path": path.name, "sha256": file_sha256(path)})
    accepted = tmp_path / "review.json"
    write_json(
        accepted,
        {
            "accepted": True,
            "stage": "after_extraction",
            "reviewed_by": "reviewer",
            "evaluation_identity": "current",
            "evidence": evidence,
            **scope,
        },
    )
    return accepted


def test_real_profile_plan_covers_inner_oof_for_each_applicable_recipe():
    from src.evaluation.campaign import benchmark_plan

    def group(arm):
        return arm["phase"], arm["model"], arm.get("pca_scope")

    def needs_inner_oof(arm):
        return arm["dataset_id"].startswith("BIN_") and arm["phase"] != "experiment_3"

    # B3 refits and B4 samples are covered by the profiles of the full-size pipelines.
    arms = [a for a in benchmark_plan() if a["phase"] in q.PROFILED_PHASES]
    expected = {group(a) for a in arms if needs_inner_oof(a) and a["width_status"] != "unsupported_width"}
    actual = {group(a) for a in q.profile_plan(arms) if needs_inner_oof(a)}
    assert actual == expected


def test_full_fold_profiles_cover_every_task_and_largest_class_vocabulary():
    from src.evaluation.campaign import benchmark_plan

    semantics = q.load_yaml("dataset_semantics")["datasets"]
    arms = [a for a in benchmark_plan()
            if a["width_status"] != "unsupported_width" and a["phase"] in q.PROFILED_PHASES]

    def group(a):
        return a["phase"], a["model"], a.get("pca_scope"), a.get("pca_components")

    def task(a):
        return semantics[a["dataset_id"]]["task_type"]

    profiles = q.profile_plan(arms)
    assert {(group(a), task(a)) for a in profiles} == {(group(a), task(a)) for a in arms}
    for key in {group(a) for a in arms}:
        expected = max(len(semantics[a["dataset_id"]].get("classes", []))
                       for a in arms if group(a) == key)
        observed = max(len(semantics[a["dataset_id"]].get("classes", []))
                       for a in profiles if group(a) == key)
        assert observed == expected


def test_full_fold_profile_reads_saved_binary_outer_and_oof_predictions(
    isolated_output, monkeypatch
):
    import json

    from sklearn.dummy import DummyClassifier

    from src.cluster.predictor_contracts import feature_cases, fixtures
    from src.evaluation.protocol import evaluate_cv, shared_splits
    from src.utils import paths

    dataset = next(d for d in fixtures() if d.metadata.task_type == "binary_classification")
    _, features, _ = next(feature_cases(dataset, "linear"))
    features.attrs["decision_artifact_checksums"] = {}
    arm = {
        "phase": "experiment_1", "model": "linear", "feature_set": "non_text",
        "dataset_id": dataset.metadata.dataset_id,
        "predictor_columns": features.shape[1],
    }
    split = shared_splits(dataset, seed=q.load_yaml("experiment_1/main")["random_seed"])
    records = evaluate_cv(
        dataset, features, DummyClassifier, specification=split,
        model_id="linear", checkpoint_id="test", feature_set="non_text",
        phase="experiment_0", allow_model_runs=True, fold_numbers=[0],
        estimator_settings={"fixture": "dummy", "execution_runtime": {"container_id": "test"}},
        representation_recipe={"provider": None, "embedding": None},
    )
    with pytest.raises(ValueError, match="lacks required prediction roles"):
        q.save_profile(arm, dataset, features, {"container_id": "test"}, records[:1])
    with pytest.raises(ValueError, match="predictor width"):
        q.save_profile(arm | {"predictor_columns": 1}, dataset, features, {}, records)
    q.save_profile(arm, dataset, features, {"container_id": "test"}, records)
    path, = (paths.phase_path("experiment_0", "manifests", "after_extraction")).glob("profile_*.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert {r["role"]: r["rows"] for r in report["roles"]} == {
        "outer_test": len(split["folds"][0]["test"]),
        "inner_oof_binary_threshold": len(split["folds"][0]["train"]),
    }
    with pytest.raises(ValueError, match="outside scratch"):
        q.verify_published_profile(report)
    durable = paths.staging_override()
    monkeypatch.setattr(paths, "durable_project_override", lambda: durable)
    q.verify_published_profile(report)
    with pytest.raises(ValueError, match="Published 0B predictions differ"):
        q.verify_published_profile(report | {"arm": report["arm"] | {"model": "different"}})
    with pytest.raises(ValueError, match="completed fold recipe"):
        q.verify_published_profile(report | {"fold_recipe_id": "wrong"})
    with pytest.raises(ValueError, match="omits or duplicates"):
        q.verify_published_profile(report | {"roles": report["roles"][:1]})
    report["roles"][0]["record_id"] = "missing"
    with pytest.raises(ValueError, match="Missing evaluation record"):
        q.verify_published_profile(report)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("missing_mode", "exact artifact IDs"),
        ("missing_pca", "exact artifact IDs"),
        ("partial_contract", "fixtures or sealed runtime"),
        ("wrong_profile_split", "full-fold profile"),
        ("wrong_profile_input", "full-fold profile"),
        ("wrong_profile_runtime", "full-fold profile"),
    ],
)
def test_checksummed_but_incomplete_or_mismatched_0b_rejected(tmp_path, reports, fault, message):
    arm, bodies = deepcopy(reports)
    if fault == "missing_mode":
        bodies[0]["decision_artifacts"][0]["artifact_ids"].pop("joint")
    elif fault == "missing_pca":
        bodies[0]["embedding_artifacts"][0]["prepared_pca_keys"] = 0
    elif fault == "partial_contract":
        bodies[1]["fixtures"].pop()
    elif fault == "wrong_profile_split":
        bodies[-1]["split_id"] = "other"
    elif fault == "wrong_profile_input":
        bodies[-1]["decision_artifact_checksums"]["joint"] = "different"
    else:
        bodies[-1]["runtime_identity"] = {"container_id": "different"}
    with pytest.raises(ValueError, match=message):
        q.benchmark_evidence(write_approval(tmp_path, bodies), arm)


def test_qualified_content_and_runtime_checked_before_benchmark_fit(tmp_path, reports):
    arm, bodies = reports
    evidence = q.benchmark_evidence(write_approval(tmp_path, bodies), arm)
    dataset = SimpleNamespace(source={"data_identity": "raw"})
    features = pd.DataFrame()
    features.attrs.update(
        feature_artifact_ids=["columns", "joint"],
        decision_artifact_checksums={"columns": "c", "joint": "j"},
    )
    runtime = {"container_id": "seal"}
    split = {"split_id": "split"}
    q.verify_inputs(evidence, dataset, features, arm, runtime, specification=split)
    with pytest.raises(ValueError, match="qualified 0B partition"):
        q.verify_inputs(
            evidence, dataset, features, arm, runtime, specification={"split_id": "other"}
        )
    features.attrs["decision_artifact_checksums"]["joint"] = "changed"
    with pytest.raises(ValueError, match="Feature artifacts"):
        q.verify_inputs(evidence, dataset, features, arm, runtime, specification=split)


def test_0b_verifier_reopens_relative_evidence_across_a_layout_move(tmp_path, reports):
    import json

    arm, bodies = reports
    root = tmp_path / "output_JEVPFN"
    current = root / "03_exp0b/manifests/after_extraction"
    review = write_approval(current, bodies)
    moved = root / "01_exp0a/manifests/preflight.json"
    moved.parent.mkdir(parents=True)
    moved.write_bytes((current / "report_0.json").read_bytes())
    body = json.loads(review.read_text(encoding="utf-8"))
    body["evidence"][0]["path"] = "../preflight.json"
    write_json(review, body)
    historic = root / "3. experiment_0/manifests/after_extraction" / review.name
    assert not historic.exists()
    q.benchmark_evidence(historic, arm)


def test_missing_text_signal_is_available_to_the_structured_baseline():
    frame = pd.DataFrame({"text": ["present", None], "number": [1, 2], "target": [2, 4]})
    dataset = SimpleNamespace(
        features=frame.drop(columns="target"),
        frame=frame,
        metadata=SimpleNamespace(
            text_columns=("text",), non_text_columns=("number",), target="target"
        ),
    )
    matrix = assemble_feature_set(dataset, [], "non_text")
    assert list(matrix.columns) == ["number", missing_column(0)]
    assert matrix[missing_column(0)].tolist() == [0, 1]


def test_raw_copies_recodings_and_duplicate_rows_are_reported_separately():
    frame = pd.DataFrame({
        "x": [1, 1, 2, 3], "copy": [0, 0, 1, 1], "inverse": [1, 1, 0, 0],
        "alias": pd.Categorical(["a", "a", "b", "b"]), "noisy": [0, 0, 1, 0],
        "target": pd.Categorical([0, 0, 1, 1]),
    })
    dataset = SimpleNamespace(
        frame=frame,
        features=frame.drop(columns="target"),
        metadata=SimpleNamespace(target="target", task_type="binary_classification"),
    )
    audit = q.source_leakage_diagnostics(dataset, {"folds": [{"train": [0, 2], "test": [1, 3]}]})
    assert audit["exact_target_copies"] == ["copy"]
    assert audit["exact_class_target_recodings"] == ["inverse", "alias"]
    assert audit["exact_feature_row_overlap_by_fold"] == [1]
    assert audit["test_rows_by_fold"] == [2]


def test_benchmark_publication_copies_only_selected_evaluation_store(tmp_path):
    from src.cluster.scratch_stage import _publish
    from src.evaluation.store import connect

    work, durable = tmp_path / "scratch" / "JEVPFN", tmp_path / "project" / "JEVPFN"
    results = work / "output_JEVPFN" / "4. experiment_1" / "results"
    evaluation = results / "evaluation"
    evaluation.mkdir(parents=True)
    for dataset in ("DATA1", "DATA2"):
        with connect(evaluation / f"{dataset}.sqlite3"):
            pass
    (results / "splits").mkdir()
    (results / "splits" / "split.json").write_text("{}", encoding="utf-8")

    _publish(durable, work, dataset_id="DATA1")
    target = durable / results.relative_to(work)
    assert (target / "evaluation" / "DATA1.sqlite3").is_file()
    assert not (target / "evaluation" / "DATA2.sqlite3").exists()
    assert (target / "splits" / "split.json").is_file()

@pytest.mark.parametrize("corrupt", ["partial_frozen", "duplicate_tar", "empty_checksum", "empty_fold", "binary_inner"])
def test_0b_requires_real_partition_coverage(reports, corrupt):
    _, bodies = reports
    preflight = bodies[0]
    if corrupt == "partial_frozen":
        preflight["embedding_artifacts"][0]["prepared_pca_keys"] -= 1
    elif corrupt == "duplicate_tar":
        preflight["tar_artifacts"][0]["prepared_pca_keys"][-1] = "p0"
    elif corrupt == "empty_fold":
        preflight["tar_artifacts"][0]["artifact_ids"][-1] = None
    elif corrupt == "binary_inner":
        preflight["datasets"][0]["task_type"] = "binary_classification"
    else:
        preflight["decision_artifacts"][0]["artifact_checksums"]["joint"] = ""
    with pytest.raises(ValueError):
        q.feature_evidence(preflight)


def test_widest_kumo_override_arm_is_a_declared_full_fold_profile():
    from src.evaluation.campaign import benchmark_plan

    arms = [a for a in benchmark_plan() if a["phase"] in q.PROFILED_PHASES]
    wide = q.kumo_override_arm(arms)
    assert (wide["model"], wide["phase"], wide["dataset_id"], wide["feature_set"]) == (
        "kumo_tabular", "experiment_2", "REG_TEXT_ROTTEN_TOMATOES", "non_text_plus_both_plus_frozen_e5")
    assert wide["predictor_columns"] == wide["kumo_max_columns_override"] == 557
    profiles = q.profile_plan(arms)
    assert wide in profiles
    # The declared case reuses the width extreme rather than adding a second 557 profile.
    assert [a for a in profiles if a.get("kumo_max_columns_override")] == [wide]
    # The phase/model profile job (benchmark_jobs.plan(profile_fold=True)) also selects it.
    kumo = [a for a in arms if a["model"] == "kumo_tabular" and a["phase"] == "experiment_2"]
    assert wide in q.profile_plan(kumo)
    assert q.kumo_override_arm([a for a in arms if a["predictor_columns"] <= 500]) is None
    assert "kumo_max_columns_override_b200_profile" in q.load_yaml(
        "experiment_0/research_questions")["rq2"]


def _b200_report(**changes):
    resources = {
        "device": "cuda", "gpu_name": "NVIDIA B200", "gpu_total_bytes": 192 * 2**30,
        "peak_gpu_allocated_bytes": 40 * 2**30, "peak_gpu_reserved_bytes": 48 * 2**30,
        "peak_rss_bytes": 2**30, "device_workers": 1, "fold_wall_seconds": 3600.0,
    }
    report = {"kumo_max_columns_override": 557, "resources": resources}
    for key, value in changes.items():
        (resources if key in resources else report)[key] = value
    return report


@pytest.mark.parametrize(
    "change",
    [
        {"kumo_max_columns_override": None},  # SelectColumns(500) was not widened
        {"kumo_max_columns_override": 500},
        {"gpu_name": "NVIDIA A100-SXM4-80GB"},  # not a real B200
        {"gpu_name": None},
        {"device_workers": 2},  # shared-device time is not a burden measure
        {"peak_gpu_reserved_bytes": None},
        {"peak_gpu_reserved_bytes": 193 * 2**30},
        {"fold_wall_seconds": None},
        {"fold_wall_seconds": 40_000.0},  # five folds would not fit the 72 h bucket
    ],
)
def test_kumo_override_profile_requires_real_b200_memory_and_time(change):
    planned = {"dataset_id": "REG_TEXT_ROTTEN_TOMATOES", "predictor_columns": 557}
    q.verify_kumo_override_profile(_b200_report(), planned)
    with pytest.raises(ValueError, match="kumo_max_columns_override_b200_profile"):
        q.verify_kumo_override_profile(_b200_report(**change), planned)


def test_signed_open_provider_scope_excludes_paid_arms_and_binds_selectors(reports, tmp_path, monkeypatch):
    from src.evaluation import acceptance

    arm, bodies = reports
    arm["provider"] = "strands"
    bodies[0]["decision_artifacts"][0]["provider"] = "strands"
    scope = {"qualified_decision_providers": ["strands"]}
    bodies[0].update(**scope, decision_input_identity="open-settings")
    monkeypatch.setattr(q, "load_yaml", lambda name: {"candidate_extractors": ["jev", "strands"], "encoders": ["encoder"]})
    monkeypatch.setattr(acceptance, "decision_input_identity", lambda providers: "open-settings")
    signed = write_approval(tmp_path, bodies, **scope)
    q.benchmark_evidence(signed, arm)
    q.benchmark_evidence(signed, arm | {"provider": None})
    with pytest.raises(ValueError, match="outside the signed"):
        q.benchmark_evidence(signed, arm | {"provider": "jev"})
    with pytest.raises(ValueError, match="signed 0B decision providers"):
        q.benchmark_evidence(write_approval(tmp_path, bodies), arm)
    signed = write_approval(tmp_path, bodies, **scope)
    monkeypatch.setattr(acceptance, "decision_input_identity", lambda providers: "changed")
    with pytest.raises(ValueError, match="selectors and artifact pins"):
        q.benchmark_evidence(signed, arm)


def _versioned_release(reports, monkeypatch, cohort, provider):
    from src.evaluation import acceptance, design

    main = {
        "candidate_extractors": ["jev", "strands"], "initial_decision_providers": ["strands"],
        "decision_extensions": {"with_jev": {"parent": "open_weight", "add": ["jev"]}},
        "encoders": ["encoder"],
    }
    monkeypatch.setattr(q, "load_yaml", lambda name: main)
    monkeypatch.setattr(acceptance, "decision_input_identity", lambda providers: "/".join(providers))
    members = list(design.decision_providers(cohort, main=main))
    versions = design.decision_versions(members)
    arm, bodies = reports
    arm["provider"] = provider
    preflight = bodies[0]
    preflight.update(decision_cohort=cohort, qualified_decision_providers=members,
                     decision_input_identity="/".join(members), decision_model_versions=versions)
    example = preflight["decision_artifacts"][0]
    preflight["decision_artifacts"] = [
        example | {key: record[key] for key in ("provider", "model", "version")} for record in versions
    ]
    return main, arm, bodies, {"decision_cohort": cohort, "qualified_decision_providers": members,
                               "decision_model_versions": versions}


@pytest.mark.parametrize(("cohort", "alias", "provider"), [
    ("with_jev", "all", "strands"),
    ("with_jev_added", "paid", "jev"),
])
def test_versioned_0b_accepts_only_the_tested_members_and_versions(
    reports, tmp_path, monkeypatch, cohort, alias, provider,
):
    from src.evaluation import design

    _, arm, bodies, signed = _versioned_release(reports, monkeypatch, cohort, provider)
    signed["decision_cohort"] = alias
    preflight = bodies[0]
    approval = write_approval(tmp_path, bodies, **signed)
    q.benchmark_evidence(approval, arm, required_cohort=alias)
    if design.incremental_cohort(cohort):
        with pytest.raises(ValueError, match="outside the signed"):
            q.benchmark_evidence(approval, arm | {"provider": None})
    else:
        q.benchmark_evidence(approval, arm | {"provider": None})
    with pytest.raises(ValueError, match="requested cohort"):
        q.benchmark_evidence(approval, arm, required_cohort="open_weight")
    with pytest.raises(ValueError, match="signed model versions"):
        q.benchmark_evidence(write_approval(tmp_path, bodies, **(signed | {"decision_model_versions": []})), arm)

    if provider == "strands":
        observed = next(item for item in preflight["decision_artifacts"] if item["provider"] == provider)
        observed["runtime_revision"] = "not-the-registered-runtime"
        with pytest.raises(ValueError, match="registered runtime revision"):
            q.feature_evidence(preflight)
        del observed["runtime_revision"]
    preflight["decision_artifacts"][0]["version"] = "another-release"
    with pytest.raises(ValueError, match="artifacts differ from the registered"):
        q.benchmark_evidence(write_approval(tmp_path, bodies, **signed), arm)
    preflight["decision_model_versions"] = []
    with pytest.raises(ValueError, match="exact registered model versions"):
        q.feature_evidence(preflight)
    del preflight["decision_cohort"], preflight["decision_model_versions"]
    with pytest.raises(ValueError, match="name the cohort"):
        q.feature_evidence(preflight)


@pytest.mark.parametrize(("alias", "frozen", "provider"), [
    ("all", "with_jev", "strands"),
    ("paid", "with_jev_added", "jev"),
])
def test_a_growing_alias_still_qualifies_exactly_its_original_release(
    reports, tmp_path, monkeypatch, alias, frozen, provider,
):
    from src.evaluation import campaign

    main, arm, bodies, signed = _versioned_release(reports, monkeypatch, alias, provider)
    accepted = write_approval(tmp_path, bodies, **signed)
    members = signed["qualified_decision_providers"]
    # The next registration is placed first. Reconstruct the original ordering for
    # tied extremes in the already approved profiles.
    main["candidate_extractors"] = ["new_model", "strands", "jev"]
    main["decision_extensions"]["next_release"] = {"parent": "with_jev", "add": ["new_model"]}
    def benchmark_plan(*, main, cohort):
        assert cohort == alias and main["candidate_extractors"][:len(members)] == members
        return [arm, arm | {"provider": "new_model"}]
    def profile_plan(*, arms):
        assert arms == [arm]
        return [arm]
    monkeypatch.setattr(campaign, "benchmark_plan", benchmark_plan)
    monkeypatch.setattr(q, "profile_plan", profile_plan)

    q.benchmark_evidence(accepted, arm, required_cohort=frozen)
    with pytest.raises(ValueError, match="requested cohort"):
        q.benchmark_evidence(accepted, arm, required_cohort=alias)
    with pytest.raises(ValueError, match="outside the signed"):
        q.benchmark_evidence(accepted, arm | {"provider": "new_model"})
    if alias == "paid":
        with pytest.raises(ValueError, match="outside the signed"):
            q.benchmark_evidence(accepted, arm | {"provider": None})
    bodies[0]["decision_artifacts"][0]["version"] = "unapproved"
    with pytest.raises(ValueError, match="artifacts differ from the registered"):
        q.benchmark_evidence(write_approval(tmp_path, bodies, **signed), arm)
    main["decision_extensions"] = {}
    with pytest.raises(ValueError, match="complete ordered decision providers"):
        q.benchmark_evidence(accepted, arm)


def test_provider_receipts_precede_cohort_defaults_and_baselines_use_full_releases(monkeypatch):
    from src.evaluation import design

    monkeypatch.setattr(design, "decision_scopes", lambda: {
        "open_weight": ("strands",), "all": ("jev", "strands", "new"),
        "paid": ("jev", "new"), "with_jev": ("strands", "jev"), "with_jev_added": ("jev",),
        "with_new": ("strands", "jev", "new"), "with_new_added": ("new",),
    })
    mapping = {"open_weight": "base", "paid": "all_paid", "with_jev_added": "new_jev",
               "with_new": "full", "providers": {"new": "new_only"}}
    assert q.acceptance_for(mapping, "strands") == q.acceptance_for(mapping, None) == "base"
    assert q.acceptance_for(mapping, "jev") == "new_jev"
    assert q.acceptance_for(mapping, "new") == "new_only"
    assert q.acceptance_for({"with_new": "full"}, None) == "full"
    assert q.acceptance_for({"paid": "all_paid", "with_jev_added": "new_jev"}, None) is None
    assert q.acceptance_for(mapping, "unregistered") is None


def test_later_phase_coverage_and_independent_c_release():
    from src.evaluation.campaign import benchmark_plan

    arms = benchmark_plan(cohort="open_weight")
    core = q.profile_plan([a for a in arms if a["phase"] not in q.GROUP_C_PHASES])
    probes = {(a["encoder"], a.get("probe_view")) for a in core if a["phase"] == "experiment_3" and a.get("probe_view")}
    assert len(probes) == 6 and all(view in {"per_column", "joint"} for _, view in probes)
    b4 = [a for a in core if a["phase"] == "data_efficiency"]
    assert {a["training_size"] for a in b4} == {128, 16384}
    assert {q.profile_fold_number(a) for a in b4} == {0, 1}
    assert {a["model"] for a in b4} == set(MAIN_MODELS)
    assert all(a["phase"] not in q.GROUP_C_PHASES and a.get("provider") != "jev" for a in core)
    for phase in q.GROUP_C_PHASES:
        later = q.profile_plan([a for a in arms if a["phase"] == phase])
        assert {a["model"] for a in later} == set(MAIN_MODELS)
        assert {"BIN", "MUL", "REG"} == {a["dataset_id"].split("_")[0] for a in later}
        if phase == "representations":
            assert len({(a["hidden_view"], a["hidden_position"], a["hidden_layer"]) for a in later}) == 12
        else:
            assert {a["adaptation_cell"] for a in later} == {"head_only", "lora_head", "lora_only"}


def test_same_width_later_arms_bind_their_actual_fold_recipe():
    runtime = {"container_id": "tested"}
    base = dict(dataset_id="data", model="linear", feature_set="fixture")
    metadata = dict(dataset_id="data", model_id="linear", feature_set="fixture",
                    feature_artifact_ids=[], estimator_settings={"execution_runtime": runtime})
    for expected, recipe, changed in [
        (dict(encoder="e", probe_view="joint"), dict(embedding={"encoder_id": "e", "probe_view": "joint"},
                                                     probe={"view": "joint", "encoder_id": "e"}), dict(probe_view="per_column")),
        (dict(dropped_text_column=1), dict(dropped_text_column=1), dict(dropped_text_column=0)),
        (dict(training_size=128, draw=0), dict(subsample={"training_size": 128, "draw": 0}), dict(draw=1)),
        (dict(adaptation_cell="head_only"), dict(adaptation={"cell": "head_only"}), dict(adaptation_cell="lora_head")),
        (dict(hidden_view="joint", hidden_position="state_end", hidden_layer=12),
         dict(hidden={"view": "joint", "position": "state_end", "layer": 12}), dict(hidden_layer=24)),
    ]:
        q.verify_arm_recipe(metadata | {"representation_recipe": recipe}, base | expected, [], runtime)
        with pytest.raises(ValueError, match="planned fold recipe"):
            q.verify_arm_recipe(metadata | {"representation_recipe": recipe}, base | expected | changed, [], runtime)


def test_profile_resources_reject_nan_and_absent_measurements():
    for value in (float("inf"), float("nan"), 0, -1):
        with pytest.raises(ValueError):
            q.verify_profile_resources(_b200_report(fold_wall_seconds=value), True)
        with pytest.raises(ValueError):
            q.verify_kumo_override_profile(_b200_report(fold_wall_seconds=value), {"predictor_columns": 557, "dataset_id": "d"})
    with pytest.raises(ValueError):
        q.verify_profile_resources(_b200_report(peak_gpu_reserved_bytes=0), True)


def test_interrupted_fixture_resumes_and_its_durable_predictions_reproduce(isolated_output, monkeypatch):
    from sklearn.dummy import DummyClassifier

    from src.cluster import predictor_contracts as pc
    from src.utils import paths

    data = next(iter(pc.fixtures()))
    case = next(pc.feature_cases(data, "linear"))
    monkeypatch.setattr(pc, "fixtures", lambda: iter([data]))
    monkeypatch.setattr(pc, "feature_cases", lambda *args: iter([case]))
    monkeypatch.setattr(pc, "estimator_factory", lambda *args, **kwargs: DummyClassifier)
    monkeypatch.setattr("src.cluster.containers.active_identity", lambda: {"container_id": "test"})
    monkeypatch.setattr("src.cluster.evidence.current_identity", lambda: {"source": "test"})
    monkeypatch.setenv("SLURM_JOB_ID", "local-fixture")
    monkeypatch.setattr(paths, "durable_project_override", paths.staging_override)
    report = pc.run("linear", execute=True)
    assert report["fixtures"][0]["partial_fold_resume_verified"]
    pc.verify_published_fixtures(report)
    genuine = report["fixtures"][0]["interrupted_record_id"]
    report["fixtures"][0]["interrupted_record_id"] = report["fixtures"][0]["records"][2].rsplit("#", 1)[1]
    with pytest.raises(ValueError, match="reused an interrupted"):
        pc.verify_published_fixtures(report)
    report["fixtures"][0]["interrupted_record_id"] = genuine
    report["fixtures"][0]["records"][0] = report["fixtures"][0]["records"][1]
    with pytest.raises(ValueError, match="duplicate or unexpected fold/role"):
        pc.verify_published_fixtures(report)


def test_real_b4_profile_binds_sample_size_parent_and_whole_test_fold(isolated_output, monkeypatch):
    import json

    from sklearn.dummy import DummyClassifier

    from src.cluster.predictor_contracts import feature_cases, fixtures
    from src.evaluation.data_efficiency import subsample_specification
    from src.evaluation.protocol import evaluate_cv, shared_splits
    from src.utils import paths

    data = next(iter(fixtures()))
    _, matrix, _ = next(feature_cases(data, "linear"))
    matrix.attrs["decision_artifact_checksums"] = {}
    parent = shared_splits(data)
    labels = data.frame[data.metadata.target].to_numpy()
    sample = subsample_specification(parent, labels, 128, 0, seed=17, classes=list(data.metadata.classes))
    arm = dict(phase="data_efficiency", model="linear", feature_set="non_text", dataset_id=data.metadata.dataset_id,
               training_size=128, draw=0, predictor_columns=matrix.shape[1])
    runtime = {"container_id": "test"}
    records = evaluate_cv(data, matrix, DummyClassifier, specification=sample,
                          model_id="linear", checkpoint_id="test", feature_set="non_text", phase="experiment_0",
                          allow_model_runs=True, fold_numbers=[1], binary_threshold_policy="fixed_0_5",
                          estimator_settings={"execution_runtime": runtime},
                          representation_recipe={"subsample": sample["subsample"]})
    q.save_profile(arm, data, matrix, runtime, records, specification=sample)
    file, = paths.phase_path("experiment_0", "manifests", "after_extraction").glob("profile_*.json")
    report = json.loads(file.read_text(encoding="utf-8"))
    assert (report["training_rows"], report["outer_fold"], report["parent_split_id"]) == (128, 1, parent["split_id"])
    assert [(r["role"], r["rows"]) for r in report["roles"]] == [("outer_test", len(sample["folds"][1]["test"]))]
    monkeypatch.setattr(paths, "durable_project_override", paths.staging_override)
    q.verify_published_profile(report, dataset=data, specification=sample)
    with pytest.raises(ValueError, match="resource use differs"):
        q.verify_published_profile(report | {"job_id": "later-allocation"}, dataset=data, specification=sample)
    with pytest.raises(ValueError, match="independently reconstructed split"):
        q.verify_published_profile(report, dataset=data, specification=parent)


def test_both_prepared_probe_layouts_are_read_for_the_exact_five_folds(isolated_output):
    import sqlite3

    import numpy as np

    from src.cluster.predictor_contracts import fixtures
    from src.evaluation import label_probe as probe
    from src.evaluation.protocol import shared_splits
    from src.utils.serialization import text_value

    data = next(iter(fixtures()))
    seed = q.load_yaml("experiment_3/tar")["random_seed"]
    rows = pd.Index([f"csv:{i}" for i in range(len(data.frame))])
    masks = [data.frame[c].map(text_value).notna().to_numpy() for c in data.metadata.text_columns]
    groups = [[f"frozen_e5__text_{i:03d}__000"] for i in range(len(masks))]
    encoder = {"encoder_id": "fixture", "dimensions": 1}
    path = probe.cache_path(data.metadata.dataset_id, "fixture")
    keys = []
    for fold in shared_splits(data, seed=seed)["folds"]:
        train = fold["train"]
        for view in probe.VIEWS:
            blocks = ([("all_text", np.any(masks, axis=0))] if view == "joint" else
                      [(f"text_{i:03d}", m) for i,m in enumerate(masks)])
            table = pd.DataFrame({f"probe__{name}__p_00": np.where(mask, .5, np.nan)
                                  for name,mask in blocks}, index=rows)
            key = probe.identity(rows[train], data.frame[data.metadata.target].iloc[train], groups,
                                 view, data.metadata.task_type, seed, ["asset"])
            probe.write(path, key, table, [])
            keys.append(key)
    assert probe.verify_prepared(data, encoder, "asset") == keys
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM features WHERE key=?", (keys[-1],))
    with pytest.raises(ValueError, match="Missing prepared"):
        probe.verify_prepared(data, encoder, "asset")


def test_kumo_override_width_is_read_from_recorded_fit_warnings():
    import json

    class Record:
        def __init__(self, messages):
            self.body = {"warnings": [{"category": "UserWarning", "message": m} for m in messages]}

        def read_text(self, encoding=None):
            return json.dumps(self.body)

    records = [Record(["other", "kumo_max_columns_override=557: widened"]), Record([])]
    assert q.kumo_overrides(records) == [557]
    assert q.kumo_overrides([Record(["other"])]) == []
    # Reopening a GPU fold on a CPU-only review node must use the original physical GPU.
    historical = {"timing": {"device": "cuda", "measured_wall_seconds": 7},
                  "resources": {"gpu_name": "NVIDIA H100", "gpu_total_bytes": 80 * 2**30}}
    assert q.profile_resources(historical)["gpu_name"] == "NVIDIA H100"
    assert q.profile_resources({"timing": {"device": "cpu", "measured_wall_seconds": 2.0},
                                "resources": {"peak_rss_bytes": 5, "device_workers": 1}}) == {
        "device": "cpu", "gpu_name": None, "gpu_total_bytes": None,
        "peak_gpu_allocated_bytes": None, "peak_gpu_reserved_bytes": None,
        "peak_rss_bytes": 5, "device_workers": 1, "fold_wall_seconds": 2.0,
    }


def test_0b_gate_checks_the_declared_kumo_override_profile(tmp_path, reports, monkeypatch):
    arm, bodies = deepcopy(reports)
    arm.update(model="kumo_tabular", predictor_columns=557, width_status="within_known_width_limit")
    bodies[-1].update(arm=arm, predictor_columns=557, **_b200_report())
    monkeypatch.setattr(q, "profile_plan", lambda **kwargs: [arm])
    monkeypatch.setattr(q, "load_catalog",
                        lambda: {"datasets": [{"id": "data", "paper_summary": {"rows": 10}}]})
    q.benchmark_evidence(write_approval(tmp_path / "pass", bodies), arm)
    bodies[-1]["resources"]["gpu_name"] = "NVIDIA H100"
    with pytest.raises(ValueError, match="kumo_max_columns_override_b200_profile"):
        q.benchmark_evidence(write_approval(tmp_path / "fail", bodies), arm)
