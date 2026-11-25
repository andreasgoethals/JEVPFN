"""The compute ledger reports compute time only, from Slurm accounting."""

from src.cluster.compute_ledger import parse_sacct, stage_of, summarize

SAMPLE = "\n".join([
    "11781957|jevpfn-features|COMPLETED|39384|billing=14,cpu=14,gres/gpu=1,mem=112000M,node=1|gpu_b200|"
    "2026-11-10T03:55:13|2026-11-10T14:51:37|mindwell|sbatch --parsable --clusters=mindwell "
    "scripts/slurm/feature_creation.slurm decision3 decision3 /gpfs1/scratch/x MUL_TEXT_WINE_REVIEW",
    "11781951|jevpfn-features|FAILED|22848|cpu=14,gres/gpu=1|gpu_b200|a|b|mindwell|"
    "sbatch scripts/slurm/feature_creation.slurm decision3 decision3 /x MUL_TEXT_US_ACCIDENTS",
    "11784791|jevpfn-scratch-release|COMPLETED|3600|cpu=2,mem=16000M|batch_graniterapids|a|b|mindwell|"
    "sbatch scripts/slurm/experiment_0_scratch_release.slurm kev --apply",
    "11784212|creditpfn-pd-train|COMPLETED|5400|cpu=8,gres/gpu=1|gpu_b200|a|b|mindwell|sbatch x",
    "11789999|jevpfn-benchmark|RUNNING|100|cpu=8,gres/gpu=1|gpu_b200|a|Unknown|mindwell|sbatch y",
    "11785143|jevpfn-feature-recovery-qa|CANCELLED by 1|60|cpu=3|batch|a|b|mindwell|sbatch z|with|pipes",
])


def test_only_terminal_jevpfn_jobs_are_counted_with_allocated_resources():
    rows = parse_sacct(SAMPLE)
    assert [r["job_id"] for r in rows] == ["11781957", "11781951", "11784791", "11785143"]
    first = rows[0]
    assert first["decision_model"] == "decision3" and first["gpus"] == 1 and first["cpus"] == 14
    assert [r["dataset_id"] for r in rows] == ["MUL_TEXT_WINE_REVIEW", "MUL_TEXT_US_ACCIDENTS", None, None]
    assert abs(first["gpu_hours"] - 39384 / 3600) < 1e-9
    assert rows[2]["gpus"] == 0 and rows[2]["cpu_core_hours"] == 2.0
    assert rows[3]["state"] == "CANCELLED"


def test_feature_job_without_dataset_selects_the_full_workload():
    line = SAMPLE.splitlines()[0].replace(" /gpfs1/scratch/x MUL_TEXT_WINE_REVIEW", "")
    row, = parse_sacct(line)
    assert row["decision_model"] == "decision3" and row["dataset_id"] is None


def test_summary_separates_successful_from_all_attempts():
    summary = {(s["stage"], s["decision_model"]): s for s in summarize(parse_sacct(SAMPLE))}
    decision3 = summary[("decision_feature_creation", "decision3")]
    assert decision3["jobs"] == 2 and decision3["successful_jobs"] == 1
    assert abs(decision3["gpu_hours"] - (39384 + 22848) / 3600) < 1e-9
    assert abs(decision3["successful_gpu_hours"] - 39384 / 3600) < 1e-9


def test_stage_mapping_prefers_specific_prefixes():
    assert stage_of("jevpfn-exp0-stage-gpfs") == "decision_feature_creation"
    assert stage_of("jevpfn-exp0-contracts") == "qualification"
    assert stage_of("jevpfn-tar-pca") == "tar"
    assert stage_of("creditpfn-x") is None


def test_c0_compute_is_separate_from_c2_training():
    from src.evaluation.study_audit import ledger_tables

    def accounting_line(job_id, name, arguments):
        return (f"{job_id}|{name}|COMPLETED|3600|cpu=4,gres/gpu=1|gpu_b200|start|end|mindwell|"
                f"sbatch scripts/slurm/c2_adaptation.slurm {arguments}")

    jobs = parse_sacct("\n".join([
        accounting_line(1, "jevpfn-c0-pilot", "pilot regression head_only"),
        accounting_line(2, "jevpfn-c2-adaptation", "pilot regression lora_only"),
        accounting_line(3, "jevpfn-c2-adaptation", "REG_TEXT_BOOK_PRICE 1 head_only"),
    ]))
    _, costs = ledger_tables({"jobs": jobs})
    assert costs.set_index("study").gpu_hours.to_dict() == {"C0 Strands qualification": 2, "C2 adaptation": 1}
