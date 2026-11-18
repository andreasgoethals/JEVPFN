import hashlib

import pytest

from src.cluster.release_recovery import filtered_reader


def test_only_reviewed_same_cluster_job_is_excluded(tmp_path):
    script = tmp_path / "source with spaces" / "check.slurm"
    script.parent.mkdir()
    script.write_text("reviewed isolated qualification")
    proof = dict(release_runtime="liquid_lfm3", independent_jobs=[dict(
        cluster="mindwell", job_id="12", script=str(script),
        sha256=hashlib.sha256(script.read_bytes()).hexdigest())])

    def read(cmd):
        if cmd[0] == "squeue":
            return "12|jevpfn-native|/data/JEVPFN\n13|jevpfn-unknown|/data/JEVPFN\n"
        return f"JobId=12 Command={script} WorkDir=/data/JEVPFN"

    checked = filtered_reader(read, proof, "liquid_lfm3")
    assert checked(["squeue", "--clusters=mindwell"]).startswith("13|")
    assert checked(["squeue", "--clusters=wice"]).startswith("12|")
    script.write_text("changed")
    with pytest.raises(ValueError, match="script changed"):
        checked(["squeue", "--clusters=mindwell"])
    with pytest.raises(ValueError, match="different workspace"):
        filtered_reader(read, proof, "intern")


def test_reviewed_array_members_require_explicit_array_proof(tmp_path):
    script = tmp_path / "array.slurm"
    script.write_text("reviewed isolated array")
    record = dict(cluster="mindwell", job_id="12", script=str(script),
                  sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    proof = dict(release_runtime="x", independent_jobs=[record])

    def read(cmd):
        if cmd[0] == "squeue":
            return "12_[0-9%10]|jevpfn|/data\n12_4|jevpfn|/data\n"
        assert cmd[-2] in {"12", "12_4"}
        return f"Command={script} WorkDir=/data"

    assert filtered_reader(read, proof, "x")(["squeue", "--clusters=mindwell"])
    record["array"] = True
    assert filtered_reader(read, proof, "x")(["squeue", "--clusters=mindwell"]) == ""


def test_changed_scheduler_command_is_rejected(tmp_path):
    proof = dict(release_runtime="x", independent_jobs=[dict(
        cluster="mindwell", job_id="1", script=str(tmp_path / "expected"), sha256="x")])
    def read(cmd):
        return "1|jevpfn|/data/JEVPFN" if cmd[0] == "squeue" else "Command=/other WorkDir=/data"
    with pytest.raises(ValueError, match="command changed"):
        filtered_reader(read, proof, "x")(["squeue", "--clusters=mindwell"])
