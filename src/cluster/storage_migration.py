"""Reviewed PROJECT cutover operations; no implicit submission or destructive cleanup."""

import argparse
import copy
import json
import os
import shutil
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from pathlib import Path

from src.utils import paths
from src.utils.serialization import digest, file_sha256, write_json


def records():
    return paths.manifests_dir("feature_creation") / "storage_migration"


def load_copy():
    p = records() / "copy_verified.json"
    r = json.loads(p.read_text())
    if r.get("status") != "copied_and_verified_requires_relocation_and_runtime_acceptance":
        raise ValueError("A successful consistent copy receipt is required")
    return p, r, Path(r["source"]), Path(r["destination"])


def relocate():
    """Only verified host wrappers/links change; original image bytes remain pinned."""
    from src.cluster.containers import MARKER
    from src.cluster.containers import relocate as move_image
    from src.cluster.control_runtime import host_view
    from src.cluster.project_migration import rewrite_links

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Relocation verification belongs in a CPU allocation")
    receipt, copied, source, destination = load_copy()
    private_modes = []
    for key in copied["source_inventory"]["files"]:
        old, new = source / key, destination / key
        control = source / "containers/control-v1"
        if old.is_relative_to(control):
            # The running Tykky image masks its host bin/ with installed
            # executables. Permission checks need the explicit read-only bind.
            old = host_view(control) / old.relative_to(control)
        mode = old.stat().st_mode & 0o777
        if not mode & 0o077:
            os.chmod(new, mode)
            if new.stat().st_mode & 0o077:
                raise ValueError("A private source file acquired group/public access")
            private_modes.append(key)
    results = []
    for marker in sorted((source / "containers").rglob(MARKER)):
        old = marker.parent
        new = destination / old.relative_to(source)
        view = host_view(old)
        result = move_image(old, new, **({"source_view": view} if view != old else {}))
        image_files = {k: v for k, v in result["files"].items() if k.endswith((".sqfs", ".sif"))}
        original = json.loads(marker.read_text())
        if image_files != {k: v for k, v in original["files"].items() if k.endswith((".sqfs", ".sif"))}:
            raise ValueError("Relocation changed an image")
        results.append({"path": str(new), "old_id": original["container_id"],
                        "new_id": result["container_id"], "images": image_files})
        print(json.dumps({"event": "container_relocated", **results[-1]}), flush=True)
    rewritten = rewrite_links(source, destination)
    # Preserve the historical cutover receipt verbatim. A derived pointer records
    # relocation lineage; CPU acceptance is still a separate real-node check.
    active = destination / "containers/control.active.json"
    previous = destination / "containers/control.before_project_migration.json"
    if not previous.exists():
        previous.write_bytes(active.read_bytes())
    original = json.loads(previous.read_text())
    original["container_id"] = json.loads((destination / "containers/control-v1" / MARKER).read_text())["container_id"]
    original["project_migration"] = {"previous": str(previous), "sha256": file_sha256(previous)}
    write_json(active, original)
    # Populate the signature cache from the verified copy manifest, avoiding a
    # second 269-GiB weight read on the first GPU allocation.
    from src.jev.checkpoint_hashes import signature

    index = destination / "checkpoints/checkpoint_hashes.sqlite3"
    with closing(sqlite3.connect(index)) as db, db:
        for key, item in copied["source_inventory"]["files"].items():
            if key.startswith("checkpoints/") and not key.endswith("checkpoint_hashes.sqlite3"):
                p = destination / key
                db.execute("INSERT OR REPLACE INTO hashes VALUES (?,?,?)", (str(p), signature(p), item["sha256"]))
        for key in copied["source_inventory"]["links"]:
            p = destination / key
            if not key.startswith("checkpoints/") or not p.is_file():
                continue
            target = p.resolve().relative_to(destination).as_posix()
            if target not in copied["source_inventory"]["files"]:
                raise ValueError("Checkpoint link has no verified target")
            db.execute("INSERT OR REPLACE INTO hashes VALUES (?,?,?)",
                       (str(p), signature(p), copied["source_inventory"]["files"][target]["sha256"]))
    result = {"status": "relocated_requires_cpu_gpu_acceptance", "containers": results,
              "owner_only_modes_preserved": private_modes,
              "rewritten_checkpoint_links": rewritten, "copy_receipt": str(receipt),
              "copy_receipt_sha256": file_sha256(receipt), "source": str(source),
              "destination": str(destination), "job_id": os.environ["SLURM_JOB_ID"]}
    write_json(records() / "relocated.json", result)
    write_json(destination / "_JEVPFN_PROJECT_MIGRATION.json", result)
    from src.jev.catalog import SELF_HOSTED

    validation = Path(os.environ["VSC_SCRATCH_GPFS1"]) / "jevpfn-migration-validation/jevpfn-work"
    for provider in SELF_HOSTED:
        # Dedicated empty diagnostic stores, never mutate historical workspaces
        # or copy production response histories merely to probe relocation.
        root = validation / provider / "JEVPFN"
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "_JEVPFN_WORKSPACE.json", {
            "runtime": provider, "durable_root": str(destination),
            "runtime_profile": "blackwell-cu130", "status": "staged_requires_b200_contracts",
            "shared_immutable_root": str(destination), "purpose": "migration_diagnostics", "files": {}})
    return result


def same_science(old_settings, new_settings):
    """Host seals may relocate; model, tokenizer, code, package and payload policy may not."""
    a, b = copy.deepcopy(old_settings), copy.deepcopy(new_settings)
    for item in (a, b):
        item.get("request_configuration", {}).pop("execution_container", None)
    return a == b


def publish_workspaces():
    """Union every old native recovery store into the new durable authority."""
    from src.cluster.maintenance import require_idle_vsc
    from src.cluster.scratch_stage import MARKER, _publish, workspace_lease

    require_idle_vsc(allow_current=True)
    _, _, source, destination = load_copy()
    if paths.checkpoints_dir().parent != destination:
        raise ValueError("Publication must select the verified new durable authority")
    parent = Path(os.environ["VSC_SCRATCH_GPFS1"]) / "jevpfn-work"
    reports = []
    for marker in sorted(parent.glob("*/JEVPFN/" + MARKER)):
        work = marker.parent
        with workspace_lease(work):
            saved = json.loads(marker.read_text())
            if saved.get("durable_root") != str(source):
                raise ValueError("Unexpected native workspace authority; preserve for review")
            result = _publish(destination, work)
            reports.append({"runtime": saved["runtime"], "working_root": str(work),
                            "old_marker_sha256": file_sha256(marker), "publication": result})
            write_json(records() / "workspace_publication_progress.json", {"workspaces": reports})
    result = {"status": "all_old_native_stores_published", "workspaces": reports,
              "source": str(source), "destination": str(destination),
              "job_id": os.environ["SLURM_JOB_ID"]}
    write_json(records() / "workspaces_published.json", result)
    return result


def cpu_accept():
    from src.cluster.containers import MARKER, probe

    job = os.environ["SLURM_JOB_ID"]
    if os.environ.get("SLURM_CLUSTER_NAME") != "mindwell":
        raise ValueError("GPFS cutover CPU validation must run natively on Mindwell")
    before = paths.manifests_dir("experiment_0") / "before_extraction"
    control = before / ("control_" + job + ".json")
    if json.loads(control.read_text()).get("status") != "control_integration_passed":
        raise ValueError("CPU image launch/integration failed")
    storage = [p for p in (before / "storage_contracts").glob("*.json")
               if (r := json.loads(p.read_text())).get("job_id") == job
               and r.get("status") == "storage_contracts_passed_requires_quota_and_backup_review"
               and r.get("storage_backend") == "gpfs" and r.get("snapshot_restore") is True]
    if len(storage) != 1:
        raise ValueError("Native two-node locking/snapshot validation is missing")
    runtimes = []
    for item in json.loads((records() / "relocated.json").read_text())["containers"]:
        folder = Path(item["path"])
        sealed = json.loads((folder / MARKER).read_text())
        actual = probe(folder / "bin/python")
        installed = {k.lower().replace("_", "-"): v for k, v in actual["packages"].items()}
        expected = {k.lower().replace("_", "-"): v for k, v in sealed["runtime"]["packages"].items()}
        if (installed != expected
                or actual.get("torch_cuda") != sealed["runtime"].get("torch_cuda")):
            raise ValueError("Relocated interpreter changed installed dependencies: " + str(folder))
        runtimes.append({"path": str(folder), "container_id": sealed["container_id"], "runtime": actual})
        print(json.dumps({"event": "relocated_interpreter_passed", "path": str(folder)}), flush=True)
    result = {"status": "native_cpu_locking_sqlite_tests_passed", "job_id": job,
              "relocated_interpreters": runtimes,
              "durable_root": str(paths.durable_project_override()),
              "evidence": [{"path": str(p), "sha256": file_sha256(p)} for p in [control, *storage]]}
    write_json(records() / "cpu_accepted.json", result)
    pointer = paths.checkpoints_dir().parent / "containers/control.active.json"
    selected = json.loads(pointer.read_text())
    selected["validation_job_id"] = job
    selected["project_migration"]["cpu_acceptance"] = {
        "path": str(records() / "cpu_accepted.json"),
        "sha256": file_sha256(records() / "cpu_accepted.json")}
    write_json(pointer, selected)
    return result


def numeric_backup():
    """Compact independent-download bundle; explicitly not a response-cache replacement."""
    from src.jev.features import load_block
    from src.jev.merge import ARTIFACT_SQL

    _, _, _, destination = load_copy()
    target = destination / "output_JEVPFN/transfers/migration_numeric_features.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError("Numeric backup already exists; verify the saved bundle instead of replacing it")
    stores = []
    with tempfile.TemporaryDirectory(prefix="jevpfn-numeric-backup-", dir=destination) as temp:
        building = Path(temp) / "numeric_features.zip"
        with zipfile.ZipFile(building, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for source in sorted((destination / "data/jev_cache").glob("*.sqlite3")):
                copied = Path(temp) / source.name
                with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src:
                    if not src.execute("SELECT 1 FROM sqlite_master WHERE name='feature_artifacts'").fetchone():
                        continue
                    with closing(sqlite3.connect(copied)) as dst, dst:
                        dst.executescript(ARTIFACT_SQL)
                        for row in src.execute("SELECT * FROM feature_artifacts"):
                            dst.execute("INSERT INTO feature_artifacts VALUES (?,?,?,?)", row)
                with closing(sqlite3.connect(copied)) as db:
                    ids = [r[0] for r in db.execute("SELECT artifact_id FROM feature_artifacts")]
                production = []
                for aid in ids:
                    block = load_block(copied, aid)
                    if block.manifest.get("purpose") != "validation_only":
                        production.append({"artifact_id": aid, "identity": block.manifest["identity"],
                                           "rows": len(block.values), "columns": len(block.values.columns)})
                stores.append({"dataset": source.stem, "sha256": file_sha256(copied),
                               "all_blocks": len(ids), "production_blocks": production})
                archive.write(copied, source.name)
                copied.unlink()
            archive.writestr("manifest.json", json.dumps({"scope": "numeric artifacts and row/request links only; no raw responses or operational queues",
                                                          "stores": stores}, indent=2))
        with zipfile.ZipFile(building) as archive:
            if archive.testzip():
                raise ValueError("Numeric backup archive failed readback")
        os.replace(building, target)
    result = {"status": "numeric_backup_ready_requires_independent_download", "path": str(target),
              "sha256": file_sha256(target), "bytes": target.stat().st_size,
              "production_blocks": sum(len(s["production_blocks"]) for s in stores), "datasets": len(stores)}
    write_json(records() / "numeric_backup.json", result)
    return result


def accept(reviewed_by):
    """Derive new acceptances; retain original 0A reports and scientific identities."""
    from src.cluster.experiment_zero import scheduler_state
    from src.cluster.maintenance import require_idle_vsc
    from src.jev.catalog import SELF_HOSTED
    from src.utils.acceptance import reviewed_acceptance

    require_idle_vsc()
    if not reviewed_by:
        raise ValueError("A named migration reviewer is required")
    copied, _, source, destination = load_copy()
    relocation = records() / "relocated.json"
    cpu = records() / "cpu_accepted.json"
    publication = records() / "workspaces_published.json"
    if (json.loads(cpu.read_text()).get("status") != "native_cpu_locking_sqlite_tests_passed"
            or json.loads(publication.read_text()).get("status") != "all_old_native_stores_published"):
        raise ValueError("CPU checks and recovery-store publication must pass")
    reports = [records() / ("gpu_" + provider + ".json") for provider in SELF_HOSTED]
    bodies = [json.loads(p.read_text()) for p in reports]
    if any(r.get("status") != "relocated_gpu_and_resume_passed"
           or r.get("settings_sha256") != digest(r.get("settings", {}))
           or file_sha256(Path(r["original_acceptance"])) != r["original_acceptance_sha256"]
           for r in bodies):
        raise ValueError("Every relocated decision runtime must pass")
    jobs = {r["job_id"] for r in bodies} | {json.loads(cpu.read_text())["job_id"]}
    for job in jobs:
        if scheduler_state(job, cluster="mindwell") != ("COMPLETED", "0:0"):
            raise ValueError("Validation allocation is not confirmed complete")
    embeddings = []
    folder = paths.manifests_dir("experiment_0") / "before_extraction/embeddings"
    for p in folder.glob("*.json"):
        r = json.loads(p.read_text())
        if r.get("job_id") in jobs and r.get("status") == "embedding_contracts_passed_requires_full_dataset_review":
            embeddings.append((p, r["encoder_id"]))
    from src.utils.config import load_yaml

    if {e for _, e in embeddings} != set(load_yaml("feature_creation/embedding_models")["models"]):
        raise ValueError("All three encoders need relocated GPU contracts")
    evidence = [copied, relocation, cpu, publication, *reports, *(p for p, _ in embeddings)]
    for provider, gpu_report in zip(SELF_HOSTED, bodies, strict=True):
        old = Path(gpu_report["original_acceptance"])
        original = reviewed_acceptance(old, stage="before_extraction")
        mapped = []
        for item in original["evidence"]:
            p = Path(item["path"])
            if p.is_relative_to(source):
                p = destination / p.relative_to(source)
            if file_sha256(p) != item["sha256"]:
                raise ValueError("Historical accepted evidence did not survive migration")
            mapped.append({"path": str(p), "sha256": item["sha256"]})
        derived = {**original, "settings_sha256": gpu_report["settings_sha256"],
                   "reviewed_by": reviewed_by,
                   "migration": {"original_acceptance": str(old), "original_sha256": file_sha256(old),
                                 "meaning": "Same scientific settings, relocated host wrappers; bounded native revalidation"},
                   "evidence": mapped + [{"path": str(p), "sha256": file_sha256(p)} for p in [old, *evidence]]}
        write_json(paths.manifests_dir("experiment_0") / "storage_migration" / ("accepted_" + provider + ".json"), derived)
    result = {"status": "accepted_requires_consumer_check_and_retirement", "reviewed_by": reviewed_by,
              "source": str(source), "destination": str(destination),
              "evidence": [{"path": str(p), "sha256": file_sha256(p)} for p in evidence]}
    write_json(records() / "accepted.json", result)
    return result


def retire_old(*, native_only=False, source_only=False):
    """Owner-authorized exact retirement, only after verified copy and native acceptance."""
    from src.cluster.maintenance import reject_links, require_idle_vsc
    from src.cluster.project_migration import compare, inventory, roots, verify_sqlite
    from src.cluster.scratch_capacity import retire_publication_snapshots
    from src.cluster.scratch_stage import MARKER, _publish, workspace_lease

    require_idle_vsc(allow_current=True)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Bulk retirement must run in a CPU allocation")
    _, copied, source, destination = load_copy()
    roots(source, destination)
    accepted = json.loads((records() / "accepted.json").read_text())
    if (accepted.get("status") != "accepted_requires_consumer_check_and_retirement"
            or accepted.get("destination") != str(destination)
            or not accepted.get("evidence")
            or paths.checkpoints_dir().parent != destination):
        raise ValueError("The new authority has not been accepted")
    for item in accepted["evidence"]:
        if file_sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Migration acceptance evidence changed")
    if not source_only:
        published = json.loads((records() / "workspaces_published.json").read_text())
        retired = []
        native_parent = Path(os.environ["VSC_SCRATCH_GPFS1"]) / "jevpfn-work"
        for item in published["workspaces"]:
            work = reject_links(Path(item["working_root"]))
            if work.parent.parent != native_parent or work.name != "JEVPFN" or work.is_symlink():
                raise ValueError("Old workspace escaped its owned native leaf")
            with workspace_lease(work):
                if file_sha256(work / MARKER) != item["old_marker_sha256"]:
                    raise ValueError("Old workspace changed after publication")
                orphaned = retire_publication_snapshots(work)
                for folder in orphaned["retained"]:
                    old = Path(folder)
                    backup = destination / "migration_recovery" / item["runtime"] / old.name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    if backup.exists():
                        raise ValueError("Recovery archive already exists; inspect before replacing")
                    shutil.copytree(old, backup, symlinks=True)
                    compare(inventory(old), inventory(backup), old, backup)
                    verify_sqlite(backup)
                # The supported cache union validates every response, origin, feature
                # blob and recovery ledger. Keep the exact old marker as provenance.
                write_json(records() / "retired_workspace_markers" / (item["runtime"] + ".json"),
                           json.loads((work / MARKER).read_text()))
                if any(p.is_mount() for p in work.rglob("*") if p.is_dir() and not p.is_symlink()):
                    raise ValueError("A native workspace contains a mount")
                retired.append({"runtime": item["runtime"], "path": str(work), "orphan_review": orphaned})
            shutil.rmtree(work)
            if not any(work.parent.iterdir()):
                work.parent.rmdir()
        validation = Path(os.environ["VSC_SCRATCH_GPFS1"]) / "jevpfn-migration-validation"
        for marker in sorted(validation.glob("jevpfn-work/*/JEVPFN/" + MARKER)):
            work = reject_links(marker.parent)
            with workspace_lease(work):
                saved = json.loads(marker.read_text())
                if saved.get("purpose") != "migration_diagnostics" or saved.get("durable_root") != str(destination):
                    raise ValueError("Unrecognized validation workspace; preserve it")
                _publish(destination, work)
                shutil.rmtree(work)
            if not any(work.parent.iterdir()):
                work.parent.rmdir()
        # Only empty parents may be removed; an unexpected file/cache is preserved.
        for p in (validation / "jevpfn-work", validation):
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
        native_result = {"status": "native_published_copies_retired", "workspaces": retired,
                         "job_id": os.environ["SLURM_JOB_ID"]}
        write_json(records() / "native_retired.json", native_result)
        if native_only:
            return native_result
    else:
        native_result = json.loads((records() / "native_retired.json").read_text())
        if native_result.get("status") != "native_published_copies_retired":
            raise ValueError("Native recovery publication/retirement must finish first")
        retired = native_result["workspaces"]
    # Re-read the protected old authority immediately before destructive action.
    current = inventory(source, workers=8)
    compare(copied["source_inventory"], current, source, source)
    require_idle_vsc(allow_current=True)  # Recheck after the potentially long hash pass.
    if source.is_mount() or source.stat().st_uid != os.getuid():
        raise ValueError("Old project root ownership/mount boundary changed")
    # Host checkpoint links were rewritten. Some historical wrapped images mount
    # their original PROJECT prefix *inside* the image, without a host dependency.
    # Permit only the exact image-executable links from an intact relocation seal.
    from src.cluster.containers import MARKER
    from src.cluster.containers import files as container_files

    internal_links = set()
    for marker in (destination / "containers").rglob(MARKER):
        folder = marker.parent
        seal = json.loads(marker.read_text())
        if (seal.get("container_id") != digest({k: v for k, v in seal.items() if k != "container_id"})
                or seal.get("files") != container_files(folder)):
            raise ValueError("A relocated container changed before retirement")
        prefix = seal.get("relocation", {}).get("image_internal_prefix_preserved")
        if prefix:
            for link in (folder / "_bin").glob("*"):
                if link.is_symlink():
                    target = Path(os.readlink(link))
                    if target.is_absolute() and ".." not in target.parts and target.is_relative_to(Path(prefix)):
                        internal_links.add(link)
    for base, dirs, files in os.walk(destination, followlinks=False):
        for name in dirs + files:
            p = Path(base) / name
            if p.is_symlink() and os.readlink(p).startswith(str(source) + "/") and p not in internal_links:
                raise ValueError("A destination link still depends on old PROJECT")
    readonly = []
    for base, _, _ in os.walk(source, followlinks=False):
        p = Path(base)
        if not p.stat().st_mode & 0o200:
            if p.stat().st_uid != os.getuid() or not p.is_relative_to(source):
                raise ValueError("Cannot retire a read-only directory outside the owned old tree")
            # Raw inputs remain read-only on the accepted new authority. Only
            # the verified redundant old directories need owner write to unlink.
            os.chmod(p, p.stat().st_mode | 0o700)
            readonly.append(p.relative_to(source).as_posix())
    shutil.rmtree(source)
    result = {"status": "old_project_and_published_native_copies_retired", "source": str(source),
              "destination": str(destination), "old_logical_bytes": sum(r["bytes"] for r in current["files"].values()),
              "workspaces": retired, "job_id": os.environ["SLURM_JOB_ID"],
              "retired_readonly_directories": readonly,
              "retained": "historical token shards and legacy images on GPFS until their individual review"}
    write_json(records() / "retired.json", result)
    return result


def rebind(previous_freeze, reviewed_by):
    """Retain completed science while changing only the reviewed storage execution plan."""
    from src.cluster import feature_admission as dispatch
    from src.cluster.experiment_zero import scheduler_state
    from src.cluster.freeze import check_if_present, runtime_pins, source_snapshot
    from src.cluster.maintenance import require_idle_vsc
    from src.jev.catalog import SELF_HOSTED
    from src.utils.config import load_yaml
    from src.utils.locking import file_lock

    require_idle_vsc()
    if not reviewed_by or not previous_freeze:
        raise ValueError("Named reviewer and original archived freeze required")
    retirement = json.loads((records() / "retired.json").read_text())
    if retirement.get("status") != "old_project_and_published_native_copies_retired":
        raise ValueError("Finish publication and scoped retirement before reopening admissions")
    old_freeze = json.loads(Path(previous_freeze).read_text())["frozen_receipt"]
    current = source_snapshot()
    changed = {k for k in old_freeze["files"].keys() | current["files"].keys()
               if old_freeze["files"].get(k) != current["files"].get(k)}
    protected = {k for k in changed if k.startswith(("src/jev/", "src/data/", "src/visualize/", "notebooks/"))
                 or (k.startswith("src/evaluation/") and k != "src/evaluation/checkpoint_paths.py")
                 or (k.startswith("config/") and not k.endswith(("experiment_0/scheduling.yaml", "experiment_0/self_hosted.yaml")))}
    routing_sources = {"src/utils/paths.py", "src/evaluation/checkpoint_paths.py", *(
        "src/cluster/" + name + ".py" for name in (
            "project_migration", "storage_migration", "storage_config", "maintenance", "scheduling", "preflight",
            "scratch_capacity", "scratch_stage", "workspace_graph", "benchmark_jobs", "experiment_zero",
            "feature_campaign", "setup_extractors", "token_shards", "accept_features", "control_runtime", "freeze",
        ))}
    protected |= {k for k in changed if k.startswith("src/") and k not in routing_sources}
    if protected:
        raise ValueError("Migration cannot authorize changed science: " + ", ".join(sorted(protected)))
    path = paths.manifests_dir("feature_creation") / "open_weight_campaign.json"
    with file_lock(path.parent / ".campaign.lock", timeout=1):
        report = json.loads(path.read_text())
        original = dispatch.plan()
        selected = report["parallel_admission"]["policy"]["split_encoders"]
        expanded = dispatch.partition_encoders(original, selected)
        old = {s["name"]: s for s in report["steps"]}
        if set(old) != {s["name"] for s in expanded}:
            raise ValueError("Migration cannot add or remove scientific jobs")
        for step in expanded:
            prior = old[step["name"]]
            for key in ("pool", "depends_on", "terminal_dependencies", "workspace_stage", "workspace_release"):
                if step.get(key) != prior.get(key):
                    raise ValueError("Migration changed experimental dependencies")
            if prior.get("status") == "starting" and not prior.get("job_id"):
                raise ValueError("Ambiguous old submission must be reconciled")
            if prior.get("job_id"):
                state = scheduler_state(prior["job_id"], cluster=prior["cluster"])
                if state is None:
                    raise ValueError("Old allocation is still active or accounting is unknown")
                prior = {**prior, "scheduler_state": state[0], "exit_code": state[1],
                         "status": "completed" if state == ("COMPLETED", "0:0") else "failed"}
                old[step["name"]] = prior
        finished = {p for p in SELF_HOSTED if len([s for s in old.values() if s.get("pool") == p]) == 18
                    and all(s.get("status") == "completed" for s in old.values() if s.get("pool") == p)}
        for step in expanded:
            prior = old[step["name"]]
            if prior.get("job_id") or prior.get("previous_attempt"):
                step["previous_attempt"] = prior
            runtime = step.get("workspace_stage") or step.get("workspace_release")
            if runtime in finished:
                step.update(status="completed", migration_satisfied_by=str(records() / "retired.json"))
            elif prior.get("status") == "completed" and prior.get("pool") in SELF_HOSTED:
                step.update({k: prior[k] for k in ("job_id", "status", "scheduler_state", "exit_code")})
                step["completed_execution"] = prior
        frozen = check_if_present()
        if not frozen.get("frozen"):
            raise ValueError("Freeze the reviewed migrated source before recovery")
        base = load_yaml("experiment_0/scheduling")
        legacy = {**base, "test_pool_limits": {**base["test_pool_limits"], **dict.fromkeys(SELF_HOSTED, 2)}}
        identity = digest(dict(frozen=frozen, runtime_pins=runtime_pins(), steps=original, settings=legacy))
        policy = {**report["parallel_admission"]["policy"], "campaign_identity": identity,
                  "source_sha256": frozen["source_sha256"], "dispatcher_sha256": file_sha256(Path(dispatch.__file__)),
                  "reviewed_by": reviewed_by, "settings": dispatch.admission_settings(base, expanded)}
        policy_sha = digest(policy)
        history = path.parent / "recovery_history" / ("before_project_migration_" + digest(report)[:16] + ".json")
        write_json(history, report)
        result = {"identity": digest(dict(origin_identity=identity, policy_sha256=policy_sha,
                                          plan=dispatch.partition_encoders(original, selected))),
                  "origin_identity": identity, "origin_steps": original, "steps": expanded,
                  "status": "migrated_requires_dispatch", "storage_migration": str(records() / "accepted.json"),
                  "parallel_admission": {"sha256": policy_sha, "policy": policy, "previous_receipt": str(history)},
                  "recovery": {"reviewed_by": reviewed_by, "previous_receipt": str(history),
                               "previous_freeze": str(previous_freeze), "operational_files_changed": sorted(changed)}}
        write_json(path, result)
        return {"status": result["status"], "origin_identity": identity,
                "source_sha256": frozen["source_sha256"], "reused_providers": sorted(finished)}


def gpu(provider):
    from src.cluster.semantic_contracts import exercise_semantics
    from src.jev.cache import ResponseCache
    from src.jev.self_hosted import SelfHostedClient, inference_environment, token_evidence

    inference_environment()
    client = SelfHostedClient(provider, allow_model_runs=True)
    evidence, _ = token_evidence(provider, client.counter)
    old_acceptance = paths.manifests_dir("experiment_0") / "before_extraction" / f"accepted_{provider}.json"
    accepted = json.loads(old_acceptance.read_text())
    old_gpu = []
    for item in accepted["evidence"]:
        p = Path(item["path"])
        if p.is_file():
            r = json.loads(p.read_text())
            if r.get("settings_sha256") == accepted["settings_sha256"] and r.get("settings"):
                old_gpu.append(r)
    if not old_gpu or not same_science(old_gpu[0]["settings"], client.settings):
        raise ValueError("Relocated runtime changed scientific settings")
    cache = ResponseCache(paths.results_dir("storage_migration", provider + ".sqlite3", phase="experiment_0"),
                          dataset_id="EXP0_SEMANTICS")
    first = exercise_semantics(client, cache)
    second = exercise_semantics(client, cache)
    if not first["passed"] or first != second:
        raise ValueError("Relocated GPU semantics or cached resume failed")
    result = {"status": "relocated_gpu_and_resume_passed", "provider": provider,
              "settings": client.settings, "settings_sha256": digest(client.settings),
              "token_audit_id": evidence["audit_id"], "semantic_probes": first,
              "cached_resume_identical": True, "original_acceptance": str(old_acceptance),
              "original_acceptance_sha256": file_sha256(old_acceptance),
              "job_id": os.environ["SLURM_JOB_ID"]}
    write_json(records() / ("gpu_" + provider + ".json"), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("relocate", "gpu", "publish-workspaces", "cpu-accept", "accept", "retire-old", "retire-native", "retire-source", "rebind", "numeric-backup"))
    p.add_argument("--provider")
    p.add_argument("--reviewed-by")
    p.add_argument("--previous-freeze", type=Path)
    a = p.parse_args()
    result = (retire_old(native_only=True) if a.action == "retire-native" else
              retire_old(source_only=True) if a.action == "retire-source" else
              numeric_backup() if a.action == "numeric-backup" else
              rebind(a.previous_freeze, a.reviewed_by) if a.action == "rebind" else
              retire_old() if a.action == "retire-old" else accept(a.reviewed_by) if a.action == "accept" else
              cpu_accept() if a.action == "cpu-accept" else relocate() if a.action == "relocate" else
              publish_workspaces() if a.action == "publish-workspaces" else gpu(a.provider))
    print(json.dumps({k: v for k, v in result.items() if k != "settings"}))


if __name__ == "__main__":
    main()
