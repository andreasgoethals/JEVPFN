"""A3. Compare frozen, same-field representations on an outcome-blind text sample."""

import argparse
import json
import zlib
from contextlib import closing
from pathlib import Path

import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from src.data import text_embeddings as em
from src.data.loaders import load_catalog, load_dataset
from src.data.selection import selected_entries
from src.evaluation.decision_comparison import normalize_group, payload_hashes
from src.evaluation.geometry import decision_distances, encoder_distances
from src.evaluation.portable_features import notebook_inventory, verified_payload_hashes
from src.evaluation.research_questions import random_probe_requests
from src.jev.features import FeatureBlock, load_block
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest, write_json


def neighbours(matrix, k):
    """Fractional membership at the cutoff gives equidistant neighbours equal weight."""
    d = matrix.copy()
    np.fill_diagonal(d, np.inf)
    threshold = np.partition(d, k - 1, axis=1)[:, k - 1, None]
    inside, tied = d < threshold, d == threshold
    share = (k - inside.sum(axis=1)) / tied.sum(axis=1)
    return inside + tied * share[:, None]


def relation(left, right, seed, permutations=19):
    a, b = squareform(left), squareform(right)
    if a.shape != b.shape or not np.isfinite([a, b]).all():
        raise ValueError("Mismatched or nonfinite pair distances")
    n = len(a)
    result = dict(
        texts=n,
        pairs=len(left),
        spearman=None,
        neighbour_overlap=None,
        null_mean=None,
        text_block_spearman_range=None,
    )
    if n < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return result | {"status": "undefined_insufficient_or_constant"}
    # Ranking once allows a correspondence permutation to reuse both rank vectors.
    ra, rb = rankdata(left), rankdata(right)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    ranks = squareform(rb)
    upper = np.triu_indices(n, k=1)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        idx = rng.permutation(n)
        null.append(np.sum(ra * ranks[idx[upper[0]], idx[upper[1]]]) / den)
    k = min(10, max(1, (n - 1) // 2))
    # Leave blocks of texts out, retaining all pairs among the remaining texts.
    # The range is a sensitivity diagnostic, never an independent-pair interval.
    sensitivity = []
    for omitted in np.array_split(rng.permutation(n), min(8, n)):
        keep = np.ones(n, dtype=bool)
        keep[omitted] = False
        mask = keep[upper[0]] & keep[upper[1]]
        if mask.sum() > 1 and np.ptp(left[mask]) and np.ptp(right[mask]):
            sensitivity.append(
                float(np.corrcoef(rankdata(left[mask]), rankdata(right[mask]))[0, 1])
            )
    return result | dict(
        status="measured",
        spearman=float(np.sum(ra * rb) / den),
        neighbour_k=k,
        neighbour_overlap=float((neighbours(a, k) * neighbours(b, k)).sum() / (n * k)),
        chance_overlap=k / (n - 1),
        null_permutations=permutations,
        null_mean=float(np.mean(null)),
        text_block_spearman_range=[min(sensitivity), max(sensitivity)] if sensitivity else None,
    )


def geometry_protocol(dataset_id, source, cfg, main, members):
    from src.evaluation.design import decision_versions
    from src.evaluation.qualification import decision_artifact_pins, decision_selector
    from src.utils.provenance import code_identity

    return {
        "source": source, "config": digest(cfg), "decision_model_versions": decision_versions(members),
        "implementation": code_identity(["src/evaluation/geometry_study.py", "src/evaluation/geometry.py",
                                         "src/evaluation/decision_comparison.py"]),
        "providers": {p: [decision_selector(main, p, dataset_id) if main["feature_selectors"].get(p) else None,
                          decision_artifact_pins(main, dataset_id, p)] for p in members},
        "decision_roster": members,
    }


def run(dataset_id, *, pilot=False, cohort=None):
    from src.evaluation.benchmark_export import cohort_key, cohort_record

    cfg = load_yaml("A3")
    main = load_yaml("B1")
    members = cohort_record(cohort, main=main)["decision_roster"]
    folder = cohort_key(cohort, main=main)
    entry = next(e for e in selected_entries(load_catalog(), main) if e["id"] == dataset_id)
    dataset = load_dataset(entry, allow_download=False)
    from src.evaluation.qualification import decision_artifact_pins, decision_selector
    protocol = geometry_protocol(dataset_id, dataset.source["data_identity"], cfg, main, members)
    pilot_path = paths.results_dir("geometry_pilots", phase="A0") / folder / (dataset_id + ".json")
    if not pilot:
        if not pilot_path.is_file():
            raise ValueError("Full geometry requires a current pilot for this dataset")
        prior = json.loads(pilot_path.read_text(encoding="utf-8"))
        if (prior.get("pilot") is not True or prior.get("protocol") != protocol
                or prior.get("missing_providers") != [] or not prior.get("comparisons")):
            raise ValueError("Geometry pilot differs from the current protocol or is incomplete")
    inventory = notebook_inventory(dataset_id)
    _, columns = em.input_rows(dataset)
    samples, distances, encoder_ids = [], {}, {}
    for name, compressed, _ in columns:
        ids = json.loads(zlib.decompress(compressed))
        positions = {k: i for i, k in enumerate(ids) if k is not None}
        keys = (
            random_probe_requests(
                positions,
                dataset_id=dataset_id,
                study=name,
                seed=cfg["seed"],
                limit=min(12, cfg["texts_per_group"]) if pilot else cfg["texts_per_group"],
            )
            if positions
            else []
        )
        samples.append(dict(field=name, text_keys=keys, rows=[f"csv:{positions[k]}" for k in keys]))
    for encoder in cfg["encoders"]:
        spec = em.encoder_config(encoder)
        path = em.cache_path(dataset_id, spec)
        em.validate_dataset(dataset, spec, source=path)
        with closing(em._connect(path, readonly=True)) as db:
            encoder_ids[encoder] = db.execute(
                "SELECT value FROM metadata WHERE key='artifact_id'"
            ).fetchone()[0]
            for sample in samples:
                if len(sample["rows"]) >= 2:
                    vectors = [
                        np.frombuffer(
                            db.execute(
                                "SELECT vector FROM embeddings WHERE key=?", (k,)
                            ).fetchone()[0],
                            dtype="<f4",
                        )
                        for k in sample["text_keys"]
                    ]
                    distances[encoder, sample["field"]] = encoder_distances(vectors)
    from itertools import combinations

    from src.jev.inputs import prepared_row

    expected_hashes = {}
    for sample in samples:
        field = sample["field"]
        values = []
        for row in sample["rows"]:
            slots = prepared_row(paths.dataset_question_path(dataset_id), int(row.removeprefix("csv:")))
            matches = [s for s in slots if s["representation"] == "per_column" and s["text_columns"] == [field]]
            if len(matches) != 1 or not matches[0]["payload_sha256"]:
                raise ValueError("Missing frozen field payload for the sampled source row")
            values.append(matches[0]["payload_sha256"])
        expected_hashes[field] = values
    references = [
        dict(dataset=dataset_id, field=s["field"], left_encoder=a, right_encoder=b,
             **relation(distances[a, s["field"]], distances[b, s["field"]], cfg["seed"]))
        for s in samples if len(s["rows"]) >= 2 for a, b in combinations(cfg["encoders"], 2)
    ]
    rows, missing, signatures = [], [], {}
    from src.evaluation.features import match_artifact_pins

    for provider in members:
        selected = inventory[
            inventory.provider.eq(provider) & inventory.representation.eq("per_column")
        ]
        if selected.empty:
            missing.append(provider)
            continue
        selected = match_artifact_pins(selected, decision_artifact_pins(main, dataset_id, provider))
        pinned = decision_selector(main, provider, dataset_id)
        for key, value in pinned.items():
            selected = selected.loc[selected[key] == value]
        if len(selected) != 1:
            raise ValueError(f"Expected one reviewed {provider} artifact matching the RQ2 selector")
        item = selected.iloc[0]
        path = Path(item.feature_store)
        block = load_block(path, artifact_id=item.artifact_id)
        if (
            block.manifest["identity"]["data_identity"] != dataset.source["data_identity"]
            or list(dataset.metadata.text_columns) != block.manifest["text_columns"]
        ):
            raise ValueError("Source rows or field identities differ")
        values = block.values.set_index("row_id", drop=False)
        requests = block.requests.set_index(["group", "row_id"], drop=False)
        for index, sample in enumerate(samples):
            if len(sample["rows"]) < 2:
                continue
            group_name = f"text_{index:03d}"
            row_ids = sample["rows"]
            sampled = FeatureBlock(
                values.loc[row_ids].reset_index(drop=True),
                requests.loc[group_name].loc[row_ids].reset_index(drop=True),
                block.manifest,
            )
            hashes = (
                verified_payload_hashes(path, sampled.requests.request_key.dropna().unique())
                if item.portable
                else payload_hashes(path, sampled)
            )
            group = normalize_group(sampled, group_name, hashes)
            positions = {row: i for i, row in enumerate(group.row_ids)}
            indices = [positions[i] for i in sample["rows"]]
            observed = group.input_hashes[indices].tolist()
            if observed != expected_hashes[sample["field"]]:
                raise ValueError("Sampled decision payload differs from the frozen row and text field")
            fingerprint = digest([group.signature, observed])
            shared_fingerprint = signatures.setdefault(sample["field"], fingerprint)
            if shared_fingerprint != fingerprint:
                raise ValueError(
                    "Decision providers disagree on the sampled question or semantic inputs"
                )
            decision = decision_distances(group.probabilities[indices])
            for encoder in cfg["encoders"]:
                result = relation(distances[encoder, sample["field"]], decision, cfg["seed"])
                rows.append(
                    dict(
                        dataset=dataset_id,
                        field=sample["field"],
                        encoder=encoder,
                        provider=provider,
                        artifact_id=item.artifact_id,
                        **result,
                    )
                )
    body = dict(
        dataset=dataset_id,
        pilot=pilot,
        data_identity=dataset.source["data_identity"],
        config=digest(cfg),
        encoders=encoder_ids,
        protocol=protocol,
        input_identities=signatures,
        samples=samples,
        comparisons=rows,
        encoder_references=references,
        missing_providers=missing,
    )
    if pilot and (missing or not rows):
        raise ValueError("A0 geometry pilot requires every selected provider and a measurable text field")
    if not pilot and (prior["encoders"] != encoder_ids or
            {r["provider"]: r["artifact_id"] for r in prior["comparisons"]} !=
            {r["provider"]: r["artifact_id"] for r in rows}):
        raise ValueError("The pilot and full geometry use different saved artifacts")
    destination = pilot_path if pilot else paths.results_dir("geometry", phase="A3") / folder / (dataset_id + ".json")
    write_json(destination, body)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cohort", metavar="COHORT")
    parser.add_argument("--pilot", action="store_true", help="A0 readback with at most 12 distinct texts per field")
    args = parser.parse_args()
    print(run(args.dataset, pilot=args.pilot, cohort=args.cohort))
