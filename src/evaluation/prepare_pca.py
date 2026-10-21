"""Prepare the reviewed PCA grid on CPU before allocating GPUs to predictors."""

import argparse
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data.loaders import load_catalog, load_dataset
from src.data.text_embeddings import encoder_config, load
from src.evaluation.design import pca_grid
from src.evaluation.embedding_control import EmbeddingControl
from src.evaluation.pca_cache import cache_path
from src.evaluation.protocol import shared_splits
from src.utils.config import load_yaml
from src.utils.serialization import canonical_json, scalar


def training_partitions(dataset, splits):
    for number, fold in enumerate(splits["folds"]):
        train = np.array(fold["train"])
        yield train
        if dataset.metadata.task_type == "binary_classification":
            labels = np.array(
                [
                    canonical_json(scalar(v))
                    for v in dataset.frame[dataset.metadata.target].iloc[train]
                ]
            )
            splitter = StratifiedKFold(3, shuffle=True, random_state=splits["seed"] + number)
            for left, _ in splitter.split(train, labels):
                yield train[left]


def run(dataset_id, encoder):
    from src.cluster.freeze import check_if_present
    from src.data.selection import selected_entries
    from src.utils import paths

    if paths.on_vsc() and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("PCA preparation needs a CPU allocation")
    check_if_present()
    cfg = load_yaml("experiment_1/main")
    entry = next((e for e in selected_entries(load_catalog(), cfg) if e["id"] == dataset_id), None)
    if entry is None:
        raise ValueError("Unselected dataset")
    dataset = load_dataset(entry, allow_download=False)
    vectors = load(dataset, encoder_config(encoder))
    groups = vectors.attrs["embedding_groups"]
    splits = shared_splits(dataset, seed=cfg["random_seed"], folds=5)
    for number, train in enumerate(training_partitions(dataset, splits)):
        for scope, components in pca_grid():
            EmbeddingControl(
                None,
                groups,
                components=components,
                scope=scope,
                seed=cfg["random_seed"],
                prepare=True,
                embedding_artifacts=vectors.attrs["feature_artifact_ids"],
                prepared_cache=cache_path(dataset_id, encoder),
            ).fit(vectors.iloc[train], None)
        print(f"{dataset_id}/{encoder}: training partition {number + 1} prepared", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--encoder", required=True)
    args = parser.parse_args()
    run(args.dataset, args.encoder)


if __name__ == "__main__":
    main()
