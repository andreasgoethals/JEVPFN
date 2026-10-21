"""Single fixed AutoGluon-Multimodal reference; no HPO or outer-test validation.

Publisher API: AutoGluon 1.6.1 MultiModalPredictor. The runtime and all encoder
files must be staged before this lazy adapter is constructed. Real compatibility
is established by Experiment 0B, not by the offline adapter tests.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import paths


class AutoMMBackend:
    def __init__(self, task, settings, encoder):
        from autogluon.multimodal import MultiModalPredictor

        self.task, self.settings, self.encoder = task, settings, encoder
        if settings.get("parameters"):
            raise ValueError("AG-MM uses one fixed recipe; review parameter overrides explicitly")
        checkpoint = settings.get("text_checkpoint_path")
        if not checkpoint:
            raise ValueError("AG-MM needs a staged text_checkpoint_path and hashed encoder assets")
        checkpoint = Path(checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = paths.checkpoints_dir() / checkpoint
        if not checkpoint.is_dir():
            raise ValueError("AG-MM text checkpoint directory is missing")
        # Refuse a supplied path outside the hash-reviewed asset tree.
        assets = []
        for pin in settings["assets"]:
            path = Path(pin["path"])
            assets.append(path if path.is_absolute() else paths.checkpoints_dir() / path)
        files = {p.resolve() for p in checkpoint.rglob("*") if p.is_file()}
        if not files or not files.issubset({p.resolve() for p in assets}):
            raise ValueError("Every AG-MM encoder file needs an explicit reviewed hash")
        self.column_types = {
            c: "text"
            if c in encoder.text_columns_
            else "numerical"
            if c in encoder.numeric_
            else "categorical"
            for c in encoder.columns_
        }
        self.label = "__jevpfn_agmm_target__"
        while self.label in encoder.columns_:
            self.label += "_"
        # Intermediate fine-tuning checkpoints are ephemeral, not thousands of
        # permanent PROJECT files. Predictions/metrics use the common result store.
        scratch = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR")
        self.workspace = tempfile.TemporaryDirectory(prefix="jevpfn-agmm-", dir=scratch)
        self.model = MultiModalPredictor(
            label=self.label,
            problem_type={
                "binary_classification": "binary",
                "multiclass_classification": "multiclass",
                "regression": "regression",
            }[task],
            presets="medium_quality",
            use_ensemble=False,
            path=str(Path(self.workspace.name) / "fit"),
            verbosity=2,
            hyperparameters={
                "model.names": ["hf_text", "numerical_mlp", "categorical_mlp", "fusion_mlp"],
                "model.hf_text.checkpoint_name": str(checkpoint),
                "env.num_gpus": 1 if settings["device"].startswith("cuda") else 0,
                "env.num_workers": settings["threads"],
            },
        )

    def _frame(self, x):
        if list(x.columns) != self.encoder.columns_:
            raise ValueError("AG-MM input schema changed")
        frame = x.copy()
        for column, kind in self.column_types.items():
            if kind == "text":
                frame[column] = frame[column].fillna("").astype(str)
            elif column in self.encoder.datetime_:
                parsed = pd.to_datetime(frame[column], format="ISO8601", utc=True, errors="coerce")
                frame[column] = parsed.dt.as_unit("ns").astype("int64").astype(float) / 1e9
                frame.loc[parsed.isna(), column] = np.nan
        return frame

    def fit(self, x, y):
        frame = self._frame(x)
        frame[self.label] = np.asarray(y)
        self.model.fit(
            train_data=frame,
            tuning_data=None,  # Internal validation uses only this training partition.
            column_types={
                **self.column_types,
                self.label: "numerical" if self.task == "regression" else "categorical",
            },
            seed=self.settings["seed"],
            time_limit=None,
            hyperparameter_tune_kwargs=None,
            standalone=False,
            clean_ckpts=True,
        )
        if self.task != "regression":
            self.classes_ = np.asarray(self.model.class_labels)
        return self

    def predict_proba(self, x):
        values = self.model.predict_proba(self._frame(x), as_pandas=True, as_multiclass=True)
        if set(values.columns) != set(self.classes_):
            raise ValueError("AG-MM probability class vocabulary changed")
        return values.loc[:, list(self.classes_)].to_numpy()

    def predict(self, x):
        return np.asarray(self.model.predict(self._frame(x), as_pandas=False))
