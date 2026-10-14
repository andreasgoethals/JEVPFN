"""Lazy, offline predictor adapters. Importing this module never loads model weights.

The factory requires explicit execution permission. Checkpoints and publisher runtimes
must be staged/verified first. Package defaults are retained except resource/seed settings
and the documented changes that prevent hidden row/feature subsampling.
"""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.utils import paths
from src.utils.serialization import canonical_json, file_sha256, scalar

MAIN_MODELS = ("catboost", "tabpfn_3_5", "tabicl_v2", "limix_2", "kumo_tabular", "linear")
TEXT_MODELS = ("tabstar", "contexttab")


class FoldInput:
    """Train-only category vocabulary; no target encoding or text detection.

    Official text selection happens before this adapter. Missing numeric values stay
    NaN here; estimators that need imputation fit that imputer on their own training fold.
    """

    def fit(self, x):
        self.columns_ = list(x.columns)
        self.numeric_ = [
            c
            for c in x
            if pd.api.types.is_numeric_dtype(x[c])
            and not isinstance(x[c].dtype, pd.CategoricalDtype)
        ]
        self.categorical_ = [c for c in x if c not in self.numeric_]
        self.vocabulary_ = {
            c: {
                v: i
                for i, v in enumerate(sorted({canonical_json(scalar(v)) for v in x[c].dropna()}))
            }
            for c in self.categorical_
        }
        return self

    def transform(self, x, *, strings=False):
        if list(x.columns) != self.columns_:
            raise ValueError("Predictor input column order/schema changed")
        out = pd.DataFrame(index=x.index)
        for c in self.columns_:
            if c in self.numeric_:
                out[c] = pd.to_numeric(x[c]).to_numpy(dtype=float, na_value=np.nan)
                out[c] = out[c].replace([np.inf, -np.inf], np.nan)
            elif strings:
                # Typed JSON preserves the distinction between literal strings and missingness.
                out[c] = x[c].map(lambda v: canonical_json(scalar(v)))
            else:
                out[c] = (
                    x[c]
                    .map(
                        lambda v, column=c: self.vocabulary_[column].get(
                            canonical_json(scalar(v)), np.nan
                        )
                    )
                    .astype(float)
                )
        # Intercept-only fallback for a dataset with no non-text columns.
        if not len(out.columns):
            out["__constant_no_non_text_features__"] = 0.0
        return out


def _checkpoint(settings, task):
    pin = settings.get("checkpoints", {}).get(task, {})
    if not pin.get("path") or not pin.get("sha256"):
        raise ValueError(f"A staged checkpoint path and SHA256 are required for {task}")
    path = Path(pin["path"])
    if not path.is_absolute():
        path = paths.checkpoints_dir() / path
    if not path.is_file() or file_sha256(path) != pin["sha256"]:
        raise ValueError("Checkpoint missing or SHA256 differs from the reviewed pin")
    return str(path)


def verify_runtime(settings):
    """Verify installed distributions without importing heavy dependencies."""
    if not settings.get("packages"):
        raise ValueError("Explicit installed-package pins are required")
    verified = {}
    for package, expected in settings.get("packages", {}).items():
        distribution = importlib.metadata.distribution(package)
        if expected.startswith("git:"):
            direct = json.loads(distribution.read_text("direct_url.json") or "{}")
            actual = direct.get("vcs_info", {}).get("commit_id")
            if actual != expected[4:]:
                raise ValueError(f"{package}: expected source commit {expected[4:]}")
        elif distribution.version != expected:
            raise ValueError(f"{package}: expected {expected}, installed {distribution.version}")
        verified[package] = distribution.version
    return verified


def _verify_assets(settings):
    """Native-text encoders may open several cached files: every one needs a saved hash."""
    assets = settings.get("assets", [])
    if not assets:
        raise ValueError("Native-text checkpoint and encoder assets need explicit staged hashes")
    for pin in assets:
        if not pin.get("path") or not pin.get("sha256"):
            raise ValueError("Incomplete native-text asset pin")
        path = Path(pin["path"])
        if not path.is_absolute():
            path = paths.checkpoints_dir() / path
        if not path.is_file() or file_sha256(path) != pin["sha256"]:
            raise ValueError("Native-text asset is missing or differs from its reviewed hash")


def _limix_config(path):
    """Use the publisher's V2 recipe, rejecting retrieval/context selection."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    pipelines = value if isinstance(value, list) else value.get("pipelines", [])
    if not pipelines:
        raise ValueError("LimiX inference config must contain explicit pipelines")
    for pipeline in pipelines:
        if pipeline.get("retrieval_config", {}).get("use_retrieval", False):
            raise ValueError("LimiX retrieval would violate the all-training-rows protocol")
    return value


def _backend(model_id, task, settings, encoder, n_train):
    """Verified public interfaces; optional modules are imported only inside this function."""
    classification = task != "regression"
    suffix = "Classifier" if classification else "Regressor"
    seed, device = settings["seed"], settings["device"]
    params = copy.deepcopy(settings.get("parameters", {}))
    if model_id == "linear":
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        numeric = encoder.numeric_ or (
            [] if encoder.categorical_ else ["__constant_no_non_text_features__"]
        )
        preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler()),
                    numeric,
                ),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), encoder.categorical_),
            ]
        )
        estimator = (
            LogisticRegression(random_state=seed, **params)
            if classification
            else LinearRegression(**params)
        )
        return make_pipeline(preprocessor, estimator)
    if model_id == "catboost":
        cls = getattr(importlib.import_module("catboost"), "CatBoost" + suffix)
        return cls(
            random_seed=seed,
            task_type="CPU",
            thread_count=settings["threads"],
            cat_features=encoder.categorical_,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
    if model_id == "tabpfn_3_5":
        cls = getattr(importlib.import_module("tabpfn"), "TabPFN" + suffix)
        version = importlib.import_module("tabpfn.constants").ModelVersion.V3_5
        return cls.create_default_for_version(
            version,
            model_path=_checkpoint(settings, task),
            device=device,
            random_state=seed,
            n_preprocessing_jobs=settings["threads"],
            inference_config={"TRANSFORM_TEXT": False},
            **params,
        )
    if model_id == "tabicl_v2":
        cls = getattr(importlib.import_module("tabicl"), "TabICL" + suffix)
        return cls(
            model_path=_checkpoint(settings, task),
            allow_auto_download=False,
            device=device,
            random_state=seed,
            n_jobs=settings["threads"],
            **params,
        )
    if model_id == "limix_2":
        predictor = importlib.import_module("inference.predictor").LimiXPredictor
        pin = settings.get("inference_configs", {}).get(task)
        if not isinstance(pin, dict) or not pin.get("path") or not pin.get("sha256"):
            raise ValueError("Select and hash the publisher's V2 no-retrieval inference config")
        config_path = Path(pin["path"])
        if not config_path.is_absolute():
            config_path = paths.checkpoints_dir() / config_path
        if not config_path.is_file() or file_sha256(config_path) != pin["sha256"]:
            raise ValueError("LimiX inference config differs from its reviewed SHA256")
        return predictor(
            device=device,
            model_path=_checkpoint(settings, task),
            inference_config=_limix_config(config_path),
            seed=seed,
            categorical_features_indices=[encoder.columns_.index(c) for c in encoder.categorical_],
            use_data_cache=False,
            **params,
        )
    if model_id == "kumo_tabular":
        return KumoBackend(task, settings, encoder.categorical_)
    if model_id == "tabstar":
        _verify_assets(settings)
        cls = getattr(importlib.import_module("tabstar.tabstar_model"), "TabSTAR" + suffix)
        if not settings.get("pretrain_dataset_or_path"):
            raise ValueError("TabSTAR needs its reviewed staged checkpoint identity")
        return cls(
            device=device,
            random_state=seed,
            pretrain_dataset_or_path=settings["pretrain_dataset_or_path"],
            output_dir=str(
                paths.results_dir("tabstar_training", uuid4().hex, phase="experiment_2")
            ),
            **params,
        )
    if model_id == "contexttab":
        _verify_assets(settings)
        cls = getattr(importlib.import_module("sap_rpt_oss"), "SAP_RPT_OSS_" + suffix)
        if not settings.get("checkpoint_filename"):
            raise ValueError("ConTextTab needs its staged checkpoint filename")
        # The default bagging=8/max_context_size=8192 silently samples. Override both.
        # Large full contexts are an unvalidated resource requirement, never a row cap.
        model = cls(
            checkpoint=settings["checkpoint_filename"],
            bagging=1,
            max_context_size=n_train,
            **params,
        )
        model.seed = seed
        model.tokenizer.random_seed = seed
        return model
    raise ValueError(f"Unknown predictor: {model_id}")


class KumoBackend:
    """NVIDIA SDM adapter using all context rows and its default ensemble recipe.

    Explicitly loads a local checkpoint without Hub access. Regression uses the mean
    of its 999 predicted quantiles, a documented approximation to the predictive mean.
    """

    def __init__(self, task, settings, categorical_columns=()):
        import sdm
        import torch

        self.torch, self.sdm = torch, sdm
        self.settings, self.task = settings, task
        self.categorical_columns = set(categorical_columns)
        torch.manual_seed(settings["seed"])
        kind = "regression" if task == "regression" else "classification"
        self.model = sdm.models.KumoTabular(
            task=kind,
            size=settings.get("size", "large"),
            pretrained=False,
            device=settings["device"],
        )
        state = torch.load(
            _checkpoint(settings, task), map_location=settings["device"], weights_only=True
        )
        self.model.models[sdm.Task(kind)].load_state_dict(state)
        self.model.eval()

    def _table(self, frame, target=False):
        stypes = {c: "categorical" if c in self.categorical_columns else "numerical" for c in frame}
        if target and self.task != "regression":
            stypes = {c: "categorical" for c in frame}
        return self.sdm.TableTensor.from_pandas(
            frame.reset_index(drop=True), stypes=stypes, device=self.settings["device"]
        )

    def fit(self, x, y):
        if x.shape[1] > 500:
            raise ValueError("Kumo default recipe selects 500 columns; refuse silent feature loss")
        self.classes_ = np.unique(y) if self.task != "regression" else None
        self.x_, self.y_ = (
            self._table(x),
            self._table(pd.DataFrame({"__target__": np.asarray(y)}), target=True),
        )
        return self

    def _predict(self, x):
        torch = self.torch
        # Batching queries leaves the complete training context intact.
        pieces = []
        batch = self.settings.get("prediction_batch_size", 256)
        for start in range(0, len(x), batch):
            generator = torch.Generator(device=self.settings["device"]).manual_seed(
                self.settings["seed"]
            )
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=self.settings["device"].startswith("cuda"),
                ),
            ):
                out = self.model(
                    x_context=self.x_,
                    y_context=self.y_,
                    x_query=self._table(x.iloc[start : start + batch]),
                    num_estimators=self.settings.get("num_estimators", 8),
                    generator=generator,
                )
            values = out.numerical.float().cpu().numpy()
            if self.task != "regression":
                names = list(out.columns[self.sdm.Stype.numerical])
                values = values[:, [names.index(str(int(c))) for c in self.classes_]]
            pieces.append(values)
        return np.concatenate(pieces)

    def predict_proba(self, x):
        return self._predict(x)

    def predict(self, x):
        return self._predict(x).mean(axis=1)


class PredictorAdapter:
    """Fresh instance per inner/outer fit; explicit class mapping and row-count checks."""

    def __init__(self, model_id, metadata, settings, *, backend_loader=_backend):
        self.model_id, self.metadata = model_id, metadata
        self.settings, self.backend_loader = copy.deepcopy(settings), backend_loader

    def fit(self, x, y):
        if len(x) != len(y) or self.metadata.target in x:
            raise ValueError("Targets must be separate and rows aligned")
        if self.model_id in MAIN_MODELS and set(x.columns) & set(self.metadata.text_columns):
            raise ValueError("Original text cannot enter a numeric feature arm")
        self.encoder_ = FoldInput().fit(x)
        task = self.metadata.task_type
        self.classes_ = np.asarray(self.metadata.classes, dtype=object)
        if task == "regression":
            encoded_y = y.to_numpy(dtype=float)
        else:
            mapping = {canonical_json(c): i for i, c in enumerate(self.metadata.classes)}
            encoded_y = np.array([mapping[canonical_json(scalar(v))] for v in y])
            if set(encoded_y) != set(range(len(mapping))):
                raise ValueError("Every class must be present in this training partition")
        self.model_ = self.backend_loader(self.model_id, task, self.settings, self.encoder_, len(x))
        converted = self._input(x)
        if self.model_id == "limix_2":
            self.train_x_, self.train_y_ = converted.to_numpy(), encoded_y
        else:
            # Preserve source row alignment, especially for native-text models using pd.concat.
            labels = pd.Series(
                y.to_numpy() if self.model_id in TEXT_MODELS else encoded_y,
                index=converted.index,
                name=y.name,
            )
            self.model_.fit(converted, labels)
        self.n_training_rows_ = len(x)
        return self

    def _input(self, x):
        if self.model_id in TEXT_MODELS:
            if list(x.columns) != self.encoder_.columns_:
                raise ValueError("Original feature schema changed")
            return x.copy()
        return self.encoder_.transform(x, strings=self.model_id in {"linear", "catboost"})

    def _limix(self, x):
        return np.asarray(
            self.model_.predict(
                self.train_x_,
                self.train_y_,
                self._input(x).to_numpy(),
                task_type="Regression"
                if self.metadata.task_type == "regression"
                else "Classification",
            )
        )

    def predict(self, x):
        return self._limix(x) if self.model_id == "limix_2" else self.model_.predict(self._input(x))

    def predict_proba(self, x):
        if self.model_id == "limix_2":
            return self._limix(x)
        output = np.asarray(self.model_.predict_proba(self._input(x)))
        if self.model_id in TEXT_MODELS:
            order = [canonical_json(scalar(v)) for v in self.model_.classes_]
            wanted = [canonical_json(v) for v in self.metadata.classes]
            if set(order) != set(wanted):
                raise ValueError("Native-text backend class vocabulary changed")
            return output[:, [order.index(v) for v in wanted]]
        order = list(self.model_.classes_)
        if set(order) != set(range(len(self.classes_))):
            raise ValueError("Backend probability vocabulary differs from the encoded target")
        return output[:, [order.index(i) for i in range(len(order))]]


def estimator_factory(model_id, metadata, settings, *, allow_model_runs=False):
    if allow_model_runs is not True:
        raise RuntimeError("Predictor execution is disabled")
    if model_id not in MAIN_MODELS + TEXT_MODELS:
        raise ValueError("Predictor is outside the approved roster")
    verify_runtime(settings)
    # All model downloads are a separate explicit staging step, never CV side effects.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(paths.checkpoints_dir("huggingface"))
    os.environ["HF_HUB_CACHE"] = str(paths.checkpoints_dir("huggingface", "hub"))
    os.environ["HF_XET_CACHE"] = str(paths.checkpoints_dir("huggingface", "xet"))
    os.environ["TORCH_HOME"] = str(paths.checkpoints_dir("torch"))
    return lambda: PredictorAdapter(model_id, metadata, settings)
