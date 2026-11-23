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
import warnings
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.evaluation.checkpoint_paths import (
    checkpoint_path,
    model_checkpoint_argument,
    verified_sha256,
)
from src.utils import paths
from src.utils.serialization import canonical_json, digest, scalar

MAIN_MODELS = (
    "catboost", "tabpfn_3_5", "tabicl_v2", "limix_2", "kumo_tabular", "linear", "lightgbm",
)
#: Predictors that receive categorical columns as pandas `category` dtype (training-fold
#: vocabulary; unseen and missing values are NaN), as their publishers document.
CATEGORY_DTYPE_MODELS = ("tabpfn_3_5", "tabicl_v2", "lightgbm")
#: Predictors that receive categories as typed-JSON strings (missing = the string "null").
STRING_CATEGORY_MODELS = ("linear", "catboost")
TEXT_MODELS = ("tabstar", "contexttab", "ag_mm")
#: Kumo's default recipe ``SelectColumns(500, method="first")`` per estimator.
KUMO_RECIPE_COLUMNS = 500


def predictor_settings_identity(settings):
    """Keep predictor/checkpoint identity independent of its diagnostic output folder."""
    return digest({key: value for key, value in settings.items() if key != "output_phase"})


class FoldInput:
    """Train-only category vocabulary; no target encoding or text detection.

    Official text selection happens before this adapter. Missing numeric values stay
    NaN here; estimators that need imputation fit that imputer on their own training fold.
    """

    def fit(self, x):
        self.columns_ = list(x.columns)
        # ISO dates are measurements, not nominal identifiers. Recognition uses only
        # this training partition; the same fixed conversion applies to unseen dates.
        self.datetime_ = [c for c in x if _is_datetime_column(x[c])]
        self.numeric_ = [
            c
            for c in x
            if c in self.datetime_
            or (
                pd.api.types.is_numeric_dtype(x[c])
                and not isinstance(x[c].dtype, pd.CategoricalDtype)
            )
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

    def transform(self, x, *, strings=False, categories=False):
        """Numeric columns as float (datetime -> seconds, +/-inf -> NaN); categorical columns as
        float codes (default), typed-JSON strings (`strings`) or pandas `category` dtype whose
        categories are the training vocabulary (`categories`). Unseen values become missing."""
        if strings and categories:
            raise ValueError("Choose one categorical representation")
        if list(x.columns) != self.columns_:
            raise ValueError("Predictor input column order/schema changed")
        out = pd.DataFrame(index=x.index)
        for c in self.columns_:
            if c in self.datetime_:
                parsed = pd.to_datetime(x[c], format="ISO8601", utc=True, errors="coerce")
                out[c] = parsed.dt.as_unit("ns").astype("int64").astype(float) / 1e9
                out.loc[parsed.isna(), c] = np.nan
            elif c in self.numeric_:
                out[c] = pd.to_numeric(x[c]).to_numpy(dtype=float, na_value=np.nan)
                out[c] = out[c].replace([np.inf, -np.inf], np.nan)
            elif strings:
                # Typed JSON preserves the distinction between literal strings and missingness.
                out[c] = x[c].map(lambda v: canonical_json(scalar(v)))
            elif categories:
                vocabulary = list(self.vocabulary_[c])
                values = x[c].map(lambda v: None if pd.isna(v) else canonical_json(scalar(v)))
                out[c] = pd.Categorical(
                    values.where(values.isin(vocabulary)), categories=vocabulary
                )
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


def _is_datetime_column(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return False
    values = series.dropna()
    if values.empty or not values.map(lambda v: isinstance(v, str)).all():
        return False
    # Do not reinterpret categorical year codes, ambiguous locale dates or arbitrary text.
    if not values.str.match(r"^\d{4}-\d{2}-\d{2}(?:$|[ T])").all():
        return False
    return bool(pd.to_datetime(values, format="ISO8601", utc=True, errors="coerce").notna().all())


def _checkpoint(settings, task):
    pin = settings.get("checkpoints", {}).get(task, {})
    if not pin.get("path") or not pin.get("sha256"):
        raise ValueError(f"A staged checkpoint path and SHA256 are required for {task}")
    path = checkpoint_path(pin["path"])
    if not path.is_file() or verified_sha256(path) != pin["sha256"]:
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
        path = checkpoint_path(pin["path"])
        if not path.is_file() or verified_sha256(path) != pin["sha256"]:
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
    if not isinstance(value, dict) or value.get("feature_encoder_mode") != "train_unknown_nan":
        raise ValueError("LimiX V2 requires training-only train_unknown_nan category encoding")
    if value.get("feature_view_strategy", "disabled") != "disabled":
        raise ValueError(
            "LimiX internal feature-view selection is outside the fixed-feature protocol"
        )
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
        # Publisher docs: declare categorical positions; `category` dtype also counts as declared.
        params.setdefault(
            "categorical_features_indices",
            [encoder.columns_.index(c) for c in encoder.categorical_],
        )
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
        if classification:
            params.setdefault("support_many_classes", True)
        return cls(
            model_path=_checkpoint(settings, task),
            allow_auto_download=False,
            device=device,
            random_state=seed,
            n_jobs=settings["threads"],
            **params,
        )
    if model_id == "limix_2":
        from src.evaluation.limix_preprocessing import TrainOnlyLimiX

        predictor = importlib.import_module("inference.predictor").LimiXPredictor
        pin = settings.get("inference_configs", {}).get(task)
        if not isinstance(pin, dict) or not pin.get("path") or not pin.get("sha256"):
            raise ValueError("Select and hash the publisher's V2 no-retrieval inference config")
        config_path = checkpoint_path(pin["path"])
        if not config_path.is_file() or verified_sha256(config_path) != pin["sha256"]:
            raise ValueError("LimiX inference config differs from its reviewed SHA256")
        params.setdefault("preprocess_num_jobs", settings["threads"])
        return TrainOnlyLimiX(
            predictor(
                device=importlib.import_module("torch").device(device),
                model_path=_checkpoint(settings, task),
                inference_config=_limix_config(config_path),
                seed=seed,
                categorical_features_indices=[
                    encoder.columns_.index(c) for c in encoder.categorical_
                ],
                use_data_cache=False,
                **params,
            )
        )
    if model_id == "kumo_tabular":
        return KumoBackend(task, settings, encoder.categorical_)
    if model_id == "lightgbm":
        return LightGBMBackend(task, settings, params)
    if model_id == "tabstar":
        if settings.get("repositories"):
            from src.evaluation.tabstar_assets import verified_settings

            repository, assets = verified_settings(settings, task)
            settings = {**settings, "pretrain_dataset_or_path": repository, "assets": assets}
        _verify_assets(settings)
        cls = getattr(importlib.import_module("tabstar.tabstarv1p1_model"), "TabSTARv1p1" + suffix)
        if not settings.get("pretrain_dataset_or_path"):
            raise ValueError("TabSTAR needs its reviewed staged checkpoint identity")
        return cls(
            device=device,
            random_state=seed,
            keep_model=False,  # Loaded adapters survive; transient epoch checkpoints are removed.
            pretrain_dataset_or_path=(settings["pretrain_dataset_or_path"]
                if settings.get("repositories") else
                model_checkpoint_argument(settings["pretrain_dataset_or_path"])),
            output_dir=str(
                paths.results_dir(
                    "tabstar_training",
                    uuid4().hex,
                    phase=settings.get("output_phase", "experiment_4"),
                )
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
            checkpoint=model_checkpoint_argument(settings["checkpoint_filename"]),
            bagging=1,
            max_context_size=n_train,
            **params,
        )
        model.seed = seed
        model.tokenizer.random_seed = seed
        return model
    if model_id == "ag_mm":
        _verify_assets(settings)
        from src.evaluation.ag_mm import AutoMMBackend

        return AutoMMBackend(task, settings, encoder)
    raise ValueError(f"Unknown predictor: {model_id}")


class LightGBMBackend:
    """LightGBM (CPU) with library defaults; native NaN and pandas-`category` handling.

    LightGBM rejects feature names with JSON special characters, so columns are renamed
    positionally (order and dtypes unchanged). `verbose=-1` only silences logging.
    """

    def __init__(self, task, settings, params):
        lightgbm = importlib.import_module("lightgbm")
        cls = lightgbm.LGBMRegressor if task == "regression" else lightgbm.LGBMClassifier
        self.model = cls(
            random_state=settings["seed"], n_jobs=settings["threads"], verbose=-1, **params
        )

    @staticmethod
    def _frame(x):
        frame = x.copy()
        frame.columns = [f"f{i:05d}" for i in range(frame.shape[1])]
        return frame

    def fit(self, x, y):
        self.model.fit(self._frame(x), np.asarray(y))
        if hasattr(self.model, "classes_"):
            self.classes_ = self.model.classes_
        return self

    def predict(self, x):
        return self.model.predict(self._frame(x))

    def predict_proba(self, x):
        return self.model.predict_proba(self._frame(x))


class KumoBackend:
    """NVIDIA SDM adapter using all context rows and its default ensemble recipe.

    Explicitly loads a local checkpoint without Hub access. Regression uses the mean
    of its 999 predicted quantiles, a documented approximation to the predictive mean.

    The default recipe keeps at most 500 columns per estimator (``SelectColumns(500)``
    after category-count expansion), a recipe default rather than an architectural
    limit. Inputs of at most 500 expanded columns use the unmodified default recipe.
    Wider inputs (owner-approved 23 Nov 2026) get a copy of that recipe whose
    ``SelectColumns.max_columns`` equals the expanded width, as the publisher's own
    ``max_columns`` wrapper parameter does, so every estimator sees every column. The
    deviation is recorded as a ``kumo_max_columns_override`` fit warning.
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

    def _recipe(self, width):
        """None (the untouched default) up to 500 columns; otherwise a widened copy."""
        if width <= KUMO_RECIPE_COLUMNS:
            return None
        recipe = self.model.default_recipe()
        selectors = [
            p for p in recipe.features.modules()
            if isinstance(p, self.sdm.processing.SelectColumns)
        ]
        if len(selectors) != 1 or selectors[0].max_columns != KUMO_RECIPE_COLUMNS:
            raise ValueError("Kumo default recipe no longer has one SelectColumns(500); review the override")
        selectors[0].max_columns = width
        return recipe

    def fit(self, x, y):
        # AddCategoryCounts(min_cardinality=50) adds one column per categorical column whose
        # fitted vocabulary (missing excluded) exceeds 50 values.
        extra_counts = sum(x[c].nunique(dropna=True) > 50 for c in self.categorical_columns)
        width = int(x.shape[1] + extra_counts)
        self.recipe_ = self._recipe(width)
        self.max_columns_override_ = width if self.recipe_ is not None else None
        if self.recipe_ is not None:
            warnings.warn(
                f"kumo_max_columns_override={width}: SelectColumns(500) widened to all "
                f"{width} expanded columns per estimator",
                UserWarning,
                stacklevel=2,
            )
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
                    # Nothing extra is passed for inputs within the default 500 columns.
                    **({} if self.recipe_ is None else {"recipe": self.recipe_}),
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
        self.encoder_.text_columns_ = list(self.metadata.text_columns)
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
        return self.encoder_.transform(
            x,
            strings=self.model_id in STRING_CATEGORY_MODELS,
            categories=self.model_id in CATEGORY_DTYPE_MODELS,
        )

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
    if paths.on_vsc() and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Predictor execution on VSC requires a compute allocation")
    if settings.get("device", "cpu").startswith("cuda"):
        from src.cluster.scheduling import reject_a100_allocation

        reject_a100_allocation()
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
