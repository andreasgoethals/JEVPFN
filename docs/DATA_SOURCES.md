# Data sources and audit definitions

Verified 29-09-2026. The first data audit used pinned official upstream sources while the
literature submodule was uninitialised. The library is now synced read-only to
`81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba`; the original data/source pins below remain unchanged.

## Primary sources and pins

- [MulTaBench official repository](https://github.com/alanarazi7/MulTaBench) directs paper
  reproduction to `paper_version`. The resolved snapshot is
  `3bb95079a6146ed35d45187e98bcad5b063273a1`.
- [Core dataset lists](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/datasets/all_multabench_datasets.py):
  `MULTABENCH_CORE_TEXT` defines the 20 IDs; no image or extended IDs are admitted.
- [Official loader](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/benchmark/load.py):
  `load_multabench_dataset` reads already-curated Kaggle `data.csv` and `metadata.json`.
- [Official summary](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/leaderboard/results/datasets_summary.csv):
  expected rows and text/structured counts, validated against every loaded dataset.
- [Official text rule](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/preprocessing/feat_types.py):
  `detect_text_features`, `_is_text_feature` and `is_date_feature`.
- Numerical precheck is from TabSTAR **1.1.15**, the version in that MulTaBench snapshot's
  `requirements.txt`: `detect_numerical_features`, `is_numerical_feature`,
  `is_mostly_numerical`, `is_numeric`. Source was inspected in the
  [published wheel](https://pypi.org/project/tabstar/1.1.15/); no TabSTAR installation or weights
  were needed.

`config/exploration/datasets.yaml` records primary-source URLs/hashes, data versions/hashes and
the column list obtained from these rules. The public official Kaggle archives are downloaded
through the versioned dataset-download endpoint, using only Python's standard library. Only
`data.csv` and `metadata.json` are extracted; no arbitrary archive paths or downloaded code run.
The CSV reader follows the official pandas defaults, with `low_memory=False` for stable types.

## Text is a benchmark definition, not a language judgement

The released metadata identifies target, task, shape and class count, but **does not provide a
text-column list**. The official paper summary computes it dynamically. Our small dependency-free
implementation of that published rule is `src/data/official_types.py`; the resulting names are
frozen in the catalog for subsequent loading, so folds never redefine text columns.

After excluding numerical and date columns, a field is text when it has at least 100 distinct
nonmissing values **or** at least 80% distinct values among nonmissing entries. The numerical
precheck also recognises more than 50 distinct digit-only strings with at most one nonnumerical
value. Date detection uses the first 1,000 valid values and a 99% parsing threshold, guarded by
date separators/alphabetic characters. We retain these rules, including their limitations, and
only stabilise column ordering to CSV order instead of set iteration.

Consequently, names, URLs, identifiers, comma-formatted numeric strings and some date strings can
be labelled text. No new natural-language detector or manual reclassification was introduced.
The frozen column names reproduce the official counts for all 20 datasets.

## Statistics

- Feature counts exclude the target; the total column count includes it.
- Missingness uses pandas CSV null semantics. No extra sentinel strings are reinterpreted.
- Unique counts exclude missing entries; the unique ratio divides by nonmissing row count.
- Numeric features use their CSV dtype; booleans are categorical; date-like strings are “other”.
  Numeric-looking strings are not transformed into floats during this audit.
- Numeric and regression summaries include count, mean, sample standard deviation, min/max and
  5th/25th/50th/75th/95th percentiles; nonfinite values are excluded from these summaries.
- Text lengths count characters and whitespace-delimited words. Missing text contributes zero.
  Approximate tokens are `ceil(characters / 4)`, configurable as a character ratio. This is not
  calibrated to Jev, especially for Unicode, codes or serialised lists.
- Combined text tokens use `ceil(sum(text characters for that row) / ratio)`; per-column tokens
  use separate ceilings. Examples are deterministic unique nonempty length ranks, clipped only
  for local display. Actual requests retain all text.
- Input-token estimates count exact compact intended JSON character lengths per row, including
  task/question/choices and JSON escapes. Non-text features are excluded; empty inputs are skipped. They exclude unknown
  provider wrappers, output tokens and retries. All logical slots are counted before deduplication.

Target distributions and regression quantiles live only in audit artifacts. Request metadata
uses the allowed distinct class labels, never their counts or any other target-derived statistic.

## Reproduction and outputs

Run `python -m src.data.prepare`, then `python -m src.utils.run_notebooks` in the
approved environment. Preparation validates all pinned hashes. The audit caches completed
datasets, writes markers last, and fingerprints its cache with data/config/code hashes and dependency
versions. Rich row examples are local/ignored; tracked summaries contain aggregate results only.
An OS-backed lock coordinates shared audit construction and length-array reads between parallel
notebooks, including release after a failed process.
Data retain source licences; the repository does not redistribute the CSVs or raw request examples.

Raw folders now use `data/raw/01_fake_job_posting/` through
`data/raw/20_zomato_restaurants/`. Catalog `directory` selects the local name, while `slug`
retains the exact official Kaggle download identifier. Renaming a local folder therefore does
not change the upstream dataset. `data.csv`, `metadata.json` and `download.json` live directly
inside each dataset folder. Dataset versions remain pinned in the catalog and `download.json`;
the loader rejects a cached download with a different source or version instead of silently
reusing it. Frozen source snapshots live under `src/data/upstream/`; their SHA-256 hashes are
unchanged, and they are never imported or executed. No Python files remain under `data/`.
