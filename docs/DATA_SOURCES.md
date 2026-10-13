# Data sources and audit definitions

Source pins verified 29-09-2026; text-reuse audit updated 30-09-2026. The library is pinned read-only to
`de324efcca55be392feb0c01830b3b6ce364587f` (existing checkout verified 13-10-2026);
the original data/source pins below remain unchanged. Earlier source review used library pin
`1ac6e6783f24b2d5cd058af7ad8cad7c1bfaf7ae`.

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

`config/datasets.yaml` records primary-source URLs/hashes, data versions/hashes and
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

The paper's printed table and its pinned executable summary are not identical: for example,
the paper lists 2/5/9 text columns for Jigsaw/Kickstarter/US Accidents, while the pinned summary
and reproduced code give 1/3/6. This project consistently uses the pinned executable definition;
it does not mix the paper table's classifications with the released CSVs. The distinction must
be stated when comparing to the paper, rather than silently adding or removing text features.

## Statistics

- Feature counts exclude the target; the total column count includes it.
- Missingness uses pandas CSV null semantics. No extra sentinel strings are reinterpreted.
- General feature unique counts exclude pandas-null entries. Text reuse separately excludes null,
  nonfinite and whitespace-only inputs; its unique ratio divides by nonempty row count.
- Numeric features use their CSV dtype; booleans are categorical; date-like strings are “other”.
  Numeric-looking strings are not transformed into floats during this audit.
- Numeric and regression summaries include count, mean, sample standard deviation, min/max and
  5th/25th/50th/75th/95th percentiles; nonfinite values are excluded from these summaries.
- Text lengths count characters and whitespace-delimited words. Missing/empty text contributes zero.
  The additional nonempty character mean excludes skipped inputs.
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

## Exact text reuse

`src/data/text_reuse.py` matches the request builder's canonical typed JSON, without target
values. It profiles every official text column and the full joint tuple of available named
fields. `reuse.csv` records nonempty/missing counts, distinct inputs, repeated copies, rows in
repeated groups, and the most common input frequencies. `text.csv` adds these to field lengths.
`summary.csv` gives per-dataset totals and joint/per-column overlap. No raw text is stored in
the reuse table. A missing field is omitted jointly, so a one-available-field joint request
shares the per-column response. Missing inputs incur no call and are not counted as reusable
nonempty text. Neither case folding nor fuzzy matching is applied.

`repeated_nonempty_rows` means nonempty slots minus distinct inputs (avoidable calls).
`rows_in_repeated_groups` includes the first occurrence too. For a value appearing five times,
these are four and five respectively. Exact input equality predicts cache reuse only with
unchanged task metadata, question and settings. Cache reuse means returning the first stored
response; it does not require a claim that repeated API executions are bit-identical.

`overview.csv` orders task/size, text availability, distinct values and per-column reuse,
nonempty text length, joint availability/reuse/length, and total extraction workload.
All text cells equal rows × text columns. Distinct text cells are unique **within each named
column**, then summed; they do not collapse equal strings across different column names.
Empty-cell percentages divide by all text cells; distinct/per-column reuse percentages divide
by nonempty cells; joint reuse divides by rows with at least one text value. Both-mode reuse
also includes identical requests shared across modes. Numeric values remain in the table/report.
The figure combines counts and percentages where useful and places all notes outside its cells.

## Long-text audit

`long_text_datasets.csv` compares p95/p99/max individual-cell and joint-row lengths;
`long_text_columns.csv` covers all official text fields. Nonempty cells/rows alone enter these
quantiles. Characters are Python string lengths; tokens remain ceil(characters/4), without
provider tokenization. Boxplots use p01/p99 whiskers, with observed maxima shown separately.
The context-limit line is a text-only screen; full request escaping/metadata can increase it.
The longest original cell per column is saved under the ignored audit cache `longest_text/`,
with dataset/column/`csv:N` identity in audit JSON. Notebook examples show beginning/middle/end
excerpts, clipped only for display. No extraction truncation policy is implied.

## Dataset issues relevant to the experiment

The pinned collection contains 821,071 rows and 90 official text columns: three binary,
seven multiclass and ten regression tasks. All official row/text/non-text counts matched the
loaded data. The generated notebook report contains the per-dataset column names and counts.

Official text includes short names, identifiers, URLs and numeric/date-like strings. Rotten
Tomatoes has 13 text fields and SciMagojr has 10; some are measurements, not prose. Review
prediction-time availability and target proxies rather than relabelling features silently.
Spotify has 114 genre classes; Data Scientist Salary has six interval-labelled classes;
Wine Review predicts 30 varieties; Mercari predicts log_price. Coded classes need factual metadata verification where their units or meaning are not evident
from the target name. Fraud=1, toxicity=True and Kickstarter=Successful are the current binary
orientations; Product Sentiment already has descriptive class names. The source-backed definitions below now make the verified units and meanings explicit.

Zomato contributes about three quarters of raw text tokens and has extreme review lengths.
Some inputs exceed the documented Jev context limit under the approximate token screen.
No automatic truncation, chunking or summarisation is implemented. Review scores may be
target proxies and warrant input-availability review. The owner selected shuffled row CV
without entity/time restrictions; this estimates that benchmark setting, not future-time or
unseen-entity deployment performance.
Audit target distributions/quantiles remain descriptive only; none enter Jev inputs.

The 30-09-2026 audit finds substantial reuse of joint inputs too: Vancouver Salaries avoids
59.36% of nonempty joint slots, Zomato 31.94%, Video Games Sales 28.20% and Data Scientist
Salary 23.78%. Across both modes, 3,985,167 nonempty slots become 2,032,351 distinct requests;
107,235 distinct joint inputs overlap per-column inputs. These counts hold for the current
untruncated text and fixed request design. They must be recomputed if that design changes.

## Extraction and experiment selection

The full exploration retains all 20 core TEXT datasets and both raw releases. Zomato
Restaurants is removed from feature creation and experiments for its extreme text burden;
Spotify Genres is removed for its 114-class output dimensionality and associated extraction
overhead. The common experiment collection therefore has 18 datasets. Selected overview
figures/tables label both as REMOVED; this does not mean files were deleted. Selection is
based on input/representation feasibility before comparative predictive results.


## Source-backed question metadata (13-10-2026)

`config/dataset_semantics.yaml` is the single maintained dictionary for the 18 selected datasets:
raw target, task, observation unit, readable task meaning, every official text column, typed raw
class values/descriptions, source URLs and unresolved facts. Pinned MulTaBench annotations use
commit `3bb95079a6146ed35d45187e98bcad5b063273a1`; their inspected source hashes are recorded.
The earlier benchmark code uses commit `77edc53ef7e8a91ea4cfdaa336b353af66e3db00` of
[sxjscience/automl_multimodal_benchmark](https://github.com/sxjscience/automl_multimodal_benchmark).

Relevant verified distinctions:

- Jigsaw uses the earlier preparation rule `target > 0.5`; its positive label is `True`.
  This documentary threshold defines the task; no row annotation is put in a request.
- Indian salary classes retain the six raw interval strings. The reproduced competition
  description specifies rupees in lakhs, in an annual-salary context. The original competition
  page is unavailable; the dataset mirror and a competition participant preserve that wording.
  Exact interval endpoint conventions remain unspecified.
- US Accidents describes traffic impact, not injuries; no invented delay boundaries or
  intermediate verbal categories replace the four original codes.
- Clothing ratings retain 1 (worst) through 5 (best). Product Sentiment keeps `Cannot Say`
  distinct from `No Sentiment`; Wine Review keeps all 30 variety strings, including aliases
  present as separate source classes, without merging them.
- Mercari preparation explicitly creates `log_price = ln(price + 1)` and `cat2`/`cat3` from
  the second/third category-path components.
- Book Price inherits an upstream documented `log10(Price + 1)` transformation, whereas its
  MulTaBench annotation only renames the target `Price in USD`. The prompt therefore describes
  log-price and does not claim the numeric values are dollars. No raw values are transformed.
- CLEAR predicts the New Dale-Chall formula, not the separate human-rated `BT Easiness` field.
- Vancouver remuneration is in Canadian dollars according to the City data dictionary.

Unresolved source details are preserved in the configuration and preparation notebook:
baby-product currency and company-field extraction conventions, CLEAR `British Words`, the
legacy movie `RatingValue` scale/provenance, salary interval endpoints, and US Accidents'
intermediate traffic-severity boundaries. Unknown units/definitions are not guessed.

Only documentary meanings enter `documented-task-v1`. Audit frequencies, target distributions,
current row labels and prediction results remain excluded. API class IDs retain original class
order and types; the human description is added alongside each raw class. The selected text
fields keep their raw names; per-column requests include only their applicable field definitions.
The previous notebook cost estimates describe the accepted `directional-v2` payload, so the
expanded question/context budget must be recomputed before paid extraction.
