# Data sources and definitions

This guide documents the pinned MulTaBench inputs, text definition and question semantics.
The [README](../README.md) describes the evaluation, and the [VSC runbook](VSC.md) gives
the verification commands.

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

`config/0. exploration/datasets.yaml` records primary-source URLs/hashes, data versions/hashes and
the column list obtained from these rules. The public official Kaggle archives are downloaded
through the versioned dataset-download endpoint, using only Python's standard library. Only
`data.csv` and `metadata.json` are extracted; no arbitrary archive paths or downloaded code run.
The CSV reader follows the official pandas defaults, with `low_memory=False` for stable types.

## How the benchmark identifies text

The official summary computes text columns from the data. `src/data/official_types.py`
reproduces that rule, and the catalog freezes the resulting names for every fold.

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

- Feature counts exclude the target. Missingness follows pandas CSV parsing. General unique
  counts exclude nulls. Text reuse also excludes nonfinite and whitespace-only values.
- The audit preserves CSV numeric dtypes and treats booleans as categorical. Numerical
  summaries and regression quantiles use finite observations. Target summaries stay in
  the audit artifacts. Requests use the original class vocabulary and documentary meanings.
- Text lengths use characters and whitespace-delimited words. Approximate tokens default to
  `ceil(characters / 4)`, with separate ceilings for per-column inputs and a row total for joint
  text. Workload estimates include the compact request JSON, including metadata and escapes.
  Exact provider token audits count the fully rendered requests separately.

## Reproduction and outputs

The catalog's `directory` selects the numbered folder under `data/raw/`. Its `slug` selects
the Kaggle source. Each folder holds `data.csv`, `metadata.json` and `download.json`.
Preparation checks the recorded source, version and hashes. Frozen references under
`src/data/upstream/` are retained for inspection. CSVs retain their source licences.

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

`overview.csv` summarizes availability, lengths, distinct values and extraction workload.
Distinct cells are counted within each named column and then summed. Empty-cell rates use
all cells, per-column reuse rates use nonempty cells, and joint reuse rates use rows with
available text. Both-mode reuse includes identical requests shared across modes.

## Long-text audit

`long_text_datasets.csv` compares p95/p99/max individual-cell and joint-row lengths;
`long_text_columns.csv` covers all official text fields. Nonempty cells/rows alone enter these
quantiles. Characters are Python string lengths; tokens remain ceil(characters/4), without
provider tokenization. Boxplots use p01/p99 whiskers, with observed maxima shown separately.
The context-limit line is a text-only screen; full request escaping/metadata can increase it.
The longest original cell per column is saved under the ignored audit cache `longest_text/`,
with dataset/column/`csv:N` identity in audit JSON. Notebook examples show beginning/middle/end
excerpts, clipped only for display. No extraction truncation policy is implied.

## Extraction and experiment selection

Exploration retains all 20 core text datasets. The 18 benchmark datasets exclude Zomato
Restaurants for its extreme text burden and Spotify Genres for its 114-class output and
extraction overhead. These choices were made before comparative predictive results.

## Dataset issues relevant to the experiment

The pinned collection contains 821,071 rows and 90 official text columns: three binary,
seven multiclass and ten regression tasks. All official row/text/non-text counts matched the
loaded data. The generated notebook report contains the per-dataset column names and counts.

Official text includes short names, identifiers, URLs and numeric/date-like strings. Rotten
Tomatoes has 13 text fields and SciMagojr has 10; some are measurements, not prose. Review
prediction-time availability and target proxies rather than relabelling features silently.
Spotify has 114 genre classes; Data Scientist Salary has six interval-labelled classes;
Wine Review predicts 30 varieties; Mercari predicts log_price. Binary orientations are
Fraud=1, toxicity=True and Kickstarter=Successful; Product Sentiment has descriptive class
names. Units and meanings of coded classes are verified in the source-backed metadata below.

Zomato contributes about three quarters of raw text tokens and has extreme review lengths.
The approximate screen flags some inputs against the Jev context limit. Decision extraction
rejects overflow. Frozen encoders cover long inputs in chunks. Review scores can be target
proxies and warrant prediction-time availability review.

The 30-09-2026 audit finds substantial reuse of joint inputs too: Vancouver Salaries avoids
59.36% of nonempty joint slots, Zomato 31.94%, Video Games Sales 28.20% and Data Scientist
Salary 23.78%. Across both modes, 3,985,167 nonempty slots become 2,032,351 distinct requests;
107,235 distinct joint inputs overlap per-column inputs. These are historical all-20-dataset
counts, not the selected 18-dataset extraction workload. Recompute them if the design changes.

## Source-backed question metadata

`config/1. question_preparation/dataset_semantics.yaml` is the single maintained dictionary for the 18 selected datasets:
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

Metadata verified 13 October 2026. Only documentary meanings enter `documented-task-v3`; source URLs remain in question-bank
provenance and are excluded from the model payload. Version 3 adds quotation marks around
binary class descriptions and uses "a lower or higher value for" to support singular and
plural regression target names. Target meanings, options, class order and the nine-level
rubric are unchanged. Earlier versions remain readable for historical
reproduction. Audit frequencies, target distributions,
current row labels and prediction results remain excluded. API class IDs retain original class
order and types; the human description is added alongside each raw class. The selected text
fields keep their raw names; per-column requests include only their applicable field definitions.
Selected-dataset budgets read the frozen prepared question version, including definitions of
nonempty fields. Removed datasets retain an audit-only template for workload comparison.
Measured hosted billing is recorded in [AGENTS_MEMORY](AGENTS_MEMORY.md).

Non-text ISO date columns are recognized on each training partition and converted to UTC
epoch seconds for the numeric predictors. Test dates use that same conversion rather than
an unseen-category code. Missing dates stay missing. This preserves, but does not repair,
Kickstarter's upstream 1970/nanosecond-formatted timestamps; no duration feature is invented.
Unseen nominal categories still map to missing for numeric-only adapters.

SciMago's comma-formatted measurements and Rotten Tomatoes' count fields remain official
text. Publisher, province and brand can provide categorical signal. The common structured
baseline controls for text availability. CLEAR's formula-related non-text inputs and
Montgomery's previous salary can leave little headroom for text gains.
Prediction-time availability depends on the intended use. Jigsaw includes comment reaction
counts, Women's Clothing includes recommendation and feedback fields, and US Accidents
includes end times. A prospective use case would need to establish when these fields become
available and prespecify which inputs it can use.

Exact text repeats can cross the selected shuffled row folds. This is not target leakage
in extraction, but it changes the interpretation of gains: repeated names/categories may
support memorization as well as useful semantic transfer. Evaluation records per-row
training-text overlap and conditional metrics without changing splits or fitting models
again. Entirely empty inputs form their own group, never a repeated semantic text group.
For example, the pinned Women's Clothing data has 666 entirely empty joint inputs but
only five nonempty rows in repeated joint groups. Unseen-joint and all-columns-unseen
results answer different questions and both retain the outer-training threshold.
