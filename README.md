# JEVPFN

JEVPFN studies **how decision models represent real text, when their representations help
prediction, and whether supervised adaptation improves them**.
Eighteen text-rich tabular datasets provide linked outcomes and a shared test bed.
The contribution connects model behaviour to predictive utility; it does not assume that
confident or agreeing models are correct.

## Study at a glance

| Research question | Experiments | Compute |
|---|---|---|
| RQ1: behaviour | Distributions/agreement, joint versus separate context, controlled equivalent-input variations | Mostly existing outputs locally; new controlled variants on VSC |
| RQ2: predictive usefulness | Decisions, frozen embeddings, supervised TAR, native-text systems, regression readout; when text helps | Predictor fits on VSC; paired analysis locally |
| RQ3: adaptation | Strands 2B calibration, head-only and LoRA plus head | Fold-local training on VSC; held-out analysis locally |

RQ2 has **6,156 configurations / 30,780 outer fits**, including 270 appendix mean-readout
ablations. The four main comparisons contribute 5,886 configurations; historical
Experiments 1–4 retain their machine identifiers. RQ3 starts with four training pilots
before a 90-run expansion. These are configured designs, not completed experiments.
See [Experiments](docs/EXPERIMENTS.md) for counts and qualification gates.

Six structured predictors compare features from fifteen extractors. Mammoth-1 and
Drex Max now have pinned native adapters; their real 0A acceptance is still required.
The comparisons use three frozen encoders
(E5-small-v2, E5-large-v2, Qwen3-Embedding-0.6B) and supervised E5-small TAR. Both embedding
tracks retain joint PCA64 and separate PCA32 per text field. TabSTAR v1.1, ConTextTab and
AutoGluon-Multimodal are descriptive native-text system references.

## Current evidence

**16 November 2026, live VSC inspection:** ten original open-weight providers have completed
all eighteen dataset jobs and publication/release; all 54 frozen encoder and 54 PCA jobs
previously passed. Liquid d1-2B staging recovered from a publication-lock failure; its
eighteen remaining dataset jobs were all submitted and queued on Mindwell at 10:47 CET.
GLiDE is locally integrated but still
needs separate real 0A acceptance and extraction. Jev full coverage remains budget-limited.
One real TAR pilot and the linear/CatBoost contracts passed; full 0B and the benchmark
have not passed or run. See [dated evidence](docs/AGENTS_MEMORY.md) for exact IDs and updates.

## Repository navigation

`config/` has `general/`, `rq1/`, `rq2/`, `rq3/`. Each RQ has a matching notebook subfolder,
with one report per experiment. General has three reports: data, features and validation.
Scientific results and figures use RQ folders. Existing results require an idle verified
layout migration; operational extraction, logs and acceptance records retain their paths. Source fingerprints change; completed scientific artifacts and
recovery references remain intact, and new evaluation code requires fresh 0B evidence.
[Model choices and evidence](docs/LITERATURE_REVIEW.md) · [Full design](docs/EXPERIMENTS.md)

## Local setup

From the repository root in Windows PowerShell, create a Python 3.12 environment once.
These dependencies support local inspection and tests; model runtimes are separate.

```powershell
py -3.12 -m venv --prompt JEVPFN .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebooks,control]"
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name jevpfn --display-name "Python 3.12 (JEVPFN)"
```

Prepare the pinned data and questions, audit the inputs, and run the notebooks.

```powershell
.\.venv\Scripts\python.exe -m src.data.prepare
.\.venv\Scripts\python.exe -m src.jev.preparation
.\.venv\Scripts\python.exe -m src.jev.question_audit
.\.venv\Scripts\python.exe -m src.utils.run_notebooks
.\.venv\Scripts\python.exe -m pytest -q
```

Check `$LASTEXITCODE` after each command and stop on failure. These commands do not launch
paid API calls or VSC inference. Preparation reuses verified downloads.
`src.jev.preparation --rebuild` regenerates the question stores transactionally while
preserving paid responses and features. For the full shared package lock and the
Windows/Linux interpreter difference, see [CPU environment parity](docs/VSC.md#cpu-environment-parity).

## Workflow and outputs

Prepare inputs, qualify them and the runtimes in **0A**, create features, then run **0B**
on the complete stores and predictor runtimes. Continue with RQ1, RQ2 and RQ3 in that
order. New controlled probes and Strands training require their own targeted pilot gates.
The [VSC runbook](docs/VSC.md) gives commands and acceptance criteria. The
[operational record](docs/AGENTS_MEMORY.md) preserves dated receipts and recovery history.

| Location | Contents |
|---|---|
| `data/raw/` | Pinned source datasets and provenance |
| `data/decision_inputs/` | One versioned question SQLite store per selected dataset |
| `data/jev_cache/` | Permanent response, accounting and feature stores, plus Jev diagnostic ledgers |
| `data/decision_features/` | Verified portable snapshots of complete models for local analysis |
| `data/text_embeddings/` | Frozen vectors per encoder and dataset |
| `output_JEVPFN/<phase>/` | Operational validation evidence and logs |
| `output_JEVPFN/rq2/<phase>/results/` | PCA, TAR and resumable benchmark records |
| `output_JEVPFN/figures/<general-or-rq>/` | Paper figures, with combined captions in `figures/Captions.md` |
| `output_JEVPFN/manifests/notebooks/<general-or-rq>/` | Figure registries and notebook execution receipts |

On VSC, DATA holds Git and compact records. GPFS PROJECT holds permanent data, models,
containers and results. Native scratch holds temporary working stores. Use the
[verified export/import workflow](docs/VSC.md#6-utilization-and-transfer) to move results.

### Notebooks

Notebooks run locally and write paper-width PDFs and printed reports. The runner supports
`--only NAME` and `--summaries-only`. Figures and captions are under
`output_JEVPFN/figures/`, and the aggregate report is `output_JEVPFN/All Results.md`.
Result figures appear as complete saved evidence becomes available.

The RQ1 Outputs and Context notebooks examine row-level vote splits, matching-class
correlations, full probability differences, confidence, text grouping and exact reuse.
They include hash-verified source examples and export all joint rows to compressed
Parquet. Only fully downloaded extractors enter comparisons; agreement is not accuracy.

| Stage | Notebooks |
|---|---|
| General | [Data](notebooks/general/01_data.ipynb), [Features](notebooks/general/02_features.ipynb), [Validation](notebooks/general/03_validation.ipynb) |
| RQ1 | [Outputs](notebooks/rq1/01_outputs.ipynb), [Context](notebooks/rq1/02_context.ipynb), [Robustness](notebooks/rq1/03_robustness.ipynb) |
| RQ2 | [Decisions](notebooks/rq2/01_decisions.ipynb), [Embeddings](notebooks/rq2/02_embeddings.ipynb), [TAR](notebooks/rq2/03_tar.ipynb), [Native text](notebooks/rq2/04_native.ipynb), [When text helps](notebooks/rq2/05_when_text_helps.ipynb) |
| RQ3 | [Calibration](notebooks/rq3/01_calibration.ipynb), [Decision head](notebooks/rq3/02_head.ipynb), [LoRA](notebooks/rq3/03_lora.ipynb) |

Regression uses **nine probabilities per text input**; the mean comparison sits in the
RQ2A appendix. RQ1C specifies 500 random source rows per dataset and one-factor changes.
RQ3 notebooks currently show plans and pending gates, not trained-model results.

## Documentation

| Document | Purpose |
|---|---|
| [Experiments](docs/EXPERIMENTS.md) | Scientific design, grid, evaluation and qualification gates |
| [Data sources](docs/DATA_SOURCES.md) | Dataset selection, text rules and question semantics |
| [Literature review](docs/LITERATURE_REVIEW.md) | Evidence, model choices and interpretation |
| [TAR](docs/TAR.md) | Upstream provenance and explicit project differences |
| [VSC operations](docs/VSC.md) | Access, storage, runtimes, commands and recovery |
| [Operational evidence](docs/AGENTS_MEMORY.md) | Dated results, receipts and unresolved validation |
| [Changelog](docs/CHANGELOG.md) | Historical implementation changes |
| [Agent instructions](AGENTS.md) | Authorization boundaries and maintenance rules |

## Citation, licensing and template

Use [CITATION.cff](CITATION.cff) to cite this software. Project code is [MIT licensed](LICENSE);
datasets, weights and vendored code retain their own licences and attribution.

The project follows the [research template](docs/TEMPLATE.md) with an approved RQ-based scientific layout and retained numbered operational folders,
`output_JEVPFN`, project-specific aggregate report names, permanent paid-response stores,
and separate local/VSC execution gates.
The project also retains its staging-root/publication interface and its explicit destructive
output-reset helper. Unlike routine template cleanup, that helper can remove result files;
use it only for an authorized reset with required backups. Automatic desktop job monitoring is disabled;
current chat and platform instructions govern any requested monitoring.
