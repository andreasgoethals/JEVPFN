# JEVPFN

JEVPFN studies **how decision models represent real text, when their representations help
prediction, and whether supervised adaptation improves them**.
Eighteen text-rich tabular datasets provide linked outcomes and a shared test bed.
A decision model reads a named text field or a group of fields and answers a fixed question about the task. We follow its answer probabilities from the original text to a held-out prediction. This lets us test whether two models that describe the same text differently also give a tabular learner different predictive information.

Prior studies compare direct decisions on whole table rows, and existing software already feeds Jev probabilities into a tabular predictor. We give the decision models the same questions and text. We then compare their probabilities with frozen and supervised text embeddings using the same tabular predictors and held-out rows. [Typed Decision Models on Tabular Data](https://arxiv.org/abs/2610.19536) · [tab-jev](https://github.com/edamame-labs/tab-jev) · [MulTaBench](https://arxiv.org/abs/2605.10616)

## Study at a glance

| Research question | Experiments | Compute |
|---|---|---|
| RQ1: behaviour | Outputs/context, controlled robustness, decision–encoder geometry | Local reports; controlled GPU calls and geometry CPU analysis on VSC |
| RQ2: predictive usefulness | Decisions, frozen embeddings, supervised TAR, native-text systems; when decision features help | Predictor fits on VSC; paired analysis locally |
| RQ3: adaptation | Strands 2B calibration, head-only, LoRA and full adaptation | Fold-local training on VSC; held-out analysis locally |

RQ2 has **6,534 configurations / 32,670 outer fits** across four comparisons; historical
Experiments 1–4 retain their machine identifiers. RQ3 is a design proposal awaiting review. Its notebooks describe candidate comparisons.
The [study configuration](config/) defines the comparisons and qualification checks.

Six tabular predictors use a shared structured baseline and features from 17 decision models: hosted Jev and 16 open-weight releases. For regression, each question produces nine probabilities over a signed evidence scale. These describe the text; the tabular predictor estimates the numeric target.
The comparisons use three frozen encoders
(E5-small-v2, E5-large-v2, Qwen3-Embedding-0.6B) and supervised E5-small TAR. Both embedding
tracks retain joint PCA64 and separate PCA32 per text field. TabSTAR v1.1, ConTextTab and
AutoGluon-Multimodal are descriptive native-text system references.

## Current evidence

**18 November 2026.** The VSC has published all eighteen datasets for each of the eleven
original open-weight decision models. All 54 frozen encoder and 54 fold-PCA stores are
also verified locally. Mammoth, Drex Max and GLiDE passed the production GPU checks and
eighteen-dataset pilots. Review 11819179 accepted all three and the extraction dispatcher
started on 18 November. KnowLine-31B and Sieve-31B-v3 have verified weights and sealed Mindwell containers. Both passed cross-node storage checks and exact token audits on all 18 datasets. Each B200 contract is followed by a dataset pilot. Full Jev remains incomplete.

One real TAR pilot and the linear/CatBoost checks passed. Full 0B and the benchmark
remain ahead.

## Repository navigation

`config/` has `general/`, `rq1/`, `rq2/`, `rq3/`. Each RQ has a matching notebook subfolder,
with one report per experiment. General has three reports: data, features and validation.
Scientific results and figures use RQ folders. Existing results require an idle verified
layout migration; operational extraction, logs and acceptance records retain their paths. Source fingerprints change; completed scientific artifacts and
recovery references remain intact, and new evaluation code requires fresh 0B evidence.

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
preserving paid responses and features. The [CPU runtime specification](config/general/3.%20experiment_0/control_runtime.yaml) records the shared dependencies.

## Workflow and outputs

Prepare inputs, qualify them and the runtimes in **0A**, create features, then run **0B**
on the complete stores and predictor runtimes. Continue with RQ1, RQ2 and RQ3 in that
order. New controlled probes and Strands training require their own targeted pilot gates.

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
containers and results. Native scratch holds temporary working stores. Transfer results with the [verified export/import tool](src/cluster/transfer.py).

### Notebooks

Notebooks run locally and write paper-width PDFs and printed reports. The runner supports
`--only NAME` and `--summaries-only`. Figures and captions are under
`output_JEVPFN/figures/`, and the aggregate report is `output_JEVPFN/All Results.md`.
Result figures appear as complete saved evidence becomes available.

The RQ1 Outputs notebook examines row-level vote splits, matching-class
correlations, full probability differences, confidence, text grouping and exact reuse.
Mindwell CPU checks the published inputs and writes the joint-row Parquet and a compact report.
The local notebook renders that report, including hash-verified examples, and identifies its model coverage.
Only extractors with both feature blocks on all 18 datasets enter the comparisons.

| Stage | Notebooks |
|---|---|
| General | [Data](notebooks/general/01_data.ipynb), [Features](notebooks/general/02_features.ipynb), [Validation](notebooks/general/03_validation.ipynb) |
| RQ1 | [Outputs](notebooks/rq1/01_outputs.ipynb), [Robustness](notebooks/rq1/02_robustness.ipynb), [Geometry](notebooks/rq1/03_geometry.ipynb) |
| RQ2 | [Decisions](notebooks/rq2/01_decisions.ipynb), [Embeddings](notebooks/rq2/02_embeddings.ipynb), [TAR](notebooks/rq2/03_tar.ipynb), [Native text](notebooks/rq2/04_native.ipynb), [When decision features help](notebooks/rq2/05_when_text_helps.ipynb) |
| RQ3 | [Calibration](notebooks/rq3/01_calibration.ipynb), [Decision head](notebooks/rq3/02_head.ipynb), [LoRA](notebooks/rq3/03_lora.ipynb), [Full adaptation](notebooks/rq3/04_full.ipynb) |

Regression uses **nine probabilities per text input**, without a mean-readout ablation. RQ1B uses 500 random semantic requests per dataset and one-factor changes; RQ1C compares all pairs of up to 500 texts per dataset/field.
RQ3 notebooks currently show plans and pending gates, not trained-model results.

## Citation and licensing

Use [CITATION.cff](CITATION.cff) to cite this software. The [MIT licence](LICENSE) covers project code. Datasets, model weights and vendored code retain their own licences and attribution.

JEVPFN uses the [research project template](https://github.com/andreasgoethals/0.-Template).
