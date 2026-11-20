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

One registry, [`config/structure.yaml`](config/structure.yaml), names every folder. `config/`,
`notebooks/` and `output_JEVPFN/` share the same folder per stage, in this order:

| Folder | Content | Internal phase IDs |
|---|---|---|
| `00_data` | Data exploration and question preparation | `exploration`, `question_preparation` |
| `01_exp0a` | Experiment 0A, qualification before feature creation | `experiment_0` |
| `02_feature_creation` | Feature creation | `feature_creation` |
| `03_exp0b` | Experiment 0B, qualification after feature creation | `experiment_0` (`after_extraction` records) |
| `exp01_outputs` … `exp03_geometry` | Decision outputs, robustness, geometry | `rq1`, routed by study |
| `exp04_decision_features` … `exp07_native_text` | Decision features, frozen embeddings, TAR, native text | `experiment_1` … `experiment_4` |
| `exp08_when_text_helps` | When decision features help | `rq2` |
| `exp10_adaptation` | Supervised Strands adaptation (Exp 10) | `rq3` |
| `synthesis` | Cross-experiment synthesis and paper figures | `synthesis` |

Exp 9 is reserved for decision-model representations (label-free decision vectors); it is
planned, not implemented, and has no folder yet. Each folder has at most one notebook,
`notebooks/<folder>.ipynb`.

Internal phase IDs stay the keys of hashes, recipes, receipts and manifests; previous config
keys (`experiment_1/main`, `rq1/behavior`) remain aliases. Paper placement is recorded under
`paper:` in the registry and never in a folder name. Existing output trees move only through
the checksum-verified [layout migration](src/utils/layout_migration.py).

## Local setup

From the repository root in Windows PowerShell, create a Python 3.12 environment once.
These dependencies support local inspection and tests; model runtimes are separate.

```powershell
py -3.12 -m venv --prompt JEVPFN .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebooks,control]"
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name jevpfn --display-name "Python 3.12 (JEVPFN)"
```

Then pin every shared package to the exact version in the VSC CPU image, so a local pass
does not fail on the cluster because of the environment. Rerun it after any other install;
`tests/test_environment_parity.py` fails with the differing packages when the venv drifts.

```powershell
.\.venv\Scripts\python.exe -m pip install @(.\.venv\Scripts\python.exe -c "from src.utils.config import load_yaml; [print(f'{k}=={v}') for k,v in load_yaml('experiment_0/control_runtime')['packages'].items()]")
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
preserving paid responses and features. The [CPU runtime specification](config/01_exp0a/control_runtime.yaml) records the shared dependencies.

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
| `output_JEVPFN/<folder>/{logs,manifests}/` | Operational validation evidence and logs |
| `output_JEVPFN/<folder>/results/` | PCA, TAR, study and resumable benchmark records |
| `output_JEVPFN/figures/<notebook>/` | Paper figures of each notebook and their `_figures.json`, with all captions in `figures/CAPTIONS.md` |
| `output_JEVPFN/<folder>/{logs,manifests,results}/notebook/` | Notebook logs, execution receipts and reports |
| `archive/` | Git-ignored previous-layout records, beside `output_JEVPFN/` |

On VSC, DATA holds Git and compact records. GPFS PROJECT holds permanent data, models,
containers and results. Native scratch holds temporary working stores. Transfer results with the [verified export/import tool](src/cluster/transfer.py).

### Notebooks

Notebooks run locally and write paper-width PDFs and printed reports. The runner supports
`--only NAME` and `--summaries-only`. Figures are under
`output_JEVPFN/figures/<notebook>/`; the one report is `output_JEVPFN/All_Results.md` and the
captions are `output_JEVPFN/figures/CAPTIONS.md`. Run them with the project venv
(`.venv\Scripts\python.exe`).
Result figures appear as complete saved evidence becomes available.

The RQ1 Outputs notebook examines row-level vote splits, matching-class
correlations, full probability differences, confidence, text grouping and exact reuse.
Mindwell CPU checks the published inputs and writes the joint-row Parquet and a compact report.
The local notebook renders that report, including hash-verified examples, and identifies its model coverage.
Only extractors with both feature blocks on all 18 datasets enter the comparisons.

| Stage | Notebooks |
|---|---|
| Preparation and qualification | [Data](notebooks/00_data.ipynb), [Features](notebooks/02_feature_creation.ipynb), [0B validation](notebooks/03_exp0b.ipynb) |
| Exp 1–3 (RQ1) | [Outputs](notebooks/exp01_outputs.ipynb), [Robustness](notebooks/exp02_robustness.ipynb), [Geometry](notebooks/exp03_geometry.ipynb) |
| Exp 4–8 (RQ2) | [Decisions](notebooks/exp04_decision_features.ipynb), [Embeddings](notebooks/exp05_frozen_embeddings.ipynb), [TAR](notebooks/exp06_tar.ipynb), [Native text](notebooks/exp07_native_text.ipynb), [When decision features help](notebooks/exp08_when_text_helps.ipynb) |
| Exp 9 | Decision-model representations: planned, not implemented |
| Exp 10 (RQ3) | [Adaptation](notebooks/exp10_adaptation.ipynb): calibration, decision head, LoRA and full adaptation in one notebook |

Regression uses **nine probabilities per text input**, without a mean-readout ablation. RQ1B uses 500 random semantic requests per dataset and one-factor changes; RQ1C compares all pairs of up to 500 texts per dataset/field.
The RQ3 notebook currently shows plans and pending gates, not trained-model results.

## Citation and licensing

Use [CITATION.cff](CITATION.cff) to cite this software. The [MIT licence](LICENSE) covers project code. Datasets, model weights and vendored code retain their own licences and attribution.

JEVPFN uses the [research project template](https://github.com/andreasgoethals/0.-Template).
