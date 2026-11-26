# JEVPFN

JEVPFN studies **how decision models represent real text, when their representations help
prediction, and whether supervised adaptation improves them**.
Eighteen text-rich tabular datasets provide linked outcomes and a shared test bed.
A decision model reads one named text column or several columns together and answers a fixed question about the task. We follow its answer probabilities from the original text to a held-out prediction. This lets us test whether two models that describe the same text differently also give a tabular learner different predictive information.

Prior studies compare direct decisions on whole table rows, and existing software already feeds Jev probabilities into a tabular predictor. We give the decision models the same questions and text. We then compare their probabilities with frozen and supervised text embeddings using the same tabular predictors and held-out rows. [Typed Decision Models on Tabular Data](https://arxiv.org/abs/2610.19536) · [tab-jev](https://github.com/edamame-labs/tab-jev) · [MulTaBench](https://arxiv.org/abs/2605.10616)

## Study at a glance

| Group | Experiments | Compute |
|---|---|---|
| General | 0 qualifies models and runtimes before feature creation; data and questions; feature creation; study audit; synthesis | Qualification and extraction on VSC; notebooks locally |
| A: behaviour | A0 pilots; A1 outputs/context; A2 controlled robustness; A3 decision–encoder geometry | Local reports; controlled GPU calls and geometry CPU analysis on VSC |
| B: predictive usefulness | B0 predictor qualification; B1 decision features; B2 frozen embeddings; B3 TAR and label-trained classifiers; B4 when decision features help; B5 amount of labelled data; B6 native-text systems | Predictor fits on VSC; paired analysis locally |
| C: Strands v1 | C0 qualifies training; C1 hidden representations; C2 zero-shot, head-only, LoRA-only and LoRA + head | C1 extraction and C0/C2 training on VSC, held-out analysis locally |

The first release contains 16 open-weight decision models. Its 77,397 Group B configurations and 2,142 Strands configurations in Group C require 397,695 outer fits. Jev 1.14.0 is optional. Adding it would require 3,843 new Group B configurations (19,215 outer fits), bringing the total to 416,910 fits. Shared controls and existing predictions carry forward. Each complete release compares decision features against three references (structured inputs alone, frozen E5-small, label-trained E5-small TAR) for each of seven predictors. These 21 paired comparisons use the saved predictions on the same 18 datasets. Adding a model can change the median over decision models and its uncertainty.

Later releases can add open-weight models, hosted models or both. Each release names its parent. The job planner schedules only the added models' fits and reuses earlier predictions and shared controls. Each release records the hosted model versions and open-weight checkpoint revisions it uses. Any release can extend the original sixteen directly, whether or not Jev has run. [Decision-model releases](config/B/B1.yaml)

Seven tabular predictors use the same structured baseline and five outer folds. For regression, each decision question produces nine ordered probabilities describing the text. The tabular predictor estimates the numeric target.
The comparisons use three frozen encoders
(E5-small-v2, E5-large-v2, Qwen3-Embedding-0.6B) and supervised E5-small TAR. Both embedding
tracks retain joint PCA64 and separate PCA32 per text column. TabSTAR v1.1, ConTextTab and
AutoGluon-Multimodal are descriptive native-text system references.

## Campaign evidence

**25 November 2026, 12:12 Brussels snapshot.** The desktop holds both feature views for 259 of 288 open-weight dataset–model pairs, plus both Jev 1.14.0 views for Product Sentiment. All 520 views match their registered model revisions. The 54 frozen encoder stores, 54 PCA stores and one real TAR pilot are also available locally. All five additional open-weight providers have signed native Experiment 0 reviews. Every remaining extraction has a Slurm job ID. The repaired jobs can checkpoint during inference and KnowLine's 100,000-row Jigsaw retry completed. The current benchmark still needs a full native B0 review. C0 qualifies Strands v1 training on the VSC before production C2.

## Repository navigation

One registry, [`config/structure.yaml`](config/structure.yaml) (layout v4, 26 November 2026),
names every folder. Each experiment has exactly one name, used for its phase, config file,
output folder and notebook (`B3`: `config/B/B3.yaml`, `output_JEVPFN/B/B3/`,
`notebooks/B/B3.ipynb`). The four top-level groups follow the study order:

| Group | Folders and notebooks | Gate |
|---|---|---|
| `general` | `0` → `1_data_and_questions` → `2_feature_creation` → `3_study_audit` → `4_synthesis` | `0` before feature creation |
| `A` | `A0` → `A1` outputs → `A2` robustness → `A3` geometry → `A_synthesis` | `A0` |
| `B` | `B0` → `B1` decision features → `B2` frozen embeddings → `B3` TAR and label-trained classifiers → `B4` when decision features help → `B5` amount of labelled data → `B6` native-text systems → `B_synthesis` | `B0` |
| `C` | `C0` → `C1` Strands hidden states → `C2` learning from labels → `C_synthesis` | `C0` |

General stages keep a config folder (`config/general/<folder>/`); each experiment has one
config file whose `sections:` hold further named configurations (`B1/models`). The study
audit lists the nine C0 pilots and their local receipts. Earlier names (layouts v1–v3, such as
`experiment_1`, `rq1` or `B/exp04_decision_features`) appear only in the checksummed
[alias map](config/layout_aliases.yaml), which locates paths recorded in earlier receipts.
Paper placement is recorded under `paper:` in the registry and never in a folder name.
Existing output trees move only through the checksum-verified
[layout migration](src/utils/layout_migration.py).

## Local setup

From the local repository root in Windows PowerShell, initialize the pinned submodules and create a Python 3.12 environment. The editable install reads configuration from the checkout. The Strands recipe tests also read publisher source from the pinned DM library. Model runtimes are separate.

```powershell
git submodule update --init
py -3.12 -m venv --prompt JEVPFN .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebooks,control]"
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name jevpfn --display-name "Python 3.12 (JEVPFN)"
```

Then pin every shared package to the exact version in the VSC CPU image, so a local pass
does not fail on the cluster because of the environment. Rerun it after any other install;
`tests/test_environment_parity.py` fails with the differing packages when the venv drifts.

```powershell
.\.venv\Scripts\python.exe -m pip install @(.\.venv\Scripts\python.exe -c "from src.utils.config import load_yaml; [print(f'{k}=={v}') for k,v in load_yaml('0/control_runtime')['packages'].items()]")
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
preserving paid responses and features. The [CPU runtime specification](config/general/0/control_runtime.yaml) records the shared dependencies.

## Workflow and outputs

Prepare inputs, qualify them and the runtimes in **0**, and create features. Each group then
starts with its gate: **A0** (native presentation and geometry pilots) before A1–A3, **B0**
(predictor contracts and full-fold profiles on the complete stores) before B1–B6, and **C0**
(Strands training pilots) before C2.

| Location | Contents |
|---|---|
| `data/raw/` | Pinned source datasets and provenance |
| `data/decision_inputs/` | One versioned question SQLite store per selected dataset |
| `data/jev_cache/` | Permanent response, accounting and feature stores, plus Jev diagnostic ledgers |
| `data/decision_features/` | Verified portable snapshots of complete models for local analysis |
| `data/text_embeddings/` | Frozen vectors per encoder and dataset |
| `output_JEVPFN/<group>/<folder>/{logs,manifests}/` | Operational validation evidence and logs |
| `output_JEVPFN/<group>/<folder>/results/` | PCA, TAR, study and resumable benchmark records |
| `output_JEVPFN/figures/<group>/<name>/` | Paper figures of each notebook and their `_figures.json`, with all captions in `figures/CAPTIONS.md` |
| `output_JEVPFN/<group>/<folder>/{logs,manifests,results}/notebook/` | Notebook logs, execution receipts and reports |
| `archive/` | Git-ignored previous-layout records, beside `output_JEVPFN/` |

On VSC, DATA holds Git and compact records. GPFS PROJECT holds permanent data, models,
containers and results. Native scratch holds temporary working stores. Transfer results with the [verified export/import tool](src/cluster/transfer.py).

### Notebooks

Notebooks run locally and write paper-width PDFs and printed reports. The runner supports
`--only NAME` and `--summaries-only`. Figures are under
`output_JEVPFN/figures/<group>/<name>/`; the one report is `output_JEVPFN/All_Results.md` and the
captions are `output_JEVPFN/figures/CAPTIONS.md`. Run them with the project venv
(`.venv\Scripts\python.exe`).
Result figures appear as complete saved evidence becomes available.

The A1 notebook examines row-level vote splits, matching-class
correlations, full probability differences, confidence, text grouping and exact reuse.
Mindwell CPU checks the published inputs and writes the joint-row Parquet and a compact report.
The local notebook renders that report, including hash-verified examples, and identifies its model coverage.
Primary results require both feature blocks for every prespecified model on all 18 datasets. Missing coverage is reported for that release.

| Group | Notebooks |
|---|---|
| General | [0 qualification](notebooks/general/0.ipynb), [Data and questions](notebooks/general/1_data_and_questions.ipynb), [Features](notebooks/general/2_feature_creation.ipynb), [Study audit](notebooks/general/3_study_audit.ipynb), [Synthesis](notebooks/general/4_synthesis.ipynb) |
| A | [A0](notebooks/A/A0.ipynb), [A1 outputs](notebooks/A/A1.ipynb), [A2 robustness](notebooks/A/A2.ipynb), [A3 geometry](notebooks/A/A3.ipynb), [A synthesis](notebooks/A/A_synthesis.ipynb) |
| B | [B0](notebooks/B/B0.ipynb), [B1 decision features](notebooks/B/B1.ipynb), [B2 frozen embeddings](notebooks/B/B2.ipynb), [B3 TAR and label-trained classifiers](notebooks/B/B3.ipynb), [B4 when decision features help](notebooks/B/B4.ipynb), [B5 amount of labelled data](notebooks/B/B5.ipynb), [B6 native text](notebooks/B/B6.ipynb), [B synthesis](notebooks/B/B_synthesis.ipynb) |
| C | [C0](notebooks/C/C0.ipynb), [C1 inside the decision model](notebooks/C/C1.ipynb), [C2 learning from labels](notebooks/C/C2.ipynb), [C synthesis](notebooks/C/C_synthesis.ipynb) |

Regression uses **nine probabilities per text input**, without a mean-readout ablation. A2 uses 500 random semantic requests per dataset and one-factor changes; A3 compares all pairs of up to 500 distinct texts within each dataset and text column.
The C0 and C2 notebooks currently show plans and pending gates, not trained-model results.

## Research sources

The pinned [decision-model synthesis](dm-library/SYNTHESIS.md) and [tabular-model synthesis](tfm-library/SYNTHESIS.md) include paper summaries, source versions and implementation records. MulTaBench selected tasks on which supervised text adaptation added signal. The 18-task analysis therefore describes a selected benchmark population. Figure 3 of the source paper compares its curated text subset with the broader candidate pool. [MulTaBench, v3](https://arxiv.org/abs/2605.10616v3)

## Citation and licensing

Use [CITATION.cff](CITATION.cff) to cite this software. The [MIT licence](LICENSE) covers project code. Datasets, model weights and vendored code retain their own licences and attribution.

JEVPFN uses the [research project template](https://github.com/andreasgoethals/0.-Template).
