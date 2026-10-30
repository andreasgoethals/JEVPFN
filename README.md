# JEVPFN

Do **label-free probabilistic text features from decision models** improve fixed tabular
predictors? Twelve frozen decision models answer one deterministic, target-aware question
about the text of each row. Their probabilities are added to the non-text features of six
tabular predictors and compared with frozen general embeddings and native text systems on
18 MulTaBench TEXT datasets.

## Status (30 October 2026)

| Stage | State |
|---|---|
| Exploration, question preparation | Complete: all 20 core datasets explored; 18 selected; question v3 frozen. |
| Experiment 0A (before extraction) | Accepted. VSC 192/192 steps passed; signed acceptance for eleven open-weight providers and three encoders. Local Jev 1.14 0A and its first-100-rows pilot passed (validation only). |
| Open-weight and encoder extraction | Enabled and submitted from frozen `954c93f`; in progress, with failures under review ([AGENTS_MEMORY](docs/AGENTS_MEMORY.md)). |
| Full Jev extraction, 0B, Experiments 1–3 | Disabled until separately reviewed and authorized. |

Exact receipts, job IDs and failures: [AGENTS_MEMORY](docs/AGENTS_MEMORY.md). Commands:
[VSC](docs/VSC.md). Never update a checkout that queued or running jobs consume.

## Research questions

| Experiment | Question | Configurations |
|---|---|---|
| 1 (RQ1) | Do decision features improve each of six fixed predictors over the non-text baseline? Twelve extractors × per-column, joint or both. | 3,996 |
| 2 (RQ2) | How do they compare with three frozen encoders (E5-small-v2, E5-large-v2, Qwen3-Embedding-0.6B), PCA-reduced per column (32) or jointly (256/128/64/32/16/8)? | 2,268 |
| 3 (RQ3) | Where do native text systems sit: TabSTAR v1.1, ConTextTab and AutoGluon-Multimodal? | 54 |

A configuration is one dataset × predictor × representation over five outer folds:
31,590 outer fits plus 15,795 binary inner fits. Experiment 1 shares its non-text baseline
across extractors; Experiment 2 has seven PCA recipes per encoder; Experiment 3 has one
original-feature setting per system. No winner is selected and no outcome-selected
combination is run. Experiment 0A (input/runtime acceptance, including robustness
diagnostics) precedes extraction; 0B (complete features, all nine predictors, full-fold
feasibility) precedes the benchmark. A synthesis notebook compares RQ1–RQ3.

## Inputs and features

Each request contains fixed task metadata, a deterministic question with the original
class options and either one named text column (**per-column**) or all available text
columns (**joint**). It never contains non-text row values, true row targets, class
frequencies, target statistics or model feedback. Class meanings come from documented
sources ([DATA_SOURCES](docs/DATA_SOURCES.md)); no generative LLM writes questions.

| Task | Features per text input |
|---|---|
| Binary | Probability supporting the fixed positive class. |
| Multiclass | All K class probabilities in the original class order. |
| Regression | Probability-weighted mean of nine ordered evidence levels, −4 … +4. |

- Probabilities are evidence scores, not calibrated target probabilities; regression scores
  carry direction, not target units. Raw responses and full vectors are retained.
- Jev serializes two decimals and Strands four. Complete vectors with a feasible rounding
  correction are projected to sum one (maximum per-entry change 0.005 and 0.00005); raw
  responses are unchanged.
- Empty text gives missing features without a call. Exact requests (context, question,
  options, named text, provider settings) are answered once and reused; a one-column joint
  request reuses its per-column response. Context overflow stops without truncation.

## Models

| Extractor | Execution |
|---|---|
| Jev 1.14.0 | Hosted TypeSafe API, run locally by the owner |
| Liquid d1-3B, Liquid d1-2B (LFM3) | Open weights, VSC B200 |
| Strands Decider 2B, Intern-Decision-4B, H2O Lightning 4B, Cloudflare Clef-flash | Open weights, VSC B200 |
| Perplexity Decider v1.1-27B, SII-Decision-27B-Full, Kev-27B, Decision3 d3-turbo | Open weights, VSC B200 (large profile) |
| Bespoke-Nimble-9B | Apache adapter on pinned Qwen3.5-9B, VSC B200 |

Pins, licences, context and prices are in `config/2. feature_creation/extractors.yaml`.
Microsoft is excluded by owner decision. All adapters keep the same canonical question,
named text and option order; native framing, calibration and wire formats differ, so raw
confidences are not interchangeable. H2O's released binary formatter floors the winning
probability at 0.801; this is preserved and disclosed. Kev uses its native pointer head,
Decision3 its prefix-bidirectional runtime. [LITERATURE_REVIEW](docs/LITERATURE_REVIEW.md)
documents each interface.

**Predictors:** CatBoost, TabPFN-3.5, TabICL v2, LimiX-2, NVIDIA Kumo Tabular and
logistic/linear regression, with fixed settings and no HPO. LimiX-2 preprocessing is
restricted to training rows (a declared adaptation of the publisher code).

**Encoders** (`config/2. feature_creation/embedding_models.yaml`): E5 uses masked mean
pooling and 512-token chunks; Qwen uses the last non-padding token and 2,048-token chunks.
Chunk vectors are token-weighted and normalized; identical text reuses its vector.
PCA is fitted on training rows only, per outer fold and binary inner fold, and stored in
PROJECT `results/pca/<encoder>/<dataset>.sqlite3`; prepare it on CPU with
`python -m src.evaluation.prepare_pca --dataset ID --encoder ENCODER`. Joint PCA imputes
missing blocks with training means; every text column keeps a missingness flag. The widest
input is Rotten Tomatoes per-column PCA32: 431 columns (433 with Kumo's category expansion),
below the 500-column guard.

**Native systems:** TabSTAR v1.1 (package 1.2.2, pinned checkpoints staged with
`python -m src.cluster.predictor_assets --execute`), ConTextTab, and AG-MM with fixed
medium-quality ELECTRA-base fusion in its own runtime. They are references, not a
replication of MulTaBench's supervised target-aware embeddings.

## Protocol and analysis

- Five-fold shuffled stratified CV (classification) or shuffled KFold (regression), all
  rows, identical folds for every arm. Binary F1 thresholds use three inner OOF splits; the
  outer test fold never selects thresholds or fits preprocessing.
- Headline metrics, fixed before results: ROC-AUC (binary), macro one-vs-rest ROC-AUC
  (multiclass), R² (regression). All other metrics, predictions, probabilities, inner-OOF
  predictions, timings and resource use are saved. Fold SD is descriptive.
- Effects are paired within dataset and predictor, including cohorts of test rows whose
  text was or was not seen in training. Optional Wilcoxon/Holm contrasts require
  prespecification (`config/7. synthesis/`). Protocol settings: `config/4. experiment_1/`.
- Limits: gains over non-text may include text-availability signal; "label-free" describes
  extraction inputs, not pretraining; no lexical, presence-only or shuffled controls.

## Workflow and notebooks

Setup → question preparation → 0A → feature creation → 0B → Experiments 1–3 → synthesis.
Folder numbers organize files; 0A runs before feature creation despite its number.
Notebooks run locally, contain no function definitions or inference, and end with the full
printed report. Figures are A4 PDFs in `output_JEVPFN/figures/<phase>/<notebook>/`, with
`figures/Captions.md` and `All Results.md`. Each header indexes its figures by question.

| Notebook | Main figures |
|---|---|
| [0 · Data exploration](<notebooks/0. exploration/01_data_exploration.ipynb>) | All 20 datasets: overview, text length, missingness, exact reuse, long text. |
| [0 · Input design](<notebooks/0. exploration/02_jev_input_design.ipynb>) | Request and feature design for every arm; regression rubric; exact requests per adapter. |
| [1 · Prepare questions](<notebooks/1. question_preparation/01_prepare_questions.ipynb>) | Request composition: fixed context against row text. |
| [2 · Feature creation](<notebooks/2. feature_creation/03_feature_creation.ipynb>) | Extraction workload and cost; predictor input widths. |
| [2 · Decision features](<notebooks/2. feature_creation/04_decision_feature_comparison.ipynb>) | Coverage, distributions and agreement once real features exist. |
| [3 · 0A evidence](<notebooks/3. experiment_0/01_before_feature_creation.ipynb>) | Runtime gates and B200 resources; pilot output profiles and agreement of all twelve models; robustness. |
| [3 · 0B evidence](<notebooks/3. experiment_0/02_after_feature_creation.ipynb>) | Representation inventory and predictor readiness. |
| [4 · Experiment 1](<notebooks/4. experiment_1/01_paired_representation_effects.ipynb>) | Coverage, effect matrices, paired effects, seen/unseen text, fit burden. |
| [5 · Experiment 2](<notebooks/5. experiment_2/01_frozen_embeddings.ipynb>) | Effect matrices, gain against PCA width, retained variance, fit burden. |
| [6 · Experiment 3](<notebooks/6. experiment_3/01_native_text_systems.ipynb>) | Coverage and native-system scores. |
| [7 · Synthesis](<notebooks/7. synthesis/01_representation_comparison.ipynb>) | Study design; decision versus embedding per task; pipelines versus native systems. |

Result figures appear only once complete saved arms exist; nothing is drawn from planned
or synthetic records. Colours and sizes come from `src/visualize/style.py`.

## Local setup

New environment only (Windows PowerShell, from this repository):

```powershell
py -3.12 -m venv --prompt JEVPFN .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebooks]"
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name jevpfn --display-name "Python 3.12 (JEVPFN)"
```

Prepare data and questions, audit them and run the notebooks:

```powershell
.\.venv\Scripts\python.exe -m src.data.prepare
.\.venv\Scripts\python.exe -m src.jev.preparation
.\.venv\Scripts\python.exe -m src.jev.question_audit
.\.venv\Scripts\python.exe -m src.utils.run_notebooks
.\.venv\Scripts\python.exe -m pytest -q
```

Preparation is deterministic and offline after download; `preparation --rebuild`
regenerates inputs transactionally without touching paid features. The audit checks every
unique payload and row/slot link. The notebook runner executes in parallel and saves
outputs into the `.ipynb` files (`--only NAME`, `--summaries-only`). Model extras live in
separate cluster runtimes, not the notebook environment. In PowerShell, check
`$LASTEXITCODE` after native commands.

## Storage

| Path | Content |
|---|---|
| `data/raw/` | 20 numbered dataset folders with provenance. |
| `data/decision_inputs/<dataset>.sqlite3` | 18 versioned question stores with row mappings. |
| `data/jev_cache/<dataset>.sqlite3` | Permanent responses, accounting and numeric feature blocks of all providers. |
| `data/jev_cache/_control/` | Permanent Jev smoke and robustness diagnostic ledgers. |
| `data/text_embeddings/[<encoder>/]<dataset>.sqlite3` | Full frozen vectors (E5-small at the top level). |
| `output_JEVPFN/<number>. <phase>/{logs,manifests,results}/` | Experiment output and 0A receipts; `results/evaluation/<dataset>.sqlite3` holds predictions, metrics, timings and resumable folds. |
| `output_JEVPFN/{figures,logs,manifests,results}/…/<phase>/<notebook>/` | Notebook figures, logs and tables. |

Rows join on dataset ID, source fingerprint and `csv:N`. Feature blocks record row IDs,
text groups, question, provider/checkpoint/settings identities and checksums; per-column
and joint blocks give the three decision arms without repeated calls. On VSC, DATA holds the
checkout and compact logs, PROJECT holds data, weights, sealed containers and large results
([VSC](docs/VSC.md)). Move results only with the validated transfer export/import, never
by copying over populated stores.

`python -m src.utils.output_cleanup` previews output cleanup; `--apply --writers-stopped`
deletes permanently (filters: `--phase`, `--kind`, `--match`, `--older-than-days`,
`--include-archives`, and `--tier data|project|both` on VSC). It never touches data,
weights, environments, containers or secrets, and refuses active jobs and linked trees.

## Execution gates

- `python -m src.cluster.experiment_zero --site local|vsc` previews 0A; `--execute` needs a
  matching frozen revision (`python -m src.cluster.freeze --record`). Slurm jobs reject
  source changes; finish a reviewed stage with `freeze --finish-stage` before reconfiguring.
- `python -m src.cluster.accept_features` signs provider acceptance only after human review
  of outputs, throughput, robustness, quota and backup; the acceptance is hash-bound to
  source, inputs, runtime and reports.
- Full Jev also needs a reviewed pilot acceptance and spending/throughput policy in its
  ignored private config. No cleanup or test dispatches API calls.
- Benchmarks require a signed 0B acceptance whose `evaluation_identity` matches the current
  implementation, settings and recipes (`python -m src.evaluation.acceptance`).

Passing finite tests does not prove the absence of bugs or full-run throughput.

## Documentation

| Document | Content |
|---|---|
| [DATA_SOURCES](docs/DATA_SOURCES.md) | Dataset provenance, text definition, semantics and caveats. |
| [LITERATURE_REVIEW](docs/LITERATURE_REVIEW.md) | Prior art, extractor interfaces and limits of interpretation. |
| [VSC](docs/VSC.md) | Cluster storage, runtimes and commands in execution order. |
| [AGENTS_MEMORY](docs/AGENTS_MEMORY.md) | Current run state, exact evidence and reusable failure lessons. |
| [CHANGELOG](docs/CHANGELOG.md) | Dated code and documentation changes. |
| [TEMPLATE](docs/TEMPLATE.md) | Canonical research template (unchanged copy). |

The project follows the [research template](https://github.com/andreasgoethals/0.-Template)
with these deviations: numbered phase folders, `output_JEVPFN`, the aggregate report names,
permanent paid-response stores and explicit local/VSC execution gates. Code is MIT;
data and weights keep their own licences. Citation: [CITATION.cff](CITATION.cff).
