# JEVPFN

Does **probabilistic text information from decision models improve tabular prediction**?
This research codebase explores all 20 core MulTaBench TEXT datasets and uses 18 for
extraction/evaluation. Zomato Restaurants and Spotify Genres remain in exploration but
are excluded from experiments for text burden and 114-class feature size.

**Stage: Experiment 0A setup, not full extraction.** The latest downloaded output, checked
22 October, confirms sealed Liquid, Strands and Perplexity containers, alongside
working Clef/Intern containers. All five decision checkpoints and three embedding checkpoints
were staged. Token job 62384579 completed all five providers with zero input overflows.
The three encoder jobs 62384604/62384609/62384611 failed in the older tokenizer code;
the five decision contracts remain queued with reason `Priority` in the owner's 22 October
snapshot. Historical Clef/Intern GPU contracts passed; fresh decision and encoder GPU
evidence is still required. Queue submission is not successful inference.
A reproduced Transformers 5 tokenizer incompatibility has been fixed: embedding chunks now
use the publisher's backend post-processor, preserving E5 CLS/SEP and Qwen end-of-text.
Updated encoder GPU contracts must pass before any full embedding extraction.
These local corrections are not available to VSC until the owner publishes them and updates
an idle checkout. The owner now excludes all A100 allocations: every GPU submission
defaults to wICE H100, and runtime guards reject A100 overrides. Cancel only JEVPFN jobs
before updating the VSC checkout; fresh H100 compatibility tests are still required.
The exhaustive question-v3 audit passed all 18 datasets, 1,698,127 unique inputs and
3,294,340 row/input links. Clef/Intern sensitivity probes show real prompt/option-order
effects that require review; an interface pass is not a claim of robustness.
Jev passed three live task probes and 15 synthetic probes. Account limits are now owner
confirmed (80 RPS, 100,000 TPS). The 13 real-text probes completed with sensitivity
to wording/injected instructions; the dataset pilot exposed rounded probability sums.
The decoder now permits only bounded two-decimal normalization, preserving raw responses.
The resumed local dataset pilot completed 288 requests across all 18 datasets, with
238,111 reported input tokens and USD 0.010000662 recorded cost. This is a small diagnostic
sample, not full feature extraction or final acceptance. Predictor runtime tests remain pending. Historical evidence is in
[AGENTS_MEMORY](docs/AGENTS_MEMORY.md); a final frozen 0A run will supersede setup evidence.
The owner handles Git commits/pushes. Full extraction and benchmark switches stay disabled.
Compact prediction stores, source/input/runtime-bound acceptance, prompt diagnostics and
checked batching passed local tests. Remaining VSC runtimes, reviewed diagnostics and a
fresh frozen 0A campaign are required before full extraction.

## Research contract

Extractors receive fixed task metadata, a deterministic question and either one named
text column or all available text columns jointly. They receive **no non-text row values,
true row target, class frequencies, target statistics or model feedback**. Class meanings
are documented from dataset sources; questions do not require a generative LLM.

| Task | Main numeric features for each text input |
|---|---|
| Binary | One probability supporting the fixed positive class over the other class. |
| Multiclass | All K class probabilities, in a fixed class order. |
| Regression | Expected direction over nine ordered levels, −4 to +4. |

Binary evidence probabilities are not assumed to be calibrated target probabilities.
Regression scores express directional evidence, not target units. Full responses and
probability vectors are retained. Jev sometimes reports rounded vectors whose sum is 0.99
or 1.01. Only complete two-decimal vectors with a feasible rounding correction are
projected to the nearest sum-one vector (each entry changes by at most 0.005). Raw
responses remain unchanged; this is a recorded decoding policy, not recovered precision. The rubric distinguishes indirect hints, qualified
support, clear direct evidence and explicit supported comparisons; zero favors neither direction.

Empty text produces missing numeric features without an inference call. Exact request
reuse includes context, question/options, named text and provider/runtime settings.
The first verified response is cached. A one-column joint request reuses its identical
per-column request; duplicate numeric columns are included only once in combined views.

## Models and comparisons

| Extractor | Execution |
|---|---|
| Jev 1.13.0 | TypeSafe API, locally |
| Liquid d1-3B | Open weights, VSC |
| Strands Decider 2B | Open weights, VSC |
| Perplexity Decider v1.1-27B | Open weights, VSC |
| Cloudflare Clef-flash | Open weights, VSC |
| Intern-Decision-4B | Open weights, VSC |

Microsoft is excluded by owner decision. Pins, licences, context and price references
are in `config/2. feature_creation/extractors.yaml`. Open weights have no vendor token
fee; GPU credits and licence conditions still apply. Compatibility requires real testing.

The six main predictors are **CatBoost, TabPFN-3.5, TabICL v2, LimiX-2, NVIDIA Kumo Tabular,
and logistic/linear regression**. All use the same rows and folds. The approved experiments are:

| Experiment | Comparison |
|---|---|
| 0A | Before extraction: local Jev interfaces/pilot; VSC storage, exact decision token counts, five decision runtimes and three embedding encoders. |
| 0B | After extraction: complete caches/row mappings; all nine predictors, PCA widths/scopes, native-text augmentation, resume and full-fold feasibility. |
| 1 | Each of six predictors: non-text alone, then joint, per-column and both decision features for each of six extractors. |
| 2 | Six predictors, three frozen encoders: per-column PCA32 and joint PCA256/128/64/32/16/8. No decision features. |
| 3 | TabSTAR, ConTextTab and AG-MM on original text and non-text features. |
| 4 | Label-blind question/context/option-order sensitivity; analysis reuses explicit real diagnostic reports. |
| 5 | Presence/statistics, shuffled decision features, probability/uncertainty controls; separate descriptive winner-selected combinations. |

Experiments 1/2/3 contain **2,052 / 2,268 / 54 configurations**: 21,870 outer fits,
plus 10,935 binary inner fits. A configuration means one dataset/predictor/representation;
it has five outer folds. No expensive HPO is performed. Experiment 1 shares its non-text
baseline across extractors. Experiment 2 has seven PCA settings, not their Cartesian
product with every decision model. Experiment 3 has one original-feature setting per model.

Optional concatenation of all six decision models is exposed with
`--control combined_extractors` (324 configurations before width exclusions).
Experiment 5's later combinations use one reviewed decision extractor/mode and one
reviewed encoder/PCA recipe, plus the three native-text models: 162 additional
configurations. Record the selected recipes, result-file checksum and selection rationale
in `config/4. experiment_1/followup_selection.yaml`. Choosing them from test results makes
those follow-ups **descriptive/exploratory**; code records that status and excludes them
from confirmatory testing. They remain absent from the executable grid until selected.
The primary experiments do not depend on a selected winner.

The encoders are **E5-small-v2 (384 dimensions), E5-large-v2 (1,024), and
Qwen3-Embedding-0.6B (1,024)**. Immutable revisions, prefix/pooling, licences and batch sizes
are in `config/2. feature_creation/embedding_models.yaml`. E5 uses masked mean pooling;
Qwen uses the last nonpadding token. All encoders are frozen and run on VSC. Identical
text reuses its vector; missing text stays missing. Long text is covered by nonoverlapping
chunks (512 tokens for E5; a configured 2,048 for Qwen, below its native 32,768 window).
Chunk vectors are content-token-weighted then normalized. This is a declared coverage
policy, not a claim of equivalence to encoding the whole text in one window.

PCA uses **32 components per original text column**, or **256/128/64/32/16/8 total
components** after concatenating the text vectors. These compare both grouping and
compression budget; no setting is claimed optimal. Neither reduces non-text or decision
features. Joint PCA imputes missing blocks with training coordinate means; every text
column retains its missingness flag. Validation/test rows transform through the same
training projection. Insufficient training rank leaves unused components missing.

Prepare PCA on a CPU allocation with `python -m src.evaluation.prepare_pca --dataset ID
--encoder ENCODER`. It prepares the exact five outer-training partitions and, for binary
tasks, the three inner-training partitions per fold. Numeric parameters are stored in
one PROJECT `results/pca/<encoder>/<dataset>.sqlite3` file and reused across predictors.
Cache identity includes training row IDs, encoder artifact, PCA settings, implementation
and scikit-learn version. Missing/stale preparations stop the benchmark, rather than
fitting PCA while a GPU waits. These parameters are rebuildable output, not paid features.

The widest Experiment 2 input is Rotten Tomatoes with per-column PCA32: **431 columns**
(13×32 components, 13 missingness flags, two non-text columns). Adding one provider's
separate and joint features later gives 445; even a conservative bound on Kumo's category
expansion is 447. The optional all-six-provider combination reaches 545 on Wine Reviews,
so Kumo refuses that arm. The recipe's 500-column guard prevents silent feature selection.
Other models still require checkpoint/resource checks; width compatibility does not prove
full-fold feasibility. AG-MM uses fixed medium-quality ELECTRA-base/fusion, no HPO or
ensemble, in its separate runtime. Actual native-model compatibility remains unverified.

Use five-fold shuffled stratified CV for binary/multiclass classification and shuffled
KFold for regression. Binary F1 thresholds use three inner out-of-fold splits, followed
by refitting on the full outer-training fold. The outer test fold never selects thresholds
or fits preprocessing. Use all rows and fixed model settings, with no HPO or silent row/
feature sampling. Full-fold feasibility must be tested before enabling the benchmark.

Save individual predictions, binary inner-OOF predictions, metrics, undefined-metric reasons,
timings, resource use and artifact identities. Analyze paired feature effects within each
predictor, including seen/unseen text cohorts. Decision and embedding arms also compare
with the same predictor using missing-text indicators only when that control is available.
Operational output folders do not change predictor/checkpoint comparison identities. Fold SD is descriptive, not a standard error.
Headline metrics and optional dataset-level Wilcoxon/Holm contrasts remain unapproved until
before comparative outcomes are reviewed. Exact protocol/settings live in
`config/4. experiment_1/`; optional analysis is in `config/8. experiment_5/`.

Extraction timings separate new inference, cache reuse and failures. Jev records client
setup, rate-limit waiting and API round trips; open-weight models and encoders record loading
and synchronized inference batches. Predictor records retain initialization, inner/final
fit and prediction times. Detailed extraction timing journals use one SQLite file per run
under the phase's PROJECT `results/timings/`; compact summaries use DATA `manifests/timings/`.
All existing classification/regression metrics are saved, not just AUC; full probabilities
and row-level predictions remain available for subsequent metrics and paired analyses.
Cheap text statistics, permutation controls and retained-probability ablations are also
available. Public-data pretraining overlap is a limitation; “label-free” describes the
inputs to extraction, not a guarantee about pretraining. See [LITERATURE_REVIEW](docs/LITERATURE_REVIEW.md).
The uncertainty view computes normalized entropy, maximum probability and, for regression,
directional variance from the saved probabilities. Raw provider confidence is retained for
traceability but is not treated as the same quantity across models.

## Workflow and notebooks

Runtime setup → one clean question-preparation/0A campaign → full feature creation →
feature inspection and Experiment 0B → benchmark. Folder numbers organize files; 0A runs
before full feature creation despite its higher folder number.

| Local notebook | Purpose |
|---|---|
| [01_data_exploration](<notebooks/0. exploration/01_data_exploration.ipynb>) | All 20 datasets: text length, missingness, exact reuse, long text and feature dimensions. |
| [02_jev_input_design](<notebooks/0. exploration/02_jev_input_design.ipynb>) | Task examples and exact request previews for all 18 datasets and six extractors. |
| [01_prepare_questions](<notebooks/1. question_preparation/01_prepare_questions.ipynb>) | Frozen wording, source meanings, grouped inputs and original-row links. |
| [03_feature_creation](<notebooks/2. feature_creation/03_feature_creation.ipynb>) | Token counts, workload, cost/throughput scenarios and feature combinations. |
| [04_decision_feature_comparison](<notebooks/2. feature_creation/04_decision_feature_comparison.ipynb>) | Coverage, distributions, agreement and disagreements once real features exist. |

Notebooks contain orchestration/display only; reusable logic is in `src/`. They perform
no inference or model training. Each ends with the complete printed report. The comparison
notebook shows coverage gaps until matching complete artifacts exist; it does not invent results.

## Local setup

For a new environment only, in Windows PowerShell from this inner repository:

```powershell
py -3.12 -m venv --prompt JEVPFN .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebooks]"
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name jevpfn --display-name "Python 3.12 (JEVPFN)"
```

Existing environments need no recreation. Select that kernel in VS Code/Jupyter. Run:

```powershell
.\.venv\Scripts\python.exe -m src.data.prepare
.\.venv\Scripts\python.exe -m src.jev.preparation
.\.venv\Scripts\python.exe -m src.jev.question_audit
.\.venv\Scripts\python.exe -m src.utils.run_notebooks
```

Preparation is local and offline. `preparation --rebuild` explicitly regenerates the active
input set transactionally; historical versions and paid features remain intact. The exhaustive
audit checks every unique payload and every original row/text-slot link. It checks mechanics,
not a proof of linguistic correctness or provider interpretation.

The no-argument notebook runner executes in parallel, prints progress and saves rich outputs
back into the source `.ipynb` files. Reload an already-open notebook after execution.
`--only 01_data_exploration` selects one; `--summaries-only` rebuilds aggregate documents.
Run `python -m pytest -q` for local tests. Install model extras in their separate environments,
not all together in the notebook environment. Full setup/transfer commands are in [VSC](docs/VSC.md).

## Storage and cleanup

| Path | Content |
|---|---|
| `data/raw/` | All 20 numbered dataset folders and their provenance. |
| `data/decision_inputs/<dataset>.sqlite3` | Exactly 18 versioned question/input stores with row mappings. |
| `data/jev_cache/<dataset>.sqlite3` | Permanent provider-specific responses, accounting and feature tables. |
| `data/jev_cache/_control/jev_smoke/` | Permanent smoke-test payment ledger and responses. |
| `data/jev_cache/_control/jev_robustness/` | Separate permanent 15-call diagnostic ledger and responses. |
| `data/jev_cache/_control/jev_dataset_robustness/` | Separate 13-call real-text sensitivity diagnostic; never production feature blocks. |
| `data/text_embeddings/<dataset>.sqlite3` | Permanent E5-small vectors and row mappings. |
| `data/text_embeddings/<encoder>/<dataset>.sqlite3` | Separate E5-large/Qwen vectors and row mappings. |
| `output_JEVPFN/figures/<phase>/<notebook>/` | PDF figures only. |
| `output_JEVPFN/{logs,manifests,results}/notebooks/<phase>/<notebook>/` | Local execution logs, figure/config receipts, full reports and analysis tables. |
| `output_JEVPFN/figures/Captions.md` | One combined captions document. |
| `output_JEVPFN/All Results.md` | One complete notebook report. |
| `output_JEVPFN/<number>. phase/{logs,manifests,results}/` | Experiment output, including 0A_local/0A_vsc receipts. |
| `<phase>/results/evaluation/<dataset>.sqlite3` | All predictions, metrics, timings and resumable fold records; Parquet payloads inside one database. |

On VSC, PROJECT holds large inputs, weights/containers, caches, exact-token databases,
predictions and large results. DATA holds the checkout, small CPU environment, compact
manifests and logs. Both use `JEVPFN/output_JEVPFN/`. Local notebooks always use local storage.
Large VSC writes fail rather than falling back to DATA. Free filesystem capacity is not quota.
Evaluation creates one PROJECT database and one small DATA locator per dataset/phase.
Fold locks use a bounded pool (at most 4,096 files per phase), not a new directory and
several files for every fit. Transfer imports merge databases transactionally and reject
conflicting records; legacy per-fold files remain readable. Failed-fit diagnostics stay visible.

The cleanup utility previews by default and never deletes data, weights, environments,
containers or secrets. Run it on the machine being cleaned:

```powershell
.\.venv\Scripts\python.exe -m src.utils.output_cleanup
.\.venv\Scripts\python.exe -m src.utils.output_cleanup --phase experiment_0 --kind logs
.\.venv\Scripts\python.exe -m src.utils.output_cleanup --include-archives --apply --writers-stopped
```

Use `--tier data`, `--tier project` or `--tier both` on VSC; auto selects both. Filters also
include repeatable phases/categories, `--match` and `--older-than-days`. Stop writers first;
active JEVPFN jobs and linked/mounted output trees are refused. Deletion is permanent, with
no archive. Paid smoke records are preserved before deleting legacy output copies. Partial
figure cleanup updates caption manifests and invalidates aggregate reports for rebuilding.
`clean_run` remains the older report-only cleaner; `fresh_start` retains archive compatibility.
In PowerShell, check `$LASTEXITCODE` after native commands; error preference alone is insufficient.

Rows join through dataset ID, source-data fingerprint and `csv:N` (original zero-based row
position). Missing slots are explicit. Provider/checkpoint/settings enter cache identities;
features contain their provenance and checksums. Use validated transfer export/import to
merge local/VSC caches, never wholesale SCP replacement of populated feature stores.

## Execution gates

`python -m src.cluster.experiment_zero --site local` or `--site vsc` previews 0A. Explicit
execution requires a matching frozen revision. Local 0A covers dependencies, full question
mapping, synthetic contracts and Jev smoke; `--dataset-pilot` adds the configured paid pilot.
Local 0A also runs a separate bounded Jev wording/repetition/instruction diagnostic
(15 attempts, at most USD 0.04032 under the configured bounds, excluding a separately
authorized authentication retry). Saved diagnostic responses survive output cleanup.
With `--dataset-pilot`, local 0A also checks 13 deterministic real-text variants across
three task types, using the reviewed private context envelope. VSC contracts check that
same label-blind sample, including aligned multiclass option reversal. These measurements
require human review; they do not select wording by predictive performance.
VSC 0A covers CPU/storage/tokens, all three encoders and five real provider contract/pilot pairs. It submits
ready independent jobs concurrently, resumes by job ID and stops new submissions after failure. See [VSC](docs/VSC.md) for commands.

Before full extraction, review real task/robustness tests, measured throughput, quota,
independent backups and provider acceptance. Jev additionally needs verified organization
RPS/TPS, context/billing bounds and the approved pilot budget. Its ignored private config
is created by `python -m src.jev.setup_pilot --configure`; the local ignored credential file
is loaded automatically and never transferred. No cleanup dispatches API calls.
Approval matches current source/config/runtime fingerprints. Supply `accept_features
--campaign PATH_TO_COMPLETED_0A_RECEIPT` to select exact job evidence when several successful
runs exist; it never selects an arbitrary “latest” report. Prompt sensitivity requires review,
not a claim that every equivalent sentence must produce identical probabilities.

Freeze source/configuration after runtime setup and cleanup with `src.cluster.freeze --record`.
Slurm jobs reject changes; local/VSC receipts compare source identity. Notebook outputs do
not change source identity. After reviewing a stage, explicitly finish its freeze before
changing next-stage configuration; do not erase passed evidence at each transition.
Experiment 0B validates complete saved features and real predictors/full-fold resource use.
Finite tests cannot establish that no bugs remain or guarantee multimillion-request throughput.

## Documentation and licence

[DATA_SOURCES](docs/DATA_SOURCES.md): provenance/semantics;
[LITERATURE_REVIEW](docs/LITERATURE_REVIEW.md): research evidence;
[VSC](docs/VSC.md): operations;
[AGENTS_MEMORY](docs/AGENTS_MEMORY.md): observed run/failure evidence;
[CHANGELOG](docs/CHANGELOG.md): code changes. Plans and requested explanations stay in chat.
Code is MIT; data/weights retain their licences. Software citation: [CITATION.cff](CITATION.cff).

## Based on the repository template

Follow the supplied [TEMPLATE](docs/TEMPLATE.md) and
[original research template](https://github.com/andreasgoethals/0.-Template).
Project-specific deviations preserve the agreed numbered phases, `output_JEVPFN`, aggregate
report names, permanent paid-response stores and explicit local/VSC execution gates. Dense
figures are paginated at final A4 width; notebook metadata/logs/results are kept outside the
figure folders. Python entry points stay under `src/`; Slurm launchers stay under `scripts/`.
