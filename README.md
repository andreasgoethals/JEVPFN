# JEVPFN

Does **probabilistic text information from decision models improve tabular prediction**?
This research codebase explores all 20 core MulTaBench TEXT datasets and uses 18 for
extraction/evaluation. Zomato Restaurants and Spotify Genres remain in exploration but
are excluded from experiments for text burden and 114-class feature size.

**Stage: preparing final Experiment 0A; full extraction and benchmarks remain disabled.**
The 29 October 14:47 download confirms all eleven open-weight decision runtimes
have bounded B200 contract/pilot evidence, including Decision3 (26 checks; 414 new
pilot requests, 36 reused, zero failures). TabSTAR v1.1 passed 16 predictor fixtures
and resume checks; full-dataset profiling remains for 0B. Three encoder contracts also
have bounded B200 passes. Scratch release succeeded, with personal GPFS usage at
8.66/500 GiB after the tests; durable weights/images and results remain preserved.
These runs used the recorded pre-notebook-review source, not the final frozen revision.
Setup and frozen 0A limit resident working sets to two and publish/release copied
inputs after their last consumer. The new bounded parallel lifecycle still needs the
fresh full-campaign run. Keep active VSC checkouts unchanged. See [VSC](docs/VSC.md).
Token audits use ten independent CPU partitions per provider and a strict merge, reusing
matching committed counts. Bounded B200 passes do not certify full-dataset extraction
or final acceptance. Source identities are recorded; the final frozen run repeats all gates.
Twelve extractors are implemented. Question-v3 evidence covers 18 datasets, 1,698,127
distinct inputs and 3,294,340 row/input links. Dated evidence is in
[AGENTS_MEMORY](docs/AGENTS_MEMORY.md).

GPU execution now defaults to **Mindwell B200**. CPU jobs build separate CUDA 13.0
containers, count tokens and copy each required working set to native GPFS scratch.
Authoritative inputs, weights and durable results remain on Lustre PROJECT; logs and
small manifests remain on DATA. Committed feature/prediction databases publish periodically
and at job exit; predictor staging also restores durable completed folds and predictions
so a recreated scratch workspace resumes them. A 200 GiB application working-set
budget and 20 GiB personal headroom check prevent staging all runtimes at once.
Scratch response queues, embeddings and predictions survive input-copy release.
Scratch is not a backup. No new
GPFS project allocation is required. Existing H100 images remain preserved; all A100
routes are excluded. See [VSC](docs/VSC.md).

The local Jev dataset pilot completed 288 diagnostic requests across all 18 datasets.
Prompt-sensitivity checks exposed wording/instruction effects requiring review. These are
bounded setup tests, not production features or final acceptance.
Since 28 October 2026, the owner permits the assistant to commit and push reviewed changes.
Never pull updated source into a VSC checkout while its jobs are active.

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
probability vectors are retained. Jev serializes two-decimal probabilities and Strands
four-decimal probabilities. Only complete vectors with a feasible rounding correction
are projected to sum one, with maximum per-entry changes of 0.005 and 0.00005 respectively.
Raw responses remain unchanged; this policy does not recover discarded precision. The rubric distinguishes indirect hints, qualified
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
| Liquid d1-2B (LFM3) | Open weights, VSC; new candidate awaiting live acceptance |
| Strands Decider 2B | Open weights, VSC |
| Perplexity Decider v1.1-27B | Open weights, VSC |
| Cloudflare Clef-flash | Open weights, VSC |
| Intern-Decision-4B | Open weights, VSC |
| H2O Lightning 4B | Open weights, VSC; publisher attention/calibration with startup anchors |
| SII-Decision-27B-Full | Open weights, VSC |
| Kev-27B | Open weights, VSC; native pointer head |
| Decision3 d3-turbo | Open weights, VSC; native prefix-bidirectional runtime |
| Bespoke-Nimble-9B | Open adapter and pinned Qwen3.5-9B base, VSC |

Microsoft is excluded by owner decision. Pins, licences, context and price references
are in `config/2. feature_creation/extractors.yaml`. Open weights have no vendor token
fee; GPU credits and licence conditions still apply. Compatibility requires real testing.
H2O, SII, Kev, Decision3 and Nimble use Apache-2.0 releases, not additional hosted APIs. Decision3 uses
the current selected d3-turbo checkpoint; Nimble uses the Apache release, not a later
noncommercial variant. H2O's released binary formatter floors the winning probability
at 0.801. Preserve and disclose that policy; it is not evidence of target calibration.
Liquid d1-2B (`liquid_lfm3`) is an additional pinned candidate alongside d1-3B;
its text-only path uses the same question v3 and explicitly requests mean score readout.
Its separate Transformers 5.19 Blackwell image serves both CPU audits and GPU inference.
It still needs its own live VSC contracts and dataset pilot; earlier Liquid evidence is for d1-3B.

All twelve adapters preserve the same canonical task metadata, question, named text, class
meanings and option order. Native framing, token IDs, probability calibration and wire
formats differ. Adapters expose the same downstream feature schema; numeric answers are
expected to differ. Raw provider confidence is not an interchangeable calibrated quantity. Conservative complete-request caps also
respect H2O's and Kev's separate state caps. Overflow stops execution without truncation.

The six main predictors are **CatBoost, TabPFN-3.5, TabICL v2, LimiX-2, NVIDIA Kumo Tabular,
and logistic/linear regression**. All use the same rows and folds. The approved experiments are:

| Experiment | Comparison |
|---|---|
| 0A | Before extraction: local Jev interfaces/pilot; VSC storage, exact decision token counts, ten decision runtimes and three embedding encoders. |
| 0B | After extraction: complete caches/row mappings; all nine predictors, PCA widths/scopes, native-text augmentation, resume and full-fold feasibility. |
| 1 | Each of six predictors: non-text alone, then joint, per-column and both decision features for each of twelve extractors. |
| 2 | Six predictors, three frozen encoders: per-column PCA32 and joint PCA256/128/64/32/16/8. No decision features. |
| 3 | TabSTAR, ConTextTab and AG-MM on original text and non-text features. |
| 4 | Label-blind question/context/option-order sensitivity; analysis reuses explicit real diagnostic reports. |
| 5 | Presence/statistics, shuffled decision features, probability/uncertainty controls; separate descriptive winner-selected combinations. |

Experiments 1/2/3 contain **3,996 / 2,268 / 54 configurations**: 31,590 outer fits,
plus 15,795 binary inner fits. A configuration means one dataset/predictor/representation;
it has five outer folds. No expensive HPO is performed. Experiment 1 shares its non-text
baseline across extractors. Experiment 2 has seven PCA settings, not their Cartesian
product with every decision model. Experiment 3 has one original-feature setting per model.

Optional concatenation of all twelve decision models is exposed with
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
policy, not a claim of equivalence to encoding the whole text in one window. Model inference
retains checkpoint precision; chunk normalization and NumPy export use float32, including
checkpoints that produce bfloat16 activations.

PCA uses **32 components per original text column**, or **256/128/64/32/16/8 total
components** after concatenating the text vectors. These compare both grouping and
compression budget; no setting is claimed optimal. Neither reduces non-text or decision
features. Joint PCA imputes missing blocks with training coordinate means; every text
column retains its missingness flag. Validation/test rows transform through the same
training projection. If there are too few nonempty training rows or embedding coordinates
for the requested width, unused output components stay missing.

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
expansion is 447. The optional all-twelve-provider combination reaches 1,085 on Wine Reviews,
so Kumo refuses that arm. The recipe's 500-column guard prevents silent feature selection.
Other models still require checkpoint/resource checks; width compatibility does not prove
full-fold feasibility. AG-MM uses fixed medium-quality ELECTRA-base/fusion, no HPO or
ensemble, in its separate runtime. Actual native-model compatibility remains unverified.
Native-text TabSTAR selects **v1.1**, through `tabstar.tabstarv1p1_model` in package 1.2.2,
with separate pinned classification/regression checkpoints and its E5 dependency. Stage
them on CPU with `python -m src.cluster.predictor_assets --execute`. The runtime resolves
verified offline cache aliases; a local directory is not passed as a Hugging Face repo ID.
Its versioned `tabstar-v1p1` image preserves older containers. Defaults include LoRA adaptation
and averaging three internally selected epochs; transient epoch checkpoints are removed.

Use five-fold shuffled stratified CV for binary/multiclass classification and shuffled
KFold for regression. Binary F1 thresholds use three inner out-of-fold splits, followed
by refitting on the full outer-training fold. The outer test fold never selects thresholds
or fits preprocessing. Use all rows and fixed model settings, with no HPO or silent row/
feature sampling. Full-fold feasibility must be tested before enabling the benchmark.

The LimiX-2 adapter restricts feature filtering, category inference/encoding and polynomial
scaling to training rows; its publisher implementation otherwise uses query rows in some
of these steps. This declared preprocessing adaptation preserves the weights and requires
real 0B validation. Unsupported publisher interfaces or multi-GPU reconstruction stop
explicitly rather than silently reverting to combined train/test preprocessing.

Save individual predictions, binary inner-OOF predictions, metrics, undefined-metric reasons,
timings, resource use and artifact identities. Analyze paired feature effects within each
predictor, including seen/unseen text cohorts. Decision and embedding arms also compare
with the same predictor using missing-text indicators only when that control is available.
Operational output folders do not change predictor/checkpoint comparison identities. Fold SD is descriptive, not a standard error.
The owner fixed MulTaBench-compatible headline metrics before comparative outcomes:
binary ROC-AUC, multiclass macro one-vs-rest ROC-AUC, and regression R². Other metrics stay
recorded, including binary validation-selected F1. Optional dataset-level Wilcoxon/Holm
contrasts still require prespecification. Exact protocol/settings live in
`config/4. experiment_1/`; optional analysis is in `config/8. experiment_5/`.

Extraction timings separate new inference, cache reuse and failures. Jev records client
setup, rate-limit waiting and API round trips; open-weight models and encoders record loading
and synchronized inference batches. Predictor records retain initialization, inner/final
fit and prediction times. Detailed extraction timing journals use one SQLite file per run
under the phase's PROJECT `results/timings/`; compact summaries use DATA `manifests/timings/`.
All existing classification/regression metrics are saved, not just AUC; full probabilities
and row-level predictions remain available for subsequent metrics and paired analyses.
Cheap text statistics, permutation controls and retained-probability ablations are also
available. The optional shuffled-feature control preserves each row's missingness pattern
and permutes complete vectors within matching patterns separately in train/test partitions.
Public-data pretraining overlap is a limitation; “label-free” describes the
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
| [02_jev_input_design](<notebooks/0. exploration/02_jev_input_design.ipynb>) | Task examples and request previews for all 18 datasets and twelve extractors. |
| [01_prepare_questions](<notebooks/1. question_preparation/01_prepare_questions.ipynb>) | Frozen wording, source meanings, grouped inputs and original-row links. |
| [03_feature_creation](<notebooks/2. feature_creation/03_feature_creation.ipynb>) | Token counts, workload, cost/throughput scenarios and feature combinations. |
| [04_decision_feature_comparison](<notebooks/2. feature_creation/04_decision_feature_comparison.ipynb>) | Coverage, distributions, agreement and disagreements once real features exist. |
| [0A evidence](<notebooks/3. experiment_0/01_before_feature_creation.ipynb>) | Local dated input/token/API/GPU evidence; never claims live cluster acceptance. |
| [0B evidence](<notebooks/3. experiment_0/02_after_feature_creation.ipynb>) | Saved representation inventory, nine predictor contracts and full-fold readiness. |
| [Experiment 1](<notebooks/4. experiment_1/01_paired_representation_effects.ipynb>) | Decision modes within fixed predictors; seen/unseen-text effects and measured width/time. |
| [Experiment 2](<notebooks/5. experiment_2/01_frozen_embeddings.ipynb>) | Three encoders, seven compression recipes, training-only PCA variance and predictive/compute effects. |
| [Experiment 3](<notebooks/6. experiment_3/01_native_text_systems.ipynb>) | Original-text reference systems on matched folds, with checkpoint and resource identities. |
| [Experiment 4](<notebooks/7. experiment_4/01_question_sensitivity.ipynb>) | Question/context/option-order sensitivity and retained directional distributions. |
| [Experiment 5](<notebooks/8. experiment_5/01_controls_and_followups.ipynb>) | Prespecified controls against no-text and exact aligned artifacts; separate exploratory follow-ups. |
| [Combined Experiments 1–3](<notebooks/9. synthesis/01_representation_comparison.ipynb>) | All matched decision-versus-embedding recipes and descriptive native-system comparisons, without selecting winners. |

Notebooks contain orchestration/display only; reusable logic is in `src/`. They perform
no inference or model training. Each ends with the complete printed report. The comparison
notebook shows coverage gaps until matching complete artifacts exist; it does not invent results.
The thirteen notebooks keep complete displayed/plot summaries, not repeated dumps of every
raw metric row. Full predictions/metrics remain in their evaluation stores. Empty all-zero
coverage heatmaps are omitted; predictive figures await genuine complete records. Synthesis
is a general notebook phase, not an extra experiment or configuration grid.
Input-design figures include dated, hash-bound saved prompt diagnostics; they are small case
studies rather than population robustness rates. Extraction costs distinguish approximate
full-run scenarios from cumulative recorded diagnostic charges. The study map counts the
actual main grid; optional controls and winner-selected follow-ups remain separate.

Decision features are **frozen, task-conditioned representations**: target names and class
meanings enter the question, but dataset labels do not train the extractor. MulTaBench's
LoRA target-aware representation learning is a distinct supervised method. See the
[literature review](docs/LITERATURE_REVIEW.md) for that distinction and scope limits.

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
On B200, large active writes use the staged GPFS working set and publish to durable
Lustre; small logs/manifests still use DATA. Large VSC writes fail rather than falling back to DATA. Free filesystem capacity is not quota.
Evaluation creates one PROJECT database and one small DATA locator per dataset/phase.
Fold locks use a bounded pool (at most 4,096 files per phase), not a new directory and
several files for every fit. Transfer imports merge databases transactionally and reject
conflicting records; legacy per-fold files remain readable. Failed-fit diagnostics stay visible.

On VSC, full output cleanup includes numbered output under GPFS working sets,
including known runtimes whose staging failed before registration, as well as DATA
and durable PROJECT. It preserves the staged inputs and containers.
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
VSC 0A covers CPU/storage/tokens, all three encoders and ten real provider contract/pilot pairs. It submits
ready independent jobs concurrently and resumes by job ID. Failed prerequisites block only
their descendants; the final report lists all failures and blocked tests. Before the final
freeze, `bash scripts/test_all.sh --execute --watch` runs the complete setup/test graph,
including downloads and containerization. Each provider's token audit has ten CPU partitions
plus a recovery check and a coverage-checked merge; up to 40 token jobs run alongside other pools. CPU, build, transfer and GPU pools are separate;
all GPU checks use B200 by default. See [VSC](docs/VSC.md) for commands.

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
Benchmark execution checks that its reviewed 0B `evaluation_identity` matches the current
implementation, model settings and feature/protocol recipes, independently of enable switches.
Finite tests cannot establish that no bugs remain or guarantee multimillion-request throughput.

The paired-results notebook also directly contrasts every decision recipe with each
embedding recipe within matching predictors/folds, displays prespecified controls and
seen/unseen-text effects, and reports measured timing and post-PCA representation width.
It selects no winners and creates no predictive plots before saved complete arms exist.
Extraction timing retains its run/hardware identity; partial runs are not presented as
complete feature-generation costs or automatically joined to predictive performance.

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
