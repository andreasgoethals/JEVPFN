# JEVPFN

Does **probabilistic text information from decision models improve tabular prediction**?
This research codebase explores all 20 core MulTaBench TEXT datasets and uses 18 for
extraction/evaluation. Zomato Restaurants and Spotify Genres remain in exploration but
are excluded from experiments for text burden and 114-class feature size.

**Stage: Experiment 0A setup, not full extraction.** Question version 3, dataset audits,
CPU/storage checks and all five open-weight tokenizer audits have passed previously.
Clef's container is now built/sealed and its A100/CUDA tensor check passed. Its model
contract test stopped before inference because the saved token-audit report was missing.
The replacement audit reports now cover all five tokenizers with zero overflows, but two
overlapping jobs collided while writing counts and both failed. Duplicate-run protection
is fixed locally; run one resumable audit to verify saved counts, then retry Clef's GPU test.
All five checkpoints were staged; only Clef/Intern's
older native runtimes have passed model inference tests. Jev 1.13 passed three live
local task probes; its larger dataset pilot and account limits remain pending. E5 and
predictor runtime tests are also pending. Historical evidence is in
[AGENTS_MEMORY](docs/AGENTS_MEMORY.md); a new frozen 0A run will supersede it.
The owner handles Git commits/pushes. Full extraction and benchmark switches stay disabled.
The 20 October implementation adds compact prediction stores, source/input/runtime-bound
acceptance, prompt diagnostics and checked batching. These changes passed local synthetic
tests; current VSC/GPU and live Jev diagnostic evidence is still required before extraction.

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
probability vectors are retained. The rubric distinguishes indirect hints, qualified
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
and logistic/linear regression**. Each uses eight feature configurations:

1. All original non-text columns, including categorical columns.
2. Non-text + joint decision features.
3. Non-text + per-column decision features.
4. Non-text + both decision representations.
5. Non-text + frozen E5 embeddings reduced by PCA.
6. Non-text + E5 + joint decision features.
7. Non-text + E5 + per-column decision features.
8. Non-text + E5 + both decision representations.

Configurations 1 and 5 are shared across extractors; the other six are repeated for each
extractor. This gives 4,104 main configurations plus 36 native-text references, each with
five outer folds. **TabSTAR and ConTextTab** use the original text and non-text columns;
they are descriptive reference results, not matched within-predictor effects.

E5-small-v2 encodes each text column once, covers long inputs with token chunks, reuses
identical text and saves 384-dimensional vectors. PCA is fitted **inside each inner/outer
training partition**, producing 32 components per text column plus a missingness flag.
Insufficient training rank leaves unused components missing. Held-out rows never fit PCA.
Non-text/decision features are not reduced. All six predictors use the same E5 source.
Every decision/E5 arm receives the same per-text-column missing indicator, exactly once
in combined arms. The non-text baseline contains no text-derived information. E5 batches
chunks across different texts; content-token weighting and row mappings are preserved.

Use five-fold shuffled stratified CV for binary/multiclass classification and shuffled
KFold for regression. Binary F1 thresholds use three inner out-of-fold splits, followed
by refitting on the full outer-training fold. The outer test fold never selects thresholds
or fits preprocessing. Use all rows and fixed model settings, with no HPO or silent row/
feature sampling. Full-fold feasibility must be tested before enabling the benchmark.

Save individual predictions, binary inner-OOF predictions, metrics, undefined-metric reasons,
timings, resource use and artifact identities. Analyze paired feature effects within each
predictor, including seen/unseen text cohorts. Fold SD is descriptive, not a standard error.
Headline metrics and optional dataset-level Wilcoxon/Holm contrasts remain unapproved until
before comparative outcomes are reviewed. Exact protocol/settings live in
`config/4. experiment_1/`; optional analysis is in `config/7. experiment_4/`.
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
| `data/text_embeddings/<dataset>.sqlite3` | Permanent E5 vectors and row mappings. |
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
VSC 0A covers CPU/storage/tokens/E5 and five real provider contract/pilot pairs. It submits
one job at a time, resumes by job ID and stops after failure. See [VSC](docs/VSC.md) for commands.

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
