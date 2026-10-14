# Operational notes

- **Source meanings:** `dataset_semantics.yaml` is the maintained dictionary. Preserve raw labels; never infer units from label magnitudes. Book Price has a misleading USD column name despite documented upstream log-price. Jev and five self-hosted decoders/materializers are implemented but live GPU compatibility and hosted multi-provider spending remain unverified.

## Current state

Experiment 0A preparation and offline tokenizer testing. Guarded SDK transport, self-hosted
GPU pilot/extraction adapters and injectable CV orchestration
are prepared; no paid calls/features or model fitting have been verified. VSC CPU and token
jobs were submitted by the owner (see table); completion is unverified. All live
switches remain disabled. The public GitHub repository is connected; generated outputs and raw data stay ignored.
The literature submodule is read-only at the existing index/checkout pin
de324efcca55be392feb0c01830b3b6ce364587f (existing index/checkout verified 13-10-2026; not modified here).
Durable method constraints are in README.md; interface evidence is in LITERATURE_REVIEW.md.
The owner handles Git commits/pushes. Proposed plans belong in chat, not temporary documents.

## Avoid repeating these failures

| Date | Cluster / job | Configuration | Observed outcome |
|---|---|---|---|
| 14-10-2026 | wICE / 62312865 | experiment_0_cpu.slurm | Submitted; log/exit status not supplied. Correct dependency install and `pip check` succeeded before submission. |
| 14-10-2026 | wICE / 62312903 | experiment_0_tokens.slurm | Submitted; owner reports running. This checkout counts the original three providers, not the newly added Clef/Intern. |

Do not update the VSC checkout while jobs use it. Record final status after inspecting
`sacct` and logs, not from submission alone. PROJECT write access is verified; quota and
independent backup remain unverified. `sam-balance` confirms account credit availability,
not storage quota or successful execution.

- **VSC installation shadowing (14-10-2026):** the owner's transcript shows a newly created
  Conda JEVPFN environment, but pip ran in TabPFNCredit's inherited venv and upgraded packages
  there. JEVPFN's clone, PROJECT write probe and tokenizer staging succeeded; no Slurm run
  is shown. Setup and jobs now share `scripts/activate_conda.sh`, which clears aliases,
  functions and inherited Python overrides, then checks the resolved prefix before exporting
  `JEVPFN_PYTHON`. Use that explicit interpreter for pip. Repair the other project's environment
  separately from its own requirements, with authorization; do not guess a rollback.
  Large CUDA environments and package caches belong on PROJECT, not personal DATA.

- **Hybrid cache transfer:** `src.cluster.transfer export/import` now snapshots both tiers,
  validates checksums and merges each dataset transactionally. Conflict means rollback, never
  replacement; retries reuse rows. Preserve original bundles and pre-merge backups. Imported
  spending records are historical evidence, never a replacement for local budgets/queues.

- **Exact token formatting (14-10-2026):** character proxies flagged Fake Job Postings rows
  that fit all three actual open-model tokenizers. Liquid encodes a complete rendered turn;
  Strands encodes state (special tokens enabled) and question separately; Perplexity uses
  its saved chat template and answer codes. Never count bare text or substitute a base tokenizer
  for an unavailable hosted tokenizer. Keep token counts resumable and traceable to input hashes.
  All 1,698,127 selected unique inputs fit: maxima Liquid 3,104, Strands 3,244 and Perplexity
  3,154 tokens. Actual GPU count parity remains a separate test; hosted counts are unverified.
  Hugging Face/Xet cache variables must be set before its first import, including during staging.

- **Preflight test environment:** the combined local preflight exposed inherited
  `JEVPFN_PHASE=experiment_0` in tests expecting exploration. Synthetic tests now clear
  ambient routing/job variables and explicitly simulate their own temporary storage;
  real host checks run outside that fixture. This also prevents writes to real VSC allocations.

- **Provider preparation boundaries:** all seven previews share semantic questions; five
  self-hosted adapters now use pinned runtimes. Perplexity uses singular `question` in each prediction row.
  Strands requires strict-window rejection and serialized inference until concurrency is verified.
- **Evaluation identity and memory:** include exact matrix/schema/row hashes, feature artifact
  IDs and resolved estimator settings in run IDs. Release each fold's estimator before loading
  its successor. Allow documented float32 probability roundoff without renormalizing outputs.

- **Temporary audit storage:** cleanup must never include a temporary directory's parent
  (the system TEMP root). Runner-owned temporary directories remove themselves; paid data remains protected.

- **Blank executed notebooks (01-10-2026):** flattening cells to a Python script never updates
  `.ipynb` outputs. Use real kernels in the invoking venv, persist successful/partial outputs,
  retain backups, and do not overwrite notebooks edited during execution.

- **Inline overview overlap (01-10-2026):** the PDF looked correct while Jupyter placed
  figure-level title/notes over cells. Keep header/table/footer in separate layout axes and
  verify an actual notebook-kernel render, not just the saved PDF.
- **Raw-data loss (30-09-2026):** the owner reported data deletion after an earlier empty-folder
  cleanup command. All 20 pinned releases were restored and hash-checked. The command is now
  an error-only stub. Never revive it or recursively clean input/cache folders.
- **Parallel audit race on Windows:** atomic replacement alone cannot replace an open NPZ.
  Retain the OS-backed audit lock for writers and readers, and the report aggregation lock.
- **Notebook report sharing violation (30-09-2026):** an interactive final cell failed at
  `os.replace` on `feature_creation/Captions.md`. Retry transient access errors, skip unchanged
  content and preserve a recovery copy on persistent failure. Aggregate writes share `.reports.lock`.
  The parallel notebook check also reproduced this on audit `text.csv`; the same atomic writer
  now protects CSV, JSON and reports, with replacement retries for NPZ outputs as well.
- **Wrong Python:** the default runtime was Python 3.14/pandas 3. Use this project's
  `.venv/Scripts/python.exe` (Python 3.12), including actual Jupyter-kernel verification.
- **Windows ReadOnly directories:** relocation/case-only renaming may fail. Resolve exact
  workspace paths and verify contents before a native PowerShell move/removal. Do not combine
  filesystem operations across shells. The dataset folders and pinned source snapshots are protected.
- **VSC environment shadowing:** explicitly use the activated Conda interpreter; an inherited
  local venv can otherwise take precedence. Keep JEVPFN and CreditPFN environments separate.
- **Cluster account:** reuse the configured CreditPFN account after checking association.
  The former Mindwell pilot account is not a fallback. Submitted jobs are recorded above.

- **Whole-group trace loss (12-10-2026):** checking only groups present in `requests` accepted
  a feature artifact after all traces for one text column were removed. Validate the expected
  group set from representation/text columns and require numeric columns for every group.
  Synthetic fault checks reproduced this; 144 tests pass after the validation fix.

Historical details remain in Git history. Keep this file short and record only reusable
operational evidence; put current task results in generated reports and edits in CHANGELOG.md.
