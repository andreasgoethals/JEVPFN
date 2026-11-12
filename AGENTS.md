# Agent instructions â€” JEVPFN

Read [README](README.md), [Experiments](docs/EXPERIMENTS.md), the relevant
[Literature review](docs/LITERATURE_REVIEW.md) and dated [Operational evidence](docs/AGENTS_MEMORY.md)
before changing this repository. Follow [TEMPLATE](docs/TEMPLATE.md); project deviations
are recorded in the README. Preserve unrelated local changes.

## Authorized work

- Since 28 October 2026, the owner permits committing and pushing reviewed JEVPFN changes.
  Later read-only/no-publication instructions take precedence. Publish only relevant
  changes; do not change branches, remotes or history without explicit instruction.
- The authorized campaign includes CPU-container cutover, verified output reset, fresh
  local/VSC 0A, review/acceptance and open-weight/encoder feature creation. The CPU and
  GPFS migrations passed; old DATA runtime and Lustre PROJECT copies were retired.
- Complete models may be downloaded as verified snapshots and merged locally, preserving
  existing Jev responses and the durable VSC copy. Partial models do not count as complete.
- Local Jev 1.14 may consume only the existing EUR 5 allowance authorized on 5 November,
  under the cumulative USD 5.50 ceiling and existing dataset allocations. Preserve all
  historical charges/reservations. No top-up or larger budget is approved. Test credit
  exhaustion and shared rate backoff before changing dispatch behavior.
- On 12 November, the owner authorized independent 0B predictor contracts and TAR
  preparation at a safe handover, alongside recovery of remaining extraction. Full
  benchmark execution still requires separate authorization and reviewed 0B. Cleanup does not
  authorize jobs or paid calls. Access alone does not authorize cancellation or migration.
- Existing environment-setup authorization persists within JEVPFN. Verify the actual
  interpreter; never change another project's jobs, files or shared environment.
- Plans and requested explanations belong in chat, not temporary planning documents.

## Scientific constraints

- The scope is Experiments 1â€“4: decisions, frozen embeddings, supervised TAR and native
  text, with prerequisite 0A/0B and synthesis. Earlier ablations are retired. No new
  models, grouped sensitivity, lexical baselines or selected-winner combinations without
  instruction. The selected roster and exact pins are in `config/`; Microsoft is excluded.
- Decision requests are deterministic and label-free: no generated questions, non-text
  row values, true row targets, class frequencies, target statistics or model feedback.
  Keep original class vocabularies and sourced meanings in `dataset_semantics.yaml`.
  Empty text retains missing features. Never silently truncate decision inputs, drop
  rows/classes/features or cap training rows.
- Among precomputed representation arms, TAR fits the encoder to dataset labels only
  inside each outer training partition.
  Preserve its vendored source byte-for-byte and the exceptions documented in `docs/TAR.md`.
  Do not deploy future benchmark changes into active extraction consumers.
- Preserve `tokenization.py` and `decision_tokenization.py` byte-for-byte while audits
  use their fingerprints. Historical availability constants do not select active providers.
- Question, model and runtime contracts require actual evidence. Synthetic tests do not
  establish checkpoint/GPU compatibility, linguistic perfection or full-stage acceptance.

## Protected files and libraries

- Raw datasets, prepared inputs, permanent responses/features and checkpoints are protected.
  Never recursively clean them or a system temporary root; never force-add data/weights.
- Credentials must not enter tracked files, notebooks, reports or tool output. The owner
  authorized ignored `secrets/typesafe.local.json`; local Jev loads it automatically.
  Never transfer it to VSC. `TYPESAFE_API_KEY` takes precedence.
- `tfm-library/` and `dm-library/` are independent, pinned, read-only submodules. Read their
  instructions, summaries and source maps; do not edit, move, refresh, lint or test them.
  Their ignored `PROJECT_SPECIFIC.md` is the only permitted local-library exception.
- `vsc-library/` is entirely read-only, with no `PROJECT_SPECIFIC` exception. Use its guides
  and examples; keep project-specific choices in `docs/VSC.md`.
- Record library pins when evidence depends on them. Cite implementation snapshots by
  symbol. Shared corrections belong in the canonical libraries under `0. Literature`.
- Frozen data-source snapshots in `src/data/upstream/` are inspected, not executed.

## VSC access and campaign continuity

Reuse the owner's native Windows OpenSSH agent. If the certificate needs renewal, ask
once for `step ssh login --context VSC` and manual KU Leuven Authenticator approval.
Do not initiate login, install another agent, configure separate project credentials,
automate MFA or schedule authentication reminders.

Follow the bounded LF stdin-script method in TEMPLATE with `ssh.exe -F none -T`, strict
host-key verification, output markers and checked exit status. Empty output with exit 0
proves nothing. This project's verified gateway workaround starts the script with
`exec 2>&1`; the dated shell workaround is in `docs/VSC.md`. Inspect scoped remote
logs, files, quotas and Slurm state directly when relevant; run computational checks
through Slurm within the authorized scope.

Keep the DATA checkout frozen while any queued/running job or publisher consumes it.
Reconcile exact IDs, process identities, locks and recovery data before cutover. Deploy
reviewed local fixes only after validation and consumer-safe handover. Failed attempts
remain historical evidence; do not erase them to make a campaign look successful.

The owner requested parallel admission of remaining extraction/embedding work. Use
`feature_admission`, all ready datasets per staged runtime, dataset-partitioned encoders,
and the existing two-workspace/byte and bulk-transfer safeguards. A reviewed scheduling
companion outside frozen source may replace the controller only with recorded hashes,
policy and original/execution identities. Never alter scientific inputs for admission.

The desktop progress monitor is disabled. Do not create an automation to watch a job
through completion. Use the current chat for requested checks; recurring reporting needs
an explicit request and grants no new execution or cleanup authority.

## Storage and repository layout

- Actual Git checkout and `.git`: DATA. Git operations remain there on VSC.
- Persistent inputs, models, sealed dependency containers, response stores, token counts,
  embeddings, predictions and large results: allocated GPFS PROJECT.
- Isolated mutable working stores and temporary build/compiler caches: native scratch.
  Share immutable GPFS PROJECT assets directly; avoid full working-set duplication.
- DATA also holds compact manifests and operational logs under the matching numbered
  `output_JEVPFN` phases. Use validated export/import, never replace populated caches wholesale.

The durable allocation comes from ignored `.vsc-storage.local.json`,
`JEVPFN_DURABLE_ROOT`, `JEVPFN_PROJECT_ROOT` or `VSC_PROJECT_ROOT`.
`JEVPFN_STAGING_ROOT` is an explicit working-set override, not durable authority. Never
silently fall back to DATA or stream bulk cross-mounted Lustre assets during B200 inference.
The owner excludes A100, including debug/MIG. Use Mindwell B200 and the validated native
CPU route. The controller image is selected by `control.active.json`; B200 model images
use `blackwell-cu130` and require staged-input, CUDA and real model contracts.

Retirement requires consumer checks, verified publication/recovery and scoped authorization.
Preserve resumable failures, referenced images and other projects. Check byte/file quotas;
never wipe shared scratch. Historical retention exceptions are documented in the VSC runbook.

Reusable Python belongs in `src/`. Config, notebook and output phases share numbers.
Do not create empty phase folders or duplicate question files. Permanent questions and
response/features use per-dataset SQLite stores. PowerShell must work on Windows
PowerShell 5.1; Bash/Slurm files use LF. Editable text is UTF-8.

## Notebooks, documentation and checks

- All notebooks run locally. Put functions/classes in source modules, not notebook cells.
  End each notebook with the complete printed report in section order.
- `src/visualize/style.py` owns colors/sizes. Use `FigureSaver`, PDF and inline display;
  inspect figures at final A4 widths and split dense views into readable panels.
  Captions describe the figure without repeating the title.
- Save figures to `output_JEVPFN/figures/<phase>/<notebook>/`, captions to
  `figures/Captions.md`, and reports to `output_JEVPFN/All Results.md`.
- Run `python -m pytest -q` before finishing. If notebooks or `src/visualize/` change,
  also run `python -m src.utils.run_notebooks` and inspect every affected figure.
  Check lint and relevant shell syntax. Separate local checks from real VSC acceptance.
- Keep the README an orientation page; `docs/EXPERIMENTS.md` owns the scientific design,
  VSC the operational procedure, and AGENTS_MEMORY dated evidence. Link rather than repeat.
  Record substantive changes concisely in CHANGELOG, newest first, DD-MM-YYYY.
- Record cluster outcomes and reusable failure lessons with exact IDs/source identities.
  Submission is not completion. Keep docs uppercase, durable and concise; preserve historical
  evidence even when the current implementation changes.
