# Agent instructions — JEVPFN

Read README.md, docs/LITERATURE_REVIEW.md and docs/AGENTS_MEMORY.md before changing this
repository, and follow docs/TEMPLATE.md with the project deviations recorded in README.

## Boundaries

- Since 28 October 2026, the owner authorizes committing and pushing reviewed JEVPFN changes
  without asking again. This supersedes the earlier no-publication preference; later
  read-only or no-publication instructions still take precedence. Publish only relevant changes.
  Never change branches/remotes/history without a separate explicit instruction.
- Proposed plans and requested explanations belong in chat, not temporary documents.
- `tfm-library/` is a pinned read-only submodule. Never edit, move, delete, lint or test it.
  The only allowed library write is its gitignored PROJECT_SPECIFIC.md. Report library errors
  to the owner. Cite code snapshots by symbol, not line number; record the pin used.
- Raw datasets, prepared inputs, permanent response/features and checkpoints are protected.
  Never recursively clean them or a system temporary root. Never force-add data/weights.
- Credentials must never enter tracked code/configuration, notebooks, reports or tool output.
  The owner explicitly authorized the Git-ignored local secrets/typesafe.local.json file.
  Local Jev commands load it automatically; never copy it to VSC or include it in transfers.
- Main inputs are deterministic and label-free: no generative questions, non-text row values,
  true row targets, class frequencies, target statistics or model feedback. Keep original
  class vocabularies; source meanings live in dataset_semantics.yaml. Skip empty text and
  retain missing features. Do not silently truncate, drop rows/classes or cap training rows.
- The selected roster and method are in README/config. Microsoft is excluded. No new models
  or changes to open design choices without instruction. Use open weights where available.
- Full extraction/benchmark switches remain disabled pending acceptance. Bounded GPU contracts
  require explicit execution, staged pins, complete exact token evidence and a VSC allocation.
  Do not submit allocations or perform paid calls as a side effect of cleanup.
- Preserve tokenization.py and decision_tokenization.py byte-for-byte while audits use their
  fingerprints. Historical availability constants there do not select active providers.
- Existing authorization for environment setup persists; verify the actual interpreter and
  do not alter another project's/shared environment without authorization.

## Authorized current run

On 29 October the owner authorized the CPU control-container rebuild/cutover,
Slurm debugging, verified final local/VSC output reset, fresh 0A, evidence review
and accepted full open-weight feature creation. Carry this out autonomously within
JEVPFN; preserve other projects. Full Jev extraction, 0B and benchmarks wait for the
owner's next stage. Preserve the old CPU environment until the replacement passes.

## Layout and operations

The active research scope is Experiments 1–3, with prerequisite 0A/0B and a synthesis
notebook. Experiments 4/5 and selected-winner combinations are removed. Do not reinstate
them without an explicit request. Keep 0A input-robustness diagnostics and the main
benchmark's statistical analysis in scope.

VSC authentication is shared and manual. Reuse the owner's native Windows OpenSSH agent.
Only when access needs renewal, ask them to run `step ssh login --context VSC` and approve
KU Leuven Authenticator. Do not initiate login, install another agent, add project-specific
credentials/configuration or schedule reminders. Follow docs/TEMPLATE.md's bounded native
`ssh.exe -F none -T` stdin-script probe, with host-key verification. Verify markers and exit
status; an empty exit-0 response proves nothing. The current gateway drops unredirected
remote stderr: the verified project workaround is `exec 2>&1` inside the LF script, retaining
error text in stdout and the remote exit status. See docs/VSC.md for the dated evidence.
Inspect this project's remote logs, files, quotas and jobs directly when relevant. Access
does not authorize new workloads, cancellation, migration or cleanup. Preserve active/queued
checkouts and recovery data; never update source consumed by a job.

Reusable Python is in src/; upstream data-source snapshots stay under src/data/upstream/ and
are never executed. Config/notebook/output phase folders use the same numbers. Do not create
empty phase folders or copied question files. All notebooks run locally; permanent questions
and responses/features each use one SQLite file per selected dataset under data/.

On VSC, the actual Git checkout and `.git` stay on DATA. PROJECT holds large inputs/weights,
immutable containers containing installed dependencies, permanent caches, token counts,
predictions and large results. DATA holds the checkout, compact manifests and operational
logs. Both use JEVPFN/output_JEVPFN/<number>. phase/. Use the validated transfer export/import,
never wholesale replacement of populated caches. See docs/VSC.md for commands/site details.
The owner excludes all A100 allocations, including debug/MIG routes. GPU jobs default to
Mindwell B200. CPU staging prepares per-runtime native GPFS scratch working sets; durable
Lustre remains authoritative. Use `JEVPFN_PROJECT_ROOT` or `VSC_PROJECT_ROOT` for the allocated
shared root; legacy `JEVPFN_STAGING_ROOT` remains an explicit working-set override so B200
jobs cannot accidentally read cross-mounted PROJECT. Never silently fall back to DATA.
The CPU controller uses a versioned PROJECT image selected by control.active.json.
An old DATA prefix may remain only until the recorded CPU and native B200 migration
gates pass. Retirement requires the owner's authorization and an idle checkout; preserve
the shared Miniconda installation and other projects. Scratch is temporary: retain failed
or resumable work until verified publication and explicit scoped retirement; never wipe shared
scratch. Check byte/file quotas and keep immutable assets needed by active jobs.
Separate blackwell-cu130 containers, staged-input receipts,
periodic/final durable cache publication and real GPU contracts are required. Never stream
bulk model data from cross-mounted Lustre during B200 inference.

PowerShell commands must remain compatible with Windows PowerShell 5.1: no &&, ternary or ??.
Slurm files are Linux Bash and must use LF endings. New/editable text files are UTF-8.

## Notebooks and validation

- Put logic in source modules; notebooks contain no function/class definitions.
- Each notebook ends with the complete printed report in section order.
- src/visualize/style.py owns colors/sizes. Use FigureSaver, PDF output and inline display;
  use final A4 widths and split dense views into readable panels. Captions describe the figure, ready for a paper.
- Save figures under output_JEVPFN/figures/<phase>/<notebook>/, aggregate captions under
  figures/Captions.md and the complete report under output_JEVPFN/All Results.md.
- Verify actual files/runtime/evidence before claiming success. Synthetic tests establish
  contracts, not CUDA/checkpoint/provider compatibility or full Experiment 0 acceptance.
- Run `python -m pytest -q` before finishing. If a notebook or src/visualize/ changed, also run
  `python -m src.utils.run_notebooks`. Check lint and relevant shell syntax.
- Record substantive changes concisely in docs/CHANGELOG.md, newest first, DD-MM-YYYY.
  Record each cluster run and reusable failures in docs/AGENTS_MEMORY.md. Submission is not
  completion; retain the exact observed status. Keep docs uppercase, durable and concise.

## Shared VSC Library

`vsc-library/` is a pinned, entirely read-only infrastructure submodule. Read its
README, source snapshot, guides and Slurm examples; never edit, refresh, lint,
format or test inside it. There is no PROJECT_SPECIFIC exception. Keep this
project's VSC choices and operating instructions in `docs/VSC.md`, outside the
submodule. Shared corrections belong in the standalone VSC Library checkout.
Record the library pin when operational evidence depends on it.


## Shared Decision Models Library

`dm-library/` is an independent pinned read-only submodule. Read
its summaries, synthesis, papers and official source snapshots for decision-model
claims; cite source paths and symbols and record the pin. Never edit, refresh,
lint or test shared files there. The sole local exception is its ignored
`PROJECT_SPECIFIC.md`, created from `PROJECT_SPECIFIC.template.md`.
Maintain shared corrections in the canonical DM Library under `0. Literature`.
Adding this reference changes no model roster, experimental protocol or execution gate.
