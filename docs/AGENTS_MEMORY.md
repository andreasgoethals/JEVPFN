# Operational evidence

This is the dated record of campaign evidence and recovery, not a live status page.
[Experiments](EXPERIMENTS.md) defines the design; the [VSC runbook](VSC.md) gives commands.
Job completion requires terminal accounting and expected artifacts. Historical entries
retain the source, settings and state observed then; later entries supersede their
pending statuses without changing the original evidence.

## Latest verified status

**Evidence through 12 November 2026. Recheck receipts and Slurm before acting.**

| Area | Last verified position | Still required |
|---|---|---|
| 0A | Accepted for eleven open-weight extractors and three frozen encoders; local Jev 1.14 pilot passed | Preserve acceptance and source identities through recovery |
| Open-weight features | Eight providers complete on all 18 datasets and verified locally, including Decision3 | Kev 16/18, Nimble 17/18 and Liquid LFM3 0/18 successful dataset steps in the live campaign |
| Frozen embeddings | All 54 encoder/dataset jobs completed and published; all 54 frozen-PCA jobs completed 0:0. | Verify all training-fold cache keys in full 0B |
| Local Jev | Prefix pilot verified; existing-balance extraction has a cumulative USD 5.50 ceiling | Read hosted receipts for live spend/coverage; full coverage needs further funding |
| Benchmark | Four-experiment design and TAR implemented locally | Real TAR bundles, predictor contracts, full-fold profiles and reviewed 0B acceptance; then separately authorized benchmark |
| Documentation/notebooks | Eight-model snapshot, 12 notebooks passed, 19 figures | Refresh when more complete evidence is imported |

### Feature recovery handover on 12 November

- Decision3 finished all 18 dataset jobs. Releases `11789401`, `11789772`, and
  `11790442` timed out on the Mercari, Kickstarter, and US Accidents publication
  locks. Its 18 scratch stores remain available. The failed release blocks Liquid LFM3.
- Kev restage `11785957` exceeded the working-set budget. Nimble US Accidents
  `11786125` ran out of node-local space during SQLite readback. Nimble release
  `11791678` is publishing the retained stores, with US Accidents readback recorded.
- Recovery waits for `11791678` and the existing controller lease to clear. It
  verifies the operational patch and runs native recovery QA before rebinding
  the original campaign. The handover log is `logs/decision3_resume_20261112.log`
  under feature creation. Local recovery suite passed 204 tests with one Linux-only skip.
- Nimble release `11791678` completed. Native QA `11791981` passed 190 tests
  and the real GPFS publication and readback check. Source `d237af6eâ€¦` rebinds
  campaign `2c6439a7â€¦`. The controller submitted Decision3 release `11791982`.
- Decision3 export `11791980` completed with 36 verified feature blocks on all
  18 datasets. The local import verified the 342,810,544-byte archive's SHA-256
  and retained the seven installed providers, yielding 288 blocks. Its receipt
  is `manifests/portable_decision3_11791980.json` under feature creation.
- All 12 local notebooks reran successfully and generated 19 figures. The
  decision-feature comparison used the eight complete portable providers.
- Retired the redundant staging from export `11774931` after verifying that
  export `11780780` and the local snapshot cover all 252 blocks. Removed 37
  scratch files (2,843,269,437 bytes) and three empty local `7. synthesis` output
  trees. The VSC receipt is `storage_cleanup/decision_feature_export_staging_20261112.json`
  under Experiment 0 manifests.

### Scoped staging and independent 0B handover on 12 November

- At 15:58 CET, the feature campaign had 247 completed steps. Kev staging was
  retrying a 200-GiB capacity estimate for all 18 stores despite only two unfinished
  datasets. Nimble has only US Accidents left; Liquid LFM3 has all 18 left.
- The owner authorized scoped staging, automatic continuation and independent 0B/TAR
  preparation. The old controller was stopped; obsolete stage attempts `11793009`
  and `11793035` were cancelled with their recovery data preserved. Frozen source
  `d237af6e…` was archived as completed revision `746a49d4…` after idle checks.
- TAR image build `11793068` failed before installation because Tykky rejected the
  controller image in PATH. The build now removes only that inherited directory,
  keeping helper calls bound to the absolute verified controller interpreter.
- Native GPFS recovery QA `11793080` passed: publication, repeated publication and
  provider-only warm-cache integrity. CatBoost image build `11793082` completed in
  9m14s. Independent CPU predictor contracts and TAR preparation have separate stores
  and must not block release of an extractor workspace.

### Frozen PCA pilot and rollout on 12 November

- Mindwell CPU job `11792883` completed in 2m21s at source `d237af6eâ€¦`. Readback
  verified SQLite integrity, payload hashes and finite numeric parameters for 20 joint
  PCA64 records and 60 separate PCA32 records (three text columns Ã— 20 training folds).
- Mindwell arrays `11792898` and `11792899` cover the remaining 42 and 11 datasetâ€“encoder
  pairs, with three concurrent tasks per array. The audited submission receipts are in
  `5. experiment_2/manifests/pca_pilot_11792883/` under DATA output. Full 0B still rechecks
  training-row, artifact and parameter identities against the reviewed features.

### Local tree deletion by Google Drive sync on 9 November

`C:\Users\U0152019\PhD Documents` is a two-way Google Drive for desktop *synced folder*
(Drive computer "My Laptop", folder id `1uLZjdywk28gk8h4AmWoU1krAvT9rw4gv`). The local
folder `Projects\5. JEVPFN` maps to the cloud folder **"0. Template - Copy"**
(`1hjawhSwN-fZU4wUFrXzmfzcln6tTSHQc`, created 29 September). It was renamed locally, but
the cloud kept the copy name. That cloud folder was moved to the Drive trash at about
14:44 UTC. `drive_fs_248.txt` shows Drive's own `local_writer.cc:RemoveDirHandlingReadOnlyRecursive`
deleting every local file from 14:44:57.112 to 14:45:15.981 UTC. The deletion ended by
attempting the `5. JEVPFN` root, covered `JEVPFN` and `JEVPFN Paper`, and skipped
directories and read-only files. The logs record no local delete upload beforehand, so
the trash arrived from the cloud side: the web, another Drive client or an API client.
Neither Claude Code sessions (no `trash_file` call) nor Codex (no 9 November session)
trashed it, and isolated pytest reruns did not reproduce the deletion. Drive Activity on
the trashed folder names the actor.

Recovery:

- Code came from GitHub `2e1e0a5` plus a replay of the uncommitted Codex and session edits.
  All 29 pre-deletion copies recovered from the Drive trash match the reconstruction;
  four differ only in CRLF and were restored byte-for-byte.
- Decision inputs and features came from VSC exports, hash-verified (`11780780`).
- The Jev cache came from the trashed Drive copies, merged to 144,653 responses.
- Hosted receipts, extraction logs, event locks and `_control` rounds came from the trash.
- Tokenizers were re-staged and match the original manifests' SHA-256 values.

Notebook outputs are regenerated. The staging folder `JEVPFN_RECOVERY/` and the My Drive
folder `JEVPFN_RECOVERY_from_trash` stay until the owner decides. Lesson: do not keep Git
checkouts or data stores in a two-way synced folder. Exclude the project, or use backup-only.

10 November check: the local `5. JEVPFN` folder and the cloud folder were created at the
same second (29 September, 12:18:25 UTC). This matches a Windows Explorer copy of
`0. Template` whose later local rename never reached the cloud. No cloud `5. JEVPFN`
exists. Drive still binds the local folder to the trashed cloud item: the new
`JEVPFN_RECOVERY` upload sits inside it, and the restored `JEVPFN`/`JEVPFN Paper`
have no cloud counterpart. The restored tree was intact (0 missing tracked files).
GitHub `7e79f05` holds the restored code, which passed 1,313 tests with 26 skips.

### VSC observation on 10 November

At 10:35 CET, dispatcher PIDs 3837469/3838367 (login-3, started 9 November 10:51) were
running source `82e19c09â€¦`. The DATA checkout is `f89688a` plus the reviewed uncommitted
patches. The campaign had 226 of 274 steps completed, with 3 submitted, 41 pending and
4 failed. Complete decision models: Liquid, Strands, Perplexity, Cloudflare, Intern, H2O
and SII. All 54 encoder jobs are complete. Kev release `11784791` was running, and
Decision3 Kickstarter `11781945` and Mercari `11781957` were running.

Four jobs failed after inference with `Insufficient native space for a validated
publication union` (`scratch_stage._merge_native_decision`):

- Kev Kickstarter `11781804`;
- Kev SCImago `11781817`;
- Decision3 US Accidents `11781951`;
- Decision3 Wine Review `11781952`.

These failures need review before retry. Nimble and Liquid LFM3 have not started. No new
complete decision model has appeared since export `11780780`.

Space diagnosis (12:00 CET):

- Running job `11781957` showed B200 `/tmp` as a 590 GiB node-local volume with 212 GiB
  used by co-located jobs. `VSC_SCRATCH_NODE` was unset in the probe step.
- Durable stores reach 62 GiB (Kickstarter, Mercari), so a union needs about 140 GiB.
- Personal GPFS scratch held 77 of 500 GiB; GPFS PROJECT `prj_00013` held 798 GiB of 2 TiB.
- Kev SCImago (3.0 GiB, 119,101 responses) stores 2.1 GiB of repeated `settings_json`,
  about 17.6 KiB per row. Most of that is the execution-container file manifest.

Job outcomes:

- Decision3 Kickstarter `11781945` completed.
- Kev release `11784791` failed with exit 76, a publication-lock timeout during that
  Kickstarter publication. The dispatcher's retry `11784836` reused two published stores.

Commit `283922d` was pushed before the owner asked to hold pushes. The owner then
approved pushing; the deployable source is `6f299b0`.

Cutover preparation (owner-authorized, 10 November):

- **Controller stopped at 12:21 CET.** On login-3, PIDs 3838367/3837469 were stopped
  while holding `.campaign.lock`. The campaign file was unchanged (227 completed,
  2 submitted, 41 pending, 4 failed). Receipt: `reviews/controller_stop_20261110.json`.
  Running jobs `11781957` and `11784836` were left untouched.
- **QA ran in an isolated DATA checkout:** `f89688a`, plus the live uncommitted patches
  reproduced byte-for-byte, plus patch `838d3e1fbb18â€¦`.
  - Recovery QA `11784949` passed 184 tests.
  - Real GPFS fallback QA `11784950` passed. The union ran in
    `publication-temporary/11784950` and the directory was removed; the leftover of
    FAILED `11781804` was retired and that of RUNNING `11781957` was kept.
  - Attempt `11784944` failed only because a test assumed directory order; it is
    retained as history.
  - The QA logs were copied to DATA logs and the QA checkout was removed.
- **Next source:** `80491e04â€¦`, changing only `src/cluster/scratch_stage.py` and two test files.
- **Cutover tool:** `controller_tools/publication_fallback_cutover_838d3e1fbb18.py`. A
  rehearsal refused at `require_idle_vsc` and left the source and freeze unchanged.
- **Detached waiter:** `publication_fallback_waiter_838d3e1fbb18.sh`, PID 3976274 on
  login-3, bounded to 30 h. Once both jobs are terminal and no JEVPFN job is queued, it
  runs the launcher: cutover, source review, then `feature_admission --recover-from 48373214â€¦`.
  Log: `logs/publication_fallback_recovery_838d3e1fbb18.log`.
- Before it fired, the waiter's patch was replaced by `fa21fa4d4d61â€¦`, which also skips
  unchanged republication. QA `11785143` (186 passed) and `11785144` (50 passed, real GPFS
  fallback, scratch left clean) passed on it.

### Cutover outcome and Kev staging (observed 11 November, 14:36 CET)

- **Cutover.** The waiter fired after Mercari `11781957` completed (14:51, 10 November) and
  Kev release `11784836` completed (17:35, published all 18 Kev stores).
  - Recovered campaign `387b9b2aâ€¦` now runs source `2ed9574câ€¦`, with live `scratch_stage.py`
    hash `51cc24caâ€¦`.
  - Decision3 retries `11785955` and `11785956` completed. Nimble staging `11786046`
    completed and 18 Nimble jobs were queued or running.
  - 231 of 274 steps are complete.
- **Kev restaging failed.** Recovery reset Kev staging/release because two Kev dataset steps
  were incomplete. Restage `11785957` failed with `JEVPFN GPFS byte budget exceeded` while
  Decision3's working set was still resident. The empty-workspace release `11786041`
  completed.
  - Kev Kickstarter and SCImago are blocked on that failed stage. Their responses are
    durable through `11784836`; a re-run would reuse them.
  - Plan: retry at the post-extraction recovery.
  - Local fix: staging now exits 77 for a full scratch budget, and the dispatcher retries
    that up to 12 times.

### Intern publication recovery on 9 November

Perplexity release `11778676` passed. Intern staging `11778900` and all 18 Intern
dataset jobs passed, as did all 18 SII dataset jobs and SII release `11779920`.
Intern release `11779759` stopped on the Kickstarter publication lock during
overlapping SII publication. The durable stores contained both feature representations
for all 18 datasets for each model. Export `11780780` added both to the verified local
snapshot, bringing it to 252 blocks across seven providers.

Recovery QA `11781019` passed 183 tests and the native GPFS publication and readback
checks. The exact patch `736a5efde78217150011d33e189f37a4e5229801349af2004a29c50b548460d5`
was applied after confirming an idle checkout and archived freeze. Source
`82e19c09dc7bd14c7a9706825987d709a80fbce6cd353b01a33f587175c881d1` rebinds campaign
`48373214ab16d241344a2756575520e8858b4a498b8ccaec705c235b3babe542`.
The recovered dispatcher retained 193 completed steps. Intern release retry `11781068`
reused the first two stores and entered publication for Kickstarter. The remaining
80 steps are eligible for admission as prerequisites and workspace limits allow.
The historical failed attempt and the previous campaign receipt remain available.

### Recovery and local analysis: 6 November

6 November, unattended recovery. H2O release `11774770` completed at 12:31 CET.
Intern stage `11774808` timed out at 14:32 CET while publishing US Accidents; all
120 Slurm success-dependent jobs were cancelled at zero runtime. Recovery
`11778590` passed SQLite quick_check on an independent rollback preview and matched
the production rollback to it. All ten complete US Accidents artifacts matched the
verified pre-timeout export. Receipt: `reviews/us_accidents_rollback_20261106.json`.

Operational commit `1d9aaeb` passed 1,292 local tests (26 environment skips) and
176 native GPFS tests in `11778670`. The first native QA attempt `11778663` lacked the
isolated checkout's storage pointer. The real timeout test verified `11778665` TIMEOUT,
replacement `11778666` COMPLETED using the published checkpoint, then dependent
`11778668` COMPLETED. QA and rollback were bounded CPU jobs.

An idle, exact-hash deployment used only the seven tested files from the published
commit and preserved the existing extraction science and runtime seals. Cutover
receipt: `reviews/unattended_cutover_d81d3aa13dab.json`. The new source is
`ee179e15e0f9409e271dba54847d132351331ed724fbe14233a9463770540320`.
The first dispatcher reattached 153 successful steps and submitted Perplexity's
release `11778671`. The release failed before publication because the wICE queue
query could not reach Slurm accounting. The verified idle cutover of `2e1e0a5`
adds a distinct retry status for this check. The current source fingerprint is
`6122143bad91868bd482f6d62acf446b2f01f2c480ba76730ef30a7d82785bc6`.
The detached login-2 dispatcher PID 3866025, start ticks 149577238, resumed the
same 274 steps. Replacement release `11778676` passed the queue guard and began
publication with a 12-hour limit. Intern staging follows in the transfer slot.
The VSC log is `logs/unattended_recovery_2e1e0a5.log`.
The isolated QA DATA checkout and synthetic GPFS fixtures were retired after
terminal accounting and preservation of the retry receipt.

The notebook expansion adds label-free vote splits, matching-class correlations,
probability concentration, text-field/duplicate sensitivity, hash-matched cases
and a complete compressed joint-row diagnostic export. It uses the same five
complete local extractors (180 blocks), without provider calls or target values.
The final local suite passed 1,309 tests (26 environment/asset skips), and all 12
notebooks executed successfully with 27 figures. Visually reviewed all 15 figures
in the feature-creation folder at 140 dpi, including 13 comparison figures; checked
665,406 source rows against 36 dataset/weighting summaries and all 54 exact case
payload hashes. Receipt: `output_JEVPFN/manifests/decision_diagnostics_review_20261106.json`.
These diagnostics neither use outcome labels nor certify downstream accuracy.

The 6 November documentation pass aligned the README, experiment guide, citation and
package metadata, operational instructions and research guides. Local validation:
1,309 tests passed, 26 environment/asset skips; Ruff, 104 local links/anchors, grid
arithmetic and metadata consistency passed. Dependency/tool configuration and the
canonical template were unchanged. No VSC inspection, deployment or workload change
was part of this documentation pass. Receipt:
`output_JEVPFN/manifests/documentation_review_20261106.json`.

### Earlier observation: 6 November, before timeout recovery

Durable output and terminal accounting confirmed all 18 datasets for
Liquid d1-3B, Strands, Clef-Flash, H2O and Perplexity, and all 54 frozen embedding
jobs. The remaining 108 decision dataset jobs and 13 prerequisites were submitted
on Mindwell as IDs 11774808â€“11774928, with dependency admission and the existing
workspace limits. At that observation, H2O release 11774770 gated the next batch. Submission
evidence is in the DATA review receipt `dependency_admission_verified_20261106.json`.
The frozen VSC source at that observation was `f89688a`; the recovery above supersedes it.

The five complete providers were exported and checksum-verified locally as 180
feature blocks in 18 portable stores. The first export 11774931 hit a SQLite
publication lock. Replacement 11775001 used shared extraction locks, verified
payload identities in SQLite row order and retained restartable dataset parts.
It completed 0:0 and produced a 1,736,158,920-byte archive. Local import verified
all member hashes before installing the snapshot. Retired only 36 redundant
transfer-stage files (2,029,685,591 bytes) after import; the archive and original
response stores remain available. All 12 notebooks completed locally. Comparison
covered 665,406 source rows with identical missing-text coverage across providers.
Local pytest passed 1,285 tests with 26 environment/asset skips. The 0B notebook
refresh is descriptive and does not execute the 0B acceptance campaign.


### Local Jev preflight: 5 November

 1,698,127 distinct prepared requests, 7,820 matching cached
responses and 1,690,307 remaining. Pilot-token/serialized-size ratios project 1.394
billion further input tokens (about USD 58.56 at USD 0.042/M); this is an estimate,
not an exact tokenizer count. Existing dataset charges are USD 0.271475568 with no
reservations. The owner authorized using the existing EUR 5 allowance only, not the
proposed USD 75 cap. Keep the USD 5.50 cumulative ceiling and existing dataset
allocations. Faster server-enforced token pacing retains hard monetary reservations
and shared 429 backoff. Partial funded output must not be reported as full features.
All 7,820 reusable Jev 1.14 responses and all eighteen raw/prepared identities passed
local preflight. Validation: 1,285 tests passed, 26 platform/asset skips, and lint passed.
The private `feature_creation/jev_funded.local` configuration targets 20 RPS/eight
workers with the existing allocations. Read its hosted/process receipts for actual
execution, spending and terminal status; these offline checks do not prove live throughput.

### Recovery handover: 5 November

The owner explicitly requested deployment/resubmission.
Stopped the verified login-4 controller (Python PID 1239677, wrapper 1239565) under
the campaign lock and verified its controller lease was released. Withdrew only
PENDING H2O IDs 11772705â€“11772722; Slurm reports cancellation with zero runtime.
At that handover, Clef publication 11772829 was running and its checkout was untouched.
Recovery revision `f89688a` passed 1,265 local tests (26 environment/asset skips),
lint and Bash syntax. Native GPFS QA `11772948` passed 169 tests and isolated
publication/union/warm-cache checks, COMPLETED 0:0 in 37 seconds. It also fixes
restaging after a failed batch's workspace was already retired. Successful inference
remains reusable. The exact-hash source review permits co-deploying disabled future
TAR code only with unchanged extraction inputs, inference, installed pins and grid.

At 10:18 CET, the guarded one-time continuation was live on tier2-p-login-1,
PID 716941/start ticks 138774836, waiting for successful publication `11772829`.
Its source is `manifests/recovery_20261105/handover.py`, SHA-256
`703077a0ffb7a85772d25f3a5e08827db75f5ae29dc7873c959732db5d248184`.
The same directory holds exact source review, local validation, process identity
and `handover_status.json`. It holds the controller lease, requires successful QA
and an idle checkout, archives the old freeze, fast-forwards only to reviewed
`f89688a`, checks runtime seals/grid, and starts the recovered feature controller.
Failure preserves evidence and stops; no speculative retry or full Jev/0B execution.
The replacement controller log is `logs/open_weight_recovery_watch_20261105.log`.
Do not claim deployment or resubmission until those receipts show actual progress.
At 10:20 CET the continuation's host PID/start identity was reverified; publication
was running and had finished its first six dataset stores. Both QA jobs were terminal
0:0. Their logs were checksummed into the canonical DATA output before retiring the
44,061,325-byte temporary DATA clone; `qa_checkout_retirement.json` records checks.

### Earlier audit: 5 November, before recovery

The DATA checkout remained clean and frozen at `3f84d7e`.
A read-only audit inspected all 774 available text logs (918,359 lines), reconciled
campaign IDs with Slurm, and checked durable embedding metadata. Liquid, Strands
and Clef completed all eighteen datasets each; Clef's final job `11769377` completed
0:0 in 4h17m06s. All 54 frozen-encoder jobs completed and logged publication; the 54
durable stores have completion identities. Full numerical transfer/0B validation remains.
At 09:25 CET, H2O's eighteen dataset jobs were queued and Clef workspace release
`11772829` was running; later provider batches were not yet admitted.

Nineteen failed steps have two causes: eighteen Perplexity jobs rejected their valid
112,000 MiB allocation because the guard only read `SLURM_MEM_PER_NODE`; Intern stage
`11772686` read a durable cache while publication held its database lock. Local fixes
accept per-CPU allocations and acquire the existing extraction lock before warm-cache
reads. At this earlier audit they had not yet been deployed. Preserve failed logs/receipts
and reconcile consumers before reviewed recovery; no active work was cancelled.

Allocation usage was 556.1 GiB/2 TiB on GPFS PROJECT and 50.57 GiB/500 GiB on GPFS scratch,
with 9,002/300,000 PROJECT files. These include other projects. No quota exhaustion
was observed. The large embedding progress log records per-batch events; it is not
an error or grounds to change the active encoder's fingerprint.

The reviewed benchmark code now uses joint PCA64 and separate PCA32. The next stage
requires verified published features, fold-local PCA and [0B evidence](VSC.md#46-experiment-0b),
including the current generated profile plan. Local CPU package parity covers all 56 control pins. The raw
input audit on 18 datasets found no exact target copies or class recodings. Exact
feature-row overlap in held-out folds was 36.2% for Data Scientist Salary and 9.6% for
Wine Review. These are diagnostics from the shuffled folds.

## Local TAR addition (4 November)

The owner added E5-small-v2 TAR as Experiment 3 and moved native text to Experiment 4.
The pinned MIT training/encoding source is copied unchanged; details and explicit
project differences are in [TAR.md](TAR.md). The grid now has 4,914 configurations,
24,570 outer predictor fits and 90 shared encoder fits. This is a local implementation,
not VSC acceptance: no running checkout was updated and no TAR jobs were submitted.
The active feature-creation evidence above remains tied to its historical source.
Local validation: 1,239 tests passed, 26 environment-dependent tests skipped;
all twelve notebooks executed with fifteen figures and no error cells. Lint,
changed-shell syntax, source hashes and current documentation links passed.
TAR training, adapter replay and full-fold predictor profiles still require real B200 0B evidence.
On 5 November, fresh official-source downloads matched all twelve vendor hashes.
0B replay now covers every text column, difficult input cases and reordered single-item
batches, and rejects missing adapters or another runtime. Offline tests also exercise
the actual upstream binary/30-class/regression example construction and split.
Validation: 1,258 local tests passed, 26 environment-dependent skips; all twelve notebooks
executed and saved fifteen figures without error cells. Lint and the four changed Bash
entry points passed. The study map and 0B qualification panel were visually inspected.

## Authority

Current execution and budget boundaries are maintained in [AGENTS](../AGENTS.md#authorized-work).
The dated migration, cleanup and recovery records below document completed actions;
they do not authorize new submissions, spending or deletion.

## GPFS PROJECT migration and resume (3â€“4 November)

The owner-approved cutover stops admissions before changing roots. Controller
806752/start ticks 121562803 on tier2-p-login-3 was verified and stopped under the
campaign lock. No inference job was cancelled. Liquid release 62527859 completed
0:0 in 2h01m07s and published all eighteen stores. The 36 completed Liquid/Strands
GPU jobs remain historical successes; remaining science is held during migration.
Immutable precopy 62530688 and full closed-store transfer 62533641 completed 0:0.
Native Mindwell verification 11767564 checked 4,579 files (531,705,668,068 logical
bytes) and all 352 SQLite databases. Relocation 62535220 preserved nineteen image
payloads, resealed host wrappers and rewrote 28 external checkpoint links. Private
metadata permissions and read-only raw inputs were verified explicitly.

CPU acceptance 11767661 passed real two-node GPFS locking/snapshot tests, data
preflight and all nineteen installed interpreters/package sets. Migration publication
11767745 completed 0:0 in 13m03s: eight CPU workers checked all eighteen dataset unions
against twelve stopped native workspaces. Every incoming record already existed in
the validated durable superset; no completed result was replaced or lost. Outputs
and recovery records were also published. Legacy-prefix check 11767804 completed 0:0 in 46s, with the old PROJECT hidden.
The independently downloaded 349,974,442-byte numerical bundle passed ZIP, SQLite,
artifact hashes and full local row-count checks for all 72 Liquid/Strands blocks;
its SHA-256 is `6952d5b650dc5b9ebb7dc2266913ca2e624a2387a3efea4be2c0f2166b37ea0b`. Local suite: 1,221 passed, 26 environment-specific skips.
Historical failed/interrupted maintenance logs remain provenance, not science failures.

Measured pre-cutover allocated usage: old PROJECT 495.211 GiB, DATA checkout
0.756 GiB, owned GPFS working sets 108.445 GiB. Destination group is lp_verbekelab
with inherited moderator ACLs and no public access. No owned jevpfn-prefixed build
directory remained on Lustre scratch.

All eleven recorded B200 jobs passed with the old host root hidden. The Cloudflare
allocation 11767880 also validated all three encoder contracts. Migration acceptance
checksummed 21 evidence files. Mindwell job 11769258 completed 0:0 and retired twelve
published native working sets. wICE job 62551120 completed 0:0 in 18m10s, verified
the old inventory and removed the old JEVPFN subtree, including 582 output files
(80.025 GiB logical). The shared Lustre allocation remained. A transient Slurm
accounting outage cleared before terminal verification.

The rebound 274-step graph reused 36 finished dataset jobs and four completed
stage/release steps. Dispatcher 1239565 (start ticks 130594300) on tier2-p-login-4
started 4 November at 10:15 CET on frozen commit `3f84d7e`. Cloudflare staging
11769300 was the first new submission. The graph includes all 162 remaining
decision dataset jobs and 54 encoder dataset jobs.

Records: `2. feature_creation/manifests/storage_migration/` on DATA. New routing uses
native GPFS PROJECT immutables directly, isolated GPFS scratch mutable stores, and
Mindwell CPU preparation. Historical receipts are preserved verbatim; derived
migration records bind relocated seals and retain the previous identities. Eleven
short independent B200 checks replace the cancelled, unstarted two-hour maintenance
allocation 11767669. Normal B200 availability remains an external queue constraint;
no other project's allocations/accounts are used.

## Release recovery (3 November, before migration)

Read all 615 available text logs (101,767 lines) and reconciled every campaign ID.
All 36 Liquid/Strands dataset jobs completed 0:0: eighteen datasets per provider;
72 full feature blocks are expected. The two release attempts, wICE 62524528 and
62524970, timed out after four hours while redundantly merging response histories
on Lustre. No new inference/CUDA/request failure appeared. The expanded campaign
stopped at 38 completed steps, two failed releases and 234 blocked descendants;
no JEVPFN job remained queued/running. Other projects were left untouched.

Operational repair: validate and merge decision snapshots on native GPFS for B200
or node-local disk for wICE transfer jobs, then use
SQLite's transactional backup to publish the union while holding the destination
extraction lock. Release temporary snapshots after each store. Warm new decision
workspaces with only their provider's responses/features, preserving complete
PROJECT history and existing native queues. Conservative provider-specific capacity
estimates retain the two-workspace/200 GiB and personal-quota/headroom guards.
Reviewed recovery reconciles every known ID, preserves completed datasets and the
54 encoder partitions, archives prior source/campaign receipts and refuses changes
to scientific code/configuration/runtime pins. Release CPU jobs use default memory
and a twelve-hour ceiling. Local verification: 1,187 full-suite passes, 25 environment-specific skips; final
34 focused capacity/admission tests passed, including the later snapshot-retirement
case. Ruff and edited Bash syntax checks passed. Numerical extraction/PCA/encoder
implementations and all model pins remain byte-identical. Real deployment/QA outcomes
are recorded below when known.
Linux QA 62527170 completed 0:0 in 6m20s: 101 regression passes, a real
native/durable union (17.3 seconds), unchanged scientific source and 12 saved-store
views with 24 validated blocks. The follow-up node-local routing change passed
61 focused tests locally. Final full-suite verification: 1,189 passed, 25 skipped;
Ruff and shell checks passed. Final Linux QA 62527352 completed 0:0 in 5m56s:
102 tests, real publication using `/tmp` (14.6 seconds), 12 store views/24 blocks.
Both QA runs retained the same numerical source and accepted runtime pins.

Deployed inference checkout is now `1381bd8fe12d07c05e345160bddacce394b3cf17`,
source `3215ad67e0605df20e687f42997747b594e3b1b3af3e8bf0f40e41cc765a9a69`.
Reviewed recovery reused all 36 successful dataset IDs and two successful stages;
reset only failed/unsubmitted work, retaining old attempts. Execution identity
`0835236be120c3df656b407d953c1b3959a3702a7ad7070fc310762fa689f2ab`;
origin identity `3064a8c129a754c8b63bf6746f31c801d86673fb81a9b997c4d6f38f5a3c22d9`.
Controller PID 806752/start ticks 121562803 on tier2-p-login-3 is live. First release
62527859 progressed from PENDING to RUNNING; final check shows normal publication
using `/tmp`, with no errors. The 611 active text logs were rescanned (99,752 lines),
with no flagged failures. Campaign counts: 38 completed steps, one submitted,
235 pending, no current failures/blocked descendants. Embedding and other providers
are not yet submitted: the dispatcher admits them after workspace/storage gates.
The desktop monitor remains absent; full Jev, 0B and benchmarks remain owner-gated.

All 36 production summaries were parsed: each provider processed/reused 1,698,127
unique requests over eighteen datasets, with zero failed requests. Evidence:
`inspection_20261103.json`, `release_recovery_review_20261103.json`,
`source_transition_20261103.json`, `recovery_qa_62527352.json`,
`controller_process.json`, and `logs/open_weight_recovery_watch_20261103.log`.
The native environment directories are empty mountpoints/39 bytes of metadata,
not installed dependency trees. A larger durable-storage move to the separate
2 TiB GPFS PROJECT allocation needs the owner's choice; none was performed.

Template matches the canonical copy byte-for-byte (SHA-256 9e113ac591a7d702bc61060a3bce703419cb7a296fcf9cf77283c1e5ffd27a20).
DATA remains the real Git checkout and compact logs/receipts; installed dependencies
remain sealed PROJECT images, persistent caches stay on PROJECT, and only owned
working copies use GPFS scratch. Snapshot: PROJECT 822.1G/1T, 58,721/150,000 files;
personal GPFS scratch 166G/500G. Failed release snapshots remain protected until
verified publication permits exact scoped retirement. No new storage migration. Eight terminal/historical logs were archived and removed
from active log folders only after archive readback and accepted-evidence-path checks.
`recovery_history/failed_logs_20261103.zip` SHA-256
`69328e816fe92a56f31ed35e505e4dd83c07fdb1022448f32dfbff4fed9d87b2`
was downloaded and independently hash-verified. Old failed-release snapshots may be
retired only when every retained database byte matches, apart from SQLite backup
header counters; mismatches/unknown files stay protected.

## Parallel admission (2 November)

The owner requested all remaining work as parallel as possible. Canonical implementation:
`src/cluster/feature_admission.py`. All ready dataset jobs per staged model can be admitted
together; three unsubmitted full encoder jobs split into 54 dataset jobs. The execution
graph has 274 steps, with the same 198 decision dataset jobs, 396 decision blocks and
54 embedding stores. Cloudflare/embedding staging takes the next genuine free workspace.
The two-resident/200 GiB storage safeguards and single bulk-transfer writer remain; Slurm
controls accepted/running jobs. Do not submit held or duplicate jobs merely to fill a queue.

Frozen inference checkout remains `e50043b8da93a435791e37c24c303fda07761f19`, source
`1e817dc6056d5a7f82c65f604f1ba65145191e824f1dadb5b276c7aed9309307`; original campaign
identity `d2c53785ea532f3231e6cc2d288946a2540f9ccd0e21b62a09e6d41443a648b6`.
The scheduling companion is deployed separately as a checksummed DATA artifact, imported
through the sealed CPU image. Its policy, old receipt, execution identity and controller
process are recorded in feature-creation output. Do not pull new main into VSC consumers.
No model pin, question, numerical feature implementation, dataset scope or Jev gate changes.

Last confirmed progress before handover: Liquid 8/18 datasets, Strands 4/18; 24 feature
blocks. Recovered jobs 11763392â€“11763395 completed and published successfully. Three
known outstanding IDs are 11763745, 11763763 and 11763671. Their current state is unknown
while Slurm accounting/cluster queries fail with `Address family '0' not supported`.
The old controller retained IDs through that outage. After queries recovered, a Strands
Product Sentiment submission timed out inside the site's balance/submit wrapper, leaving
an ambiguous `starting` receipt and stopping the old controller. Reconcile both successful
accounting and live-queue queries before retrying; preserve the diagnostic and review in
the campaign. No fresh acceptance of a
submission may be claimed until an actual Slurm receipt exists. The official status page
was generated 30 October and does not resolve the directly observed 2 November failure.
Snapshot: PROJECT 767.8G/1T, 53,995/150,000 files; personal GPFS scratch 79.93G/500G.
This change introduces no asset migration or additional scratch deletion; the existing
reviewed publication/release jobs remain in the graph.
The desktop progress automation stays removed. Current truth is in
`2. feature_creation/manifests/open_weight_campaign.json` and its controller receipt.
Local validation: 1,182 passed, 25 environment-specific skips; six focused admission tests
also passed on the final artifact. Ruff and diff checks passed. Real VSC admission and
accepted IDs must be verified separately from these local tests.

Deployment verified 2 November, 08:22 UTC: controller PID 2743308, host tier2-p-login-3,
start ticks 112683660. Canonical implementation commit `37a9ea9`; deployed artifact
`controller_tools/feature_admission_1240371794716a88.py`, SHA-256
`1240371794716a884ae5bd6f03a2edcf9f4ddcf40cac21ce4807cd62b07a7fc8`.
Execution identity `f59f7fb6851b709898054d75fc96c48fa6b8cc37983e3e6552c352ba52bafa57`;
policy SHA `695c07f9f6f3ee76c3894b4862fea9746339c01503880b16c392d568c79694ae`.
Three original queued IDs are preserved; the expanded graph has 14 completed steps
(12 decision datasets plus two stages), three submitted and 257 pending. No new Slurm ID
has yet been accepted from the parallel policy. The controller retries the definite
`QOSMaxSubmitJobPerUserLimit` refusal every 60 seconds without crashing or duplicating work.

Live Mindwell `scontrol show assoc_mgr` confirms effective QoS `lp_verbekelab` limits of
500 submitted jobs per user/account and 16 running jobs per user/account. Expanded arrays
show 500 tasks already submitted: 484 pending, 16 running across this user's projects.
Slurm/accounting is responding again; admission capacity is now the blocker. Other projects
were neither cancelled nor reprioritized. Proof and current state:
`parallel_deployment_20261102.json`, `parallel_reconciliation_20261102.json`,
`controller_process.json`, and `logs/open_weight_parallel_watch.log` in feature-creation output.

## Operational recovery (1 November)

Direct accounting confirms no JEVPFN jobs queued/running before repair. Both Fake Job
Posting jobs completed (Liquid 11757487; Strands 11757556), and Liquid Kickstarter
11759420 completed on 31 October, 23:04 CET. Both Jigsaw attempts failed during publication
(11757488, 11757580). Liquid's 99,661 Jigsaw responses and two numeric blocks were saved;
Strands retained 67,000 responses and needs to finish the dataset. No model/CUDA or
recorded request failure appeared in the 21 production text logs scanned.

The controller stopped on the definite `QOSMaxSubmitJobPerUserLimit` refusal for Strands
Kickstarter. Its manifest remained stale; scheduler accounting, not that `running` label,
establishes the old attempts' terminal state. Recovery must reuse successful IDs and
saved responses, retry the two failed jobs and the refused submission, and continue the
unchanged 223-step grid. None of the three full encoder jobs has started yet.

Repairs replace the global publication lock with destination-specific locks, restrict
decision publication to the assigned dataset, and keep source snapshots on native scratch.
A busy periodic target defers without killing inference; final publication still verifies
the transaction. Named submission ceilings wait for the next poll; uncertain acceptance
still stops for review. `--recover --reviewed-by NAME` records old attempts and reconciles
all IDs before reusing completed work. Inference, question, embedding and merge-validation
implementations are unchanged. Recovery QA and exact replacement IDs live in the phase's
`manifests/recovery_qa_*.json` and `open_weight_campaign.json`; submission is not completion.

Quota snapshot: PROJECT 760.3G/1T, 52,415/150,000 files; personal GPFS scratch 59.79G/500G;
Lustre scratch 4.262G/500G; DATA 46,278M/76,800M. No quota is exceeded. GPFS PROJECT
`prj_00013` is now visible (2T, empty); no migration to it is implied or performed.
Keep scratch responses, token recovery shards and accepted 0A evidence. The scheduled
desktop monitor remains removed at the owner's request.

Local repair checks: 1,176 tests passed, 25 environment-specific skips; the focused
34 publication/campaign checks, Ruff and both edited/new Slurm scripts' Bash syntax passed.
Linux job 62516047 completed 0:0 in 7m46s: 64 regressions, unchanged numerical-source
checks and saved-block integrity inspection. Lustre random reads fell below 1 MiB/s;
the inspection finished just before a stop request reached Slurm, so no allocation was
interrupted. Future inspections use node-local SQLite snapshots and print per-store
progress (syntax checked; not needed for this already completed review). This was a QA
performance issue, not an inference or integrity failure.

## Initial open-weight feature campaign (historical)

Production freeze: Git `954c93fdb29290e85cb98bf3a374674d466e5ef8`, source
`9b216097e5780189b8e3e547b52a69ee6dc9f27661df7ac686f92b3eccc3332d`. Handover CPU job
62478168 completed 0:0. Campaign record on DATA:
`output_JEVPFN/2. feature_creation/manifests/open_weight_campaign.json` (223 steps:
198 decision Ã— dataset B200 jobs, 3 encoder jobs, stage/release pairs; 396 decision
blocks and 54 embedding stores expected). Controller PID 538048 on a login node, receipt
`2. feature_creation/manifests/controller_process.json`.

Monitor `jevpfn-feature-creation-progress` was created 14:22 UTC and **removed at the
owner's request** at 14:43 UTC without cancelling VSC work; delivery authorization is kept.
Local record: `output_JEVPFN/2. feature_creation/manifests/progress_monitor.json`.

Direct read-only inspection, 17:20 UTC:

| Step | Job | Observed |
|---|---|---|
| liquid stage | wICE 62478247 | COMPLETED 0:0 |
| strands stage | wICE 62478308 | COMPLETED 0:0 |
| liquid Fake Job Posting | Mindwell 11757487 | COMPLETED 0:0, 25m05s |
| strands Fake Job Posting | Mindwell 11757556 | COMPLETED 0:0, 41m40s |
| liquid Jigsaw | Mindwell 11757488 | **FAILED 1:0** at 17:08 UTC after 99,648 new + 13 reused responses, 0 failed requests |
| strands Jigsaw | Mindwell 11757580 | **FAILED 1:0** at 17:10 UTC after 66,990 new + 10 reused responses |
| liquid Kickstarter | Mindwell 11759420 | PENDING (Priority) |
| strands Kickstarter | none | `starting`, no job ID: the controller's `sbatch` returned exit 1 |

- Both Jigsaw failures are `TimeoutError` waiting 600 s for the global
  `/lustre1/project/stg_00211/JEVPFN/.scratch-publication.lock`, then
  `RuntimeError: Periodic durable publication failed; scratch is retained`. Inference
  completed; the responses remain in the GPFS working sets and are not yet on Lustre.
- The watcher crashed at 16:07 UTC with `CalledProcessError` from that `sbatch`
  (`open_weight_campaign_watch.log`); the manifest was last written then and still lists
  both Jigsaw jobs as `submitted`. No controller process was visible from login node
  tier2-p-login-4. No JEVPFN job was cancelled or resubmitted.
- Open issues for the owner: lock contention between concurrent publications, the
  ambiguous `starting` step and controller restart. 11759420 still consumes the frozen
  checkout, so no source may be pulled there yet. Scratch responses must be published
  before the 30-day purge.

## Gates passed

Historical acceptance evidence, primarily 29â€“30 October. Later relocation/recovery gates
are recorded above; these passes do not qualify a changed benchmark runtime automatically.

| Evidence | Result and scope |
|---|---|
| CPU image build 62468642 | Passed, 27m02s; sealed PROJECT controller. |
| Controller integration 62469013 | Passed, 3m44s; DATA-to-PROJECT cutover recorded. |
| Input install 62470782 | Passed; eighteen independently validated question stores. |
| Native GPFS 11753505 | Passed; cross-node exclusion/reacquisition, release and snapshot restore. |
| Basic B200 migration 11753506 | Passed CUDA 13.0, native sm_100 and BF16; not decision inference. |
| Corrected controller integration 62471351 | Passed, 4m04s; four launch checks, two-node storage, eighteen datasets, 1,139 CPU contracts. |
| Frozen VSC 0A | 192/192 steps, all 0:0, on `73807ee`: CPU/data/question checks, exact token audits (eleven providers Ã— eighteen datasets, zero overflow), eleven decision contracts/pilots, three encoder contracts. |
| 0A operational review | CPU review/export 62477605 passed eleven evidence gates and backup; approval 62477630 signed eleven provider and three encoder acceptances. |
| Jev 1.14 local 0A | All eight steps on `d24b79c`; 1,163 tests (25 skips), 31 live diagnostics, 288 dataset-pilot calls. Bounded review accepted. |
| Jev 1.14 first-100-row pilot | Eighteen datasets, 7,643 requests, 36 numeric blocks, every cell rechecked against its raw response. Validation only. |
| Earlier local 0A (Jev 1.13) | All eight steps on `73807ee`; superseded for new Jev calls. |
| Historical predictor setup | TabSTAR v1.1 bounded checks passed; full predictor feasibility is 0B work. |

## Experiment 0A evidence

VSC 0A: Git `73807ee6c193377b99134196e69e6ee9538410be`, frozen source
`eb20d0e666fd1c9a0bed5e12c73b043221f293eaf4ae5ab855e3e1a0178cd2b3`, receipt
`3. experiment_0/manifests/0A_vsc/20261029T215821Z_8d152b91.json`. The campaign status
stays `vsc_checks_completed_requires_review`; the separate signed acceptances approve the
providers. Job IDs, report checksums and the removed monitor `jevpfn-0a-progress` are in
`3. experiment_0/manifests/progress_monitor.json`.

- Review 62477535 exposed a filename collision (`tokens_liquid_*` also matched
  `tokens_liquid_lfm3_*`), not bad counts. Selection now requires exact provider metadata,
  with two regression cases. Both validation freezes were closed explicitly.
- Question v3 covers **1,698,127 unique inputs and 3,294,340 row/slot links**; no labels or
  non-text values enter payloads; empty text has no request. Eight missing upstream source
  snapshots were restored from Git and verified against catalogued SHA-256 values.
- Backup bundle `transfers/JEVPFN_20261030T131627Z_0654b30c.zip`, 312,931,407 bytes,
  SHA-256 `c96df498ccdbf86ed4b9a516e8b8fd84af27a88f512dbad8c2ba201483d549c9`, downloaded and
  imported locally (raw responses, reports, fixtures; merged token databases stay on PROJECT).
  The twelve signed acceptances have a local ZIP copy, SHA-256
  `c4c073773ef942cdbe8187a414616082ab19755409c7597b8e5ad06a194fa1e4`.
- Review retained instruction/option sensitivity and warm median request latencies of
  0.026â€“0.298 s; batch sizes and pins stay fixed. Bounded observations are not population
  robustness or a full-run guarantee.
- The backup first imported the VSC freeze as active local control. The receipt was kept
  under `manifests/imported_revisions/`; imports now always route foreign freezes there
  (receipt `storage_cleanup/imported_freeze_recovery_20261030.json`).
- Before handover: 1,167 local tests passed (25 skips), 92 focused checks after enabling
  extraction, all notebooks executed. Inference adapters, tokenization, checkpoint readers,
  encoders and publication code are byte-identical to the tested `73807ee`.

Local Jev 1.14: source `c2c72bfd0a49d26e2446f19df0529897fa1ea4efd3e8914add26027afc0a6b8e`,
receipt `0A_local/20261030T081306Z_c9a72019.json`, review
`before_extraction/accepted_jev14_pilot.json`. It does not validate the VSC source.

## Jev accounting

Historical validation charges below are not a live account balance. Current funded-run
spend and reservations must be read from the hosted receipt and dataset ledgers.

- 1.14.0 at USD 0.042/M input tokens, 64k/request, 32k state plus question, 32 MiB body
  (checked 30 October). Organization limits 80 RPS/100,000 input TPS do not remove the
  per-attempt reservations. There is no exact local Jev tokenizer.
- 1.13 dataset pilot: 288 requests, 238,111 reported input tokens, USD 0.010000662; fresh
  diagnostics USD 0.000667506. All ledgers, responses and historical acceptances are kept.
- Prefix pilot: the first attempt stopped on missing raw US Accidents inputs after six
  datasets' numeric blocks and the seventh's responses. Resumed receipt
  `hosted/25afa73213ed4041b1574daa2b691833.json` completed on unchanged source, model and
  budget (first seven datasets cache hits; PID 14100 exited). All 7,643 responses decode;
  461 empty row/groups stay missing. Dataset-ledger charges including historical tests
  total USD 0.271475568, zero reservations.
- Real-text sensitivity persisted in 1.14: instruction change moved a probability by 0.48
  with a class flip; multiclass option reversal had total variation 0.07 without a flip.
  Reviewed explicitly; neither semantic consistency nor calibration is guaranteed.

## Storage

Historical reset, controller and retention evidence from 29â€“30 October. The later
[GPFS migration](#gpfs-project-migration-and-resume-34-november) supersedes the old
allocation paths and usage; none of these totals is a current quota check.

- Controller identity `f04a2aef749aea1e56a397289aab237465a6f9dbc4cd6a8494d500bf305a2b7d`
  under `PROJECT/JEVPFN/containers/control-v1`. The old `DATA/conda_envs/JEVPFN` prefix is
  retired after 11753506; shared Miniconda and other projects were preserved.
- Post-0A inspection, 30 October: no JEVPFN jobs; shared PROJECT 643.1G/1T and
  49,516/150,000 files; personal DATA 45,945M/76,800M, GPFS 9.876G/500G, Lustre scratch
  4.262G/500G. JEVPFN uses 753M DATA, 422G PROJECT, 2.6G GPFS recovery stores.
- Storage audit, 30 October 10:01 CET (no changes made): PROJECT 421.1 GiB (268.7 GiB
  checkpoints, 66.7 GiB containers, 78.4 GiB token evidence incl. 39.0 GiB recovery shards);
  GPFS 90.7 GiB; build scratch 28.07 GiB. Live timestamps confirm the reset: DATA output
  from 21:57:23 UTC, 0A from 21:58:21 UTC, PROJECT from 22:00 UTC on 29 October.
- 29 October: 416 GiB of JEVPFN working sets plus 84 GiB of CreditICL exhausted the 500 GiB
  personal GPFS allocation; this motivated the two-resident/200 GiB budget.
- Approved maintenance 62474729 (0:0, 11m12s) verified twelve B200 image seals and removed
  25 build leaves (30,141,542,400 bytes); receipt
  `storage_cleanup/build_retirement_20261030T092018Z.result.json`.
- Token-shard retirement is deferred: scans 62477662, 62477905, 62477933 were stopped
  (deep SQLite verification on Lustre too slow). Keep ~39 GiB through the first verified
  production run (`storage_cleanup/token_shards_retirement_deferred_20261030.json`).
- The final authorized reset removed local output and 84.2 GB of VSC output before 0A.
  Independent compact backup `PROJECT/uploads/local_0A_eb20d0e6.zip`, SHA-256
  `ab0e4140ebee5d3d7485f7bf5cd3155f530e4726078d2a2c73abcfae61aa651a`.
- Local raw-data incident, 30 October 10:10â€“10:11 CET: all sixty raw CSV/metadata/marker
  files were recycled; the owner did not delete them and no responsible process was found.
  Do not attribute it to another agent or claim a compromise. All sixty were restored from
  the Recycle Bin and hash-checked (`storage_cleanup/local_raw_restore_20261030T094200Z.json`).

## Reusable failure lessons

- A single global publication lock with a 600 s timeout fails long jobs that finish
  together; inference survives on scratch, but the job reports failure.
- A controller that crashes on `sbatch` exit 1 leaves a `starting` step without a job ID.
  Check accounting before resubmitting; never retry an ambiguous submission blindly.
- Git Bash `ssh` cannot use the Windows agent; call `C:\Windows\System32\OpenSSH\ssh.exe`.
- `srun` stdout is not pure JSON. Use job-bound receipts; retry only absent receipts.
- Immutable prepared SQLite files must copy byte-for-byte; use snapshots for mutable caches.
- Merge stores transactionally; never overwrite populated caches or reset ambiguous paid jobs.
- Disable checkpoint bytecode. Linux hash reuse includes ctime; Windows rehashes. Preserve
  pinned publisher code and frozen tokenizers.
- Slurm scripts set roots and runtime explicitly; exports are not inherited. Verify the
  interpreter, not the Conda prompt.
- Fix only a confirmed blocked directory's permissions, never recursive ACLs.
- Output cleanup rejects links/mount escapes; never recursively remove empty data folders.
- Filename prefixes are not provider identities; match exact metadata.
- Notebooks need atomic writes, shared locks, real kernels and partial-output recovery.
- Empty text is its own overlap cohort; report cohort metrics without changing splits.
- Require every row, text group and class order; missing extraction is failure, not missing text.
