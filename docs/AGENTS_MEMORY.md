# Current operational evidence

Updated 29 October 2026. README owns the method, VSC owns commands,
DATA_SOURCES owns dataset meanings and LITERATURE_REVIEW owns research claims.
Older details/test counts remain in CHANGELOG/Git. Submission is not completion.

Owner instruction, 28 October 2026: the assistant may commit and push reviewed JEVPFN
changes without asking again. This supersedes the former no-commit/no-push preference.
Read-only task boundaries still apply; never update a checkout used by active VSC jobs.

## Authorized control migration, 29 October

The owner now authorizes recreating/testing the CPU control image, live Slurm debugging,
verified cutover, final local/VSC output reset, full 0A, evidence review/acceptance and
open-weight feature creation. Full Jev extraction waits for the owner. Read-only SSH
at 16:44 found JEVPFN idle, PROJECT 538.3G/1T and 45,347 files, GPFS scratch 8.634G/500G.
The existing control Python is 3.12.15. Its package versions are frozen in
experiment_0/control_runtime.yaml. CPU image build 62468642 passed (27m02s), with sealed ID
f04a2aef749aea1e56a397289aab237465a6f9dbc4cd6a8494d500bf305a2b7d.
Read-only upload validation 62468736 passed all 18 members; none matched the old
installed file bytes, so a later installation replaced them. Local question audit passed
18 datasets, 1,698,127 inputs and 3,294,340 links. Control test 62468965 passed Slurm
accounting/scheduling, two-node Lustre locks and CPU/data contracts; its overall
status failed because the monitored model wrapper rejects another enclosing image.
Earlier self-integrity failure was Tykky masking bin: preserve the read-only
/JEVPFN_CONTROL_HOST bind and verify all original bytes there. The native wrapper
relocation now has policy version 2, recognizing its own installation prefix before
assuming it is already inside its target image. Original PROJECT GPU image seals
remain unchanged. Job 62469013 passed all integration/storage/data checks (3m44s),
and the PROJECT control image is now selected by its persistent cutover receipt.
Input install 62470733 failed before replacement because --noheader also hides
Slurm cluster labels. Maintenance now queries both clusters separately, exempting
only its exact current allocation. Input install 62470782 passed (2m25s),
replacing all eighteen files with the independently validated upload. A queued
dependent consumer correctly blocks installation too; submit staging only after
the globally quiescent installer finishes. Local suite: 1,144 passed, 25 platform/
optional skips; compact-backup and uncertain full-campaign submission checks also pass.
Native stage 62470816 passed (4m32s); GPFS storage 11753505 passed (1m43s), and
B200 control/model-image launch 11753506 passed (1m06s) with CUDA 13.0, native sm_100
and BF16 kernels. This basic migration check is not full model inference acceptance.
The old DATA/conda_envs/JEVPFN prefix is absent; the persistent retirement receipt
embeds the B200 evidence. Shared Miniconda and all other environments are preserved.
Working-copy release 62470859 failed safely because the executing CPU image masks
host wrapper files. Release now verifies and reads the unmasked host view as staging
already does; retry 62470887 passed (47s), publishing all eighteen cache stores and
removing 32,202,732,578 bytes of temporary inputs/images. The completed control build tree was
removed, with a persistent control.build-cleanup.json receipt.
At 21:29 eight tracked MulTaBench reference snapshots disappeared locally during
concurrent workspace activity. Their deletion is uncommitted and was not made by this
task. The owner confirmed the deletions were unintended; all eight files were restored
from Git and every catalogued SHA-256 matches. The local source discrepancy is resolved.
The latest full local suite passed 1,147 tests with 25 platform/optional skips.
Fresh-round Jev diagnostics and 0A orchestration additionally passed 22 focused tests.
Full extraction and predictive execution remain disabled until the fresh 0A review.

First clean source dadf9a5: local 0A passed all eight steps, including 31 new Jev
diagnostic calls (USD 0.000667506) and 288 reused dataset-pilot responses. Known real-text
instruction and option-order sensitivity was reviewed and retained as a limitation.
VSC CPU 62470909 passed the exhaustive questions/data checks but failed three synthetic
legacy-activation fixtures: they inherited the live control route instead of selecting
their fake Conda prefix. Make that selection explicit; do not weaken image verification.
Also clear a stale interpreter before any activation branch. The remaining independent
token jobs continue on the old frozen VSC source; no remote pull while they run.
The interactive CPU route hit QOSMaxCpuPerUserLimit. A recorded pending-job reroute to
batch ran successfully through the tests. Future CPU preflight defaults to batch with
three cores and partition-default memory (3400 MiB/core verified live), avoiding Slurm's
implicit expansion of the previous two-core/8 GiB request. A final corrected 0A is still required.
The broader local run exposed two cleanup-fixture failures after real diagnostic rounds
existed: switching isolated_output to local mode still read real compact output. Keep
its local manifests in its temporary notebook-output root as well. Eighty focused
cleanup/ledger/path tests passed; no production route or paid ledger was bypassed.

## Current scope and direct access, 29 October

Owner removed Experiments 4/5. Only 0A/0B and Experiments 1–3 remain; statistical
analysis and the combined notebook are under `7. synthesis`. Do not reintroduce
winner-selected combinations, permutation/presence/length controls or a lexical
baseline without instruction. The primary grid remains 3,996/2,268/54 configurations.

TEMPLATE.md matched the canonical `0. Template/docs/TEMPLATE.md` byte-for-byte,
SHA256 285c7c321b2cd52d8d415e30f41fe775ceb47113f39007c2490fdb0ec23fc8d8.
Manual shared Windows-agent access worked without renewal. Native ssh.exe, `-F none
-T`, BatchMode, host-key checks, bounded timeout and an LF script on stdin preserved
stdout and a deliberate exit 7. This gateway dropped separate remote stderr;
`exec 2>&1` preserved it in stdout with the exit status. Use that tested workaround.
Never initiate authentication; ask the owner for `step ssh login --context VSC`
only if the shared certificate needs renewal. Instructions are in AGENTS/VSC.

Read-only inspection around 15:41 local time: actual checkout and .git reside on
DATA; remote HEAD 5b7768f969ec9159ab192788706bf0d18dfa4c2d, clean. No JEVPFN jobs
were active. Quota: PROJECT 538.2G/1T and 45,343/150,000 files; DATA 56,233M/76,800M;
GPFS scratch 8.634G/500G. The persistent CPU control environment is still
`$VSC_DATA/conda_envs/JEVPFN`; replacing it with a verified PROJECT CPU container
requires a separately reviewed migration/cutover. No remote mutation, environment
migration, cleanup or job submission was performed during this change.

The supplied cleanup log records 2,571 output files removed (84,510,558,770 bytes).
Raw/prepared data, paid caches, weights and runtimes survived. At 16:11, read-only SSH confirmed the uploaded ZIP (1,118,154,339 bytes), its checksum
sidecar and frozen receipt, all timestamped 15:51. No JEVPFN jobs were active. The
sidecar reports 8c63720e68a5e7e3c0e6a0a971d917f739f102a0f49d4ac24e54726fa9e7084d;
the installer must still validate the ZIP. The inputs were prepared with a22a785 and
the uploaded frozen source receipt is obsolete.
Question generation itself is unchanged by the scope/access edits: reuse verified
inputs, but record/upload a new freeze after local and VSC source agree. Do not
accept the old receipt as evidence for the revised code.

## Execution decision and storage

The owner selected Mindwell B200 without requesting a new GPFS allocation.
Their shell confirmed /gpfs1/scratch/383/vsc38338 and no /gpfs1/project/stg_00211.
The durable master remains /lustre1/project/stg_00211/JEVPFN. CPU preparation
stages per-runtime native working sets under
$VSC_SCRATCH_GPFS1/jevpfn-work/<runtime>/JEVPFN. DATA keeps logs/manifests.
Committed decision/embedding/evaluation SQLite results are snapshotted/merged to
durable Lustre periodically and at command exit; failures retain scratch for recovery.
Scratch is not a backup. Never stream bulk model/data inputs from Lustre on B200.

Separate blackwell-cu130 images preserve existing cu126 images. Exact pinned
wheel availability is checked before building; absent wheels stop the build.
Startup checks actual B200, native sm_100, BF16 operations and float export.
These checks are not real model acceptance. Both native GPFS and durable Lustre
need two-node lock/snapshot tests; every provider/encoder needs B200 execution.
The final index check on 24 October found the exact Torch 2.12.1/vision 0.27.1
and Perplexity Torch 2.14.0/vision 0.29.0 wheels under cu130, not cu128.
The B200 profile therefore uses CUDA 13.0, preserving package pins, and requires
a live driver-580-or-newer check plus actual GPU kernels before model loading.

No A100 route is allowed. Default jobs no longer use H100. Cancellation checks
live ownership, H100 partition and checkout paths; CPU and other projects stay.
Do not update source used by active jobs. No remote cancellation/submission was
performed during this implementation.

Owner quota report, 29 October 10:24: Lustre PROJECT 628 GB/1 TB and 46,067 files;
DATA 56,234/76,800 MB; personal GPFS scratch 499.8/500 GB. JEVPFN working copies
used 416 GB, CreditICL 84 GB. Containerization fixed the earlier inode problem,
but retaining every runtime's byte-heavy working set exhausted scratch space.
Do not touch other projects. Native GPFS I/O remains required by the VSC Library
STORAGE guide at pin 1c29213086a0ae76b8f95cdd8eccbb9db8d6ca87, rechecked 29 October.

Setup/frozen 0A now bound staging-to-release residency to two workspaces, with a
200 GiB JEVPFN budget and 20 GiB personal headroom. Release publishes results then
removes copied inputs/images, preserving mutable queues/features/predictions and all
durable assets. Failed publication blocks new stages. Predictor staging omits decision
token databases. Manual live release passed; the bounded full graph still needs VSC validation.

## Latest downloaded VSC evidence

Downloads/output_JEVPFN.zip modified 29 October 14:47:58, 3,000,218 bytes:
- CPU 62465912 passed dataset/question checks and its contract suite.
- Decision3 GPFS 11751621 passed; B200 contract 11751677 passed all 26 checks
  (124.9 seconds). Pilot 11751721 covered 18 datasets: 414 new, 36 reused,
  zero failed requests. Token merge 62463241 had already passed in 2h47m55s.
- TabSTAR GPFS 11751766 passed; predictor 11751786 passed 16 fixtures: binary,
  3/30-class multiclass and regression, each with original text and three augmentation
  modes. Resume without refitting passed; full_dataset_profile=false.
- Cleanup now applied successfully for all existing runtimes, followed by successful
  Decision3/TabSTAR restaging and release (62466455 and 62466583). Personal GPFS
  was 8.66/500 GiB at 14:11, versus 499.8 before recovery. Durable assets and mutable
  results were retained. No inference bug appears in these new logs.
- Decision3 uses correct but slower reference kernels when causal_conv1d and
  flash-linear-attention are absent. No dependency change was made. Pilot GPU
  telemetry averaged 36% (peak 100%); a bounded test is not a throughput benchmark.
- Decision3 real-input diagnostics changed the multiclass top answer after an embedded
  instruction. This is a recorded sensitivity, not proof of invariant interpretation.
Local review of f359d40 retained its notebook redesign but fixed empty-cache reporting,
duplicate dataset counting across arms/splits, lifetime-cache provenance labels and
serialized text-length accounting. Request length does not establish information content.
Validation: 1,153 passed, 25 skipped; Ruff passed; 13 notebooks and 15 figures checked.

All these receipts share source identity dc721879d9e6bd1705509b198d9c6c52585542d365feb5d30fca8b99d7b61147
and precede Claude's f359d40 notebook changes and this review. They establish setup
compatibility, not a fresh frozen 0A. The new two-workspace graph still needs a full
campaign validation. No remote jobs, deletion, downloads or paid calls were run here.

The preceding full setup ended with 174 completed, one failed and one blocked job.
The table below is historical evidence from that campaign, not current-source acceptance.
It matches pre-fix checkout 5a50643; all twelve selected extractors need current-source
acceptance after the final freeze. The ZIP contains DATA reports, not full PROJECT databases.

| Check | Latest observed result |
|---|---|
| CPU, 62447461 | Passed: 1,074 tests, 33 optional/platform skips, four notebook deselections. Dataset/question checks also passed. |
| Lustre, 62447462 | Two-node lock/reacquisition/snapshot checks passed. |
| All ten token audits | All final merges passed, including the five earlier timeouts. Each covers 1,698,127 inputs over 18 datasets, with zero overflow. Maxima range from 3,027 tokens (Kev) to 4,551 (Nimble). |
| Native GPFS storage | All ten runtime-specific working sets passed two-node storage checks. Quota and independent backup remain separate reviews. |
| Nine B200 decision contracts | Liquid 11744467, Strands 11744468, Perplexity 11745277, Clef 11744669, Intern 11744758, H2O 11747508, SII 11745712, Kev 11747524 and Nimble 11745716: 26 checks each passed on actual B200. |
| Original five pilots | Liquid 11744793, Strands 11745024, Perplexity 11745337, Clef 11745025, Intern 11745260: each covered 18 datasets, reused 450 requests and reported zero failed requests. These reruns were cache reuse, not fresh pilot inference. |
| Four new pilots | H2O 11747514, SII 11745721, Kev 11747525, Nimble 11745722: each covered 18 datasets, completed 414 new requests, reused 36 and reported zero failed requests. |
| Decision3, 11745759 | Failed before inference: the publisher calls torch.cuda.set_device and rejects indexless cuda. Its dataset pilot was blocked, not run. |
| Three B200 embedding contracts | E5-small 11744670, E5-large 11744671, Qwen3-Embedding-0.6B 11744672: four-row fixtures passed, with missing-input skipping, long-text chunks and cached resume. These are not full-dataset embeddings. |
| TabSTAR staging, 62447473 | Classification download completed; regression download lost the connection (httpx.RemoteProtocolError). No new TabSTAR image/GPU acceptance is present. |

Decision3 now receives cuda:<current logical index>, after Slurm's device mapping.
Local regression tests reproduce the old error and cover logical devices 0 and 2.
TabSTAR staging retries transport failures at most three times, reusing the pinned
Hub cache; HTTP/integrity errors still stop and no incomplete manifest is accepted.
Neither fix changed tokenization, questions or model pins. Their successful 29 October
retries are recorded above; these failures are retained only as historical diagnosis.

The 28 October targeted Decision3 watcher's accounting failure was confirmed by a
manual query at 10:04: slurmdbd connection refused, not a model failure.
The updated watcher pauses/retries read-only accounting errors, retains known job IDs
and diagnostics, and defers new submissions. Ambiguous sbatch responses still require
review. That older attempt was superseded by the successful retry above. Never infer job
failure or success from a scheduler outage; preserve receipts before recovery.

Real-text sensitivity remains a research limitation requiring review. Earlier Clef
instruction/context variants and Intern option reversal changed predictions despite
passing synthetic semantic anchors. Fixed questions and class order remain required;
do not tune main questions against target labels to conceal diagnostic sensitivity.
Final acceptance still requires the frozen campaign and robustness/storage review.

Historical lessons: the token merge's bulk inserts, sequential token scans and compact
lookup index resolved the five merge timeouts. GPFS BSD flock allowed simultaneous
owners; POSIX record locking has real two-node GPFS/Lustre passes. Perplexity staging
now preserves source/uv.lock; Tykky sealing/relocation normalize links identically.
Compact images addressed the former ~111k-file environment problem. H100 evidence is
historical; current receipts demonstrate bounded B200 inference for eleven decision
models and three encoders.

## Jev and permanent accounting

Pinned jev-1.13.0 synthetic contracts passed: 1,051 reported tokens and
USD 0.000044142 recorded cost. A prior HTTP401/reservation remains in history.
The later local pilot completed 288 requests across 18 datasets, 238,111 input
tokens, USD 0.010000662, no pending reservations; receipt
hosted/9f844b4ee73c4ece9c3dacff37fb611e.json. These are diagnostic samples;
full extraction remains disabled.

The owner confirmed 80 RPS/100,000 input TPS on 21 October from the console.
Recheck before extraction. Conservative 64k per-request billing reservations
are separate from account per-second limits; do not replace them with an
unverified tokenizer estimate. Timeout does not imply an unbilled request.
Jev two-decimal and Strands four-decimal rounding have separate bounded
projections; preserve raw outputs/scores/usage, never claim recovered precision.
Other providers retain their validated policies.

Credentials stay in the explicitly authorized ignored local secret file.
Never print/upload them. Environment overrides take precedence. Paid responses,
queues, recovery/spending history and smoke state are permanent data; cleanup
preserves them. Unknown outcomes require reviewed recovery, not blind retries.

## Inputs and literature

Question v3 supersedes v2 evidence. Mechanical audit: 665,406 selected rows,
80 text columns across datasets, 1,698,127 unique inputs, 3,294,340 slots,
79,461 empty slots, 1,516,752 reused nonempty slots. All 20 raw datasets remain;
Spotify/Zomato are excluded from extraction/evaluation. Exact counts differ by tokenizer.

All twelve extractors receive the same canonical task/question/text/class meanings.
Native framing, tokens, attention, readouts and confidence fields differ. Shared
numeric schema does not imply equal probabilities. Exclude provenance URLs,
non-text row values and row targets from model text; retain sources separately.
Mechanical checks cannot prove language interpretation or calibrated confidence.

The 24 October literature/methodology review used read-only pins:
DM 5f1b3f8f1611681579aba052a86c2626a6a17b96;
TFM 702d9ab0b66f6639ba8d443ff9c4a91231964465;
VSC 3bcdf55e576d10161d6189b40ac39380bbba9eca.
The 27 October storage/Slurm check used VSC Library
1c29213086a0ae76b8f95cdd8eccbb9db8d6ca87 (upstream
7c04105490d610fee13125c8c70fa4bddd854a88); no pins were changed in this task.
Reviewed sibling manuscript, CreditICL figure recommendations, MulTaBench v2 and
relevant library guides. LoRA TAR is supervised; frozen task conditioning here
does not train on dataset targets. TILE/STRABLE are adjacent work; do not claim
firstness. Libraries are dated selections, not exhaustive current literature.
H2O's 0.801 winning-probability floor is publisher behavior, not measured confidence.

## Validation and next gates

Current scope/access validation: 1,134 tests passed, 25 optional/platform skips;
Ruff, pip check and all 31 shell syntax/LF checks passed. Eleven notebooks execute
with 13 saved PDFs, captions and complete printed reports. The revised study map
was visually inspected. Concurrent execution from the owner's PowerShell caused
three protected recovery saves; those notebooks passed after the owner closed it.
Separate 0A/0B notebooks and the synthesis remain.
Missing benchmark results stay pending; no synthetic performance enters stores.

Methodological protections retained: evaluation_identity binds 0B approval to code,
settings, protocol and exact feature recipes; transforms/PCA fit only training rows,
including binary inner folds. LimiX preprocessing was corrected to use training
rows only. Missing text skips inference; missing extraction is a failure. Joint
PCA curves compare matched complete datasets. Headline metrics are binary ROC-AUC,
multiclass macro OvR ROC-AUC and regression R-squared. Full predictions and secondary
metrics remain stored. Statistical contrast families still require approval before
outcomes; native-system results are descriptive. The removed controls cannot be
cited as evidence that any gain is specifically semantic.

TabSTAR v1.1, package 1.2.2 with Transformers 4.57.6, has historical bounded B200
fixtures and resume evidence. Its separate pinned classification/regression models
and E5 cache aliases remain required. Current 0B and full-fold feasibility are pending
for all predictors. ConTextTab and AG-MM still require their own staged runtime tests.

The final sequence is reviewed setup (including the control-runtime storage migration),
a deliberate clean output reset, prepare/audit/upload unchanged question-v3 inputs,
a matched source/runtime freeze, local and VSC 0A, accepted full feature creation,
complete-feature/predictor 0B, then Experiments 1–3. Do not reset again while uploads,
question audits or jobs are running. The latest task permits documentation/config edits
and read-only remote inspection, not silent migration, scratch deletion or workloads.
Paid response/accounting state always survives output cleanup. Revalidation may reuse
cached requests and must disclose that reuse rather than claim fresh inference.

## Liquid d1-2B addition and latest review

The owner authorized Liquid d1-2B alongside d1-3B on 28 October. The active catalogue
is `src/jev/catalog.py`; the fingerprinted registry.py, tokenization.py and
decision_tokenization.py remain unchanged. d1-2B uses its own `liquid_lfm3` assets,
image and cache identities, pinned at cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41.
CPU token jobs and GPU tests use its isolated Transformers 5.19 image. Explicit mean
score readout preserves the expected-direction design; the publisher's default is median.
Its lfm3.py answer-lead registration matters and is read without importing Torch on CPU.
All full-extraction/evaluation gates remain disabled. Model weights and live B200 tests
were not run locally. The twelve-extractor grid has 3,996/2,268/54 configurations in
Experiments 1/2/3; the current all-runtime setup has 215 jobs, 204 with all images reusable
(including eleven workspace release jobs).

Claude's TabSTAR concern was checked against the pinned publisher source. Epochs were
selected on TabArena fold 0, not an unspecified suite; documented pretraining filtering
includes MulTaBench. This is not independent proof of absence of pretraining exposure.
No experimental redesign or unrequested statistical contrasts were introduced.

Local validation of this addition: 1,124 tests passed and 22 skipped (unavailable old
tokenizer assets/Torch or POSIX-specific behavior). The d1-2B publisher tokenizer/source
fixture was explicitly exercised on CPU, without weights or inference. All 13 notebooks
executed, all 25 PDFs were visually inspected, and Ruff, pip check and 30 shell syntax/LF
checks passed. Historical tokenizer files remain byte-for-byte unchanged. These local checks alone did not establish runtime acceptance. The subsequent bounded
B200 pass is historical setup evidence; full-dataset acceptance remains pending.

## Reusable failure lessons

- srun stdout is not pure JSON. Use job/probe-bound receipts; briefly retry only
  absent receipts after success, never repeat completed writes.
- Immutable prepared SQLite files must copy byte-for-byte: backup changes headers
  and invalidates token evidence. Use snapshots for mutable caches.
- Merge stores transactionally; preserve conflicts/accounting. Do not overwrite
  populated caches with partial remote copies or reset ambiguous paid jobs.
- Disable checkpoint bytecode. Linux hash reuse includes ctime; Windows rehashes.
  Preserve pinned publisher code and frozen tokenizer implementations.
- Slurm scripts must establish roots/runtime explicitly; submitting exports are
  not inherited automatically. Verify the interpreter, not the Conda prompt.
- Raw inputs need reads; installation needs writes. Fix only a confirmed owner's
  exact blocked directory, never recursively alter ACLs.
- Output cleanup rejects links/mount escapes and touches output only. The separate
  scratch-release helper removes verified working copies as described above. Windows
  attribute recovery must not change POSIX ACLs. Never recursively remove empty data folders.
- Notebooks need atomic writes/shared locks, real kernels, partial-output recovery
  and protection against concurrent user edits.
- Empty text is a separate overlap cohort. Repetition across row folds is an
  interpretation concern; report paired cohort metrics without changing splits.
- Require every expected row/text group and class order. Missing extraction is
  failure, not missing text. A finite passing suite never proves all bugs absent.
