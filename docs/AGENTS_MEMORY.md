# Current operational evidence

Updated 29 October 2026. README owns the method, VSC owns commands,
DATA_SOURCES owns dataset meanings and LITERATURE_REVIEW owns research claims.
Older details/test counts remain in CHANGELOG/Git. Submission is not completion.

Owner instruction, 28 October 2026: the assistant may commit and push reviewed JEVPFN
changes without asking again. This supersedes the former no-commit/no-push preference.
Read-only task boundaries still apply; never update a checkout used by active VSC jobs.

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
Neither fix changes tokenization, questions or model pins. Decision3's sealed image
and completed token counts can be reused. Retry its branch and the independent TabSTAR
staging/container/GPU chain as documented in VSC; do not clean completed work yet.

The 28 October targeted Decision3 watcher's accounting failure was confirmed by a
manual query at 10:04: slurmdbd connection refused, not a model failure.
The updated watcher pauses/retries read-only accounting errors, retains known job IDs
and diagnostics, and defers new submissions. Ambiguous sbatch responses still require
review. This older stopped attempt needs an idle restart after accounting recovers
and the fix is pulled; never erase receipts, cancel jobs or infer success from this outage.

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

The 29 October import-cache cleanup correction passed 1,151 local tests with 25 skips for optional
staged assets/Torch or POSIX-only behavior. Ruff, dependency, Bash syntax and diff
checks passed. New tests cover input-copy release, preserved mutable queues, failed
publication, queued consumers, capacity budgets, bounded graph residency and cleanup
after failed consumers. No remote deletion or jobs were executed locally. Live release
and the bounded B200 lifecycle remain unverified. Reviewed source changes are authorized
for Git publication. The tokenizer modules and registry were unchanged.

The current-source review confirmed Claude's completed-fold restoration, SII label/cue
checks, token-merge validation, failed-semantic-probe propagation, paired representation
views and runtime-specific GPFS evidence. It found two further issues: the permutation
control moved NaNs between rows, and 0B approval was not bound to the tested evaluation
pipeline. Permutation now preserves complete missingness patterns within each partition;
0B reports/approval carry evaluation_identity for source, model pins/settings, protocol,
selectors and encoder/PCA recipes. Finalize these before 0B; only activation switches and
the acceptance path may change afterward without invalidating that identity.
The earlier review found that TabSTAR selected the original generation. Following the
owner's instruction, it now selects v1.1 explicitly through tabstar 1.2.2 and its distinct
module, with Transformers 4.57.6. CPU staging pins separate classification/regression
snapshots and E5, including offline HF-cache aliases required by the publisher's warm-cache
check. A versioned tabstar-v1p1 container preserves earlier images; native GPFS staging
copies the declared assets. Container retirement proof uses the actual versioned image
path. Bounded predictor contracts are permitted with --execute; full-fold/benchmark gates
remain disabled. This new runtime/checkpoint combination needs real B200 tests.
The review used DM 4cf49431cab921fb727e883760ed535ca8e77f72, TFM
e604ade99f7cf135f41c17a49e1f91da528e835d and VSC
1c29213086a0ae76b8f95cdd8eccbb9db8d6ca87 without moving pins.
The scientific comparison is task-conditioned frozen feature extraction, not supervised
representation learning. The owner declined word/character TF-IDF plus SVD additions;
keep the existing three pretrained embedding encoders and existing cheap-statistics control.

On 26 October, cached local Jev responses were revalidated without network calls:
three synthetic contracts passed; 15 synthetic and 13 real-text robustness cases
decoded successfully and still require semantic review. This does not replace a
fresh source-bound local 0A or re-run the 288-request dataset pilot.

All thirteen notebooks ran locally with saved outputs, 25 PDF figures, aggregated captions
and complete printed reports. Each experiment has its own notebook (0A/0B separate), with
a final synthesis for Experiments 1-3. Empty performance plots and repeated figures were
removed; missing benchmark results remain pending. Every generated PDF was visually checked.
Future result-panel layouts were checked with explicitly synthetic fixtures outside research
output. Their values must never enter research stores. PCA fits training rows only.
Following the owner, headline metrics now match MulTaBench: binary ROC-AUC, multiclass
macro OvR ROC-AUC and regression R-squared. All secondary metrics remain recorded.
Regression headline deltas are R-squared differences, with no relative-RMSE conversion.
Statistical contrast families still require approval. Native-system comparisons are
descriptive; aligned/shuffled/full-distribution controls retain exact artifact pairing.

The 24 October methodological review found combined train/query preprocessing in
the pinned LimiX-2 implementation. The adapter now fits filtering, categorical
inference/encoding and polynomial scaling on training rows only, uses torch.device
and the allocated CPU thread budget, and rejects unreviewed API/multi-GPU paths.
The publisher's RebalanceFeatureDistribution already uses training rows and was
not changed. Real 0B testing remains required. Presence/statistics controls now use
the same empty/nonfinite-text policy as extraction; nonfinite regression targets
are rejected before splitting. Joint-PCA summary curves use matched dataset/split
cohorts across widths, preventing partial completion from changing the comparison.
The 26 October check also aligned embedding input construction with the same
missing-text policy: numeric infinity is missing; literal "inf"/"nan" remain text.
CPU checkpoint staging now accepts CPU Slurm jobs while rejecting GPU allocations;
encoder staging is refused after freeze. Per-column PCA32 and joint-PCA ablations
retain separate training-only projections, including each binary inner partition.

The owner authorized an interim clean debugging rerun once active jobs finish;
use output_cleanup and test_all, without freezing yet. Finish setup before the final
clean campaign. Then stop writers, delete local/DATA/Lustre/GPFS output (including
known runtimes with incomplete staging), regenerate/audit/upload questions, freeze matching source
and run fresh local/VSC 0A. Preserve data/features/weights/images. Acceptance
requires current matching reports and quota/backup/throughput/robustness review.
Create features, run 0B including real encoder/predictor/complete-feature checks,
then benchmark. Full-fold resources and native predictor runtimes remain unproven
until their real gates pass. A source/runtime change requires a reviewed stage
transition, not deletion of mismatch evidence. The owner granted standing commit/push authorization on 28 October; active VSC
checkouts must still remain unchanged until their jobs stop.

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
checks passed. Historical tokenizer files remain byte-for-byte unchanged. These checks
do not establish the new Transformers 5.19/B200 runtime or full-dataset acceptance.

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
