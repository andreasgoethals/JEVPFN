# Current operational evidence

Updated 27 October 2026. README owns the method, VSC owns commands,
DATA_SOURCES owns dataset meanings and LITERATURE_REVIEW owns research claims.
Older details/test counts remain in CHANGELOG/Git. Submission is not completion.

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

Owner quota report, 26 October 08:44: shared Lustre PROJECT 473.3 GB/1 TB and
40,313/150,000 files; DATA 55,045/76,800 MB and 528k/10M files; GPFS scratch
9.537/500 GB. These are dated observations, not current capacity. Compact
containers addressed the previous ~111k-file native-environment problem.
Never delete working images or other projects to make space.

## Latest downloaded VSC evidence

Inspected Downloads/output_JEVPFN.zip, modified 27 October 2026 13:53:18,
1,526,721 bytes, 833 members. It contains DATA logs/manifests, not complete PROJECT
count/response databases. The setup receipt ended setup_completed_with_failures:
151 completed, five failed, 20 blocked, all terminal. Its new attempt began
27 October 07:05 UTC; job receipts share source fingerprint
422b33e71f48ca08f25377350e37587ab83969c1f1b8b2e6137a1b310b289621,
runtime pins 42474dae6611c00955f851a072da2b996ecbc4d5f479c677f5bd55e0badc6e84,
prepared inputs 9ea10eed02750810c48765e692eabb1afef70ed841b253d517ca92bd63b17bf8.
These precede the new merge changes; do not present them as current-source passes.

| Check | Latest observed result |
|---|---|
| Recovery | Recorded accepted GPFS job 11739854 at 07:05:36 UTC; saved Slurm user/account/checkout/command match. Restart succeeded. |
| CPU, 62440843 | Passed. |
| Lustre, 62440844 | POSIX two-node lock/reacquisition/snapshot checks passed; the earlier controller failure was retried successfully. |
| Liquid/Strands/Perplexity/Clef/Intern token audits | All merged: 62440903/04/27/28 and 62441362. Each covers 1,698,127 distinct inputs with zero overflow. |
| All ten tokenizers | All 100 partition receipts completed. H2O/SII/Kev/Decision3/Nimble still need their strict final merges. |
| Added-provider merges | H2O 62441778, SII 62441719, Kev 62441464, Decision3 62441465, Nimble 62441727 exhausted 4-hour limits. TIMEOUT with ExitCode 0:0 is a failure. |
| Five GPFS working sets | Liquid 11741548, Strands 11741549, Perplexity 11741576, Clef 11741555, Intern 11741560 passed. Quota/independent backup remain separate reviews. |
| Five B200 decision contracts | Liquid 11741550, Strands 11741551, Perplexity 11741577, Clef 11741556, Intern 11741563: 26 diagnostic groups each passed, actual NVIDIA B200, runtime token-count parity confirmed. |
| Five B200 dataset pilots | Liquid 11741553, Strands 11741554, Perplexity 11741578, Clef 11741562, Intern 11741565: each covered 18 datasets, 414 completed requests, 36 reused and no failed requests. |
| Three B200 embedding contracts | E5-small 11741557, E5-large 11741558, Qwen3-Embedding-0.6B 11741559 passed on four-row fixtures: three unique texts encoded, missing inputs skipped, resume without inference. Not complete-dataset embeddings. |
| Five added providers | Their scratch, GPFS lock, B200 contract and dataset pilot stages were blocked by token merges, not attempted model failures. |

Real-text sensitivity remains a research limitation requiring review. Clef changed
its top answer on irrelevant-context/instruction variants (max probability change
about 0.555); Intern changed its multiclass answer after option reversal (about
0.517). Synthetic semantic anchors passed. Keep fixed questions and class order;
do not tune the main questions against target labels to conceal these diagnostics.

Merge bottleneck: the old path made per-input count/blob lookups against large
prepared-token B-trees and rolled back at timeout. It now bulk-compares/inserts
counts, sequentially scans token blobs once per partition in key order and verifies
checksums/counts in bounded batches. A covering key index stays inside the same
SQLite file for final coverage joins; cache is bounded to 256 MiB. Conflicts and
incomplete partitions still roll back the entire merge. Added merges get 24 hours
and phase/progress/elapsed logs. Complete partition receipt/hash reuse skips costly
monolithic-prefix recopying. Tokenizers/registry/questions remain unchanged, so all
completed count work can be reused. Live performance of the fix is not yet measured.

After publishing/pulling with idle JEVPFN jobs, restart the five unfinished provider
branches using setup_extractors --provider flags, retaining previous evidence in the
same receipt. This is 87 diagnostic steps with images reused, not fresh feature
creation. Do not clean yet. Once all setup gates and robustness/storage reviews
pass, perform the agreed clean regeneration/freeze and complete all-provider 0A.

Historical lessons: GPFS BSD flock allowed simultaneous owners; POSIX record locking
now has real two-node GPFS and Lustre passes. Perplexity staging omitted source/uv.lock;
its corrected scratch copy and B200 tests now passed. Tykky link normalization uses
the same rules for sealing and relocation. Compact images addressed the former
~111k-file native-environment problem. H100 evidence is historical; B200 receipts
above now demonstrate bounded inference for five decision models and three encoders.

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

All eleven extractors receive the same canonical task/question/text/class meanings.
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

The latest 27 October update passed 1,089 local tests in 266.46 seconds,
with 22 platform/optional-runtime skips. Ruff, dependency and diff checks passed.
Regression tests exercise incomplete/altered partitions, damaged/conflicting saved
tokens, atomic rollback, bounded/bulk copying and verified partition resume. The
reviewed tokenization modules and registry remain byte-identical to HEAD; their
cache identities are unchanged. Library pins were not changed.
No remote execution, paid calls, cleanup or Git publication occurred in this task.
Local tests do not establish the new merge's Lustre performance or added-model B200
compatibility. Earlier validation counts remain in CHANGELOG.

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
transition, not deletion of mismatch evidence. Publication requires the owner's
explicit authorization for the current task; the owner handles Git publication.

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
- Cleanup rejects links/mount escapes and touches output only. Windows attribute
  recovery must not change POSIX ACLs. Never recursively remove empty data folders.
- Notebooks need atomic writes/shared locks, real kernels, partial-output recovery
  and protection against concurrent user edits.
- Empty text is a separate overlap cohort. Repetition across row folds is an
  interpretation concern; report paired cohort metrics without changing splits.
- Require every expected row/text group and class order. Missing extraction is
  failure, not missing text. A finite passing suite never proves all bugs absent.
