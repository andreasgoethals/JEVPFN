# Current operational evidence

Updated 26 October 2026. README owns the method, VSC owns commands,
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

Inspected Downloads/output_JEVPFN.zip, modified 26 October 2026 10:58:09,
1,404,271 bytes, 474 members with valid ZIP CRCs. It contains DATA logs/manifests,
not all PROJECT data/caches. The archive was read without importing over local evidence.
Do not infer current scheduler state or full-cache validation from this archive.

| Job | Check | Observed result |
|---|---|---|
| 62427444 | CPU/data/questions | Passed; 959 tests, 33 skips, four deselections; exhaustive 18-file question audit. |
| 62427445 | Two-node Lustre lock/snapshot | Passed; quota/independent backup remain unverified. |
| 62427446 | All three encoder checkpoints | Staged/fingerprinted; previous CPU-allocation guard failure resolved. |
| 62427447/48/55/56/60/61/64/65/68/69 | Ten decision checkpoints | Completed in the coordinator receipt. |
| 62427451/53/77 | Liquid/Strands/Perplexity Blackwell images | Logs end with successful seals; Perplexity not yet reconciled in the coordinator. |
| 62427452/54/57/58/62 | Original five exact-token jobs | Completed/reused counts; no GPU inference. |
| 62427476/92 | Liquid/Strands GPFS staging | Failed: copied-link checksum normalization mismatch, fixed locally. |
| 62427463/66/67/70/71 | H2O/SII/Kev/Decision3/Nimble exact tokens | Still progressing in this partial archive; completion unverified. |

The coordinator stopped at 62427476 before any native-GPFS storage or B200 job.
Independent already submitted CPU jobs continue. Do not cancel advancing token jobs
or pull code into their checkout. The owner supplied a read-only diagnostic showing
the mismatches are Tykky `_bin/*` links with duplicate separators (`bin//python`):
sealing used Path normalization but relocation compared raw readlink text. Both now
share the original seal normalization; changed images/links still fail. Do not rebuild
or delete sealed containers. CPU/storage receipts identify source
ab7453fb8590c70734df77ea11c5ddfd3826be0e1cbeb23e65003a49c1a8a2e9,
runtime pins 086f1143adaa6b8811ff01a751db11db70164b293af6670f76c60ad6ffabe232,
and prepared inputs 9ea10eed02750810c48765e692eabb1afef70ed841b253d517ca92bd63b17bf8.
They do not certify the later local fixes. Tokenizer files and registry identity are
unchanged. Explicit `setup_extractors --restart`
requires idle jobs and accounted submissions, keeps previous attempts in the same
receipt, reuses verified sealed images and resumes existing token databases.
The owner subsequently requested a clean parallel debugging rerun. Cleanup is permitted
only after current jobs/manual writers stop; it deliberately discards rebuildable token
output but preserves questions, paid caches, checkpoints and containers. The final frozen
campaign still follows successful setup. No live cluster operation was performed here.

The new coordinator collects independent failures rather than stopping at the first failed
job. Ten separate GPFS probes avoid a global first-model dependency; separate CPU/GPU/
download/build/transfer pools allow 32 queued/running jobs. Setup has 66–76 jobs depending
on reusable images, final 0A has 55. These are implemented plans, not observed VSC passes.
The owner explicitly authorized Git publication for this update; future publication again
requires explicit authorization.
Local validation of this update: 1,003 tests passed, 20 optional-runtime/platform skips;
Ruff, dependency consistency, pinned libraries and changed Bash syntax/LF checks passed.
All six notebooks ran with 25 saved figures, captions and complete summaries. New direct
comparison/control/overlap panels were rendered using isolated synthetic fixtures because
benchmark predictions do not yet exist. These checks do not certify real B200 execution.

Claude's missing durable evaluation-store staging finding was confirmed: recreated
predictor workspaces could refit completed folds and conflict at publication. Staging
now unions the durable prediction/completion SQLite stores for all experiment phases;
end-to-end synthetic CV tests verify reuse and publication. SII's publisher already
filters single-token labels beyond A–Z; new runtime guards also require distinct IDs
and the reviewed cue, without changing tokenized inputs. Statistical reports expose
ideal exact resolution and Holm step-down ranks; analysis remains unapproved. The
review's claim that all A100 references vanished is literally false: rejection guards
and their tests intentionally mention A100; no A100 route was enabled.

Historical evidence retained in this ZIP:

| Job | Check | Observed result |
|---|---|---|
| 62408630 | CPU/data/questions | Passed. |
| 62408631 | Two-node Lustre lock/snapshot | Passed. |
| 62408634 | Liquid H100 contracts | Passed requiring review; 102 diagnostic requests. |
| 62408638 | Strands H100 contracts | Failed on four-decimal rounding; fixed locally, rerun needed. |
| 62408642 | Perplexity H100 contracts | Passed requiring review; 106 requests. |
| 62408646 | Clef H100 contracts | Passed requiring review; 106 requests. |
| 62408650 | Intern H100 contracts | Passed requiring review; 102 requests. |
| 62408653 | H2O image | Sealed; not GPU acceptance. |
| 62408658/63 | SII/Kev images | Empty Miniforge version discovery; now explicit URL/checksum, rebuild needed. |
| 62408635/43/47/51 | Liquid/Perplexity/Clef/Intern H100 dataset pilots | Completed requiring review across 18 datasets. |
| 62408696/97/98 | E5-small/E5-large/Qwen H100 contracts | Passed; Qwen float export now exercised. |
| 62408654/59/64/69/74 | Added-provider token jobs | All stopped at four-hour time limit; committed batches retained. |

No added-provider GPU/pilot completion or B200 completion is present. The four
decision contract/pilot pairs name source fingerprint
8c348bdaf743ac676b1b6cd6a6ff8a550d23a64bf7e5ef8b55eab3d511097b95,
not the current working tree. Encoder reports can be correlated to their job logs,
but lacked explicit source/job fields; new reports record validation identity, job,
cluster and start/end timestamps. Do not infer their exact source from download time.
The five original tokenizers each completed 1,698,127 unique inputs with no overflow.
The five added tokenizers remain partial; do not extrapolate that result to them.
New CPU preparation uses eight workers with a single transactional SQLite writer
and 24-hour segments for the added renderers. Tokenizer source and cache identity
are unchanged, allowing exact continuation of earlier batches.
Relocated B200 container evidence now retains its durable source identity; environment
retirement verifies that identity, the approved GPFS path and unchanged image bytes.
This fixes a false mismatch between original and relocated container IDs without
allowing unverified environment deletion. Real B200 execution is still pending.

The 24 October terminal pulled d12ffda but the B200 implementation was still local.
It rejected --watch before launch. PowerShell commands pasted into Bash also failed;
the helper was not uploaded by that attempt. The owner must publish all local changes
before the corresponding VSC pull. No fresh B200 job was launched by that transcript.

Some telemetry failed because a single
container GPU was index 0 while Slurm used a host index. Only that unambiguous
case now maps to its UUID; missing samples never mean idle GPU. Perplexity had
44 samples, mean utilization 3.84%, peak 68%; diagnostics are not extraction throughput.

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

Read-only pins: DM 5f1b3f8f1611681579aba052a86c2626a6a17b96;
TFM 702d9ab0b66f6639ba8d443ff9c4a91231964465;
VSC 3bcdf55e576d10161d6189b40ac39380bbba9eca.
Reviewed sibling manuscript, CreditICL figure recommendations, MulTaBench v2 and
relevant library guides. LoRA TAR is supervised; frozen task conditioning here
does not train on dataset targets. TILE/STRABLE are adjacent work; do not claim
firstness. Libraries are dated selections, not exhaustive current literature.
H2O's 0.801 winning-probability floor is publisher behavior, not measured confidence.

## Validation and next gates

The latest 26 October local suite passed 993 tests, with 20 optional skips
(15 tokenizer assets, four PyTorch, one POSIX link). Ruff, dependency and diff
checks passed. The prior 25 October check verified all 28 unchanged Bash/Slurm
files. Synthetic/local tests do not establish Blackwell compatibility. Both
historical tokenizer modules and registry remain byte-identical to HEAD.
Earlier validation counts belong in CHANGELOG.

On 26 October, cached local Jev responses were revalidated without network calls:
three synthetic contracts passed; 15 synthetic and 13 real-text robustness cases
decoded successfully and still require semantic review. This does not replace a
fresh source-bound local 0A or re-run the 288-request dataset pilot.

Six notebooks run locally, prioritizing text/reuse, exact questions, feature
widths, saved pilot behavior and paired effects. Missing benchmark results stay
pending. PCA fits training rows only. Across-dataset regression figures use
relative RMSE changes, not averages mixing salary/price/score units. Synthetic
layout data must not be stored as research results.

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
explicit authorization; it was given for the current parallel-testing update.

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
