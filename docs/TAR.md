# Experiment 3: target-aware representations

TAR adapts E5-small-v2 using the dataset's training targets, then feeds its vectors to
the same six predictors used in Experiments 1 and 2. It is supervised representation
learning; the frozen decision features remain task-conditioned and label-free.

## Source and workflow

The MIT-licensed training, encoding and dependency files in
`src/evaluation/vendor/` are copied byte-for-byte from
[MulTaBench commit bc9e871](https://github.com/alanarazi7/MulTaBench/tree/bc9e87130df9884ccd49d04055599c0eda9f3238).
`SOURCE.json` records every file hash; `tar.upstream()` verifies them before import.
The integration is in `src/evaluation/tar.py`. The relevant upstream path is
`embeddings/compute.py::_finetune` → `finetune_e5` → `finetune_with_lora`, followed
by `encode_texts_with_e5` and `encoders/column_encoders.py::ColumnEncoder.fit`.

For each of the five shared outer folds, one E5 adapter is trained jointly across
the sorted text-column names. Only outer-training rows and their targets enter training.
The internal validation split is made by source row before expanding rows into text-column
examples. Upstream uses 10%, capped at 1,000, with two additional binary validation rows.
Regression targets are discretized into 20 quantile bins within the outer-training set.
These bins also use the internal early-stopping holdout, as in upstream; outer test labels
remain excluded. The best validation-loss checkpoint is retained, without a final encoder refit.

The original LoRA settings are rank 16, alpha 32, dropout 0.1, last three layers,
batch 256, learning rate 0.0001, at most 50 epochs and patience three. The pinned
**code** uses weight decay 0.001; this differs from the paper's 0.01. Seeds and the
four data-loader workers are preserved. The training head is discarded. Inference
uses masked mean pooling and L2-normalized 384-dimensional vectors.

Each fold's adapter and vectors are prepared once and reused by all six predictors
and both PCA recipes. There are **18 × 5 = 90 encoder fits** and
**18 × 6 × 2 = 216 predictor configurations**, each with five outer fits (1,080).
Upstream does not train encoders inside an inner threshold-CV loop. TAR therefore uses
fixed-0.5 binary decision metrics; it produces no inner-OOF threshold predictions.
Compare default-threshold metrics across families; AUC does not depend on the threshold.

## Explicit project differences

- We retain JEVPFN's shared all-row five-fold CV, dataset selection and fixed text-column schema, predictor versions,
  missing-text flags, metrics and storage contracts. MulTaBench uses bounded random holdouts.
- Upstream standardizes each embedding coordinate using training rows. TAR preserves
  that, then uses the requested separate PCA32 or joint PCA64, instead of its per-column
  PCA30. Only text vectors enter PCA; non-text columns bypass it. Its sklearn auto solver
  and PCA seed 42 are preserved; the shared outer splits retain the project split seed.
- Upstream's `passage: <column>: <value>` prefix, 512-token truncation and missing-value
  placeholder are preserved and truncation is counted. Experiment 2 instead uses its
  frozen encoding/chunking policy without coordinate standardization. TAR versus frozen
  E5 is consequently a comparison of complete pipelines, not an isolated fine-tuning effect.
- Interrupted training keeps its incomplete directory for diagnosis. Only complete,
  checksummed bundles are reusable; an incomplete adapter is never declared finished.

## Storage and 0B evidence

Immutable bundles live on native GPFS PROJECT under
`output_JEVPFN/6. experiment_3/results/tar/<dataset>/fold_<n>/`: adapter files,
one float32 vector array per text column, and a manifest binding source, checkpoint,
training/test row IDs, training-label digest, optimizer/validation membership,
runtime seal, timing, truncation and file hashes. Each dataset has one transactional
`tar_pca/*.sqlite3` for its five folds' training-only scaling/PCA parameters.
Predictors read these shared assets directly; their mutable prediction stores retain
the established scratch/publication workflow. Temporary Trainer checkpoints use the
job's owned native scratch cache. Standard result export includes completed durable
TAR assets; interrupted bundles stay on PROJECT for diagnosis and are not exported.

0B must validate all 90 real bundles, disjoint internal training/validation membership,
complete finite unit-length vectors, adapter save/reload agreement on **every text column**, both PCA recipes,
and the dedicated sealed `e5_tar` runtime. Reload probes include missing/blank text,
Unicode, duplicates and the longest tokenized input, and compare reordered single-item
inference with batched inference. Both serialized adapter weights and configuration
must exist; a partial save or missing column cannot pass. Its generated
full-fold profiles include TAR for every predictor and PCA recipe. Local tests exercise
identity/corruption rejection, row alignment, unchanged held-out transforms, cached
scaling against an independent sklearn transform, missing flags, fixed-threshold scoring
and resume without refitting. The actual upstream example construction and internal split
are exercised for binary, 30-class and regression targets with a substituted trainer;
these local checks do not train weights. A fresh comparison on 5 November verified all
12 vendored files against the official pinned source byte-for-byte.
These tests do not substitute for B200 training and predictor acceptance.

## VSC preparation after current extraction finishes

Do not update the currently active VSC checkout. After publication and consumer checks,
close the extraction freeze and deploy the reviewed source. Build the TAR runtime
before recording a new freeze: image construction is refused in a frozen checkout.
Then enable `prepare_enabled` in `config/6. experiment_3/tar.yaml` and the authorized
0B switches, and freeze the 0B stage before training.
The TAR benchmark `enabled` switch remains false until 0B is accepted.
Use the current [VSC freeze and review procedure](VSC.md#46-experiment-0b).

Build the separate runtime once, with the already staged E5-small checkpoint:

```bash
mkdir -p "output_JEVPFN/3. experiment_0/logs" "output_JEVPFN/6. experiment_3/logs"
sbatch --wait scripts/slurm/experiment_0_container_build.slurm e5_tar new /gpfs1/project/prj_00013 blackwell-cu130
```

For each selected dataset, submit its five-fold training array and dependent CPU PCA
array. Start with one dataset to validate real runtime compatibility before admitting
the remaining datasets. The job ID is printed and must be retained, not guessed:

```bash
jevpfn_tar_job=$(sbatch --parsable scripts/slurm/prepare_tar.slurm BIN_TEXT_JIGSAW_TOXICITY)
jevpfn_tar_job=${jevpfn_tar_job%%;*}
sbatch --dependency="afterok:$jevpfn_tar_job" scripts/slurm/prepare_tar_pca.slurm BIN_TEXT_JIGSAW_TOXICITY
```

Repeat for the other selected IDs in `config/0. exploration/datasets.yaml`. Queue completion alone is insufficient: the 0B preflight reads
the artifact checksums and PCA stores. Generate predictor-profile arrays through
`src.cluster.benchmark_jobs --phase experiment_3 --model MODEL --profile-fold --write`.
Review and accept them before enabling the final benchmark. No TAR jobs were launched
by the local implementation task.
