# Jev feature design

Updated 30-09-2026 after the owner's text-only and missing-input decisions. This is still
preparation: no inference client, real responses or feature artifacts exist.

## Input modes

| Mode | One Jev input | Maximum request slots per row with t text columns |
|---|---|---:|
| Per-column | One named text column plus task metadata | t |
| Joint | All available named text columns plus task metadata | 1 |
| Both | The two preceding modes | t + 1 |

Non-text features never enter Jev. TabPFN receives them later alongside selected Jev features.
Task metadata contains dataset/target names, task type and class labels where applicable.
It excludes true row targets, frequencies, target statistics, correlations and model errors.

Skip null/nonfinite or whitespace-only text. For per-column mode, the corresponding feature
columns are missing (`NaN`). Joint mode omits missing fields and uses the available fields;
if every text field is missing, its features are missing too. Missing text has its own trace
status and no paid request. A missing response for **nonempty** text is an error, never silently
converted to missing features. Strings such as literal "[]" remain nonempty inputs; no extra
semantic missing-value detector is invented.

Content hashes share identical requests across rows and modes. In particular, a joint request
with only one available field reuses that field's per-column response. The combined model input
joins the saved separate and joint outputs; it requires no third extraction.

## Numeric outputs

The documented [Noul](https://docs.typesafe.ai/primitives/noul),
[Choice](https://docs.typesafe.ai/primitives/choice) and
[Score](https://docs.typesafe.ai/primitives/score) mechanisms are mapped in `src/jev/provider.py`.
The mapping is inspected offline; live behaviour has not been tested.

| Task | Jev output per request | Proposed first TabPFN view, still for review |
|---|---|---|
| Binary | Noul: probability supporting one fixed class over the other | One probability |
| Multiclass | Choice: all K class probabilities, plus confidence | K probabilities |
| Regression | Score: probabilities over nine levels, plus confidence and expected direction | One expected direction |

Keep the raw response and complete probability vector even if the initial model view uses fewer
columns. Binary orientation is mechanical canonical label order. Choice IDs map back to exact
labels. Confidence is retained for later consideration, not automatically selected for modelling.

For t text columns, let d = 1 for the proposed binary/regression view and d = K for multiclass.
Per-column adds t*d columns, joint adds d, and both add (t+1)*d before removing exact duplicate
views. With two text columns in a binary task, that is **2 separate probabilities + 1 joint
probability = 3 new features**. A K-class vector needs one request, not K requests.

Across the 20 separate schemas, the proposed compact view has 565 per-column and 181 joint
columns. These are sums across datasets, not one pooled matrix. Lossless storage is wider:
1,170 per-column plus 288 joint numeric columns, including all regression probabilities and
available confidence. The plan has **40 tables: 20 datasets times two text representations**.
It does not mean 40 numeric features. Single-text views coincide and need not be duplicated
in a downstream matrix. Selecting/removing those duplicate model columns remains part of the
future experiment assembler; cache keys already avoid duplicate paid extraction.

## What the regression wording issue means

The nine scores describe **direction and strength of evidence**, not target units or predictions.
For a price target, -4 means strong evidence for a lower price, 0 means no directional evidence,
and +4 means strong evidence for a higher price. A score of +2 does not mean two euros.

Currently, -1/-2/-3 and +1/+2/+3 differ mainly by their numeric markers in the descriptions.
The descriptions should also distinguish the strength of evidence. A proposed wording is
weak / moderate / clear / strong evidence on each side of zero. **This proposal is not yet
applied to the frozen templates:** review its interpretation and intermediate wording before
the pilot. No target mean, quantile, range or current-row target will define the levels.

Jev indexes the nine descriptions 0..8. The saved expected direction is
`sum((i - 4) * p[i] for i in range(9))`. Full probabilities remain available even when this
scalar is used. The exact downstream choice is still open.

## Tables and reuse

```text
data/jev_cache/
  responses.sqlite3             future paid responses; no persistent demo database
  features/<artifact_id>/
    features.parquet            csv:N + numeric Jev columns, NaN for empty text
    requests.parquet            csv:N + text group + request_key + status
    manifest.json               dataset/schema/model/template/mapping/file hashes
```

Each dataset has its own tables. `csv:N` is the position in the original pinned CSV, fixed
before splitting. Per-column names identify `text_000`, `text_001`, etc.; the manifest maps
them to source names. Example: `jev__per_column__text_only__text_000__p_positive`.
Joint columns use `all_text`. Neither table includes the target.

The model matrix joins original **non-text features** and selected Jev columns by row ID.
Targets are loaded separately. A comparison using native TabPFN text preprocessing would be
a separate proposed arm, not an implicit change to Jev input.

The cache hashes exact state/question/model/settings and retains origins, timestamps and raw
responses. Transactional completed responses survive restart. Feature files are immutable,
hash-checked Parquet with a completion manifest written last. Mock responses cannot enter
real feature tables. Single-writer transport, reservations, bounded retries, spending control
and reconciliation of server-accepted/client-lost responses still need implementation. Exactly-once
billing across that last interruption window is not claimed. Back up paid artifacts independently.

## Offline cost estimate for the agreed inputs

[Model documentation](https://docs.typesafe.ai/models), checked 30-09-2026, lists
`jev-1.13.0` at **USD 0.042 per million input tokens**, with free output tokens. This is a dated,
configurable assumption in `config/exploration/audit.yaml`, overridden by
`JEV_INPUT_PRICE_PER_MILLION`. Verify the actual price immediately before paid calls.

`python -m src.jev.review` uses the actual 20 datasets, canonical input deduplication and full
API-preview JSON lengths. The central approximation is four characters/token.

| Scope | Requests after empty-input skipping | Distinct requests | Approx. tokens after reuse | Estimated USD |
|---|---:|---:|---:|---:|
| Per-column alone | 3,164,764 | 1,401,423 | 500,299,639 | 21.01 |
| Joint alone | 820,403 | 738,163 | 371,359,956 | 15.60 |
| Both, sharing identical inputs | 3,985,167 | 2,032,351 | 853,391,404 | **35.84** |

A total of 98,493 empty slots are skipped. Do not add all three rows: the last row already
includes both modes. Its cost is below the sum of the first two because it reuses overlapping
per-column/joint requests. No row labels participate in reuse decisions.

Sensitivity assumptions of 3-5 characters/token, 0-100 extra tokens/request and 0-10% billed
retries give **USD 28.68-61.94**. This is a planning range, not a confidence interval, spending
cap or quote. The exact tokenizer, revised wording, overflow policy, taxes and actual usage
can change cost. A first/middle/last-row preview gives 315 slots, 303 distinct requests and
about 152,272 tokens (USD 0.0064); it is an illustration, not the approved pilot sample.

At the dated published 40 requests/second, 2,032,351 requests alone imply at least 14.1 hours;
latency, token limits, account limits and retries can make execution much longer. No duration
has been measured and no spending is authorized by this estimate.

Machine-readable estimates and sensitivity scenarios are in
`output_JEVPFN/feature_creation/results/review/`. The feature plan and approximate context
screen are in the same phase's `results/`. The older four-input-variant costs are superseded.

## Important unresolved data issues

Zomato contributes about 75% of raw text tokens; some rows substantially exceed the documented
32,000-token state-plus-question limit. No silent truncation or chunking is implemented.
The current estimates still include those inputs and must be recalculated after a length
policy is agreed. Official text includes names, URLs, identifiers and numeric-looking strings.
Spotify has 114 classes; salary bands are classification; Mercari predicts log-price. Inspect
these task meanings and prediction-time proxies before freezing prompts or evaluation.
