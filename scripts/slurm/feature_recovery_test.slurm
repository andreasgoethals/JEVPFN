#!/bin/bash -l
# Operational recovery QA only: no inference, API calls or accepted-report replacement.
#SBATCH --job-name=jevpfn-feature-recovery-qa
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=00:20:00
#SBATCH --output="output_JEVPFN/2. feature_creation/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/2. feature_creation/logs/%x_%j.log"
set -euo pipefail
cd "${VSC_DATA:?}/JEVPFN"
export JEVPFN_PHASE=feature_creation
export JEVPFN_STAGING_ROOT=/lustre1/project/stg_00211
export JEVPFN_DURABLE_ROOT="$JEVPFN_STAGING_ROOT"
export JEVPFN_RUNTIME_PROFILE=blackwell-cu130
export JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh
"${JEVPFN_PYTHON:?}" -m pytest -q tests/test_feature_campaign.py tests/test_scratch_stage.py tests/test_cache_merge.py tests/test_parallel_diagnostics.py tests/test_freeze.py
"$JEVPFN_PYTHON" - <<'PY'
import json, os, sqlite3, subprocess
from contextlib import closing
from pathlib import Path
from src.cluster.freeze import source_snapshot
from src.jev.features import load_block
from src.utils import paths
from src.utils.serialization import write_json

unchanged = [
    'src/jev/self_hosted.py', 'src/jev/extended_runtimes.py', 'src/jev/catalog.py',
    'src/jev/checkpoints.py', 'src/jev/tokenization.py', 'src/jev/decision_tokenization.py',
    'src/jev/registry.py', 'src/jev/provider.py', 'src/jev/requests.py', 'src/jev/inputs.py',
    'src/jev/question_bank.py', 'src/jev/features.py', 'src/jev/merge.py',
    'src/data/text_embeddings.py', 'src/cluster/token_shards.py',
]
for name in unchanged:
    old = subprocess.check_output(['git', 'show', '954c93f:' + name])
    assert old.replace(b'\r\n', b'\n') == Path(name).read_bytes().replace(b'\r\n', b'\n'), name
stores = []
for runtime in ('liquid', 'strands'):
    for tier, root in (
        ('scratch', Path(os.environ['VSC_SCRATCH_GPFS1']) / 'jevpfn-work' / runtime / 'JEVPFN'),
        ('durable', Path(os.environ['JEVPFN_DURABLE_ROOT']) / 'JEVPFN'),
    ):
        for dataset in ('BIN_TEXT_FAKE_JOB_POSTING', 'BIN_TEXT_JIGSAW_TOXICITY', 'BIN_TEXT_KICKSTARTER_FUNDING'):
            p = root / 'data/jev_cache' / (dataset + '.sqlite3')
            if not p.is_file():
                continue
            with closing(sqlite3.connect(p.as_uri() + '?mode=ro', uri=True)) as db:
                count = db.execute("SELECT COUNT(*) FROM responses WHERE kind='verified_response'").fetchone()[0]
                tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                ids = list(db.execute('SELECT artifact_id,manifest_json FROM feature_artifacts')) if 'feature_artifacts' in tables else []
            blocks = []
            for aid, encoded in ids:
                m = json.loads(encoded)
                if m['identity']['provider'] != runtime or m.get('purpose') == 'validation_only':
                    continue
                block = load_block(p, aid)  # Verifies checksums, identity and saved table shapes.
                blocks.append({'artifact_id': aid, 'rows': len(block.values),
                               'mode': m['identity']['representation'], 'status': m['status']})
            stores.append({'runtime': runtime, 'tier': tier, 'dataset': dataset,
                           'verified_responses_all_providers': count, 'blocks': blocks})
report = {'status': 'operational_recovery_checks_passed', 'job_id': os.environ['SLURM_JOB_ID'],
          'source': source_snapshot(), 'unchanged_scientific_files': unchanged, 'stores': stores,
          'meaning': 'Operational QA and saved-block integrity; no new inference or new 0A acceptance'}
destination = paths.manifests_dir('feature_creation') / ('recovery_qa_' + os.environ['SLURM_JOB_ID'] + '.json')
write_json(destination, report)
print(json.dumps({'status': report['status'], 'report': str(destination), 'stores': len(stores)}))
PY
