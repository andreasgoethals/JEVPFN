#!/bin/bash -l
# Operational recovery QA only: no inference, API calls or accepted-report replacement.
#SBATCH --job-name=jevpfn-feature-recovery-qa
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=01:00:00
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
"${JEVPFN_PYTHON:?}" -m pytest -q tests/test_feature_campaign.py tests/test_feature_admission.py tests/test_scratch_stage.py tests/test_scratch_capacity.py tests/test_cache_merge.py tests/test_parallel_diagnostics.py tests/test_freeze.py
"$JEVPFN_PYTHON" - <<'PY'
import json, os, shutil, sqlite3, subprocess, tempfile
from contextlib import closing
from pathlib import Path
from src.cluster.freeze import source_snapshot
from src.cluster.scratch_stage import _publish
from src.jev.features import load_block
from src.jev.merge import snapshot
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

# Exercise real native merge + transactional PROJECT backup on a small completed
# dataset before retrying large release jobs. No GPU/model/API call is made.
publication = _publish(
    Path(os.environ['JEVPFN_DURABLE_ROOT']) / 'JEVPFN',
    Path(os.environ['VSC_SCRATCH_GPFS1']) / 'jevpfn-work/liquid/JEVPFN',
    dataset_id='MUL_TEXT_PRODUCT_SENTIMENT',
)
assert publication['decision_stores'] == 1

def inspect_store(source, runtime):
    # Lustre random reads were <1 MiB/s in the first inspection. Stream a
    # consistent snapshot once, then check its indexes/blobs on node-local disk.
    local = Path(os.environ.get('SLURM_TMPDIR') or '/tmp')
    if shutil.disk_usage(local).free < source.stat().st_size + 512 * 2**20:
        raise OSError('Insufficient node-local space for recovery inspection')
    print(json.dumps({'event': 'inspect_store_started', 'source': str(source)}), flush=True)
    with tempfile.TemporaryDirectory(prefix='jevpfn-recovery-', dir=local) as temporary:
        p = snapshot(source, Path(temporary) / source.name)
        with closing(sqlite3.connect(p.as_uri() + '?mode=ro', uri=True)) as db:
            count = db.execute("SELECT COUNT(*) FROM responses WHERE kind='verified_response'").fetchone()[0]
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            ids = list(db.execute('SELECT artifact_id,manifest_json FROM feature_artifacts')) if 'feature_artifacts' in tables else []
        blocks = []
        for aid, encoded in ids:
            m = json.loads(encoded)
            if m['identity']['provider'] != runtime or m.get('purpose') == 'validation_only':
                continue
            block = load_block(p, aid)
            blocks.append({'artifact_id': aid, 'rows': len(block.values),
                           'mode': m['identity']['representation'], 'status': m['status']})
    print(json.dumps({'event': 'inspect_store_completed', 'source': str(source), 'blocks': len(blocks)}), flush=True)
    return count, blocks

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
            count, blocks = inspect_store(p, runtime)
            stores.append({'runtime': runtime, 'tier': tier, 'dataset': dataset,
                           'verified_responses_all_providers': count, 'blocks': blocks})
report = {'status': 'operational_recovery_checks_passed', 'job_id': os.environ['SLURM_JOB_ID'],
          'source': source_snapshot(), 'unchanged_scientific_files': unchanged, 'stores': stores,
          'native_publication': publication,
          'meaning': 'Operational QA and saved-block integrity; no new inference or new 0A acceptance'}
destination = paths.manifests_dir('feature_creation') / ('recovery_qa_' + os.environ['SLURM_JOB_ID'] + '.json')
write_json(destination, report)
print(json.dumps({'status': report['status'], 'report': str(destination), 'stores': len(stores)}))
PY
