#!/bin/bash -l
# Operational recovery QA only: no inference, API calls or accepted-report replacement.
#SBATCH --job-name=jevpfn-feature-recovery-qa
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=01:00:00
#SBATCH --output="output_JEVPFN/2. feature_creation/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/2. feature_creation/logs/%x_%j.log"
set -euo pipefail
# A separately published DATA checkout permits QA while the old publisher finishes.
cd "${1:-${VSC_DATA:?}/JEVPFN}"
case "$(pwd -P)/" in
    "$(cd "${VSC_DATA:?}" && pwd -P)/"*) ;;
    *) echo 'Recovery QA checkout must be on this account DATA.' >&2; exit 2 ;;
esac
source scripts/storage_environment.sh
export JEVPFN_PHASE=feature_creation
export JEVPFN_STAGING_ROOT=${JEVPFN_PROJECT_ROOT:?Select the allocated durable PROJECT root}
export JEVPFN_DURABLE_ROOT="$JEVPFN_STAGING_ROOT"
export JEVPFN_RUNTIME_PROFILE=blackwell-cu130
export JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh
"${JEVPFN_PYTHON:?}" -m pytest -q tests/test_feature_campaign.py tests/test_feature_admission.py tests/test_scratch_stage.py tests/test_scratch_capacity.py tests/test_cache_merge.py tests/test_parallel_diagnostics.py tests/test_freeze.py tests/test_shell_environment.py
"$JEVPFN_PYTHON" - <<'PY'
import json, os, sqlite3, subprocess, sys, tempfile
from pathlib import Path
from src.cluster.freeze import source_snapshot
from src.cluster.scratch_stage import _publish, _provider_cache
from src.utils import paths
from src.utils.serialization import write_json

# Synthetic helpers on real isolated GPFS files; production stores stay untouched.
sys.path.insert(0, str(paths.REPO_ROOT / "tests"))
from test_cache_merge import cache, request, response
unchanged = [
    'src/jev/self_hosted.py', 'src/jev/extended_runtimes.py', 'src/jev/catalog.py',
    'src/jev/checkpoints.py', 'src/jev/tokenization.py', 'src/jev/decision_tokenization.py',
    'src/jev/registry.py', 'src/jev/provider.py', 'src/jev/requests.py', 'src/jev/inputs.py',
    'src/jev/question_bank.py', 'src/jev/features.py', 'src/jev/merge.py',
    'src/data/text_embeddings.py',
]
for name in unchanged:
    old = subprocess.check_output(['git', 'show', '3f84d7e:' + name])
    assert old.replace(b'\r\n', b'\n') == Path(name).read_bytes().replace(b'\r\n', b'\n'), name

project = paths.results_dir('recovery_qa', phase='feature_creation')
project.mkdir(parents=True, exist_ok=True)
with (
    tempfile.TemporaryDirectory(prefix='fixture-', dir=project) as durable_tmp,
    tempfile.TemporaryDirectory(prefix='jevpfn-recovery-', dir=os.environ['VSC_SCRATCH_GPFS1']) as native_tmp,
):
    durable, work = Path(durable_tmp) / 'JEVPFN', Path(native_tmp) / 'JEVPFN'
    target = cache(durable / 'data/jev_cache/fixture.sqlite3', request('liquid'))
    source = cache(work / 'data/jev_cache/fixture.sqlite3', request('intern'))
    first = _publish(durable, work, dataset_id='fixture')
    again = _publish(durable, work, dataset_id='fixture')
    assert first['decision_stores'] == again['decision_stores'] == 1
    assert target.get(request('liquid'), kind='verified_response') == response()
    assert target.get(request('intern'), kind='verified_response') == response()
    warmed = work / 'warm.sqlite3'
    _provider_cache(target.path, warmed, 'intern')
    with sqlite3.connect(warmed) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert db.execute('SELECT COUNT(*) FROM responses').fetchone()[0] == 1
    assert source.path.exists()
report = {'status': 'operational_recovery_checks_passed', 'job_id': os.environ['SLURM_JOB_ID'],
          'source': source_snapshot(), 'unchanged_scientific_files': unchanged,
          'native_publication': first, 'idempotent_publication': again,
          'warm_cache_integrity': True, 'production_caches_modified': False,
          'meaning': 'Synthetic GPFS recovery QA; no new inference or new 0A acceptance'}
destination = paths.manifests_dir('feature_creation') / ('recovery_qa_' + os.environ['SLURM_JOB_ID'] + '.json')
write_json(destination, report)
print(json.dumps({'status': report['status'], 'report': str(destination)}))
PY
