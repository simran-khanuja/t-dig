"""Submit a preflighted job matrix once, journaling each receipt.

Each job runs the submit_command stored in matrix.json (T_DIG_SUBMIT at preparation: a scheduler
submit command such as sbatch, or bash to run in the foreground).
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hpdv2.storage import atomic_json, read_json, run_lock
from baselines.assets import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--expected-jobs', type=int, default=8)
    a = p.parse_args()
    root = a.matrix.resolve()
    matrix = read_json(root / 'matrix.json')
    preflight = read_json(root / 'preflight.json')
    if not preflight.get('passed') or len(preflight['checks']) != a.expected_jobs:
        raise ValueError('Every requested configuration must pass preflight first')
    # Every job of a matrix asks for the same node count (2 by default, --nodes at preparation).
    per_job = {j['nodes'] for j in matrix['jobs']}
    if (
        len(matrix['jobs']) != a.expected_jobs
        or len(per_job) != 1
        or matrix['nodes'] != per_job.pop() * a.expected_jobs
    ):
        raise ValueError('Job count differs from the explicitly requested submission size')
    if file_hash(Path(matrix['dataset']) / 'manifest.json') != matrix['dataset_manifest_sha256']:
        raise ValueError('Dataset changed after preparation')
    for name, expected in read_json(root / 'source-hashes.json').items():
        if file_hash(root / 'source' / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    print(
        json.dumps(
            dict(
                execute=a.execute,
                nodes=matrix['nodes'],
                gpus=matrix['gpus'],
                jobs=[j['name'] for j in matrix['jobs']],
            ),
            indent=2,
        ),
        flush=True,
    )
    if not a.execute:
        return
    with run_lock(root / 'submission'):
        if (root / 'submission/receipts.json').exists():
            raise ValueError(
                'Submission receipts already exist; inspect the scheduler before any further submission'
            )
        receipts = []
        for job in matrix['jobs']:
            result = subprocess.run(job['submit_command'], text=True, capture_output=True)
            receipt = dict(
                name=job['name'],
                submitted=time.time(),
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                command=job['submit_command'],
            )
            receipts.append(receipt)
            atomic_json(root / 'submission/receipts.json', receipts)
            if result.returncode:
                raise RuntimeError('Submission failed; previous receipts preserved')
            # sbatch prints "Submitted batch job <id>"; bash prints the job log.
            lines = result.stdout.strip().splitlines()
            receipt['job_id'] = lines[-1].split()[-1] if lines and lines[-1].split() else ''
            atomic_json(root / 'submission/receipts.json', receipts)
            print(job['name'], receipt['job_id'], flush=True)
        atomic_json(
            root / 'submission/status.json',
            dict(state='submitted', jobs=len(receipts), nodes=matrix['nodes'], gpus=matrix['gpus']),
        )


if __name__ == '__main__':
    main()
