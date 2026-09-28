#!/usr/bin/env python3
"""Prepare immutable MPCGPU workloads; run ONLY in an authorized quiet window."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

from build import ROOT, build, source_digest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def identity():
    # HEAD may advance by a receipt-only commit; tracked source tree must not.
    return {'headers': source_digest(), 'files': {
        str(p.relative_to(ROOT)): sha(p)
        for folder in ('tools', 'include', 'examples/trajfiles')
        for p in sorted((ROOT/folder).rglob('*'))
        if p.is_file() and p.suffix in ('.py', '.cu', '.cuh', '.hpp', '.csv', '.traj')
    }}


def prepare(directory, horizons, repeats):
    if repeats < 3 or any(n < 2 for n in horizons):
        raise ValueError('At least three repeats and horizons >=2 are required')
    directory.mkdir(parents=True, exist_ok=False)
    workloads = []
    for n in horizons:
        for backend in ('pcg', 'qdldl'):
            for mode in ('reuse', 'fresh'):
                binary = build(backend, knots=n, profile='timing',
                    output=directory/f'{backend}-{n}-{mode}.exe',
                    extra='-DMPCGPU_NO_REUSE=1' if mode == 'fresh' else '')
                workloads.append({'id': f'{backend}-{n}-{mode}', 'backend': backend,
                    'knots': n, 'mode': mode, 'binary': str(binary), 'sha256': sha(binary)})
    plan = {'protocol': 1, 'commit_at_prepare': git('rev-parse','HEAD'),
            'identity': identity(), 'reference': str(ROOT/'examples/trajfiles/0_0'),
            'repeats': repeats, 'workloads': workloads,
            'note': 'Internal SQP time excludes portions of setup/cleanup. Fresh is a same-source workspace baseline, not old main.'}
    (directory/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(directory/'plan.json')


def metrics(log, prefix, backend='pcg'):
    match = re.search(r'RESULT offsets=(\d+) mean=([\d.eE+-]+) max=([\d.eE+-]+) final=([\d.eE+-]+)', log)
    if not match or int(match[1]) != 1202:
        raise ValueError('Missing/incomplete tracking result')
    quality = [float(v) for v in match.groups()[1:]]
    if not all(math.isfinite(v) and v >= 0 for v in quality) or quality[1] > 1:
        raise ValueError('Nonfinite/divergent tracking result')
    samples = [float(x) for x in Path(str(prefix)+'_0_sqp_times.result').read_text().split()]
    if len(samples) < 100 or not all(math.isfinite(x) and x > 0 for x in samples):
        raise ValueError('Missing/invalid internal SQP samples')
    stats = Path(str(prefix)+'_0_stats.result').read_text()
    count = re.search(r'control_updates: (\d+)',stats)
    if not count or int(count[1]) != len(samples):
        raise ValueError('Incomplete SQP sample collection')
    ordered = sorted(samples)
    result = {'samples': len(samples), 'median_us': statistics.median(samples),
              'p90_us': ordered[math.ceil(.9*len(ordered))-1],
              'tracking_mean_l2': quality[0], 'tracking_max_l2': quality[1],
              'tracking_final_l2': quality[2]}
    for name in ('pcg_iters', 'pcg_exits', 'sqp_iters', 'sqp_exits'):
        values = [float(x) for x in Path(str(prefix)+f'_0_{name}.result').read_text().split()]
        if backend == 'qdldl' and name.startswith('pcg_') and not values:
            result[name+'_mean'] = None  # Direct solver has no PCG iteration/exit stream.
            continue
        if not values or not all(math.isfinite(x) for x in values):
            raise ValueError(f'Invalid {name} samples')
        result[name+'_mean'] = statistics.mean(values)
    return result


def validate(plan):
    if identity() != plan['identity']:
        raise ValueError('Source/dependency/reference changed: prepare a new plan')
    for workload in plan['workloads']:
        if sha(workload['binary']) != workload['sha256']:
            raise ValueError('Prepared binary changed')


def run(plan_path, output, resume=None, dry_run=False):
    plan = json.loads(plan_path.read_text())
    validate(plan)
    jobs = [(w, repeat) for repeat in range(plan['repeats'])
            for w in (plan['workloads'] if repeat % 2 == 0 else list(reversed(plan['workloads'])))]
    completed = set()
    if resume:
        previous = json.loads((resume/'run.json').read_text())
        if previous['plan_sha256'] != sha(plan_path):
            raise ValueError('Resume plan does not match')
        for record in resume.glob('*/verdict.json'):
            verdict = json.loads(record.read_text())
            if verdict.get('ok'):
                completed.add(record.parent.name)
    jobs = [(w,r) for w,r in jobs if f"{w['id']}-r{r}" not in completed]
    if dry_run:
        print(json.dumps({'pending_repeats':len(jobs), 'plan':str(plan_path),
                          'output':str(output), 'no_execution':True},indent=2))
        return
    if os.environ.get('MPCGPU_QUIET_WINDOW') != '1':
        raise RuntimeError('Coordinator must explicitly set MPCGPU_QUIET_WINDOW=1')
    if git('status','--porcelain','--ignore-submodules=untracked'):
        raise RuntimeError('Timing requires a clean source/receipt tree')
    subprocess.run([str(ROOT/'.venv/bin/gpu-proof'), 'verify', '--receipt', 'gpu-proof.json',
        '--repo','.', '--policy','test/gpu-proof-policy.yaml',
        '--expected-skips','test/expected_skips.txt','--require-gpu'],cwd=ROOT,check=True)
    with open('/tmp/a2rlab-timing.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True,exist_ok=False)
        machine = subprocess.check_output(['nvidia-smi'],text=True)
        (output/'run.json').write_text(json.dumps({'plan_sha256':sha(plan_path),
            'commit':git('rev-parse','HEAD'), 'receipt_sha256':sha(ROOT/'gpu-proof.json'),
            'machine':machine, 'resume_from':str(resume) if resume else None},indent=2)+'\n')
        env = {**os.environ,'LD_LIBRARY_PATH':str(ROOT/'qdldl/build/out')+':'+os.environ.get('LD_LIBRARY_PATH','')}
        for workload,repeat in jobs:
            if (output/'STOP').exists():
                print('STOP boundary reached; remaining repeats need a fresh output directory.'); return
            case = output/f"{workload['id']}-r{repeat}"
            case.mkdir()
            prefix = case/'tracking'
            # Timeout/interrupt/failed result produces no success verdict.
            start = time.perf_counter()
            with (case/'stdout.log').open('w') as stream:
                subprocess.run([workload['binary'],plan['reference'],str(prefix)],
                    cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=600)
            wall = time.perf_counter()-start
            result = metrics((case/'stdout.log').read_text(),prefix,workload['backend'])
            result.update(ok=True, process_wall_seconds=wall, workload=workload, repeat=repeat)
            (case/'verdict.json').write_text(json.dumps(result,indent=2)+'\n')
            print(case.name, result['median_us'], 'us', flush=True)
        print('Collection complete. Review tracking and repeat distributions before publishing.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('directory',type=Path)
    prep.add_argument('--horizons',type=int,nargs='+',default=[32,64,128,256,512])
    prep.add_argument('--repeats',type=int,default=3)
    execute = commands.add_parser('run')
    execute.add_argument('plan',type=Path)
    execute.add_argument('output',type=Path)
    execute.add_argument('--resume',type=Path)
    execute.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.command == 'prepare': prepare(args.directory.resolve(),args.horizons,args.repeats)
    else: run(args.plan.resolve(),args.output.resolve(),args.resume.resolve() if args.resume else None,args.dry_run)


if __name__ == '__main__':
    main()
