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
        for folder in ('tools', 'include', 'examples')
        for p in sorted((ROOT/folder).rglob('*'))
        if p.is_file() and p.suffix in ('.py', '.cu', '.cuh', '.hpp', '.csv', '.traj')
    }}


# ICRA 2024 paper-task batch (docs/icra-replication.md). "linsys" workloads reproduce the Figure 4/5
# measurement: per-solve linear-system time at 500 Hz with the 2024 SQP cap of 20. "iters" workloads
# reproduce Figure 6: SQP iterations per control step at 250/500/1000 Hz with the cap of 40.
ICRA_RATES_HZ = (250, 500, 1000)
ICRA_N128_TOLERANCES = ('5e-5', '1e-5')   # Figure 5 variants beside the default 1e-4


def icra_workloads(directory, horizons):
    workloads = []
    def add(backend, n, kind, extra, args, rate):
        tag = f"icra-{backend}-{n}-{kind}-{rate}hz"
        binary = directory/f'{tag}.exe'
        if not binary.exists():
            build(f'icra-{backend}', knots=n, profile='timing', output=binary, extra=extra)
        suffix = f"-tol{args[1]}" if args else ''
        workloads.append({'id': tag + suffix, 'task': 'icra', 'backend': backend, 'knots': n,
            'kind': kind, 'rate_hz': rate, 'args': args, 'binary': str(binary), 'sha256': sha(binary)})
    for n in horizons:
        for backend in ('pcg', 'qdldl'):
            add(backend, n, 'linsys', '', [], 500)
            if backend == 'pcg' and n == 128:
                for tolerance in ICRA_N128_TOLERANCES:
                    add(backend, n, 'linsys', '', ['--tol', tolerance], 500)
            for rate in ICRA_RATES_HZ:
                period = 1_000_000 // rate
                add(backend, n, 'iters', f'-DTIME_LINSYS=0 -DSIMULATION_PERIOD={period} -DSQP_MAX_TIME_US={period}',
                    [], rate)
    return workloads


def prepare(directory, horizons, repeats, task='fig8'):
    if repeats < 3 or any(n < 2 for n in horizons):
        raise ValueError('At least three repeats and horizons >=2 are required')
    directory.mkdir(parents=True, exist_ok=False)
    if task == 'icra':
        plan = {'protocol': 1, 'task': 'icra', 'commit_at_prepare': git('rev-parse','HEAD'),
                'identity': identity(), 'reference': str(ROOT/'examples/icra/pick_place'),
                'repeats': repeats, 'workloads': icra_workloads(directory, horizons),
                'note': 'ICRA 2024 pick-and-place protocol on current hardware; not the published numbers.'}
        (directory/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        print(directory/'plan.json')
        return
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


def icra_metrics(summary_path, workload):
    summary = json.loads(Path(summary_path).read_text())
    error = summary['l2_error_m']
    if not summary['config']['timers'] or summary['offsets'] != summary['reference_rows']:
        raise ValueError('Missing timers or incomplete circuit')
    if not all(math.isfinite(v) and v >= 0 for v in error.values()) or error['max'] > 1:
        raise ValueError('Nonfinite/divergent tracking result')
    timing = summary['timing_us']
    if workload['kind'] == 'linsys' and timing['linsys']['count'] < 100:
        raise ValueError('Missing linear-system samples')
    if timing['sqp']['count'] != summary['control_updates']:
        raise ValueError('Incomplete SQP sample collection')
    return {'tracking_mean_l2': error['mean'], 'tracking_max_l2': error['max'], 'tracking_final_l2': error['final'],
            'sqp_iters_mean': summary['sqp']['mean_iters'], 'sqp_rho_exits': summary['sqp']['rho_exits'],
            'linsys_us': timing['linsys'], 'sqp_us': timing['sqp'], 'median_us': timing['sqp']['median'],
            'pcg_iters_mean': summary['linsys']['mean_pcg_iters'] if workload['backend'] == 'pcg' else None}


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
            icra = workload.get('task') == 'icra'
            command = ([workload['binary'], '--reference', plan['reference'], '--out', str(case/'icra'), *workload['args']]
                       if icra else [workload['binary'], plan['reference'], str(prefix)])
            with (case/'stdout.log').open('w') as stream:
                subprocess.run(command, cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=600)
            wall = time.perf_counter()-start
            result = (icra_metrics(case/'icra'/'summary.json', workload) if icra
                      else metrics((case/'stdout.log').read_text(),prefix,workload['backend']))
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
    prep.add_argument('--task',choices=['fig8','icra'],default='fig8')
    execute = commands.add_parser('run')
    execute.add_argument('plan',type=Path)
    execute.add_argument('output',type=Path)
    execute.add_argument('--resume',type=Path)
    execute.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.command == 'prepare': prepare(args.directory.resolve(),args.horizons,args.repeats,args.task)
    else: run(args.plan.resolve(),args.output.resolve(),args.resume.resolve() if args.resume else None,args.dry_run)


if __name__ == '__main__':
    main()
