"""Exercise timing fail-closed behavior without CUDA, clocks or workloads."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import timing

pytestmark = pytest.mark.gpu_proof


@pytest.mark.parametrize('failure',['none','nan','short','missing','divergent'])
def test_timing_metrics(tmp_path,failure):
    prefix=tmp_path/'tracking'
    Path(str(prefix)+'_0_stats.result').write_text('control_updates: 100\n')
    samples=['2']*100
    if failure=='nan': samples[0]='nan'
    if failure=='short': samples=samples[:3]
    if failure!='missing': Path(str(prefix)+'_0_sqp_times.result').write_text('\n'.join(samples))
    for name in ('pcg_iters','pcg_exits','sqp_iters','sqp_exits'):
        Path(str(prefix)+f'_0_{name}.result').write_text('1\n0\n')
    log='RESULT offsets=1202 mean=.03 max=.05 final=.01'
    if failure=='divergent': log=log.replace('max=.05','max=2')
    if failure=='none': assert timing.metrics(log,prefix)['median_us']==2
    else:
        with pytest.raises((ValueError,FileNotFoundError)): timing.metrics(log,prefix)


def test_dry_run_and_authorization(tmp_path,monkeypatch):
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps({'workloads':[],'repeats':3}))
    monkeypatch.setattr(timing,'validate',lambda plan:None)
    monkeypatch.delenv('MPCGPU_QUIET_WINDOW',raising=False)
    output=tmp_path/'out'
    timing.run(plan,output,dry_run=True)
    assert not output.exists()
    with pytest.raises(RuntimeError,match='Coordinator'): timing.run(plan,output)


def test_binary_tamper(tmp_path,monkeypatch):
    binary=tmp_path/'workload.exe'; binary.write_text('original')
    plan={'identity':{},'workloads':[{'binary':str(binary),'sha256':timing.sha(binary)}]}
    monkeypatch.setattr(timing,'identity',lambda:{})
    timing.validate(plan)
    binary.write_text('replaced')
    with pytest.raises(ValueError,match='binary'): timing.validate(plan)


def test_qdldl_empty_pcg_metrics(tmp_path):
    prefix=tmp_path/'tracking'
    Path(str(prefix)+'_0_stats.result').write_text('control_updates: 100\n')
    Path(str(prefix)+'_0_sqp_times.result').write_text('2\n'*100)
    for name in ('pcg_iters','pcg_exits','sqp_iters','sqp_exits'):
        Path(str(prefix)+f'_0_{name}.result').write_text('' if name.startswith('pcg') else '1\n'*100)
    result=timing.metrics('RESULT offsets=1202 mean=.03 max=.05 final=.01',prefix,'qdldl')
    assert result['pcg_iters_mean'] is None
