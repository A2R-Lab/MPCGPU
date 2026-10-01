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


@pytest.mark.parametrize('failure',['none','untimed','partial','divergent','nan','samples'])
def test_icra_timing_metrics(tmp_path,failure):
    summary={'config':{'timers':failure!='untimed'},'offsets':666,'reference_rows':666 if failure!='partial' else 667,
             'control_updates':5204,'l2_error_m':{'mean':.02,'max':.1 if failure!='divergent' else 2,'final':.001},
             'sqp':{'mean_iters':8,'rho_exits':0},'linsys':{'mean_pcg_iters':6},
             'timing_us':{'linsys':{'count':40000 if failure!='samples' else 3,'median':50},
                          'sqp':{'count':5204,'median':1900}}}
    if failure=='nan': summary['l2_error_m']['mean']=float('nan')
    path=tmp_path/'summary.json'; path.write_text(json.dumps(summary))
    workload={'kind':'linsys','backend':'pcg'}
    if failure=='none': assert timing.icra_metrics(path,workload)['median_us']==1900
    else:
        with pytest.raises(ValueError): timing.icra_metrics(path,workload)


@pytest.mark.parametrize('kind,iters,max_error,expect',[
    ('iters',0.0,1.6,'flagged'),      # budget admits <1 iteration and the arm drifted: rate not met
    ('iters',0.0,0.018,'sustained'),  # <1 iteration on average but still tracking: a valid rate
    ('iters',1.0,1.6,'error'),        # divergence with a full iteration per step is a failure
    ('linsys',0.0,1.6,'error')])      # linear-system workloads never excuse divergence
def test_icra_rate_not_met(tmp_path,kind,iters,max_error,expect):
    summary={'config':{'timers':True},'offsets':666,'reference_rows':666,'control_updates':10408,
             'l2_error_m':{'mean':.8,'max':max_error,'final':1.0},'sqp':{'mean_iters':iters,'rho_exits':0},
             'linsys':{'mean_pcg_iters':0},'timing_us':{'linsys':{'count':500,'median':1200},
                                                        'sqp':{'count':10408,'median':1238}}}
    path=tmp_path/'summary.json'; path.write_text(json.dumps(summary))
    workload={'kind':kind,'backend':'qdldl'}
    if expect=='error':
        with pytest.raises(ValueError): timing.icra_metrics(path,workload)
    else:
        assert timing.icra_metrics(path,workload)['rate_not_met'] is (expect=='flagged')
