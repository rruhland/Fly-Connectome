"""Approved STP preflight; training must not run if observation targets change."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from controlled_visual import make_network,run_sequence
from temporal_visual import oscillation
from timing_transfer import load_model
from short_term import DepressionNetwork,FacilitationNetwork
from fly_connectome.plasticity import LearningConfig
from fly_connectome.data import checksum


OUT=Path('runs/short-term-v1')
KINDS={'baseline':'area-matched-excitation-v1','D':'short-term-D-v1','F':'short-term-F-v1'}


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def record_release(n,m):
    """Diagnostic wrapper only; records actual arrivals, never advances release."""
    from fly_connectome.sensor import Retina
    retina=Retina(**m['retina']);cfg=LearningConfig(**m['learning'])
    rows=[];step=n.step
    def capture(sensory_current,*,capture_increments=False):
        a=step(sensory_current,capture_increments=capture_increments)
        keys=n.arrival_keys
        env,edges=keys.div(n.e,rounding_mode='floor'),keys.remainder(n.e)
        observed=cfg.observation(a,n.config.threshold,retina.injected,m['config']['sensory_gain'])
        rows.append(dict(tick=np.full(len(keys),n.step_index-1),edges=edges.numpy().copy(),
            gains=n.arrival_gains.numpy().copy(),
            state=n.release_state[env,n.release_index[edges]].numpy().copy(),
            observed=observed[env,n.post[edges]].numpy().copy()))
        return a
    n.step=capture
    return rows


def preflight():
    OUT.mkdir(exist_ok=False);crop,m,_=load_model()
    target=int(np.searchsorted(crop['graph'].body_ids,82450))
    # Neutral control exercises real-crop learning with standard warmup.
    frames=oscillation(12);reference=None;neutral={}
    for name in ['baseline','D','F']:
        if name=='baseline':n=make_network(crop,m,predictive_kinetics=KINDS['baseline'])
        else:
            cls=DepressionNetwork if name=='D' else FacilitationNetwork
            from fly_connectome.dynamics import NeuronConfig
            n=cls(crop['graph'],crop['delays'],crop['pathways'],unit_release=True,
                config=NeuronConfig(**m['neurons']),cell_types=crop['retina']['cell_types'])
            n.magnitudes.copy_(crop['weights'])
            for _ in range(m['config']['warmup_steps']):n.step(torch.zeros_like(n.voltage))
        r=run_sequence(n,crop,m,frames,[target],learning=True,visual_schedule='frame-horizon-v1')
        if reference is None:reference=(r,n.magnitudes.clone())
        else:
            for key in ['prediction','target','spikes','eligibility','local_update_sums']:
                np.testing.assert_array_equal(r[key],reference[0][key])
            torch.testing.assert_close(n.magnitudes,reference[1],rtol=0,atol=0)
        neutral[name]=True
    save(OUT/'neutral.json',neutral)
    blanks=np.random.default_rng(9080).integers(12,37,size=5)
    results={};all_equal=True
    for dwell in [2,3,4,6]:
        frames=torch.cat([oscillation(int(b),dwell=dwell) for b in blanks]);baseline=None
        for name in KINDS:
            n=make_network(crop,m,predictive_kinetics=KINDS[name])
            rows=record_release(n,dict(m,retina=crop['retina'])) if name!='baseline' else None
            started=time.perf_counter()
            # Audit all local targets used by learning, not only the scored L3.
            r=run_sequence(n,crop,m,frames,list(range(n.n)),learning=False,deadline=started+300)
            seconds=time.perf_counter()-started
            assert torch.equal(n.magnitudes,crop['weights'])
            if name=='baseline':baseline=r
            different=r['target']!=baseline['target'];all_equal&=not different.any()
            key=f'{name}-{dwell}';path=OUT/f'{key}.npz'
            data={k:v for k,v in r.items() if k!='stability'}
            stats={}
            if rows is not None:
                data.update({'release_'+k:np.concatenate([row[k] for row in rows]) for k in rows[0]})
                gain=data['release_gains'];state=data['release_state'];events=data['release_observed']!=0
                assert np.isfinite(state).all() and ((state>=0)&(state<=1)).all()
                lo,hi=(0,1) if name=='D' else (1,2)
                assert ((gain>=lo)&(gain<=hi)).all()
                for label,mask in [('event',events),('quiet',~events)]:
                    stats[label]=dict(count=int(mask.sum()),gain_quantiles=np.quantile(gain[mask],[0,.1,.5,.9,1]).tolist() if mask.any() else [],
                        state_quantiles=np.quantile(state[mask],[0,.1,.5,.9,1]).tolist() if mask.any() else [])
                stats['persistent_release_bytes']=sum(t.numel()*t.element_size() for t in [n.release_state,n.release_last,n.release_index])
            np.savez_compressed(path,**data)
            results[key]=dict(seconds=seconds,frames=len(frames),stability=r['stability'],
                target_mismatches=int(different.sum()),target_samples=int(different.size),
                scored_target_mismatches=int(different[:,target].sum()),
                first_mismatch_ticks=np.flatnonzero(different.any(axis=1))[:10].tolist(),
                affected_body_ids=crop['graph'].body_ids[different.any(axis=0)].tolist(),
                spike_count=int(r['spikes'].sum()),release=stats,sha256=checksum(path))
            save(OUT/'preflight-results.json',results)
            print(key,'target mismatches',int(different.sum()),flush=True)
    save(OUT/'preflight-decision.json',dict(passed=bool(all_equal),neutral_learning_exact=neutral,
        blank_frames=blanks.tolist(),seed=9080,
        decision='Eligible for fixed training' if all_equal else 'STOP: observation targets changed; no training authorized under this protocol'))


if __name__=='__main__':
    torch.set_num_threads(1)
    preflight()
