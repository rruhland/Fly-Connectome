"""Offline held-out local-state distinguishability audit; no neural learning."""
import json
from pathlib import Path
from dataclasses import replace
import time
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from controlled_visual import crop_payload, make_network
from temporal_visual import oscillation, recurrent_boundaries
from frame_prediction import FramePrediction
from fly_connectome.plasticity import LearningConfig
from fly_connectome.sensor import Retina, EventCamera
from fly_connectome.data import checksum


class NeighborProbe:
    """Fixed 15-neighbor diagnostic, with training-only standardization."""
    def __init__(self,x,y,k=15):
        self.mean=x.mean(0)
        self.scale=x.std(0)
        self.keep=self.scale>1e-8
        self.scale=np.where(self.keep,self.scale,1.)
        self.tree=cKDTree(((x-self.mean)/self.scale)[:,self.keep])
        self.y=np.asarray(y)!=0
        self.k=min(k,len(x))

    def score(self,x):
        _,indices=self.tree.query(((x-self.mean)/self.scale)[:,self.keep],k=self.k)
        if self.k==1:
            indices=indices[:,None]
        return self.y[indices].mean(1)


def detection_metrics(scores,labels,threshold):
    event=labels!=0
    called=scores>=threshold
    positive=int(event.sum());negative=int((~event).sum())
    ranks=rankdata(scores)
    auc=float((ranks[event].sum()-positive*(positive+1)/2)/(positive*negative)) if positive and negative else None
    return dict(on_count=int((labels<0).sum()),off_count=int((labels>0).sum()),quiet_count=negative,
                on_recall=float(called[labels<0].mean()),off_recall=float(called[labels>0].mean()),
                quiet_false_positive_rate=float(called[~event].mean()),auc=auc,
                precision=float(event[called].mean()) if called.any() else None)


def calibrate_threshold(scores,labels):
    best=None
    for threshold in np.r_[np.unique(scores),scores.max()+1e-6]:
        m=detection_metrics(scores,labels,threshold)
        if m['quiet_false_positive_rate']<=.05:
            key=(min(m['on_recall'],m['off_recall']),m['on_recall']+m['off_recall'],
                 -m['quiet_false_positive_rate'],threshold)
            if best is None or key>best[0]:best=(key,float(threshold))
    return best[1]


def motion_phase(frame,blank,dwell):
    relative=frame-int(blank)
    if not 0<=relative<8*dwell:
        return 'blank'
    offset=relative%(2*dwell)
    if offset==0:return 'on'
    if offset==dwell:return 'off'
    return f'after_on_{offset}' if offset<dwell else f'after_off_{offset-dwell}'


@torch.no_grad()
def collect(crop,metadata,weights,blanks,dwell=3):
    net=make_network(crop,metadata,weights,predictive_kinetics='area-matched-excitation-v1')
    retina=Retina(**crop['retina']);camera=EventCamera(1,32,64)
    cfg=replace(LearningConfig(**metadata['learning']),eta_prediction=0.,eta_reward=0.,homeostasis_rate=0.)
    rule=FramePrediction(net,cfg,sensory_mask=retina.injected,sensory_gain=metadata['config']['sensory_gain'])
    target=int(np.searchsorted(crop['graph'].body_ids,82450))
    incoming=((net.post==target)&(net.pathways==1)).nonzero().flatten()
    lookup=torch.full((net.e,),-1,dtype=torch.long);lookup[incoming]=torch.arange(len(incoming))
    state_names=['predictive_current','excitatory_prediction','inhibitory_prediction',
                 'sensory_state','feedforward_current','behavioral_current','voltage','adaptation','refractory']
    names=state_names+['local_sensory_increment','local_feedforward_increment','own_spike','own_post_trace','own_rate']
    for prefix in ['eligibility','arrival_trace','arrival_now']:
        names.extend(f'{prefix}_{int(crop["graph"].body_ids[net.pre[e]])}' for e in incoming)
    features=[];trial_ids=[];phases=[];predictions=[];targets=[];spikes=[]
    for trial,blank in enumerate(blanks):
        for frame,image in enumerate(oscillation(int(blank),dwell=dwell)):
            injection=retina.project(camera.observe(image))*metadata['config']['sensory_gain']
            for tick in range(8):
                a=net.step(injection if tick==0 else torch.zeros_like(injection),capture_increments=True)
                observed=cfg.observation(a,net.config.threshold,retina.injected,metadata['config']['sensory_gain'])
                predictions.append(float(cfg.encode(a.predicted[0,target],net.config.threshold)))
                targets.append(float(observed[0,target]));spikes.append(a.spikes[0].numpy().copy())
                rule.observe(a,torch.zeros(1))
                if tick==0:
                    e=torch.zeros(len(incoming));arr_trace=e.clone()
                    positions=lookup[rule.keys];keep=positions>=0
                    e[positions[keep]]=rule.values[keep];arr_trace[positions[keep]]=rule.arrival_trace[keep]
                    arrivals=torch.bincount(a.arrival_edges,minlength=net.e)[incoming]
                    row=[float(getattr(net,n)[0,target]) for n in state_names]
                    row += [float(a.sensory_input[0,target]),float(a.feedforward_arrivals[0,target]),
                            float(a.spikes[0,target]),float(rule.post_trace[0,target]),float(rule.rates[0,target])]
                    features.append(row+e.tolist()+arr_trace.tolist()+arrivals.tolist())
                    trial_ids.append(trial)
                    phases.append(motion_phase(frame,blank,dwell))
            rule.synchronize()
    assert torch.equal(net.magnitudes,weights)
    features=np.asarray(features);assert np.isfinite(features).all()
    issue=recurrent_boundaries(blanks,dwell=dwell);target_frames=(issue+8)//8
    # Trial IDs and phase labels belong to the TARGET frame, never to features.
    return dict(x=features[issue//8],y=np.asarray(targets)[issue+8],trial=np.asarray(trial_ids)[target_frames],
                phase=np.asarray(phases)[target_frames],prediction=np.asarray(predictions)[issue],
                all_prediction=np.asarray(predictions),all_target=np.asarray(targets),all_spikes=np.asarray(spikes),
                feature_names=np.asarray(names),blank_frames=blanks)


def evaluate(probe,data,columns,threshold,noise=0.):
    x=data['x'][:,columns].copy()
    if noise:
        x+=np.random.default_rng(9030).normal(size=x.shape)*probe.scale*noise
    scores=probe.score(x)
    result=detection_metrics(scores,data['y'],threshold)
    result['quiet_by_phase']={phase:dict(count=int(mask.sum()),false_positive_rate=float((scores[mask]>=threshold).mean()))
        for phase in ['after_on_1','after_on_2','after_off_1','after_off_2','blank']
        if (mask:=((data['phase']==phase)&(data['y']==0))).any()}
    return result


def main():
    torch.set_num_threads(1)
    source=Path('runs/temporal-area-matched-excitation-v1')
    r=json.loads((source/'results.json').read_text());manifest=r['manifest']
    payload=torch.load(manifest['source_checkpoint'],weights_only=True);m=payload['metadata']
    ids=np.array(m['graph']['body_ids'])
    crop,_,_=crop_payload(payload,np.searchsorted(ids,manifest['body_ids']))
    assert crop['graph'].identity()==manifest['graph_sha256']
    weights=torch.from_numpy(np.load(source/'training.npz')['weights_trained'])
    old=np.load(source/'trained.npz')
    output=Path('runs/local-information-v1');output.mkdir(exist_ok=False)
    started=time.perf_counter()
    data=collect(crop,m,weights,old['blank_frames'])
    for key in ['prediction','target','spikes']:
        np.testing.assert_array_equal(data['all_'+key],old[key].reshape(data['all_'+key].shape))
    np.savez_compressed(output/'recorded.npz',**data)
    print('Original frozen prediction, target and spike traces match exactly',flush=True)
    fresh=collect(crop,m,weights,np.random.default_rng(9024).integers(12,37,size=50))
    np.savez_compressed(output/'fresh.npz',**fresh)
    split={name:{k:v[mask] for k,v in data.items() if k in ['x','y','trial','phase','prediction']}
           for name,mask in [('train',data['trial']<25),('calibration',(data['trial']>=25)&(data['trial']<35)),('test',data['trial']>=35)]}
    groups={'memory_currents':np.arange(3),'cell_state':np.arange(14),
            'synaptic_state':np.r_[np.arange(6),np.arange(14,data['x'].shape[1])],
            'all_local':np.arange(data['x'].shape[1])}
    results={}
    for name,columns in groups.items():
        train,cal=split['train'],split['calibration']
        probe=NeighborProbe(train['x'][:,columns],train['y'])
        threshold=calibrate_threshold(probe.score(cal['x'][:,columns]),cal['y'])
        results[name]=dict(features=data['feature_names'][columns].tolist(),threshold=threshold,
            retained_dimensions=int(probe.keep.sum()),calibration=evaluate(probe,cal,columns,threshold),
            test=evaluate(probe,split['test'],columns,threshold),fresh=evaluate(probe,fresh,columns,threshold),
            fresh_noise_1pct=evaluate(probe,fresh,columns,threshold,noise=.01))
    columns=groups['all_local'];train=split['train'];cal=split['calibration']
    shuffled=np.random.default_rng(9029).permutation(train['y'])
    null=NeighborProbe(train['x'],shuffled)
    threshold=calibrate_threshold(null.score(cal['x']),cal['y'])
    results['shuffled_training_labels']=dict(threshold=threshold,fresh=evaluate(null,fresh,columns,threshold))
    assert checksum(manifest['source_checkpoint'])==manifest['source_sha256']
    result=dict(seconds=time.perf_counter()-started,source_checkpoint_sha256=manifest['source_sha256'],
        trained_weights_artifact_sha256=checksum(source/'training.npz'),feature_timing='after issue tick, target +8 ticks',
        splits='seed9023 trials1-25 fit,26-35 calibrate,36-50 test; fresh50trials seed9024',
        unchanged_frozen_arrays=True,source_unchanged=True,probe='15-neighbor binary event detection; train-only standardization; no neural updates',
        results=results)
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:r.get('fresh') for name,r in results.items()},indent=2))


if __name__=='__main__':main()
