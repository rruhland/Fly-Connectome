"""Fixed four-observation visual velocity estimator in existing transport slots."""
from collections import deque
import torch
from fly_connectome.vision.interface import VisualStateEncoder
from fly_connectome.control.pong_evaluation import FreshPongEvaluator
from fly_connectome.control.codec import fingerprint

TRANSPORT='visual-transport-v1:observed-four-point-LS-velocity'


class FourObservationEncoder(VisualStateEncoder):
    def reset_state(self):
        super().reset_state();self.observed_history={}

    def encode(self,state,frame):
        out=super().encode(state,frame);sample=state['sample'];active={e['id'] for e in out['entities']};self.observed_history={k:v for k,v in self.observed_history.items() if k in active}
        for entity in out['entities']:
            history=self.observed_history.setdefault(entity['id'],deque(maxlen=4))
            if entity['observed']:history.append((sample,entity['position'].clone()))
            known=len(history)>=2;velocity=torch.zeros(2)
            if known:
                t=torch.tensor([s-history[-1][0] for s,p in history],dtype=torch.float64)*self.period;t-=t.mean();positions=torch.stack([p for s,p in history]).double()/self.scale;positions-=positions.mean(0);velocity=(torch.sum(t[:,None]*positions,0)/torch.sum(t*t)).to(entity['position'].dtype)
            entity.update(observed_velocity=velocity,velocity_known=known,velocity_age_seconds=(sample-history[-1][0])*self.period if known else None)
        vector=torch.zeros(self.dimension);vector[out['indices']]=out['values'];current=vector[slice(*self.layout['current'])].view(7,16,16);relations=vector[slice(*self.layout['relations'])].view(5,9,9);current[2:]=0;relations[1:]=0;ordered=sorted(out['entities'],key=lambda e:(*e['normalized_position'].tolist(),e['observed'],*e['observed_velocity'].tolist(),e['velocity_known']))
        for entity in ordered:
            p=entity['normalized_position'];v=entity['observed_velocity']
            if bool((p.abs()>self.half_extent).any()):continue
            y,x=self._bin(p,16)
            if entity['velocity_known']:
                current[2:6,y,x]+=torch.stack((v[0].clamp(min=0),(-v[0]).clamp(min=0),v[1].clamp(min=0),(-v[1]).clamp(min=0)));current[6,y,x]+=1
            neighbors=sorted((other for other in ordered if other is not entity),key=lambda other:float((other['normalized_position']-p).square().sum()))[:4]
            for other in neighbors:
                if entity['velocity_known'] and other['velocity_known']:
                    y,x=self._bin((other['normalized_position']-p)/2,9);dv=other['observed_velocity']-v;relations[1:,y,x]+=torch.stack((dv[0].clamp(min=0),(-dv[0]).clamp(min=0),dv[1].clamp(min=0),(-dv[1]).clamp(min=0)))
        indices=vector.nonzero().flatten();out.update(indices=indices,values=vector[indices],experimental_transport=TRANSPORT);return out


class FreshLSEvaluator(FreshPongEvaluator):
    def __init__(self,weights,config):
        super().__init__(weights,config);self.encoder=FourObservationEncoder(64,64,.02);self.initial_learned_hash=self.learned_hash()

    def learned_hash(self):return fingerprint((super().learned_hash(),TRANSPORT,4,.02))


def synthetic_diagnostic():
    def state(sample,x,identity=1):return dict(sample=sample,forecasts=[],entities=[dict(id=identity,position=torch.tensor([30.,x]),observed=True,observation_age=0)])
    encoder=FourObservationEncoder(64,64,.02);constant=[]
    for sample in range(8):constant.append(encoder.encode(state(sample,25.+sample*.5),None)['entities'][0]['observed_velocity'][1].item())
    expected=.5/(64*.02);assert max(abs(v-expected) for v in constant[1:])<1e-6;encoder.reset_state();velocities=[]
    positions=[25.,25.5,26.,26.5,26.,25.5,25.,24.5]
    for sample,p in enumerate(positions):velocities.append(encoder.encode(state(sample,p),None)['entities'][0]['observed_velocity'][1].item())
    first_reversed=next(i for i in range(4,len(velocities)) if velocities[i]<0);encoder.reset_state();fresh=encoder.encode(state(0,35.),None)['entities'][0];assert not fresh['velocity_known'];changed=encoder.encode(state(1,36.,identity=2),None)['entities'][0];assert not changed['velocity_known']
    return dict(passed=True,constant_velocity_expected=expected,constant_estimates=constant,reversal_positions=positions,reversal_estimates=velocities,first_post_reversal_sample=4,first_negative_estimate_sample=first_reversed,sign_reversal_lag_samples=first_reversed-4,scene_reset_clears_velocity=True,new_track_starts_unknown=True,nominal_period_seconds=.02,observations_per_fit_at_most=4,no_rebound_specific_rule=True,new_cameras=0)
