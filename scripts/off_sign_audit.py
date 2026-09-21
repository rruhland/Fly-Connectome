"""Read-only source-artifact audit; replay one trial on disposable online copies."""
import sys,json,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'scripts')
from controlled_visual import crop_payload,make_network
from temporal_visual import oscillation
from frame_prediction import FramePrediction
from fly_connectome.sensor import Retina,EventCamera
from fly_connectome.plasticity import LearningConfig
from fly_connectome.data import checksum

torch.set_num_threads(1)
p=Path('runs/temporal-frame-horizon-v1'); manifest=json.loads((p/'results.json').read_text())['manifest']
payload=torch.load(manifest['source_checkpoint'],weights_only=True); m=payload['metadata']
ids=np.array(m['graph']['body_ids']); crop,_,_=crop_payload(payload,np.searchsorted(ids,manifest['body_ids']))
target=int(np.searchsorted(crop['graph'].body_ids,82450)); blank=int(np.random.default_rng(9021).integers(12,37))
frames=oscillation(blank); events=[(blank+c*6+3)*8 for c in range(1,4)]
rows=[]; summaries=[]
for condition,weights in [('initial',crop['weights']),('frame-trained',torch.from_numpy(np.load(p/'training.npz')['weights_trained']))]:
 net=make_network(crop,m,weights); retina=Retina(**crop['retina']); camera=EventCamera(1,32,64)
 rule=FramePrediction(net,LearningConfig(**m['learning']),sensory_mask=retina.injected,sensory_gain=m['config']['sensory_gain'])
 incoming=((net.post==target)&(net.pathways==1)).nonzero().flatten(); sources=net.pre[incoming]; contributions=torch.zeros(len(incoming))
 for frame,image in enumerate(frames):
  camera_events=camera.observe(image); injection=retina.project(camera_events)*m['config']['sensory_gain']
  for tick in range(8):
   t=frame*8+tick; before=net.magnitudes[incoming].clone(); activity=net.step(injection if tick==0 else torch.zeros_like(injection),capture_increments=True)
   arrivals=torch.bincount(activity.arrival_edges,minlength=net.e)[incoming]
   contributions.mul_(net.current_decay[target]).add_(arrivals*before*net.signs[incoming])
   prediction=float(activity.predicted[0,target]); observed=float(rule.config.observation(activity,net.config.threshold,retina.injected,m['config']['sensory_gain'])[0,target])
   forecast=rule.forecast
   issue_trace=torch.zeros(len(incoming)); issue_prediction=None
   if forecast is not None:
    keys,values,pred=forecast
    for j,edge in enumerate(incoming):
     found=(keys==edge).nonzero().flatten()
     if len(found): issue_trace[j]=values[found[0]]; issue_prediction=float(pred[found[0]])
   rule.observe(activity,torch.zeros(1)); delta=torch.zeros(len(incoming))
   if rule.last_visual_update is not None:
    edges,y,d=rule.last_visual_update
    for j,edge in enumerate(incoming):
     hit=edges==edge
     if hit.any(): delta[j]=d[hit].sum()
   if tick==7: rule.synchronize()
   if any(event-24<=t<=event+8 for event in events):
    residual=prediction-float(contributions.sum()); assert abs(residual)<1e-6,residual
    row=dict(condition=condition,tick=t,frame=frame,subtick=tick,target=observed,prediction=prediction,retained_prediction=issue_prediction,
       reconstructed_current=float(contributions.sum()),residual=residual,
       source_spikes=activity.spikes[0,sources].tolist(),arrivals=arrivals.tolist(),signed_contributions=contributions.tolist(),
       source_sensory_input=activity.sensory_input[0,sources].tolist(),source_sensory_state=net.sensory_state[0,sources].tolist(),
       source_observed_current=activity.observed[0,sources].tolist(),source_recurrent_current=activity.predicted[0,sources].tolist(),
       source_voltage=net.voltage[0,sources].tolist(),source_adaptation=net.adaptation[0,sources].tolist(),
       weights_at_arrival=before.tolist(),retained_signed_eligibility=issue_trace.tolist(),weight_delta=delta.tolist())
    rows.append(row)
   if t in events:
    assert observed==1
    assert issue_prediction is not None
    expected=rule.config.eta_prediction*(observed-issue_prediction)*issue_trace
    torch.testing.assert_close(delta,expected,rtol=2e-7,atol=1e-10)
    assert torch.all(delta[net.signs[incoming]>0]>=0)
    assert torch.all(delta[net.signs[incoming]<0]<=0)
    summaries.append(dict(condition=condition,event_tick=t,forecast=issue_prediction,target=observed,
      sources=[dict(body=int(crop['graph'].body_ids[s]),sign=float(net.signs[e]),eligibility=float(issue_trace[j]),delta=float(delta[j])) for j,(s,e) in enumerate(zip(sources,incoming)) if issue_trace[j]!=0]))
result=dict(blank_frames=blank,events=events,incoming=[dict(edge=int(e),source=int(crop['graph'].body_ids[s]),cell_type=crop['retina']['cell_types'][s],sign=float(net.signs[e]),delay=int(net.delays[e])) for s,e in zip(sources,incoming)],
 summaries=summaries,rows=rows,notes='Short online diagnostic copies of initial and learned networks; original weights/checkpoint untouched. Per-edge current integrates weights at arrival. Warmup residue verified <1e-6 in all recorded rows.')
assert checksum(manifest['source_checkpoint'])==manifest['source_sha256']
Path('runs/off-sign-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summaries,indent=2))
