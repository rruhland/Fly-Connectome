"""Frozen same-magnitude control for the area-matched kinetics experiment."""
import sys,json,time,copy
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from controlled_visual import crop_payload,make_network,run_sequence,score_forecasts
from temporal_visual import oscillation,recurrent_boundaries
from fly_connectome.data import checksum

torch.set_num_threads(1)
prior=Path('runs/temporal-slow-excitation-v1-warmup')
r=json.loads((prior/'results.json').read_text()); manifest=r['manifest']
payload=torch.load(manifest['source_checkpoint'],weights_only=True); m=payload['metadata']
ids=np.array(m['graph']['body_ids']); crop,_,_=crop_payload(payload,np.searchsorted(ids,manifest['body_ids']))
assert crop['graph'].identity()==manifest['graph_sha256']
target=int(np.searchsorted(crop['graph'].body_ids,manifest['target_body']))
w=torch.from_numpy(np.load(prior/'training.npz')['weights_trained'])
reference=np.load(prior/'trained.npz'); blanks=reference['blank_frames']; frames=torch.cat([oscillation(int(b)) for b in blanks])
output=Path('runs/area-matched-frozen-control-v2');output.mkdir(exist_ok=False)
started=time.perf_counter();metrics={'slow-excitation-v1':r['metrics']['trained']}
for kinetics in ['original','uniform-fast-control','area-matched-excitation-v1']:
 if kinetics=='uniform-fast-control':
  cold=copy.deepcopy(m); cold['config']['warmup_steps']=0
  net=make_network(crop,cold,w,predictive_kinetics='slow-excitation-v1')
  net.excitatory_decay=net.inhibitory_decay
  for _ in range(m['config']['warmup_steps']): net.step(torch.zeros_like(net.voltage))
 else:
  net=make_network(crop,m,w,predictive_kinetics=kinetics)
 result=run_sequence(net,crop,m,frames,[target],learning=False,deadline=started+180)
 assert torch.equal(net.magnitudes,w)
 assert np.array_equal(result['target'],reference['target'])
 score=score_forecasts(result['target'],result['prediction'],recurrent_boundaries(blanks),8)
 metrics[kinetics]=score
 np.savez_compressed(output/(kinetics+'.npz'),**{k:v for k,v in result.items() if k!='stability'},blank_frames=blanks)
assert checksum(manifest['source_checkpoint'])==manifest['source_sha256']
result=dict(seconds=time.perf_counter()-started,metrics=metrics,weights_source=str(prior/'training.npz'),weights_source_sha256=checksum(prior/'training.npz'),
 note='All conditions use identical magnitudes learned by unscaled20ms model. No retraining; fresh warmup and same evaluation trajectories. Slow-excitation reference reused from saved frozen evaluation. Uniform-fast-control sets both predictive decays to5ms before warmup, eliminating original class-dependent decay differences.',source_unchanged=True)
(output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
