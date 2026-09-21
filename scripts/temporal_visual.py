"""Fixed temporal visual capability test on a directly driven measured motif."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from controlled_visual import crop_payload, make_network, run_sequence, score_forecasts, passed
from visual_preflight import delayed_traces
from fly_connectome.data import checksum
from fly_connectome.sensor import Retina


def oscillation(blank, cycles=4, dwell=3):
    frames = torch.zeros(blank+2*cycles*dwell+1, 1, 32, 64, dtype=torch.bool)
    for frame in range(2*cycles*dwell):
        frames[blank+frame, 0, 30, 39 if (frame//dwell)%2 == 0 else 38] = True
    return frames


def recurrent_boundaries(blanks, cycles=4, dwell=3):
    lengths = np.asarray(blanks)+2*cycles*dwell+1
    keep = np.ones(int(lengths.sum()), dtype=bool)
    start = 0
    for blank, length in zip(blanks, lengths):
        keep[start+blank:start+blank+2*dwell] = False
        start += length
    future = np.flatnonzero(keep)
    return (future[future > 0]-1)*8


def trace_available(sign, trace, blank_trace):
    return bool(((sign*trace > .001) & (abs(trace-blank_trace) > .001)).any())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/event-v1-combined-rate-initial.pt')
    parser.add_argument('--output', default='runs/temporal-visual-v1')
    parser.add_argument('--training-trials', type=int, choices=(200,1000), default=200)
    parser.add_argument('--visual-schedule', choices=('tick-v1', 'frame-horizon-v1'), default='tick-v1')
    args = parser.parse_args()
    torch.set_num_threads(1)
    sha = checksum(args.checkpoint)
    payload = torch.load(args.checkpoint, weights_only=True)
    m = payload['metadata']
    if m['config']['neural_steps'] != 8:
        raise ValueError('fixed protocol requires eight neural ticks per camera frame')
    g = m['graph']
    ids, pre, post = (np.array(g[k]) for k in ('body_ids', 'pre', 'post'))
    paths = np.array(m['pathways'])
    target_parent = int(np.flatnonzero(ids == 82450)[0])
    sources = pre[(post == target_parent) & (paths == 'predictive')]
    retina = Retina(**m['retina'])
    columns = retina.pixel_bins.reshape(32, 64)[30, 38:40].numpy()
    if columns[0] == columns[1] or m['retina']['neuron_columns'][target_parent] != columns[1]:
        raise ValueError('two-position stimulus does not isolate target column')
    sensory = np.flatnonzero(np.isin(m['retina']['neuron_columns'], columns))
    ancestors = pre[np.isin(post, sources)]
    crop, nodes, edges = crop_payload(payload, np.unique(np.concatenate((sensory, sources, ancestors, [target_parent]))))
    if len(nodes) > 512 or len(edges) > 8192:
        raise ValueError('fixed motif exceeds small-test size limit')
    target = int(np.searchsorted(nodes, target_parent))
    incoming = np.flatnonzero((crop['graph'].post == target) & (np.array(crop['pathways']) == 'predictive'))
    source = crop['graph'].pre[incoming]
    signs = crop['graph'].signs[source]
    if set(signs.tolist()) != {-1, 1}:
        raise ValueError('both fixed prediction signs required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started+(900 if args.training_trials == 1000 else 600)
    base = make_network(crop, m)
    decay = base.current_decay[base.post[incoming]].numpy()
    delays = base.delays[incoming].numpy()
    weights = base.magnitudes[incoming].numpy().copy()
    manifest = dict(source_checkpoint=args.checkpoint, source_sha256=sha, graph_sha256=crop['graph'].identity(),
        neurons=len(nodes), edges=len(edges), body_ids=crop['graph'].body_ids.tolist(),
        parent_edge_indices=edges.tolist(), target_body=82450, target_type='L3',
        cut_incoming=int((~np.isin(pre,nodes)&np.isin(post,nodes)).sum()),
        cut_outgoing=int((np.isin(pre,nodes)&~np.isin(post,nodes)).sum()),
        sources=[dict(body=int(crop['graph'].body_ids[i]), cell_type=crop['retina']['cell_types'][i],
                      sign=int(s), weight=float(w), delay=int(d)) for i,s,w,d in zip(source,signs,weights,delays)],
        config=m['config'], neurons_config=m['neurons'], learning=m['learning'],
        stimulus=dict(row=30, target_x=39, other_x=38, dwell_frames=3, cycles=4),
        seeds=dict(preflight=9021, training=9022, evaluation=9023), primary_lead_ticks=8,
        training_trials=args.training_trials, evaluation_trials=50, visual_schedule=args.visual_schedule)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    preflight, reconstruction_error = [], 0.
    for trial, blank in enumerate(np.random.default_rng(9021).integers(12,37,size=10)):
        frames = oscillation(int(blank))
        pair = []
        for condition, images in (('full', frames), ('blank', torch.zeros_like(frames))):
            net = copy.deepcopy(base)
            r = run_sequence(net,crop,m,images,[target],learning=False,deadline=deadline)
            if not torch.equal(net.magnitudes,base.magnitudes):
                raise AssertionError('preflight changed weights')
            trace = delayed_traces(r['spikes'],source,delays,signs,decay)
            np.savez_compressed(output/f'preflight-{trial}-{condition}.npz',
                **{k:v for k,v in r.items() if k!='stability'}, signed_trace=trace)
            pair.append((r,trace))
        for cycle in range(1,4):
            for polarity, sign, offset in (('on',-1,0),('off',1,3)):
                event=(int(blank)+cycle*6+offset)*8
                issue=event-8
                if pair[0][0]['target'][event,0] != sign:
                    raise AssertionError('unexpected camera event')
                for r,trace in pair:
                    error=abs(np.clip(trace[issue]@weights/base.config.threshold,-1,1)-r['prediction'][issue,0])
                    reconstruction_error=max(reconstruction_error,float(error))
                    if error>1e-6:
                        raise AssertionError('physical trace reconstruction failed')
                preflight.append(dict(trial=trial,cycle=cycle,polarity=polarity,blank_frames=int(blank),
                    event_tick=event,issue_tick=issue,trace=pair[0][1][issue].tolist(),
                    blank_trace=pair[1][1][issue].tolist(),prediction=float(pair[0][0]['prediction'][issue,0]),
                    available=trace_available(sign,pair[0][1][issue],pair[1][1][issue])))
    counts={p:sum(r['available'] for r in preflight if r['polarity']==p) for p in ('on','off')}
    result=dict(manifest=manifest,preflight=preflight,available_counts=counts,
                max_reconstruction_error=reconstruction_error,training_performed=False)
    print(f'Frozen motif: {len(nodes)} neurons/{len(edges)} edges, availability {counts}/30',flush=True)
    if min(counts.values()) < 24:
        result['status']='preflight_fail'
    else:
        schedules=[]
        for number,seed in ((args.training_trials,9022),(50,9023)):
            blanks=np.random.default_rng(seed).integers(12,37,size=number)
            schedules.append((torch.cat([oscillation(int(b)) for b in blanks]),blanks))
        net=make_network(crop,m)
        print(f'Training {args.visual_schedule} on {args.training_trials} trials',flush=True)
        train=run_sequence(net,crop,m,schedules[0][0],[target],learning=True,deadline=deadline,
                           visual_schedule=args.visual_schedule)
        learned=net.magnitudes.clone()
        np.savez_compressed(output/'training.npz',**{k:v for k,v in train.items() if k!='stability'},
            weights_initial=crop['weights'].numpy(),weights_trained=learned.numpy(),blank_frames=schedules[0][1])
        metrics={}
        frames,blanks=schedules[1]
        boundaries=recurrent_boundaries(blanks)
        targets=None
        for name,w in (('frozen',crop['weights']),('trained',learned)):
            net=make_network(crop,m,w)
            r=run_sequence(net,crop,m,frames,[target],learning=False,deadline=deadline)
            if not torch.equal(net.magnitudes,w):
                raise AssertionError('evaluation changed weights')
            if targets is not None and not np.array_equal(targets,r['target']):
                raise AssertionError('evaluation inputs differ')
            targets=r['target']
            metrics[name]=score_forecasts(targets,r['prediction'],boundaries,8)
            metrics[name]['all_frames']=score_forecasts(targets,r['prediction'],np.arange(0,len(targets),8),8)
            metrics[name]['stability']=r['stability']
            np.savez_compressed(output/(name+'.npz'),**{k:v for k,v in r.items() if k!='stability'},blank_frames=blanks)
        for name,pred in (('zero',np.zeros_like(targets)),('persistence',targets)):
            metrics[name]=score_forecasts(targets,pred,boundaries,8)
        result.update(status='pass' if passed(metrics['trained'],metrics['frozen'],metrics['zero'],metrics['persistence']) else 'learning_fail',
            training_performed=True,metrics=metrics,training_stability=train['stability'],
            train_frames=len(schedules[0][0]),evaluation_frames=len(frames),
            local_update_sums_on_off_quiet=train['local_update_sums'].sum(axis=1).tolist())
    if checksum(args.checkpoint)!=sha:
        raise AssertionError('source checkpoint changed')
    result.update(seconds=time.perf_counter()-started,source_unchanged=True)
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','seconds','training_performed')},indent=2),flush=True)


if __name__=='__main__':
    main()
