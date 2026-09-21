"""Frozen longer-horizon and bounded next-event diagnostics; no training changes."""
import argparse
import json
from pathlib import Path
import torch
from audit_supported_forecasts import audit,group_scores


def fixed_horizons(targets,timeline,positive,negative,neural_steps,dt,horizons=(30,50,100)):
    leads={str(ms):max(1,round(ms/(1000*dt))) for ms in horizons}
    bases=torch.arange(len(targets))*neural_steps
    frames=(bases-max(leads.values())+1>=1).nonzero().flatten()
    if not len(frames):
        raise ValueError('sequence too short for fixed horizons')
    permutation=torch.randperm(len(frames),generator=torch.Generator().manual_seed(421))
    result={}
    for name,lead in leads.items():
        indices=bases[frames]-lead+1
        prediction=timeline[indices]
        # timeline[i] follows neural tick i-1; only that tick's frame is observed.
        observed_frames=(indices-1)//neural_steps
        score=lambda p:group_scores(targets[frames],p,positive,negative)
        result[name]=dict(actual_lead_ms=lead*dt*1000,model=score(prediction),
            shuffled=score(prediction[permutation]),frame_persistence=score(targets[observed_frames]))
    return dict(target_frames=frames.tolist(),scores=result,
        convention='Lead is time from the recorded neural tick to the future target tick. Common target frames at every horizon; no post-event activity.')


def next_events(targets,horizon):
    if horizon<1 or len(targets)<=horizon:
        raise ValueError('positive horizon and complete future windows required')
    count=len(targets)-horizon
    labels=torch.zeros_like(targets[:count])
    waits=torch.zeros_like(labels,dtype=torch.int64)
    for offset in range(1,horizon+1):
        future=targets[offset:offset+count]
        first=(waits==0)&(future!=0)
        labels=torch.where(first,future,labels)
        waits=torch.where(first,offset,waits)
    return labels,waits


def diagnose(targets,timeline,positive,negative,neural_steps,dt):
    fixed=fixed_horizons(targets,timeline,positive,negative,neural_steps,dt)
    horizon=max(1,round(.1/(neural_steps*dt)))
    labels,waits=next_events(targets,horizon)
    count=len(labels)
    # Decision after all neural ticks of frame f; target search starts at f+1.
    prediction=timeline[(torch.arange(count)+1)*neural_steps]
    last=torch.zeros_like(targets[0])
    history=[]
    for frame in targets[:count]:
        last=torch.where(frame!=0,frame,last)
        history.append(last)
    history=torch.stack(history)
    permutation=torch.randperm(count,generator=torch.Generator().manual_seed(421))
    score=lambda p:group_scores(labels,p,positive,negative)
    controls=dict(model=prediction,shuffled=prediction[permutation],
                  last_polarity=history,opposite_last_polarity=-history,frame_persistence=targets[:count])
    return dict(fixed=fixed,next_event=dict(horizon_frames=horizon,horizon_ms=horizon*neural_steps*dt*1000,
        decision_frames=count,windows_with_event=int((waits>0).sum()),windows_without_event=int((waits==0).sum()),
        mean_wait_frames=float(waits[waits>0].double().mean()) if (waits>0).any() else None,
        scores={name:score(p) for name,p in controls.items()},
        convention='At every frame end, score the next nonzero signed input to the same sensory neuron within 100ms; zero if none. Exclude incomplete tail windows. Labels use future observations only for offline scoring. Windows overlap and repeat events; this is not independent evidence or an online learning rule. Quiet windows and polarity controls are included.'),
        interpretation='Exploratory frozen current readout. Does not test retraining at another horizon or whether other internal states contain decodable information. No new readout, altered traces, model fitting, or final-seed use.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--seeds',default='1101,1102')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    result=audit(args.checkpoint,args.steps,[int(s) for s in args.seeds.split(',')],diagnostic=diagnose)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(training_steps=result['training_steps'],
        fixed={h:v['model']['all']['model_mse'] for h,v in result['additional_diagnostic']['fixed']['scores'].items()},
        next_event=result['additional_diagnostic']['next_event']['scores']['model']['all'])),flush=True)
