"""Compare chronological warm and cold target learning with one public duration per point."""

import argparse

import numpy as np
from scipy.special import betaincinv

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_transfer import MACRO,STREAMS
from .tt22_joint_practice import draw
from .tt23_coach import PRACTICE as COACH
from .tt23_coach_trial import ReplyRanking
from .tt23_opening_motor import CHECKPOINT as STROKE
from .tt23_opening_sequence import CorrectedOpeningStation
from .tt23_optical_contact import OUT
from .tt23_sequence import MODES, setup, summarize,verify
from .tt23_sequence_learning import SOURCE

STREAM_SEEDS=(186219799,1641730608,1305364061,733645001)


def stream_schedule(stream):
    seed=STREAM_SEEDS[stream]
    rng=np.random.default_rng(seed)
    by_band=[draw(rng,6,band) for band in STREAMS.values()]
    starts=[dict(seed=case,lane=lane,band=k) for pair in zip(*by_band,strict=True)
            for k,(case,lane) in zip(STREAMS,pair,strict=True)]
    return starts,rng.uniform(1.e-7,1.-1.e-7,size=(12,3))


def play(stream):
    path=OUT/f'online-rally-stream-{stream+1}.json.gz'
    common=setup()
    scored=_read(SOURCE)
    ranks=np.array(scored['rank'])
    start={'pong':.25+.25*(ranks-1),'cold':np.full(len(MODES),.5)}
    counts={k:[2*mean,2*(1.-mean)] for k,mean in start.items()}
    schedule,uniforms=stream_schedule(stream)
    if set(s['seed'] for s in schedule)&{s['seed'] for s in common['starts']}:
        raise ValueError('reused reference course')
    for archive in (STROKE,COACH):
        if set(s['seed'] for s in schedule)&{r['seed'] for r in _read(archive).get('rows',[]) if 'seed' in r}:
            raise ValueError('reused target practice')
    weights=_read(STROKE)['model']
    station=CorrectedOpeningStation(_read(MACRO)['parameters'],common['evidence'],common['far_profile'],weights)
    station.rival.confirm=ReplyRanking(station.rival,_read(COACH)).confirm
    lineage=dict(source_hash=_hash(SOURCE),stroke_hash=_hash(STROKE),coach_hash=_hash(COACH),starts=schedule,
                 sampling=uniforms.tolist(),opening_gains=common['gains'],stream=stream)
    doc=dict(lineage=lineage,cases=[])
    if path.exists():
        doc=_read(path)
        if doc['lineage']!=lineage:
            raise ValueError('cannot resume from a different policy or physical schedule')
    try:
        for i,startpoint in enumerate(schedule):
            if i>=len(doc['cases']):
                doc['cases'].append(dict(startpoint,results={},choices={}))
            case=doc['cases'][i]
            for k,(a,b) in counts.items():
                chosen=int(np.argmax(betaincinv(a,b,uniforms[i])))
                option=MODES[chosen]
                if k in case['results']:
                    if case['choices'][k]!=option: raise ValueError('saved choice cannot be regenerated')
                else:
                    case['choices'][k]=option
                    other='cold' if k=='pong' else 'pong'
                    if case['choices'].get(other)==option and other in case['results']:
                        row=case['results'][other]
                    else:
                        row=station.play(case['seed'],case['lane'],common['gains'][option])
                    case['results'][k]=row
                    _write(path,doc)
                    print(stream+1,i+1,k,option,row['longest'],flush=True)
                duration=len(case['results'][k]['actions'])
                utility=min(duration/300.,1.)
                a[chosen]+=utility
                b[chosen]+=1.-utility
        doc['verified']=verify(doc['cases'])
        doc['summary']={k:summarize([c['results'][k] for c in doc['cases']]) for k in counts}
        doc['posteriors']={k:[a.tolist(),b.tolist()] for k,(a,b) in counts.items()}
        _write(path,doc)
    finally:
        station.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream',type=int,choices=range(1,5),required=True)
    play(p.parse_args().stream-1)
