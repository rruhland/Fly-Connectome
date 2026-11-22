"""Score the optical flight fit on replays, then test it in uninterrupted physical points."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT,_read,_write
from .tt22_flight_inverse import FlightInverse
from .tt22_rally import RallyPlayer
from .tt22_rally_trial import Station,replay

DIR=ROOT/'artifacts/tt22'


def audit():
    result=[]
    infer=FlightInverse()
    for row in _read(DIR/'imaging-offline.json.gz')['rows']:
        history=[]
        for s in row['samples']:
            e=s['estimates']['stock']
            history.append((s['tick'],e['position']))
            if s['active'] and e['velocity'] is not None:
                found=infer.fit(history)
                if found is not None:
                    _,v,spin,rms=found
                    result.append(dict(seed=row['seed'],tick=s['tick'],phase=s['bounce_phase'],
                                       stock=e['velocity'],estimated=v.tolist(),ref=s['velocity'],
                                       spin=spin.tolist(),rms_mm=rms))
    _write(DIR/'flight-inverse-audit.json.gz',dict(rows=result))
    for phase in set(r['phase'] for r in result):
        rows=[r for r in result if r['phase']==phase]
        err=[[np.linalg.norm(np.subtract(r[k],r['ref'])) for k in ['stock','estimated']] for r in rows]
        print(phase,len(rows),np.median(err,axis=0),np.quantile(err,.9,axis=0),flush=True)


class InversePlayer(RallyPlayer):
    def __init__(self,law):
        super().__init__(law)
        self.infer=FlightInverse()

    def search(self,estimate,observation,target):
        fit=self.infer.fit(self.history)
        if fit is None:
            return super().search(estimate,observation,target)
        p,v,w,rms=fit
        # Optical z has a calibration bias, so keep the usual measured center at decision time.
        revised=replace(observation,velocity=v,spin=w)
        choice=super().search((estimate[0],v/30.),revised,target)
        if choice is not None:
            choice[1]['flight_fit']=dict(velocity=v.tolist(),spin=w.tolist(),rms_mm=rms)
        return choice


def play(count):
    doc=dict(spread=.30,pace=.82,rival='anchor',privileged=False,rows=[])
    station=Station(doc['spread'],doc['pace'],doc['rival'])
    law=station.player.law
    station.player.close()
    station.player=InversePlayer(law)
    try:
        for s in np.random.default_rng(220381117).integers(2**31,size=count):
            row=station.point(int(s))
            doc['rows'].append(row)
            _write(DIR/'flight-inverse-held.json.gz',doc)
            print(s,row['longest'],flush=True)
    finally:
        station.close()
    doc['verified']=replay(doc['rows'],doc['spread'],doc['pace'],doc['rival'])
    _write(DIR/'flight-inverse-held.json.gz',doc)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--play',type=int,default=0)
    args=p.parse_args()
    play(args.play) if args.play else audit()
