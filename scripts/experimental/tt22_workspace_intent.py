"""Test a coarse motor goal from native Pong with fine control inside the chosen physical workspace cell."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT,_read,_write
from .tt19_flight import FlightLaw
from .tt22_opponent import ANCHOR,VARIED
from .tt22_spatial import SpatialPlayer
from .tt22_spatial_anonymous import Station,verify,DURATION_SOURCE
from .tt22_spatial_trial import SOURCE
from .tt22_rally_trial import OUTPUT


class CellPlayer(SpatialPlayer):
    def lateral_contact(self,estimate,center):
        requested=super().lateral_contact(estimate,center)
        cell=.08*float(np.clip(np.rint(requested/.08),-3,3))
        return float(np.clip(center[0],cell-.04,cell+.04))


class WorkspaceStation(Station):
    def __init__(self):
        super().__init__(np.zeros(4))
        self.profile=np.array(_read(OUTPUT/'tt22_opponent-learned.json.gz')['fit']['profile'])
        self.rival.target=lambda:tuple(self.profile+.5*(
            np.array(VARIED[(self.arena.seed+self.arena.contacts-1)%len(VARIED)])-ANCHOR))
        self.player.close()
        self.player=CellPlayer(FlightLaw(),direct=True)


def run(seed,count,path):
    source=_read(SOURCE)
    index=np.argmax(np.array(source['totals'])/source['counts'])
    warm=np.asarray(source['coefficients'][index])
    choices={'pong':warm.tolist(),'neutral':[0.,0.,0.],'opposed':(-warm).tolist()}
    station=WorkspaceStation()
    doc=dict(seed=seed,cell_m=.08,source=choices,profile=station.profile.tolist(),rows={k:[] for k in choices})
    if path.exists():
        previous = _read(path)
        if any(doc[key] != previous[key] for key in ("seed", "cell_m", "source", "profile")):
            raise ValueError("incompatible saved course")
        doc = previous
    try:
        for i,point in enumerate(np.random.default_rng(seed).integers(2**31,size=count)):
            keys=list(choices)
            for k in keys[i%3:]+keys[:i%3]:
                if len(doc["rows"][k]) > i:
                    if doc["rows"][k][i]["seed"] != int(point):
                        raise ValueError("incompatible saved seed")
                    continue
                row=station.point(int(point),choices[k])
                # Parent's diagnostic labels describe the unquantized proposal. Record both explicitly.
                for shot in row['shots']:
                    x=shot['contact_x']
                    cell=.08*float(np.clip(np.rint(x['requested']/.08),-3,3))
                    x['cell_center']=cell
                    x['issued_x']=float(np.clip(x['nominal'],cell-.04,cell+.04))
                doc['rows'][k].append(row)
                print(i,k,row['longest'],row['outcome']['legal_returns'],flush=True)
                _write(path,doc)
    finally:
        station.close()
    doc['verified']=verify([row for rows in doc['rows'].values() for row in rows])
    _write(path,doc)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count',type=int,default=12);p.add_argument('--seed',type=int,default=147529001)
    p.add_argument('--output',type=Path,default=OUTPUT/'workspace-intent.json.gz')
    a=p.parse_args();run(a.seed,a.count,a.output)
