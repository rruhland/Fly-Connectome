"""Revisit wide-feed stalls with additional edge-of-blade motor proposals."""

import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_factory import CONTACT_OFFSET, moving_joint_rate
from .tt19_shoot import COURT, HORIZON
from .tt22_imaging_trial import verify_live
from .tt22_rally import DEPTHS, LEADS, PACES, face, rates_for
from .tt22_rally_trial import CONFIG, OUTPUT
from .tt22_value_calibration import CALIBRATION
from .tt22_value_player import LIMIT, ValuePlayer
from .tt22_value_support import SupportedValue
from .tt22_value_trial import REPLICATION_RECEIPT, SOURCE, SUPPORT_RECEIPT, Station, report


RECEIPT = OUTPUT/'value-wide-development.json.gz'
STALLS = {SUPPORT_RECEIPT:(1774760871,), REPLICATION_RECEIPT:(2054549508,584279213)}


class WidePlayer(ValuePlayer):
    """Offer reachable blade-edge poses when the standard deck has no proposals."""

    def candidate_tapes(self,estimate,target):
        tapes,params = super().candidate_tapes(estimate,target)
        if len(tapes):
            return tapes,params
        points,speeds = self.law.trajectory(*estimate,HORIZON,bounds=COURT)
        timeline = np.arange(HORIZON)
        goals,params = [],[]
        for depth in DEPTHS:
            hits = np.flatnonzero((points[:-1,1]>=depth)&(points[1:,1]<depth))
            if not len(hits):
                continue
            i = hits[0]
            fraction = (depth-points[i,1])/(points[i+1,1]-points[i,1])
            eta = i+fraction
            center = points[i]+fraction*(points[i+1]-points[i])
            if not (1.4<eta<HORIZON-2 and .33<=abs(center[0])<.47 and .056<center[2]<.45):
                continue
            speed = speeds[i]+fraction*(speeds[i+1]-speeds[i])
            default_yaw = self.rally_yaw(center,speed)
            for pitch,edge,steer in product((.18,.32,.50,.68),(.02,.045,.07),(-.20,0.,.20)):
                yaw = default_yaw+steer
                normal = face(pitch,yaw)
                tangent = np.array((1.,0.,0.))-normal*normal[0]
                edge_shift = np.copysign(edge,center[0])*tangent/np.linalg.norm(tangent)
                anchor = center-CONTACT_OFFSET*normal-edge_shift
                pose = self.factory.solve(anchor,normal,target)
                recovered,facing,_,_ = self.factory.pose(pose)
                error = np.linalg.norm(recovered-anchor)
                if error>.057 or facing@normal<.77:
                    continue
                for pace,lead in product(PACES,LEADS):
                    rate = moving_joint_rate(self.factory,pose,pace)
                    phase = np.clip(timeline-eta+lead,-3.1,3.4)
                    goals.append(pose+phase[:,None]*rate)
                    params.append(dict(proposal='edge',pitch=pitch,yaw=yaw,eta=float(eta),depth=depth,
                                       anchor=center.tolist(),lead=lead,pace=pace,alignment=float(error),edge=edge))
        if not goals:
            return tapes,params
        selection = np.linspace(0,len(goals)-1,min(len(goals),LIMIT)).astype(int)
        return rates_for(np.asarray(goals)[selection],target,self.factory.low,self.factory.high), [
            params[index] for index in selection]


def run(output):
    started = perf_counter()
    checkpoint = _read(CALIBRATION)
    model = SupportedValue(checkpoint,'kernel')
    gains = _read(SOURCE)['candidates']['duration'][1:]
    data = dict(schema='tt23-value-wide-development-1',arena=CONFIG,source=CALIBRATION.name,
                source_sha256=_hash(CALIBRATION),spread=.30,pace=.82,rival='anchor',
                recorded_starts={path.name:list(seeds) for path,seeds in STALLS.items()},
                comparison_receipts={path.name:_hash(path) for path in STALLS},portable_gains=gains,pairs=[])
    station = Station(dict(wide=model),gains,data['spread'],data['pace'],checkpoint['horizon'],True,
                      player_type=WidePlayer)
    try:
        for path,seeds in STALLS.items():
            previous = {pair['kernel']['seed']:pair['kernel'] for pair in _read(path)['pairs']}
            for seed in seeds:
                regular = previous[seed]
                wide = station.point(seed,'wide')
                first_edge = next((shot['tick'] for shot in wide['shots'] if shot['proposal']=='edge'),None)
                prefix = len(regular['actions']) if first_edge is None else first_edge
                if regular['actions'][:prefix]!=wide['actions'][:prefix]:
                    raise ValueError('the development replay diverged before its first wide proposal')
                data['pairs'].append(dict(kernel=regular,wide=wide))
                data['summary'] = {mode:report([rows[mode] for rows in data['pairs']]) for mode in ('kernel','wide')}
                _write(output,data)
                print('wide proposal',seed,regular['longest'],wide['longest'],first_edge,flush=True)
    finally:
        station.close()
    data['verified'] = verify_live(data)
    data['elapsed_seconds'] = perf_counter()-started
    _write(output,data)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RECEIPT)
    run(parser.parse_args().output)
