"""Choose real strokes under a distribution of errors learned from predicted and observed video."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_flight import FlightLaw
from .tt19_shoot import HORIZON
from .tt22_far_cache import FarCache
from .tt22_flight_inverse import FlightInverse
from .tt22_match import CHECKPOINT
from .tt22_ordinary_game import DIR, FrozenMemory, StandardMatch
from .tt22_rally import RallyPlayer
from .tt22_after_service import verify_rows
from .tt23_optical_contact import OFFSETS, OUT, PRACTICE

SCALE = .035
START = 1185630613
MODEL = OUT/'risk-weights.json.gz'
LIVE = OUT/'risk-native.json.gz'


def posterior(likelihood, prior):
    weights = prior.copy()
    for _ in range(12):
        responsibility = likelihood * weights
        responsibility /= responsibility.sum(axis=1,keepdims=True)
        weights = (responsibility.sum(axis=0) + 8*prior) / (len(likelihood)+8)
    return weights


def train():
    practice = _read(PRACTICE)
    rows = practice['rows']
    error = np.array([s['errors_m'] for s in rows])
    seed = np.array([s['seed'] for s in rows])
    likelihood = np.exp(-error/SCALE)
    prior = np.full(error.shape[1],1./error.shape[1])
    scores = []
    for donor in np.unique(seed):
        test = seed == donor
        weights = posterior(likelihood[~test],prior)
        scores.extend(-np.log(np.column_stack((likelihood[test,0], likelihood[test]@prior, likelihood[test]@weights))))
    weights = posterior(likelihood,prior)
    # Use a fixed subset covering all three axes. Assign each full-grid cell to its nearest representative.
    ids = [0] + [i for i,off in enumerate(OFFSETS) if np.all(off!=0.) and abs(off[2])==.008]
    norm = np.array([.006,.010,.008])
    # Split exact-distance ties symmetrically.
    distance = np.linalg.norm((OFFSETS[:,None]-OFFSETS[ids])/norm,axis=2)
    mask = np.isclose(distance, np.min(distance,axis=1,keepdims=True))
    assignment = mask/mask.sum(axis=1,keepdims=True)
    result = dict(practice_sha256=_hash(PRACTICE), scale_m=SCALE, donors=len(set(seed)), shots=len(rows),
                  score_order=['single_zero','uniform','learned'], mean_held_score=np.mean(scores,axis=0).tolist(),
                  offsets=OFFSETS[ids].tolist(), weights=(weights@assignment).tolist(), uniform=(prior@assignment).tolist())
    _write(MODEL,result)
    return result


class RiskPlayer(RallyPlayer):
    def __init__(self,model):
        super().__init__(FlightLaw())
        self.infer = FlightInverse()
        self.scenarios = np.column_stack((np.array(model['offsets']), np.zeros((len(model['offsets']),6))))
        self.weights = np.asarray(model['weights'])

    def search(self, estimate, observation, target):
        from dataclasses import replace
        fit = self.infer.fit(self.history)
        if fit is not None:
            observation = replace(observation,velocity=fit[1],spin=fit[2])
            estimate = estimate[0],fit[1]/30.
        tapes,params = self.goal_tapes(estimate,target)
        if not len(tapes):
            return None
        lengths = np.clip(np.ceil([p['eta'] for p in params]).astype(int)+3,3,HORIZON)
        self.sim.current = self.sim.twin.state()
        original = self.sim.predict(tapes,lengths,observation,np.zeros((len(tapes),9)))
        cost = np.array([self.memory.rank(r) for r in original])
        picks = np.argsort(cost)[:8]
        if not original[picks[0]]['hit']:
            return None
        n = len(self.weights)
        trials = self.sim.predict(np.repeat(tapes[picks], n, axis=0), np.repeat(lengths[picks],n),
                                  observation, np.tile(self.scenarios,(len(picks),1)))
        risk = np.array([self.memory.rank(row) for row in trials]).reshape(len(picks),n)
        chosen = int(picks[np.argmin(risk@self.weights)])
        prediction = original[chosen]
        if not prediction['legal'] and estimate[0][1]>-.64:
            return None
        return tapes[chosen,:lengths[chosen]],dict(params[chosen], prediction=prediction, candidate_count=len(tapes))


def test(path,count):
    model = _read(MODEL)
    seeds = np.random.default_rng(START).integers(2**31,size=count).tolist()
    assert not set(seeds)&{s['seed'] for s in _read(PRACTICE)['rows']}
    feed = _read(DIR/'tt22_opponent-learned.json.gz')['fit']['profile']
    memory = FrozenMemory(_read(CHECKPOINT)['evidence'])
    station = StandardMatch(feed)
    cache = FarCache(station.rival)
    original = station.player
    risk = RiskPlayer(model)
    names = ['stock','uniform','learned']
    data = dict(model_sha256=_hash(MODEL),seeds=seeds,rows={k:[] for k in names})
    if path.exists():
        prior = _read(path)
        if prior['seeds']!=seeds or prior['model_sha256']!=data['model_sha256']:
            raise ValueError('incompatible saved training or evaluation')
        data = prior
    try:
        for i,seed in enumerate(seeds):
            for k in names[i%3:]+names[:i%3]:
                if len(data['rows'][k])>i:
                    if data['rows'][k][i]['seed']!=seed:
                        raise ValueError('incomplete prior seed schedule')
                    continue
                station.player = original if k=='stock' else risk
                if k!='stock':
                    risk.weights = np.asarray(model['uniform' if k=='uniform' else 'weights'])
                row = station.play(seed,1,memory)
                data['rows'][k].append(row)
                _write(path,data)
                print(i,k,row['longest'],flush=True)
        allrows = [r for rows in data['rows'].values() for r in rows]
        verify_rows(allrows)
        data['verified']=len(allrows)
        data['profile']=feed
        data['opponent_cache']=dict(hits=cache.hits,misses=cache.misses,audited=cache.checked)
        _write(path,data)
    finally:
        station.player=original
        risk.close()
        station.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--count',type=int,default=6)
    p.add_argument('--output',type=Path,default=LIVE)
    a=p.parse_args()
    if a.run:
        test(a.output,a.count)
    else:
        print(train())
