"""Test the mean arm return count across independent duration fits under shuffled experience."""

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_transfer import MACRO
from .tt22_motor_transfer import MotorPrimitive
from .tt23_anonymous_action import CHECK, WIDTH
from .tt23_anonymous_replication import TRAIN
from .tt23_anonymous_target import target_context
from .tt23_direction import ARM, ARM_BANK, MIRROR, SOURCE, durations, marginal_votes
from .tt23_optical_contact import OUT


RECEIPT = OUT/'direction-permutation.json.gz'
DRAWS = 4096
SEED = 167425599


def kernel(query,source):
    distance = ((query[:,None]-source)/WIDTH)**2
    return np.exp(-.5*np.minimum(distance.sum(axis=-1),100.))


def options(weight,threshold,votes,seen):
    preference = weight@votes.T
    picked = np.where(preference>threshold[:,None],4,np.where(preference<-threshold[:,None],2,3))
    picked[~seen] = 3
    return picked


def run():
    frozen,arm,bank = (_read(path) for path in (SOURCE,ARM,ARM_BANK))
    if arm['source_sha256']!=_hash(SOURCE) or arm['bank_sha256']!=_hash(ARM_BANK):
        raise ValueError('the evaluation changed after the frozen direction policy was scored')
    if any(frozen['sources'][path.name]!=_hash(path) for path in (CHECK,TRAIN)):
        raise ValueError('the native source archives changed after fitting')
    old,replicated = _read(CHECK),_read(TRAIN)
    archives = [(1559000000,old['training']),*((r['offset'],r['rows']) for r in replicated['sets'])]
    ready = MotorPrimitive(np.asarray(_read(MACRO)['parameters']),None).ready_world
    contexts = [target_context(row['sight'],ready) for row in bank['rows']]
    seen = np.array([x is not None for x in contexts])
    query = np.array([x if x is not None else np.zeros(4) for x in contexts])
    outcomes = np.array([[choice['legal']>0 for choice in row['options']] for row in bank['rows']])
    neutral = int(outcomes[:,3].sum())
    indices = np.arange(len(query))[:,None]
    recorded = []
    for (offset,archive),assessed in zip(archives,arm['independent'],strict=True):
        if offset!=assessed['offset']:
            raise ValueError('the native training groups changed order')
        x,ticks,_ = durations(archive)
        positive,reflected = kernel(query,x),kernel(query,x*MIRROR)
        weights = positive-reflected
        threshold = 1.e-12*(positive+reflected).sum(axis=1)
        real = options(weights,threshold,marginal_votes(ticks)[None,:],seen)[:,0]
        expected = assessed['scores']['learned']
        if real.tolist()!=expected['choices']:
            raise ValueError('vectorized scoring changed an actual frozen motor choice')
        rng = np.random.default_rng(np.random.SeedSequence([SEED,offset]))
        permutations = rng.random((DRAWS,*ticks.shape)).argsort(axis=2)
        reward = np.take_along_axis(ticks[None,:,:],permutations,axis=2)
        votes = np.sign(reward[:,:,4:]-reward[:,:,2::-1]).mean(axis=2)
        picks = options(weights,threshold,votes,seen)
        null = outcomes[indices,picks].sum(axis=0)
        recorded.append(dict(offset=offset,real_legal=expected['legal'],null_legal=null.tolist(),
                             probability_beats_center=float(np.mean(null>neutral)),
                             probability_matches_fit=float(np.mean(null>=expected['legal']))))
    observed = np.array([row['real_legal'] for row in recorded])
    shuffles = np.array([row['null_legal'] for row in recorded])
    mean_null = shuffles.mean(axis=0)
    above_mean = int(sum(mean_null>=observed.mean()))
    all_above_center = int(np.all(shuffles>neutral,axis=0).sum())
    all_match_fit = int(np.all(shuffles>=observed[:,None],axis=0).sum())
    result = dict(schema='tt23-direction-permutation-1',arm_sha256=_hash(ARM),bank_sha256=_hash(ARM_BANK),
                  seed=SEED,draws=DRAWS,neutral=neutral,source_fits=recorded,observed_mean=float(observed.mean()),
                  null_mean=float(mean_null.mean()),null_quantiles=np.quantile(mean_null,[.05,.5,.95]).tolist(),
                  joint_mean_exceedances=above_mean,joint_mean_p=(above_mean+1)/(DRAWS+1),
                  all_five_exceed_center=all_above_center,all_five_exceed_center_p=(all_above_center+1)/(DRAWS+1),
                  all_five_match_fit=all_match_fit,all_five_match_fit_p=(all_match_fit+1)/(DRAWS+1))
    _write(RECEIPT,result)
    print({k:v for k,v in result.items() if k!='source_fits'},flush=True)


if __name__=='__main__':
    run()
