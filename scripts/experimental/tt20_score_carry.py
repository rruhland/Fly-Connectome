"""Choose a serve from past point rewards and verify exact policy matches on paired tosses."""
from hashlib import sha256
import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt20_camera_carry import EYES, OUTPUT
from .tt20_carry_serves import paired


def rewarded_choice(rows, prior):
    # The program library is the same at both cameras. Ties keep the factory
    # choice if possible, then use stable program order.
    count = np.bincount([r['candidate'] for r in rows], minlength=4)
    wins = np.bincount([r['candidate'] for r in rows], weights=[r['reward']>0 for r in rows], minlength=4)
    scores = (wins+1.)/(count+2.)
    best = np.flatnonzero(scores == max(scores))
    return int(prior if prior in best else best[0]), count.tolist(), wins.astype(int).tolist()


def run():
    sources = {}
    for eye in EYES:
        path = OUTPUT/f'camera-carry-{eye}-practice.json.gz'
        practice = _read(path)
        choice, counts, wins = rewarded_choice(practice['points'],practice['factory_pick'])
        sources[eye] = dict(practice_sha256=_hash(path),choice=choice,attempts=counts,wins=wins)
    result = {}
    for eye in EYES:
        other = next(e for e in EYES if e != eye)
        path = OUTPUT/f'camera-carry-{eye}-paired-serves.json.gz'
        data = _read(path)
        index = sources[other]['choice']
        rows = data['rows']
        for group in rows:
            if group[str(index)]['candidate'] != index:
                raise ValueError('the saved policy did not execute the program ranked on the other eye')
        digest = sha256()
        for group in rows:
            digest.update(np.asarray(group[str(index)]['actions'],dtype='<f8').tobytes())
        result[eye] = dict(held_sha256=_hash(path),tape_sha256=digest.hexdigest(),choice=index,
                           metrics={f:paired(rows,data['choices']['factory'],index,f) for f in ('wins','legal')})
    _write(OUTPUT/'score-only-carry.json.gz',dict(source=sources,destination=result))
    return result


if __name__ == '__main__':
    print(run())
