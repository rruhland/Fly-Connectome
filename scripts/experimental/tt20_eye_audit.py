"""Independently replay recorded single-camera opening experiments and verify their splits."""

import argparse
from hashlib import sha256

import numpy as np

from .tt4_transfer_trial import _read, _write, _hash
from .tt20_single_eye import SAMPLES
from .tt20_single_eye_trial import EYES, OUT, Trial, summary


def verify():
    sources = {}
    data = dict(eyes={}, physical_tapes=0)
    for eye in EYES:
        path = OUT/f'one-eye-{eye}-practice.json.gz'
        source = _read(path)
        sources[eye] = _hash(path)
        station = Trial(eye)
        memories = station.memories()
        counts = {}
        try:
            def replay(row):
                arena = station.arena
                arena.reset(seed=row['seed'], server=row['role'], render=False)
                tape = memories[row['role']].commands(row['selection'], row['cue'], row['role'], arena.max_steps)
                actions = tape[:row['elapsed']]
                if sha256(actions.tobytes()).hexdigest() != row['action']:
                    raise ValueError('stored selection does not reproduce the executed motor program')
                digest = sha256()
                for tick, a in enumerate(actions):
                    if tick in SAMPLES[row['role']]:
                        images = arena.render_cameras()
                        if tuple(images) != (eye,):
                            raise ValueError('unexpected stream in replay')
                        digest.update(images[eye])
                    _,reward,done,capped,_ = arena.step(a,render=False)
                if (not (done or capped) or arena.point_result() != row['outcome']
                        or reward != row['reward'] or digest.hexdigest() != row['digest']):
                    raise ValueError(f"physical or camera evidence changed on {eye}/{row['seed']}")
                return 1
            for row in source['practice']:
                data['physical_tapes'] += replay(row)
            practice = {r['seed'] for r in source['practice']}
            if len(practice) != len(source['practice']):
                raise ValueError('duplicate practice points')
            for path in sorted(OUT.glob(f'one-eye-{eye}-*-*-*.json.gz')):
                saved = _read(path)
                held = {r['seed'] for r in saved['held']}
                local = {r['seed'] for r in saved['local_practice']}
                if held & (practice | local) or local & practice:
                    raise ValueError('practice overlaps with held data')
                score = summary(saved['held'])
                if score != saved['summary']:
                    raise ValueError('outcome summary changed')
                for row in saved['local_practice'] + saved['held']:
                    data['physical_tapes'] += replay(row)
                counts[f"{saved['condition']}-{saved['target_points']}"] = score
            data['eyes'][eye] = dict(practice=source['summary'], held=counts)
        finally:
            station.arena.close()
    data['source_sha256'] = sources
    _write(OUT/'one-eye-audit.json.gz',data)
    return data['physical_tapes']


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print('Verified point tapes and exact sensor frames:',verify())
