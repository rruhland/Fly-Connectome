"""Check how much of the full-game rally capacity comes from carried optical practice."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read,_write
from .tt8_service_arena import ServiceScore
from .tt22_match import CHECKPOINT,summarize
from .tt22_ordinary_game import DIR,SEED,StandardMatch,FrozenMemory,verify


def play(station,memory,path,condition):
    rng=np.random.default_rng(SEED)
    score=ServiceScore(first_server=-1)
    doc=dict(seed=SEED,condition=condition,rows=[],initial_evidence=memory.evidence.copy())
    if path.exists():
        previous = _read(path)
        if previous["seed"] != SEED or previous["condition"] != condition or previous["initial_evidence"] != doc["initial_evidence"]:
            raise ValueError("incompatible prior game")
        doc = previous
    for i, s in enumerate(rng.integers(2**31,size=90)):
        if i < len(doc["rows"]):
            row = doc["rows"][i]
            if row["seed"] != int(s) or row["role"] != score.server:
                raise ValueError("incompatible prior point")
            score.record(row["outcome"])
            if score.finished:
                break
            continue
        before=list(score.score)
        row=station.play(int(s),score.server,memory)
        after=list(score.record(row['outcome']))
        doc['rows'].append(dict(row,score_before=before,score_after=after))
        _write(path,doc)
        print(len(doc['rows']),after,row['longest'],flush=True)
        if score.finished:
            break
    doc['verified']=verify(doc['rows'])
    doc['summary']=summarize(doc['rows'])
    _write(path,doc)
    return doc


if __name__=='__main__':
    station=StandardMatch(_read(DIR/'tt22_opponent-learned.json.gz')['fit']['profile'])
    try:
        play(station,FrozenMemory(guard_empty=True),DIR/'ordinary-zero-practice.json.gz','zero prior with same guard')
    finally:
        station.close()
