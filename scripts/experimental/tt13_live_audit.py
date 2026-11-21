"""Score saved games against physical port assignments only after gameplay."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .tt4_transfer_trial import _read
from .tt6_body_arena import TT6Wiring
from .tt10_series_trial import _game_summary


def assignment(width, original, current):
    arm = SimpleNamespace(action_size=width)
    source, target = TT6Wiring(arm, original), TT6Wiring(arm, current)
    order = np.argsort(source.order)[target.order]
    return [order.tolist(), (target.signs * source.signs[order]).tolist()]


def audit(receipt):
    data = _read(receipt)
    name = receipt.name.removesuffix('.json.gz')
    memory = _read(receipt.with_name(name+'-player.json.gz'))['port_memory']
    keys = memory['keys']
    width = len(keys[0][0])
    segments = []
    for game_index, game in enumerate(data['games'],start=1):
        if _game_summary(game['points']) != game['metrics']:
            raise ValueError('score summary diverged')
        for installation in dict.fromkeys(row['installation'] for row in game['points']):
            rows = [row for row in game['points'] if row['installation']==installation]
            actual = assignment(width,data['arena']['seed'],installation)
            correct = [keys[row['choice']['active']]==actual for row in rows]
            segments.append(dict(game=game_index, changed=installation!=data['arena']['seed'],
                                 points=len(rows), wins=sum(r['points_for'] for r in rows),
                                 returns=sum(r['player_returns'] for r in rows),
                                 sustained=sum(r['summary']['sustained'] for r in rows),
                                 used_correct=sum(keys[r['model_used']]==actual for r in rows),
                                 correct_after=''.join('+' if ok else '-' for ok in correct),
                                 switches=sum(r['model_used']!=r['choice']['active'] for r in rows)))
    return segments


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipts',nargs='+',type=Path)
    for receipt in parser.parse_args().receipts:
        print(receipt.name,audit(receipt))
