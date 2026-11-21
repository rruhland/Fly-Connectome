"""Measure shaded-stereo throughput and RGB differences on identical physical actions."""
import argparse
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from .tt4_arena import TT4Arena
from .tt4_transfer import calibration_commands
from .tt5_hall_trial import simplify_hall_lighting


def measure():
    cases = []
    for i in range(4):
        captures = {}
        stats = {}
        for fast in ((True, False) if i % 2 else (False, True)):
            arena = TT4Arena(69237+i, width=192, height=144, render_mode='mujoco', venue='hall', baseline=.28)
            if fast:
                simplify_hall_lighting(arena)
            try:
                arena.reset(seed=69237+i)
                frames = []
                cpu = process_time()
                wall = perf_counter()
                for action in calibration_commands(69251+i, 34):
                    img, _, term, trunc, _ = arena.step(action)
                    frames.append(img)
                    if term or trunc:
                        break
                stats['fast' if fast else 'normal'] = dict(cpu_ms=1000*(process_time()-cpu)/len(frames),
                                                          wall_ms=1000*(perf_counter()-wall)/len(frames))
                captures[fast] = (np.stack(frames),arena.data.qpos.copy())
            finally:
                arena.close()
        normal, fast = captures[False], captures[True]
        np.testing.assert_array_equal(normal[1],fast[1])
        cases.append(dict(seed=69237+i, frames=len(normal[0]), rgb_mae=float(np.abs(
            normal[0].astype(np.int16)-fast[0]).mean()), **stats))
    return cases


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.write_text(json.dumps(measure(),indent=2)+'\n')
