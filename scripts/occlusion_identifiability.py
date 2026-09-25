"""Causal matched-prefix test for variable-length visual disappearance."""

import json
import time
from pathlib import Path

import torch

from correlation_latent_robustness import scene_sequence
from diverse_visual_experience import TEST_SHAPES
from history_gated_code import DIRECTIONS


OUT = Path('docs/experiments/2026-09-25-occlusion-identifiability-results.json')
WINDOWS = ((7, 8), (7, 8, 9), (7, 8, 9, 10))


def matched_prefix_suite():
    rows = []
    for shape in TEST_SHAPES:
        for direction, (dy, dx) in DIRECTIONS.items():
            for speed in (1, 2):
                for background in (True, False):
                    for center in ((13, 27), (16, 32), (19, 37)):
                        sequences = []
                        for hidden in WINDOWS:
                            obj = dict(shape=shape, center=center,
                                       before=(dy*speed, dx*speed),
                                       after=(dy*speed, dx*speed),
                                       hidden=hidden)
                            sequences.append(scene_sequence(
                                [obj], background=background))
                        rows.append((shape, direction, speed, background,
                                     center, sequences))
    return rows


def compare_pair(rows, left_index, right_index, through):
    result = dict(cases=0, identical_prefixes=0,
                  different_targets=0, target_disagreement_pixels=0,
                  visible_target_events=0, hidden_target_events=0)
    for _, _, _, _, _, sequences in rows:
        left, right = sequences[left_index], sequences[right_index]
        result['cases'] += 1
        result['identical_prefixes'] += int(all(
            torch.equal(left[t], right[t]) for t in range(through+1)))
        target_left, target_right = left[through+1], right[through+1]
        different = not torch.equal(target_left, target_right)
        result['different_targets'] += int(different)
        result['target_disagreement_pixels'] += int(
            (target_left != target_right).sum())
        result['visible_target_events'] += int(target_left.sum())
        result['hidden_target_events'] += int(target_right.sum())
    return result


def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    rows = matched_prefix_suite()
    result = dict(
        configurations=len(rows),
        two_vs_three=compare_pair(rows, 0, 1, through=8),
        three_vs_four=compare_pair(rows, 1, 2, through=9),
        elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
