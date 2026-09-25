"""Audit local credit coverage at intrinsically ambiguous gap exits."""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from correlation_input_latent import correlation_sequence
from diverse_visual_experience import unseen_shape_sequences
from gap_timing_transfer import MODEL_OUT
from local_visibility_likelihood import LocalVisibilityHead, nearby_events
from occlusion_identifiability import matched_prefix_suite
from run_local_hidden_transition import train as train_transition
from run_local_visibility_likelihood import clean_training_cases


OUT = Path('docs/experiments/2026-09-25-visibility-feature-audit-results.json')


def empty():
    return dict(frames=0, active_frames=0, sites=0, local_positive=0,
                target_pixels=0, covered_pixels=0,
                age_sum=0., mass_sum=0., evidence_sum=0.)


@torch.no_grad()
def inspect(transition, events, decision_t, row):
    transition.reset_state()
    head = LocalVisibilityHead()
    for t, code in enumerate(correlation_sequence(events[:decision_t+1])):
        transition.step(code)
        head.step(transition, events[t])
    row['frames'] += 1
    target = events[decision_t+1]
    pixels = target.sum(0) > 0
    row['target_pixels'] += int(pixels.sum())
    sites = head.previous_sites
    if sites is None:
        return
    row['active_frames'] += 1
    y, x = sites.T
    row['sites'] += len(sites)
    row['local_positive'] += int(nearby_events(target)[y, x].sum())
    features = head.previous_features
    row['age_sum'] += float(features[:, 2].sum())
    row['mass_sum'] += float(features[:, 1].sum())
    row['evidence_sum'] += float(features[:, 3].sum())
    footprint = torch.zeros((32, 64))
    footprint[y, x] = 1
    covered = F.max_pool2d(footprint[None, None], 5,
                           stride=1, padding=2)[0, 0] > 0
    row['covered_pixels'] += int((pixels & covered).sum())


def finish(row):
    return {**row,
            'local_positive_fraction': row['local_positive']/max(row['sites'], 1),
            'pixel_coverage': row['covered_pixels']/max(row['target_pixels'], 1),
            'mean_age_feature': row['age_sum']/max(row['sites'], 1),
            'mean_mass_feature': row['mass_sum']/max(row['sites'], 1),
            'mean_recent_evidence': row['evidence_sum']/max(row['sites'], 1)}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    clean = clean_training_cases()
    transition = train_transition(
        saved['models']['learned_split'].code.motion,
        [(kind, '', events) for kind, events in clean])['learned']
    rows = {name: empty() for name in
            ('clean_visible', 'two_vs_three', 'three_vs_four')}
    for *_, events in unseen_shape_sequences():
        for t in (5, 6, 10):
            inspect(transition, events, t, rows['clean_visible'])
    for *_, sequences in matched_prefix_suite():
        inspect(transition, sequences[0], 8, rows['two_vs_three'])
        inspect(transition, sequences[1], 9, rows['three_vs_four'])
    result = dict(groups={name: finish(row) for name, row in rows.items()},
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
