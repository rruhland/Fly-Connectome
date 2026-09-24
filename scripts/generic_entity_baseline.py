"""Opt-in generic event-occupancy and visual-continuity baseline."""

from collections import deque
import json
from pathlib import Path

import torch

from generic_motion_probe import SHAPES, object_mask
from generic_motion_stress import path_events


OUT = Path('docs/experiments/2026-09-23-generic-grouping-results.json')


class EventOccupancy:
    def __init__(self):
        self.contrast = torch.zeros((32, 64))

    def step(self, events):
        self.contrast.add_(events[1]-events[0]).clamp_(-1, 1)
        return self.contrast.abs() > .5


def components(mask):
    active = mask.tolist()
    seen = [[False]*64 for _ in range(32)]
    groups = []
    for y in range(32):
        for x in range(64):
            if not active[y][x] or seen[y][x]:
                continue
            seen[y][x] = True
            queue = deque([(y, x)])
            sy = sx = size = 0
            while queue:
                cy, cx = queue.popleft()
                sy += cy
                sx += cx
                size += 1
                for ny, nx in ((cy-1, cx), (cy+1, cx),
                               (cy, cx-1), (cy, cx+1)):
                    if (0 <= ny < 32 and 0 <= nx < 64 and
                            active[ny][nx] and not seen[ny][nx]):
                        seen[ny][nx] = True
                        queue.append((ny, nx))
            groups.append((sy/size, sx/size, size))
    return sorted(groups, key=lambda group: group[1])


class VisualContinuity:
    """Transparent unlabeled grouping ceiling; not a biological circuit."""

    def __init__(self):
        self.tracks = []

    def step(self, groups):
        if not self.tracks:
            self.tracks = [dict(center=(y, x), velocity=(0., 0.),
                                observed=(y, x), missed=0)
                           for y, x, _ in groups]
            return self.tracks
        predicted = [(track['center'][0]+track['velocity'][0],
                      track['center'][1]+track['velocity'][1])
                     for track in self.tracks]
        if len(groups) < len(self.tracks):
            for track, center in zip(self.tracks, predicted):
                track['center'] = center
                track['missed'] += 1
            return self.tracks
        pairs = sorted(((sum((predicted[i][axis]-group[axis])**2
                             for axis in (0, 1)), i, j)
                        for i in range(len(self.tracks))
                        for j, group in enumerate(groups)))
        assigned_tracks = set()
        assigned_groups = set()
        for distance2, i, j in pairs:
            if i in assigned_tracks or j in assigned_groups or distance2 > 36:
                continue
            track = self.tracks[i]
            center = groups[j][:2]
            elapsed = track['missed']+1
            track['velocity'] = tuple((center[axis]-track['observed'][axis])
                                      / elapsed for axis in (0, 1))
            track['center'] = center
            track['observed'] = center
            track['missed'] = 0
            assigned_tracks.add(i)
            assigned_groups.add(j)
        for i, track in enumerate(self.tracks):
            if i not in assigned_tracks:
                track['center'] = predicted[i]
                track['missed'] += 1
        self.tracks.extend(dict(center=group[:2], velocity=(0., 0.),
                                observed=group[:2], missed=0)
                           for j, group in enumerate(groups)
                           if j not in assigned_groups)
        return self.tracks


def evaluate(sequence, truth_masks):
    memory = EventOccupancy()
    continuity = VisualContinuity()
    rows = []
    for frame, (events, masks) in enumerate(zip(sequence, truth_masks)):
        observed_groups = components(memory.step(events))
        tracks = continuity.step(observed_groups)
        if 2 <= frame < 15:
            event_groups = components(events.sum(0) > 0)
            expected = []
            for mask in masks:
                points = mask.nonzero()
                expected.append(tuple(points.float().mean(0).tolist()))
            rows.append(dict(frame=frame, expected_count=len(expected),
                             occupancy_count=len(observed_groups),
                             event_count=len(event_groups),
                             track_count=len(tracks),
                             centers=[list(track['center']) for track in tracks],
                             expected_centers=expected))
    return rows


def masks_for(objects):
    masks = []
    for frame in range(18):
        masks.append([object_mask(shape, path[frame]) for shape, path in objects]
                     if 2 <= frame < 15 else [])
    return masks


def main():
    cases = {}
    for shape in SHAPES:
        path = [None]*18
        for frame in range(2, 15):
            path[frame] = (16, 20+frame-2)
        for background in (True, False):
            key = f'{shape}:{"dark" if background else "bright"}'
            cases[key] = evaluate(path_events([(shape, path)],
                                              background=background),
                                  masks_for([(shape, path)]))
    static_path = [None]*18
    for frame in range(2, 15):
        static_path[frame] = (16, 32)
    static_events = path_events([('square', static_path)])
    static_masks = masks_for([('square', static_path)])
    cases['static'] = evaluate(static_events, static_masks)
    rng = torch.Generator().manual_seed(0)
    noisy = []
    for events in static_events:
        retained = events * (torch.rand(events.shape, generator=rng) >= .1)
        false = (torch.rand(events.shape, generator=rng) < .001).float()
        noisy.append(torch.maximum(retained, false))
    cases['static_noisy'] = evaluate(noisy, static_masks)
    left = [None]*18
    right = [None]*18
    for frame in range(2, 15):
        left[frame] = (16, 26+2*(frame-8))
        right[frame] = (16, 38-2*(frame-8))
    crossing_objects = [('square', left), ('square', right)]
    cases['crossing'] = evaluate(path_events(crossing_objects),
                                 masks_for(crossing_objects))

    summary = {}
    for name, rows in cases.items():
        errors = []
        switches = 0
        for row in rows:
            if len(row['centers']) >= len(row['expected_centers']):
                for index, expected in enumerate(row['expected_centers']):
                    actual = row['centers'][index]
                    errors.append(sum((actual[axis]-expected[axis])**2
                                      for axis in (0, 1))**.5)
                if name == 'crossing' and row['frame'] >= 12:
                    first, second = row['centers'][:2]
                    expected_first, expected_second = row['expected_centers']
                    own = sum((first[axis]-expected_first[axis])**2
                              for axis in (0, 1))
                    other = sum((first[axis]-expected_second[axis])**2
                                for axis in (0, 1))
                    switches += int(other < own)
        summary[name] = dict(frames=len(rows),
            occupancy_count_correct=sum(row['occupancy_count'] ==
                                        row['expected_count'] for row in rows),
            event_count_correct=sum(row['event_count'] ==
                                    row['expected_count'] for row in rows),
            track_count_correct=sum(row['track_count'] ==
                                    row['expected_count'] for row in rows),
            mean_center_error=sum(errors)/len(errors) if errors else None,
            identity_switch_frames=switches)
    OUT.write_text(json.dumps(dict(summary=summary, frames=cases), indent=2,
                              allow_nan=False)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
