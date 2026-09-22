"""Capped development Pong M1A training and frozen same-seed comparison."""
import argparse
import json
from pathlib import Path

import torch

from full_context_pong import OpenLoopContextRun
from fly_connectome.data import checksum


OUT = Path('runs/full-context-m1a-v1')
SOURCE = Path('checkpoints/event-v1-combined-rate-initial.pt')


def advance(run, checkpoint, source_sha, total):
    if run.frame > total:
        raise ValueError('checkpoint passed requested frame count')
    while run.frame < total:
        run.run(min(100, total-run.frame))
        run.save(checkpoint, source_sha)
        print('saved', checkpoint, 'frame', run.frame, 'fps',
              round(run.summary()['frames_per_second'], 2), flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=500)
    args = parser.parse_args()
    if args.frames < 2 or args.frames % 100:
        raise ValueError('pilot frame count must be a positive multiple of 100')
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    OUT.mkdir(parents=True, exist_ok=True)
    train_path = OUT/f'pilot-train-{args.frames}.pt'
    train = (OpenLoopContextRun.load(train_path, payload, source_sha) if train_path.exists()
             else OpenLoopContextRun(payload, 1101, 1.))
    advance(train, train_path, source_sha, args.frames)
    if train.seed != 1101 or train.eta != 1.:
        raise ValueError('training checkpoint protocol mismatch')
    evaluations = {}
    runs = {}
    for name, components in (('initial', None), ('trained', train.net.components.clone())):
        path = OUT/f'pilot-{name}-{args.frames}.pt'
        run = (OpenLoopContextRun.load(path, payload, source_sha) if path.exists()
               else OpenLoopContextRun(payload, 1102, 0., components=components))
        if run.seed != 1102 or run.eta != 0.:
            raise ValueError('evaluation checkpoint protocol mismatch')
        advance(run, path, source_sha, args.frames)
        evaluations[name] = dict(checkpoint_sha256=checksum(path), **run.summary())
        runs[name] = run
    if runs['initial'].events != runs['trained'].events:
        raise AssertionError('initial and trained runs saw different camera targets')
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    result = dict(source=str(SOURCE), source_sha256=source_sha,
        train_seed=1101, evaluation_seed=1102, frames=args.frames,
        horizon_ticks=8, selected_eta=1.,
        training=dict(checkpoint_sha256=checksum(train_path), **train.summary()),
        evaluations=evaluations, camera_event_sequences_equal=True,
        final_seeds_untouched=[1201, 1202, 1203, 1204],
        script_sha256=checksum(Path(__file__)))
    output = OUT/f'pilot-{args.frames}.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
