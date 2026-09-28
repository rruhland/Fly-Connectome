"""Hash complete fixed-drive Pong perception trajectories across source revisions."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmark_pong_vision import dense_events, load_runtime, physics_ticks_for_sample


def digest(value):
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            result.update(b'tensor')
            result.update(str(item.dtype).encode())
            result.update(str(tuple(item.shape)).encode())
            result.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            result.update(b'dict')
            result.update(str(len(item)).encode())
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(type(item).__name__.encode())
            result.update(str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            result.update(type(item).__name__.encode())
            result.update(repr(item).encode())

    visit(value)
    return result.hexdigest()


@torch.no_grad()
def stream(game_type, vision, encoder_type, samples, *, learn):
    game = game_type([1101])
    encoder = encoder_type(64, 64, .02)
    previous = torch.zeros(64, 64, dtype=torch.bool)
    drive = torch.zeros(1)
    hashes = []
    for sample in range(samples):
        frame = game.render(64, 64)[0]
        events = dense_events(previous, frame)
        previous.copy_(frame)
        image = frame.float()
        state = vision.step(events, image, learn=learn)
        encoded = encoder.encode(state, image)
        outcomes = []
        for _ in range(physics_ticks_for_sample(sample)):
            outcome = game.step(drive)
            outcomes.append((outcome.scores, outcome.hits, outcome.rally_steps))
        hashes.append(digest((frame, events, state, encoded, outcomes,
                              game.rng, game.ball, game.ball_velocity, game.opponent,
                              game.opponent_velocity, game.body.position,
                              game.body.activation, game.body.velocity, game.rally_steps)))
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--samples', type=int, default=240)
    args = parser.parse_args()
    torch.set_num_threads(1)
    Pong, ProbabilisticVisualState, VisualStateEncoder, load_default, source = load_runtime(args.source_root)
    online = load_default()
    online_hashes = stream(Pong, online, VisualStateEncoder, args.samples, learn=True)
    checkpoint = args.output.with_suffix('.online.pt')
    online.save(checkpoint)
    saved = digest(torch.load(checkpoint, weights_only=True))
    evaluation = ProbabilisticVisualState.load(checkpoint)
    eval_hashes = stream(Pong, evaluation, VisualStateEncoder, args.samples, learn=False)
    eval_checkpoint = args.output.with_suffix('.eval.pt')
    evaluation.save(eval_checkpoint)
    eval_saved = digest(torch.load(eval_checkpoint, weights_only=True))
    result = dict(online=dict(samples=online_hashes, checkpoint=saved),
                  evaluation=dict(samples=eval_hashes, checkpoint=eval_saved))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(source_package=source, trajectory_digest=digest(result),
                          checkpoint_digest=saved)), flush=True)


if __name__ == '__main__':
    main()
