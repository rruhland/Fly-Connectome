"""Bounded Linux-only reconstruction proof. No policy or vision training.

Pickles are trusted, locally produced runtime checkpoints, not portable imports.
The active snapshot currently embeds the visual object graph; deduplicating its
immutable banks and adding delta replay are later engineering gates.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fly_connectome.control import FrozenLinearPolicy, Journal
from fly_connectome.pong import Pong
from fly_connectome.vision import VisualStateEncoder, load_default


def fingerprint(value):
    """Canonical graph digest including repeated-object aliases and RNG state."""
    seen = {}
    retained = []  # Keep temporary RNG state tensors alive while assigning alias IDs.
    def visit(obj):
        if obj is None or isinstance(obj, (str, bool, int, float)):
            return obj
        identity = id(obj)
        if identity in seen:
            return ['ref', seen[identity]]
        seen[identity] = len(seen)
        retained.append(obj)
        if isinstance(obj, torch.Tensor):
            return ['tensor', str(obj.dtype), list(obj.shape), obj.detach().contiguous().numpy().tobytes().hex()]
        if isinstance(obj, np.ndarray):
            return ['array', str(obj.dtype), list(obj.shape), obj.tobytes().hex()]
        if isinstance(obj, bytes):
            return ['bytes', obj.hex()]
        if isinstance(obj, torch.Generator):
            return ['torch-generator', visit(obj.get_state())]
        if isinstance(obj, dict):
            return ['dict', [[visit(k), visit(v)] for k, v in sorted(obj.items(), key=lambda kv: repr(kv[0]))]]
        if isinstance(obj, (list, tuple, deque)):
            return [type(obj).__name__, [visit(x) for x in obj]]
        if hasattr(obj, '__dict__'):
            fields = vars(obj)
            if '_variance_version' in fields:
                fields = dict(fields)
                # Torch version counters are process-local cache invalidation hints.
                fields['_variance_version'] = fields['_variance_version'] == fields['_variance_keys']._version
            return [type(obj).__module__+'.'+type(obj).__qualname__, visit(fields)]
        raise TypeError(type(obj))
    return hashlib.sha256(json.dumps(visit(value), sort_keys=True).encode()).hexdigest()


def fresh():
    random.seed(1101)
    np.random.seed(1101)
    torch.manual_seed(1101)
    encoder = VisualStateEncoder(64, 64, .02)
    weights = np.random.default_rng(1101).normal(0, .001, (3, encoder.dimension))
    game = Pong([1101])
    return dict(game=game, body_alias=game.body, vision=load_default(), encoder=encoder,
                policy=FrozenLinearPolicy(weights), previous=torch.zeros(64, 64, dtype=torch.bool),
                transition=0, physics_ticks=0, pending_action=None, points=0, misses=0,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state())


def run(path, stop, resume):
    torch.set_num_threads(1)
    started = time.perf_counter()
    with Journal(path) as journal:
        saved = journal.load()
        if resume:
            if saved is None:
                raise ValueError('cannot resume empty journal')
            state = pickle.loads(saved[1])
            if saved[0] != state['transition']-1:
                raise ValueError('continuation sequence mismatch')
            random.setstate(state['python_rng'])
            np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'])
        else:
            if saved is not None:
                raise ValueError('fresh proof requires empty journal')
            state = fresh()
        if not state['transition'] < stop <= 100:
            raise ValueError('proof must advance to at most 100 transitions')
        model = Path(__file__).resolve().parents[1]/'src/fly_connectome/vision/data/default.pt'
        model_hash = journal.put_artifact(model.read_bytes())
        initial_weights = state['policy'].weights
        learned = lambda: (state['vision'].observer.weights, state['vision'].observer.bias,
                           tuple(getattr(state['vision'].state.memory, name) for name in ('keys', 'values', 'metric')),
                           state['vision'].dynamics)
        learned_before = fingerprint(learned())
        while state['transition'] < stop:
            if time.perf_counter()-started > 120:
                raise TimeoutError('worker exceeded bounded proof time')
            game, vision, encoder = state['game'], state['vision'], state['encoder']
            assert state['body_alias'] is game.body
            frame = game.render(64, 64)[0]
            events = torch.stack((state['previous'] & ~frame, ~state['previous'] & frame)).float()
            state['previous'].copy_(frame)
            image = frame.float()
            encoded = encoder.encode(vision.step(events, image, learn=False), image)
            action, q = state['policy'].choose(encoded['indices'].numpy(), encoded['values'].numpy())
            state['pending_action'] = action
            index = state['transition']
            ticks = (index+1)*12//5-index*12//5
            reward, terminated = 0., False
            for _ in range(ticks):
                outcome = game.step(torch.tensor([(-1., 0., 1.)[action]]))
                score = float(outcome.scores[0])
                reward += score
                state['points'] += int(score > 0)
                state['misses'] += int(score < 0)
                state['physics_ticks'] += 1
                if score:
                    terminated = True
                    # Pong auto-reset: clear sensor and all transient visual histories.
                    vision.reset_state()
                    encoder.reset_state()
                    state['previous'].zero_()
                    break
            state['pending_action'] = None
            state['transition'] += 1
            # Exercise global RNG continuation even though the frozen policy is deterministic.
            rng_probe = [random.random(), float(np.random.random()), float(torch.rand(()))]
            state.update(python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                         torch_rng=torch.get_rng_state())
            sample = dict(transition=index, action=action, q=q.tolist(), reward=reward,
                          terminated=terminated, physics_ticks=state['physics_ticks'], rng_probe=rng_probe,
                          model_hash=model_hash)
            journal.commit(index, sample, pickle.dumps(state, protocol=5))
        assert np.array_equal(initial_weights, state['policy'].weights)
        assert fingerprint(learned()) == learned_before, 'frozen visual parameters changed'
        result = dict(transitions=stop, state_hash=fingerprint(state), model_hash=model_hash,
                      elapsed_seconds=time.perf_counter()-started, learned_updates=0)
    print(json.dumps(result))


def proof(directory):
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    def worker(name, stop, resume=False):
        command = [sys.executable, __file__, '--db', str(directory/name), '--stop', str(stop)]
        if resume:
            command.append('--resume')
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=max(.01, 180-(time.perf_counter()-started)))
        if result.returncode:
            raise RuntimeError(result.stderr)
        return json.loads(result.stdout)
    full = worker('full.db', 48)
    part = worker('split.db', 24)
    resumed = worker('split.db', 48, True)
    assert full['state_hash'] == resumed['state_hash'], 'full continuation graph mismatch'
    with Journal(directory/'full.db') as a, Journal(directory/'split.db') as b:
        assert a.samples() == b.samples(), 'audit trajectory mismatch'
    report = dict(status='passed', proof='48 uninterrupted vs 24 + 24 fresh-process resumed',
                  total_executed_transitions=96, full=full, first_segment=part, resumed=resumed,
                  wall_seconds=time.perf_counter()-started, development_observations=0,
                  limitations=['frozen evaluation only', 'trusted same-platform pickle',
                               'immutable vision banks still embedded in active snapshot',
                               'no delta replay or learning-state proof yet'])
    (directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--db', type=Path)
    parser.add_argument('--stop', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.proof:
        proof(args.proof)
    elif args.db and args.stop:
        run(args.db, args.stop, args.resume)
    else:
        parser.error('--proof or --db and --stop required')
