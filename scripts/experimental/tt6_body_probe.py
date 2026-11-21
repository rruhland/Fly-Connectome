"""Collect action and stereo-image pairs for an unfamiliar receiver interface."""

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from .tt4_arena import mujoco
from .tt4_transfer_trial import ROOT, _hash


def probe_commands(seed, horizon, action_size):
    """Excite each public channel and vary the combinations without moving endlessly."""
    rng = np.random.default_rng(seed)
    position = np.zeros(action_size)
    commands = []
    for block in range((horizon + 3) // 4):
        if block % 5 == 4:
            goal = np.zeros(action_size)
        elif block % 3 == 0:
            goal = np.zeros(action_size)
            goal[rng.integers(action_size)] = rng.choice([-3.7, 3.7])
        else:
            goal = rng.uniform(-2.9, 2.9, action_size)
        command = np.clip((goal - position) / 4., -1., 1.)
        commands.extend([command] * 4)
        position += 4. * command
    return np.array(commands[:horizon])


def _collect(wiring, schedule, horizon):
    histories, receipt = [], []
    for index, seed in enumerate(schedule['calibration']):
        frames = wiring.reset(seed=seed)
        action_size = wiring.arena.action_size
        history = [(np.zeros(action_size), frames)]
        for action in probe_commands(schedule['player_seed'] + index, horizon, action_size):
            frames, _, terminated, truncated, _ = wiring.step(action)
            history.append((action, frames))
            if terminated or truncated:
                break
        histories.append(history)
        receipt.append(dict(seed=seed, steps=len(history) - 1, frames=len(history),
                            actions=[action.tolist() for action, _ in history[1:]]))
    return histories, receipt


def probe_histories(wiring, schedule, geometry, cache=None):
    horizon = geometry['probe_horizon']
    if cache is None:
        return _collect(wiring, schedule, horizon)
    names = ('tt4_arena.py', 'tt4_arena.xml', 'tt4_hall.py', 'tt4_camera.py', 'tt6_body_arena.py',
             'tt6_body_gantry.xml', 'tt6_body_offset.xml', 'tt6_body_probe.py', 'tt6_render.py')
    if wiring.arena.body == 'rotary':
        names += ('tt6_body_rotary.xml',)
    code = [_hash(ROOT / 'scripts' / 'experimental' / name) for name in names]
    manifest = dict(geometry=geometry, code=code, mujoco=mujoco.__version__,
                    schedule={key: value for key, value in schedule.items() if key != 'tests'})
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    path = cache / f'{key}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            frames, actions, offsets = data['frames'], data['actions'], data['offsets']
            rows = json.loads(data['receipt'].item())
        return [list(zip(actions[a:b], frames[a:b], strict=True))
                for a, b in zip(offsets[:-1], offsets[1:], strict=True)], rows
    histories, rows = _collect(wiring, schedule, horizon)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, frames=np.stack([f for hist in histories for _, f in hist]),
                            actions=np.stack([a for hist in histories for a, _ in hist]),
                            offsets=np.cumsum([0] + [len(hist) for hist in histories]), receipt=json.dumps(rows))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return histories, rows
