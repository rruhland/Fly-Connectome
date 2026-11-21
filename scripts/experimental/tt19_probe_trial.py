"""Fit on short RGB movements, then measure one-step-ahead optical displacement."""
import argparse
from pathlib import Path
import numpy as np
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt18_online import restore_body
from .tt19_online_body import InverseModel, blade_pose, kernel_for
from .tt19_probe import command_tape
from .tt19_view import ViewTransport
from .tt19_view_trial import SO101ViewArena


DONOR = ROOT/'artifacts/tt17/motor-gantry-coupling-travel-player.json.gz'


def read_tape(arena, transport, tape, seed):
    tracker = transport.tracker()
    frames = arena.reset(seed=seed, render=True, server=-1)
    tracker.update(frames)
    poses = [blade_pose(tracker)]
    for action in tape:
        frames, _, ended, truncated, _ = arena.step(action)
        tracker.update(frames)
        poses.append(blade_pose(tracker))
        if ended or truncated:
            break
    return dict(actions=tape[:len(poses)-1].tolist(), poses=[None if pose is None else
                [None if not np.isfinite(x) else float(x) for x in pose] for pose in poses], rgb=tracker.digest.hexdigest())


def feed(model, sample, learn):
    model.reset()
    model.learning = learn
    errors = []
    displacement = []
    for i, (action, raw) in enumerate(zip([None, *sample['actions']], sample['poses'], strict=True)):
        model.advance(action)
        if raw is None:
            continue
        pose = np.asarray(raw, dtype=float)
        if model.last is not None:
            predicted = model.live_pose()
            errors.append(np.linalg.norm(predicted[:3] - pose[:3]))
            displacement.append(np.linalg.norm(model.last[2][:3] - pose[:3]))
        model.measure(pose)
    return np.asarray(errors), np.asarray(displacement)


def audit(folder, seed, length=36):
    donor = _read(DONOR)
    motor = donor['series']['model']
    kernel = kernel_for(restore_body(donor['models'][motor]))
    camera_mode = 'wrist+fixed'
    spec = dict(seed=seed, frameskip=8, max_steps=160, venue='hall', width=320, height=240, detail='preview')
    arena = SO101ViewArena(view_mode=camera_mode, **spec)
    try:
        transport = ViewTransport(camera_mode)
        training = read_tape(arena, transport, command_tape(6, length, kernel=kernel), seed)
        rng = np.random.default_rng([seed, 11485])
        targets = []
        for trial in range(6):
            commands = np.repeat(rng.uniform(-.80,.80,(7,6)), 3, axis=0)
            targets.append(read_tape(arena, transport, commands, seed + 10 + trial))
    finally:
        arena.close()
    summaries = []
    for count in sorted(set([0, 12, min(24,length), length])):
        model = InverseModel(6,kernel,'secant',seed)
        prefix = dict(actions=training['actions'][:count], poses=training['poses'][:count+1])
        feed(model,prefix,True)
        errors, shifts = [], []
        for target in targets:
            err, delta = feed(model, target, False)
            errors.extend(err)
            shifts.extend(delta)
        jacobian = model.jacobian(np.zeros(6))
        summaries.append(dict(frames=len(prefix['actions']), samples=model.samples, optical_pairs=len(errors),
                              jacobian_singular_values=np.linalg.svd(jacobian[:3], compute_uv=False).tolist(),
                              rms_error_m=float(np.mean(np.square(errors))**.5) if errors else None,
                              mean_error_m=float(np.mean(errors)) if errors else None,
                              mean_hold_last_m=float(np.mean(shifts)) if shifts else None))
    result=dict(schema='tt19-optical-probe-1',seed=seed, donor_sha256=_hash(DONOR),
                arena=spec, camera_mode=camera_mode, kernel=kernel.tolist(), training=training,
                targets=targets, summaries=summaries)
    _write(folder/f'probe-{seed}.json.gz',result)
    print(summaries, flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--seed',type=int,default=190121)
    p.add_argument('--length',type=int,default=36)
    a=p.parse_args()
    a.folder.mkdir(exist_ok=True,parents=True)
    audit(a.folder,a.seed,a.length)
