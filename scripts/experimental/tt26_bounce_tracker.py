"""Test a frozen bounce action model on six full-size receiving motions."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_so101 import AXES
from .tt19_view_full_game import digest_json
from .tt24_rally_trial import PublicBody, STEP_SIZE
from .tt25_pose_demonstrations import OBSERVE
from .tt25_recorded_flight import OUTPUT as TABLE
from .tt25_reference_receiver import ReferenceReceiver
from .tt26_bounce_effects import CHECKPOINT, CONTROL_INTERVAL, JointEffects, ORDINARY, OUTPUT


RECEIPT = OUTPUT / 'bounce-tracker.json.gz'
PILOT_RECEIPT = OUTPUT / 'bounce-tracker-pilot.json.gz'
DEMONSTRATIONS = TABLE / 'recorded-pose-imitation.json.gz'
REFERENCE = TABLE / 'recorded-secondary-reference.json'
DRIVES = {
    'compliant': ((.72, .68, .78, .66, .85), (1., 1., 1., 1., 1.)),
    'damped': ((.90, .98, .86, .97, .82), (1.16, 1.25, 1.12, 1.31, 1.23)),
}
AXIS_COUNT = 5
MAX_CORRECTION = .32
RIDGE = .022


def encoders(arena, dofs):
    return arena.joints[:AXIS_COUNT].copy(), arena.data.qvel[dofs].copy()


def replay(arena, shot, goal, dofs, *, mode='frozen', reference=None, effects=None, lag=0., anchored=False):
    """Play the original shot while following a complete reference motor path."""
    arena.launch(shot)
    issued = PublicBody(arena)
    samples, actions, applied_actions = [], [], []
    previous = np.zeros(arena.action_size)
    steps = round(arena.control_dt / CONTROL_INTERVAL)
    if not np.isclose(steps * CONTROL_INTERVAL, arena.control_dt):
        raise ValueError('the target control interval must be a multiple of the source interval')
    for tick in range(arena.max_steps):
        q, qvel = encoders(arena, dofs)
        target = issued.target[:AXIS_COUNT].copy()
        samples.append((q, qvel, target))
        command = np.zeros(arena.action_size)
        if reference is None:
            if tick >= OBSERVE:
                command[:AXIS_COUNT] = np.clip((goal - target) / STEP_SIZE[:AXIS_COUNT], -1., 1.)
        else:
            nominal = reference['actions'][min(tick, len(reference['actions']) - 1)][:AXIS_COUNT]
            command[:AXIS_COUNT] = nominal
            if tick < reference['first_contact']:
                ref_q, ref_qvel, ref_target = reference['samples'][tick]
                anchor = (ref_target - target) / STEP_SIZE[:AXIS_COUNT] if anchored else np.zeros(AXIS_COUNT)
                if mode == 'encoder':
                    e = ref_q - q + .4 * arena.control_dt * (ref_qvel - qvel)
                    correction = e / STEP_SIZE[:AXIS_COUNT]
                elif effects is not None:
                    # Transport only the local response. The reference and
                    # current states use the same source model, so their
                    # shared extrapolation errors cancel to first order.
                    aligned = np.clip(nominal + anchor, -1., 1.)
                    prediction = effects.advance(q, qvel, target, aligned, steps=steps)[0]
                    desired = effects.advance(ref_q, ref_qvel, ref_target, nominal, steps=steps)[0]
                    _, jacobian = effects.linearize(q, qvel, target, steps=steps)
                    normal = jacobian.T @ jacobian + RIDGE ** 2 * np.eye(AXIS_COUNT)
                    correction = np.linalg.solve(normal, jacobian.T @ (desired - prediction))
                else:
                    correction = np.zeros(AXIS_COUNT)
                    anchor = np.zeros(AXIS_COUNT)
                if anchored:
                    correction = anchor + .22 * correction
                command[:AXIS_COUNT] = np.clip(nominal + np.clip(correction, -MAX_CORRECTION, MAX_CORRECTION), -1., 1.)
        applied = (1. - lag) * command + lag * previous
        _, _, done, capped, _ = arena.step(applied, render=False)
        issued.issued(command)
        actions.append(command)
        applied_actions.append(applied)
        previous = command
        if done or capped or arena.legal_returns:
            break
    q, qvel = encoders(arena, dofs)
    samples.append((q, qvel, issued.target[:AXIS_COUNT].copy()))
    first_contact = next((event['step'] for event in arena.events
                          if event['kind'] == 'contact' and event['side'] == 1), None)
    return dict(samples=np.asarray(samples), actions=np.asarray(actions), applied_actions=np.asarray(applied_actions),
                first_contact=first_contact, contact=arena.contacts, legal=arena.legal_returns,
                incoming=arena.rival_returns, events=digest_json(arena.events), frames=len(actions))


def tracking_error(reference, replay):
    end = min(reference['first_contact'], len(replay['samples']))
    errors = (replay['samples'][:end, 0] - reference['samples'][:end, 0]) * 1000.
    return dict(precontact_ticks=end, rms_mrad=float(np.sqrt(np.mean(errors ** 2))),
                final_five_rms_mrad=float(np.sqrt(np.mean(errors[-5:] ** 2))),
                latest_axis_errors_mrad=errors[-1].tolist())


def packed(row):
    return dict(row, **{key: row[key].tolist() for key in ('samples', 'actions', 'applied_actions')})


def completed_summary(cases):
    result = {}
    for drive in DRIVES:
        result[drive] = {}
        for mode in cases[0]['drives'][drive]:
            rows = [case['drives'][drive][mode] for case in cases]
            result[drive][mode] = dict(
                average_rms_mrad=float(np.mean([row['rms_mrad'] for row in rows])),
                average_final_five_rms_mrad=float(np.mean([row['final_five_rms_mrad'] for row in rows])),
                contacts=sum(row['contact'] > 0 for row in rows), legal_returns=sum(row['legal'] for row in rows),
            )
    return result


def run(path, *, pilot=False):
    reference_hash = json.loads(REFERENCE.read_text())['sha256']
    if _hash(path) != reference_hash:
        raise ValueError('the recorded launch source differs from the TT25 reference')
    shots = {row['id']: row for row in json.loads(path.read_text())}
    archive = _read(DEMONSTRATIONS)
    selected = [(row['id'], next(trial for trial in row['trials'] if trial['legal']))
                for row in archive['teachers'] if any(trial['legal'] for trial in row['trials'])]
    arena = ReferenceReceiver()
    sites = arena._arm_ids[1][2][:AXIS_COUNT]
    dofs = np.asarray([arena.model.joint('learner_' + axis).dofadr[0] for axis in AXES[:AXIS_COUNT]])
    base_gains = arena.model.actuator_gainprm[sites].copy()
    base_bias = arena.model.actuator_biasprm[sites].copy()
    models = {'ordinary': JointEffects.load(ORDINARY), 'varied': JointEffects.load(CHECKPOINT)}
    lags = dict(compliant=.0, damped=.0) if pilot else dict(compliant=.45, damped=.65)
    cases = []
    try:
        for shot_id, teacher in selected:
            arena.model.actuator_gainprm[sites] = base_gains
            arena.model.actuator_biasprm[sites] = base_bias
            shot = shots[shot_id]
            reference = replay(arena, shot, np.asarray(teacher['goal']), dofs)
            if any(reference[key] != teacher[key] for key in ('legal', 'contact', 'incoming', 'events', 'frames')):
                raise ValueError(f'original physical return differs from TT25 demonstration {shot_id}')
            results = {}
            for condition, (stiffness, damping) in DRIVES.items():
                arena.model.actuator_gainprm[sites, 0] = base_gains[:, 0] * stiffness
                arena.model.actuator_biasprm[sites, 1] = base_bias[:, 1] * stiffness
                arena.model.actuator_biasprm[sites, 2] = base_bias[:, 2] * damping
                results[condition] = {}
                for mode in ('frozen', 'encoder', *models):
                    kwargs = dict(mode=mode, reference=reference, effects=models.get(mode),
                                  lag=lags[condition], anchored=not pilot)
                    branch = replay(arena, shot, None, dofs, **kwargs)
                    packed_branch = packed(branch)
                    if packed_branch != packed(replay(arena, shot, None, dofs, **kwargs)):
                        raise ValueError(f'changed servo replay diverged on {shot_id}/{condition}/{mode}')
                    results[condition][mode] = dict(packed_branch, **tracking_error(reference, branch))
            cases.append(dict(id=shot_id, reference=packed(reference), drives=results))
            print('Tracker', shot_id, {drive: {mode: (round(row['rms_mrad'], 1), row['legal'])
                                               for mode, row in outcomes.items()}
                                     for drive, outcomes in results.items()}, flush=True)
    finally:
        arena.close()
    summary = completed_summary(cases)
    _write(PILOT_RECEIPT if pilot else RECEIPT,
           dict(schema='tt26-bounce-tracker-1', source_sha256=reference_hash,
                         demonstration_sha256=_hash(DEMONSTRATIONS), shots=[sid for sid, _ in selected],
                         bounce_model_sha256={p.name: _hash(p) for p in (CHECKPOINT, ORDINARY)},
                         target_dimensions_m=[2.74, 1.525], drives=DRIVES, command_lags=lags, summary=summary,
                         servo_target_anchor=not pilot, feedback_gain=1. if pilot else .22,
                         model_observations='current joint angles, joint velocity, and own issued actuator targets',
                         controller_target='each original full-size demonstration joint path and command sequence',
                         target_refit_transitions=0, no_ball_state_to_tracking_controller=True, cases=cases))
    print('Held tracking', summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rallies', type=Path)
    parser.add_argument('--pilot', action='store_true', help='reproduce the unanchored trial without input lag')
    args = parser.parse_args()
    run(args.rallies, pilot=args.pilot)
