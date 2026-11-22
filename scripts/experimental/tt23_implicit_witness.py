"""Audit and replay the 62-return point and two fixed jog counterfactuals."""

import argparse

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_joint_practice import READS
from .tt22_motor_transfer import MotorPrimitive
from .tt23_anonymous_action import proposal
from .tt23_anonymous_target import target_context
from .tt23_implicit_online import CONFIRM_SEEDS, JogLearner, NAMES, live_schedule, paired_draws
from .tt23_implicit_online import SOURCE as NATIVE_SOURCE
from .tt23_jog_station import ExplicitJogStation
from .tt23_opening_motor import MACRO, OUTPUT
from .tt23_sequence import verify


SOURCE = OUTPUT / 'implicit-confirm-16.json.gz'
RECEIPT = OUTPUT / 'implicit-witness.json.gz'
PRIOR = OUTPUT / 'implicit-witness-prior.json'
VIDEO = OUTPUT / 'implicit-witness-62.mp4'
SEED = 1309653644
JOGS = {'center': 0., 'left': -.09}


def witness():
    record = _read(SOURCE)
    if record['verified'] != sum(len(case['results']) for case in record['cases']):
        raise ValueError('expected a completed and verified online source run')
    case = record['cases'][3]
    if case['seed'] != SEED or case['choices']['learned'] != NAMES.index('right'):
        raise ValueError('the archived fourth point did not choose the expected witness jog')
    row = case['results']['learned']
    if (row['longest'], len(row['actions']), row['outcome']['legal_returns'], row['outcome']['rival_returns']) != (
            62, 1140, 31, 31):
        raise ValueError('the selected point no longer matches the verified witness')
    return record, case


def collect():
    _, case = witness()
    right = case['results']['learned']
    metadata = dict(schema='tt23-implicit-witness-1', source=SOURCE.name, source_sha256=_hash(SOURCE),
                    seed=SEED, lane=case['lane'], opening=right['opening'],
                    experiment='retrospective fixed jogs without learner updates')
    data = (_read(RECEIPT) if RECEIPT.exists() else
            dict(metadata=metadata, results={'right': dict(right, jog=.09, execution='archived')}, verified=0))
    if data['metadata'] != metadata:
        raise ValueError('existing counterfactual receipt refers to a different source point')
    station = ExplicitJogStation(lambda action, _sight, _ready: action)
    try:
        for name, jog in JOGS.items():
            if name in data['results']:
                continue
            row = station.play(SEED, case['lane'], jog)
            if (row['opening'] != right['opening'] or row['actions'][:READS] != right['actions'][:READS]
                    or row['rival_actions'][:READS] != right['rival_actions'][:READS]):
                raise ValueError('the fixed jogs changed the shared native optical and physical prefix')
            data['results'][name] = dict(row, jog=jog, execution='live')
            _write(RECEIPT, data)
            print(name, row['longest'], len(row['actions']), flush=True)
    finally:
        station.close()
    data['verified'] = verify([data])
    _write(RECEIPT, data)
    print('verified', data['verified'], flush=True)


def audit_prior():
    record, case = witness()
    source_hash = _hash(NATIVE_SOURCE)
    if record['source'] != source_hash:
        raise ValueError('the archived run used another native Pong checkpoint')
    generator = CONFIRM_SEEDS[15]
    visits = [(row['seed'], row['lane']) for row in record['cases']]
    if visits != live_schedule(generator)[:len(visits)]:
        raise ValueError('the observed points do not match the confirmation schedule')
    model = _read(NATIVE_SOURCE)['pooled']
    ready = MotorPrimitive(_read(MACRO)['parameters'], None).ready_world

    def optical_context(point):
        sight = point['results']['learned']['opening']
        value = None if sight is None else target_context(sight, ready)
        archived = None if value is None else value.tolist()
        if archived != point['context']:
            raise ValueError('the archived optical prefix did not regenerate its selector context')
        return value

    learners = {mode: JogLearner(model, mode) for mode in ('learned', 'flat')}
    draws = paired_draws(generator + 1, n=1)
    history = []
    for i, earlier in enumerate(record['cases'][:3]):
        x = optical_context(earlier)
        choice = np.asarray([earlier['choices']['learned']])
        if not np.array_equal(choice, learners['learned'].choose(x, draws[i])):
            raise ValueError('the native prior did not regenerate an archived history choice')
        ticks = len(earlier['results']['learned']['actions'])
        for learner in learners.values():
            learner.update(x, choice, [ticks])
        history.append(dict(seed=earlier['seed'], context=earlier['context'], action=NAMES[choice[0]], ticks=ticks))
    x = optical_context(case)
    if x is None:
        raise ValueError('the archived witness requires an observed optical mover')
    native_vote = proposal(model['learned'], x)[2:5]
    chosen = {mode: NAMES[learner.choose(x, draws[3])[0]] for mode, learner in learners.items()}
    if chosen['learned'] != NAMES[case['choices']['learned']]:
        raise ValueError('the native prior did not regenerate the observed 62-return choice')
    result = dict(schema='tt23-implicit-witness-prior-1', stream_sha256=_hash(SOURCE), native_sha256=source_hash,
                  seed=SEED, point=4, prior_audit='retrospective comparison on the same realized three-point history',
                  context=case['context'], history=history, order=list(NAMES), draw=draws[3, 0].tolist(),
                  frozen_votes=native_vote.tolist(), frozen_choice=NAMES[int(np.argmax(native_vote))],
                  prior_means={mode: learner.prior(x).tolist() for mode, learner in learners.items()},
                  choices=chosen, witness_outcome_excluded=True)
    _write(PRIOR, result)
    print(result, flush=True)


def render():
    from .tt23_sequence_coach_movie import render_tape

    _, case = witness()
    render_tape(case['results']['learned'], SOURCE, VIDEO, branch='learned',
                subtitle='Right jog selected online · Learned far coach',
                terminal_caption='Point ended · Near arm landed on its own half', maxrate='380k')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('collect', 'prior', 'video', 'verify'))
    args = parser.parse_args()
    if args.phase == 'collect':
        collect()
    elif args.phase == 'prior':
        audit_prior()
    elif args.phase == 'video':
        render()
    else:
        print('verified', verify([_read(RECEIPT)]))


if __name__ == '__main__':
    main()
