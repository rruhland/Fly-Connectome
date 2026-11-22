"""Learn a signed Pong motion option from complete cross-court physical points."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import betaincinv

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_flight import FlightLaw
from .tt22_after_service import IntentPlayer, verify_rows
from .tt22_far_cache import FarCache
from .tt22_match import CHECKPOINT, HANDOVER
from .tt22_ordinary_game import DIR, FrozenMemory, StandardMatch

OUTPUT = DIR.parent / 'tt23'
SOURCE = DIR / 'duration-pong.json.gz'
RIVAL = DIR / 'tt22_opponent-learned.json.gz'
RECEIPT = OUTPUT / 'source-lanes-physical.json.gz'
CURVES = OUTPUT / 'source-lanes-curves.json.gz'
SCHEDULE = 1787082463
IDENTITIES = ('6', '10', '7')
SIDES = {'left': -1., 'right': 1.}
TRAIN_PER_SIDE = 2
HELD_PER_SIDE = 2
DESIRED = .12
DRAWS = 512


def cases():
    rng = np.random.default_rng(SCHEDULE)
    return [dict(seed=int(rng.integers(2**31)), cohort=cohort, split=split)
            for split, count in (('held', HELD_PER_SIDE), ('practice', TRAIN_PER_SIDE))
            for _ in range(count) for cohort in SIDES]


def lane_choice(arena, side, profile):
    return (side * (-1) ** arena.rival_contacts * DESIRED, *profile[1:])


def referee_lanes(row):
    """Evaluate where each actual rival reply crosses the near optical stroke corridor."""
    seen = []
    observations = row['optical']
    contacts = [event['step'] for event in row['events'] if event['kind'] == 'contact']
    for index, event in enumerate(e for e in row['events'] if e['kind'] == 'contact' and e['side'] == -1):
        following = next((tick for tick in contacts if tick > event['step']), len(row['actions']))
        corridor = [(tick, xyz) for tick, xyz in observations if event['step'] < tick <= following and xyz[1] < -.55]
        crossing = min(corridor, key=lambda sample: abs(sample[1][1] + .77), default=None)
        seen.append(dict(tick=event['step'], return_index=index,
                          visible_y=crossing[1][1] if crossing else None,
                          visible_x=crossing[1][0] if crossing else None))
    return seen


def repeat_observation_tape(reference, player, memory):
    """Certify the same action sequence against the captured RGB track."""
    optical = dict(reference['optical'])
    player.reset()
    player.outbound = False
    player.memory, player.update_memory = memory, True
    for tick, command in enumerate(reference['actions']):
        position = np.asarray(optical[tick]) if tick in optical else None
        if tick < HANDOVER:
            player.history.append((tick, position))
            if position is not None and position[1] > .07:
                player.outbound = True
            player.sim.twin.advance(np.asarray(command))
        elif player.command(tick, position).tolist() != command:
            return None
    return dict(reference, shots=player.shots, observation_reuse=True)


def play(path):
    source = _read(SOURCE)
    gains = {name: source['candidates'][name] for name in IDENTITIES}
    score = {name: source['screen_duration_ticks'][name] for name in IDENTITIES}
    profile = _read(RIVAL)['fit']['profile']
    calibration = _read(CHECKPOINT)
    target = calibration['evidence']
    schedule = cases()
    fresh = {case['seed'] for case in schedule}
    if len(fresh) != len(schedule) or fresh & {row['seed'] for row in calibration['practice']}:
        raise ValueError('cross-court starts overlap each other or the shared calibration')
    report = dict(schema='tt23-source-lanes-physical-1', schedule=SCHEDULE, requested_x=DESIRED,
                  cases=schedule, options=gains, source_scores=score, profile=profile, rows=[],
                  hashes={p.name: _hash(p) for p in (SOURCE, RIVAL, CHECKPOINT)})
    if path.exists():
        previous = _read(path)
        if any(previous[key] != value for key, value in report.items() if key != 'rows'):
            raise ValueError('incompatible earlier cross-court receipt')
        report = previous
    station = StandardMatch(profile)
    cache = FarCache(station.rival)
    station.player.close()
    station.player = IntentPlayer(FlightLaw())
    begins = perf_counter()
    try:
        for i, point in enumerate(schedule):
            if i == len(report['rows']):
                report['rows'].append(dict(**point, options={}, shared_frames=None))
            bundle = report['rows'][i]
            if any(bundle[k] != v for k,v in point.items()):
                raise ValueError('unexpected earlier point')
            side = SIDES[point['cohort']]
            station.rival.target = lambda side=side: lane_choice(station.arena, side, profile)
            order = IDENTITIES[i % 3:] + IDENTITIES[:i % 3]
            for name in order:
                if name in bundle['options']:
                    continue
                station.player.gains = gains[name]
                reference = next(iter(bundle['options'].values()), None)
                row = repeat_observation_tape(reference, station.player, FrozenMemory(target)) if reference else None
                if row is None:
                    row = dict(station.play(point['seed'], 1, FrozenMemory(target)), observation_reuse=False)
                row['lanes_seen'] = referee_lanes(row)
                verify_rows([row])
                bundle['options'][name] = row
                _write(path, report)
                print(i, point['split'], point['cohort'], name, row['longest'],
                      row['outcome']['legal_returns'], row['observation_reuse'], round(perf_counter() - begins, 1), flush=True)
            ref = bundle['options'][IDENTITIES[0]]
            duration = min(len(ref['actions']), HANDOVER)
            prefix = ref['actions'][:duration], ref['rival_actions'][:duration]
            if any((row['actions'][:HANDOVER], row['rival_actions'][:HANDOVER]) != prefix
                   for row in bundle['options'].values()):
                raise ValueError('treatment changed shared service motions')
            bundle['shared_frames'] = duration
            _write(path, report)
    finally:
        station.close()
    report['verified'] = sum(len(row['options']) for row in report['rows'])
    report['cache'] = dict(hits=cache.hits, misses=cache.misses, audited=cache.checked)
    _write(path, report)
    return report


def adapt(report, path, physical_path):
    schedule = report['rows']
    if len(schedule) != len(report['cases']) or any(set(r['options']) != set(IDENTITIES) for r in schedule):
        raise ValueError('every planned case requires a physical point for every option')
    if report.get('verified') != len(schedule) * len(IDENTITIES) or any(row['shared_frames'] is None for row in schedule):
        raise ValueError('cannot evaluate without physical replay and shared service checks')
    practice = [i for i, row in enumerate(schedule) if row['split'] == 'practice']
    held = {side: [i for i, row in enumerate(schedule) if row['split'] == 'held' and row['cohort'] == side] for side in SIDES}
    length = np.array([[row['options'][name]['longest'] for name in IDENTITIES] for row in schedule])
    legal = np.array([[row['options'][name]['outcome']['legal_returns'] for name in IDENTITIES] for row in schedule])
    reward = -np.expm1(-length / 5.)
    source = np.array([report['source_scores'][name] for name in IDENTITIES])
    rank = np.argsort(np.argsort(source)) / (len(source) - 1)
    prior = .20 + .40 * rank
    initial = {'pong': prior, 'cold': np.full(len(IDENTITIES), .40), 'permuted': np.roll(prior, 1)}
    rng = np.random.default_rng(SCHEDULE + 1)
    records = {name: [] for name in initial}
    choice_records = {name: [] for name in initial}
    for _ in range(DRAWS):
        order = rng.permutation(practice)
        uniforms = rng.uniform(.000001, .999999, size=(len(order), len(IDENTITIES)))
        ties = rng.uniform(0., 1.e-9, size=(len(order) + 1, len(IDENTITIES)))
        for name, start in initial.items():
            a, b = 2. * start.copy(), 2. * (1. - start)
            selected, feedback, greedy = [], [], []
            for budget in range(len(order) + 1):
                greedy.append(int(np.argmax(a / (a + b) + ties[budget])))
                if budget == len(order):
                    break
                option = int(np.argmax(betaincinv(a, b, uniforms[budget])))
                physical_case = int(order[budget])
                score = reward[physical_case, option]
                a[option] += score
                b[option] += 1. - score
                selected.append(option)
                feedback.append(int(length[physical_case, option]))
            records[name].append(dict(order=order.tolist(), practice=selected, physical_return_counts=feedback, greedy=greedy))
            choice_records[name].append(greedy)
    summaries = {}
    cohorts = dict(held, all=sum(held.values(), []))
    for key, trials in choice_records.items():
        gain = np.asarray(trials)
        summaries[key] = {}
        for cohort, held_indices in cohorts.items():
            ring = length[held_indices][:, gain]
            near = legal[held_indices][:, gain]
            summaries[key][cohort] = dict(alternating=ring.mean(axis=(0, 1)).tolist(),
                                          near=near.mean(axis=(0, 1)).tolist(),
                                          ten=(ring >= 10).mean(axis=(0, 1)).tolist(),
                                          twenty=(ring >= 20).mean(axis=(0, 1)).tolist())
    result = dict(schema='tt23-source-lanes-curves-1', schedule=SCHEDULE,
                  physical=physical_path.name, physical_sha256=_hash(physical_path),
                  practice_indices=practice, held_indices=held,
                  seeds=[row['seed'] for row in schedule],
                  prior={name:value.tolist() for name,value in initial.items()}, identities=IDENTITIES,
                  strict_alternating=length.tolist(), near_legal=legal.tolist(), repetitions=DRAWS,
                  records=records, summaries=summaries)
    _write(path, result)
    print(summaries, flush=True)


def chart(curves, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary = curves['summaries']
    fig, panels = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True, layout='constrained')
    modes = {'pong': ('Pong rank', '#137a8d', 'o'),
             'cold': ('Uniform', '#282a36', 's'),
             'permuted': ('Permuted Pong rank', '#c05737', '^')}
    for ax, cohort, title in zip(panels, ('all', 'left', 'right'), ('All held points', 'Left first', 'Right first')):
        for name, (label, color, marker) in modes.items():
            curve = summary[name][cohort]['alternating']
            ax.plot(range(len(curve)), curve, marker=marker, color=color, label=label, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel('Completed practice points')
        ax.set_xticks(range(len(curve)))
        ax.set_ylim(bottom=0.)
        ax.grid(axis='y', alpha=.25)
    panels[0].set_ylabel('Mean alternating legal returns')
    panels[-1].legend(loc='best', fontsize=8)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical', type=Path, default=RECEIPT)
    parser.add_argument('--curves', type=Path, default=CURVES)
    parser.add_argument('--curves-only', action='store_true')
    parser.add_argument('--chart', type=Path)
    args = parser.parse_args()
    report = _read(args.physical) if args.curves_only else play(args.physical)
    adapt(report, args.curves, args.physical)
    if args.chart is not None:
        chart(_read(args.curves), args.chart)


if __name__ == '__main__':
    main()
