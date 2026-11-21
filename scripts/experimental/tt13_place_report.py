"""Summarize return placement and check cross-fixture receipt provenance."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import median

from .tt4_transfer_trial import _hash, _read, _write
from .tt13_place_trial import openings, summary


def compact(section):
    optical = section['optics_audit']
    matched = optical['visible']
    observed = [value['optical'] for value in matched]
    physical = optical['physical_by_action']
    return dict(metrics=section['metrics'], by_server=section['metrics_by_server'],
                optical=dict(assigned_returns=optical['legal_associated'], matches=len(matched),
                             unmatched=optical['unmatched_optics'],
                             median_depth_m=None if not observed else median(observed),
                             median_absolute_error_m=None if not observed else
                             median(abs(value['optical'] - value['scorer']) for value in matched)),
                placed=[dict(count=len(depths), median_m=median(depths) if depths else None)
                        for depths in physical])


def first_landings(rows):
    bounces = {}
    for row in rows:
        first = next((event for event in row['events']
                      if event['kind'] == 'legal_return' and event['side'] == 1), None)
        if first is not None:
            bounces[row['seed']] = first['location'][1]
    return dict(by_seed=bounces, median=median(bounces.values()) if bounces else None)


def paired_results(report):
    sections = report['sections']
    matched = [row for row in sections['fixed']['points'] if openings(row)]
    results = {'fixed': summary(matched)}
    results['first_landing_m'] = {'fixed': first_landings(matched)}
    for policy in ('frozen', 'reward'):
        active = sections[policy]['points']
        wins = Counter((str(left['points_for']) + str(right['points_for']))
                       for left, right in zip(matched, active, strict=True))
        by_role = {str(side): summary([row for row in active if row['server'] == side]) for side in (-1, 1)}
        results['first_landing_m'][policy] = first_landings(active)
        results[policy] = dict(metrics=summary(active), by_server=by_role,
                               paired_wins={key: wins[key] for key in ('00', '01', '10', '11')},
                               optics=compact(sections[policy])['optical'])
    baseline_depths = results['first_landing_m']['fixed']['by_seed']
    results['paired_depth_change_m'] = {}
    for policy in ('frozen', 'reward'):
        new_depths = results['first_landing_m'][policy]['by_seed']
        changes = [new_depths[seed] - before for seed, before in baseline_depths.items() if seed in new_depths]
        results['paired_depth_change_m'][policy] = dict(count=len(changes),
                                                       median=median(changes) if changes else None)
    results['fixed_by_server'] = {str(side): summary([row for row in matched if row['server'] == side])
                                  for side in (-1, 1)}
    return results


def game_results(fixed, active):
    fixed_policy, candidate = (next(iter(record['sections'])) for record in (fixed, active))
    a, b = fixed['sections'][fixed_policy], active['sections'][candidate]
    if any(fixed[key] != active[key] for key in ('source_sha256', 'seed', 'arena', 'schedule')):
        raise ValueError('game control and learner used different starting setup')
    if any(a[key] != b[key] for key in ('first_server', 'race_to')) or not (a['finished'] and b['finished']):
        raise ValueError('game comparators did not both complete the same race')
    setup = {}
    for side in (-1, 1):
        scheduled = fixed['schedule'][str(side)]
        left, right = ([r for r in result['points'] if r['server'] == side] for result in (a, b))
        for rows in (left, right):
            if [[r['seed'], r['phase']] for r in rows] != scheduled[:len(rows)]:
                raise ValueError('recorded openings departed from their side-specific schedule')
        setup[str(side)] = min(len(left), len(right))
    return dict(fixture=fixed['seed'], comparable_openings=setup,
                control=dict(score=a['score'], **compact(a)), candidate=dict(score=b['score'], **compact(b)),
                prior_optical_samples=len(active['memory_before']['observations']),
                post_optical_samples=len(b['memory_after']['observations']),
                carry_source=active['placement_source_sha256'])


def run(folder, additional):
    def read(name):
        path = folder / f'place-{name}.json.gz'
        artifact = _read(path)
        if artifact['replayed'] != sum(len(s['points']) for s in artifact['sections'].values()):
            raise ValueError(f'incomplete replay coverage in {name}')
        return artifact

    def match(name):
        return read(f'game-{name}-fixed'), read(f'game-{name}-online')

    practice = read('practice')
    pairs = read('pairs')
    expected = _hash(folder / 'place-practice-player.json.gz')
    if pairs['placement_source_sha256'] != expected or practice['checkpoint_sha256'] != expected:
        raise ValueError('paired comparison was not drawn from the recorded practice')
    if pairs['source_sha256'] != practice['source_sha256']:
        raise ValueError('practice and matched comparisons did not use the same installation state')
    games = {name: match(name) for name in ('a', 'b', *additional)}
    if any(games[name][1]['placement_source_sha256'] != expected for name in ('a', 'b')):
        raise ValueError('independent gantry games did not start from the same placement experience')
    if 'transfer' in games:
        carried = _hash(folder / 'place-game-a-player.json.gz')
        if (games['transfer'][1]['placement_source_sha256'] != carried or
                games['a'][1]['checkpoint_sha256'] != carried or
                games['transfer'][1]['arena'] == games['a'][1]['arena']):
            raise ValueError('transfer used the wrong checkpoint or installation')
    if 'revisit' in games:
        carried = _hash(folder / 'place-game-transfer-player.json.gz')
        if (games['revisit'][1]['placement_source_sha256'] != carried or
                games['transfer'][1]['checkpoint_sha256'] != carried or
                games['revisit'][1]['arena'] != games['a'][1]['arena']):
            raise ValueError('revisit used the wrong checkpoint or installation')

    seen = set()
    fixtures = [practice, pairs, *(value[0] for value in games.values())]
    for fixture in fixtures:
        seeds = {seed for cohort in fixture['schedule'].values() for seed, _ in cohort}
        if seen & seeds or len(seeds) != sum(map(len, fixture['schedule'].values())):
            raise ValueError('fixture and practice schedules overlap')
        seen.update(seeds)
    output = dict(practice=dict(values=practice['values'], **compact(practice['sections']['online'])),
                  pairs=paired_results(pairs),
                  games={name: game_results(*receipts) for name, receipts in games.items()},
                  tape_replays=practice['replayed'] + pairs['replayed'] +
                  sum(result['replayed'] for match in games.values() for result in match))
    _write(folder / 'place-comparison.json.gz', output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--additional', nargs='*', choices=('transfer', 'revisit'), default=[])
    args = parser.parse_args()
    comparison = run(args.folder, args.additional)
    print(comparison)
