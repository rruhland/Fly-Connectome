"""Replay the complete SO-101 match tapes and account for target practice."""
import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_model_rollout_trial import replay
from .tt19_retention import compare_gantry, compare_pong


def practice(folder, bank):
    def read(name):
        return _read(folder / (name + '.json.gz'))
    rows = []
    labels = []
    if bank == 'full-from-wide':
        courses = ['center-prior', 'wide-from-center', bank]
    else:
        courses = [] if bank == 'local' else [bank]
    for initial in (['local', 'gantry'] if courses else ['local']):
        data = read(f'context-{initial}-practice')
        rows.extend((row[0], row[2]) for row in data['rows'])
        labels.append(f'context-{initial}')
        qualified = read(f'consolidated-{initial}-bank')
        for group in qualified['banks'].values():
            rows.extend((t[1], t[2]) for t in group['trials'])
    for course in courses:
        data = read(f'feeder-{course}-practice')
        for epochs in data['archive'].values():
            for epoch in epochs:
                rows.extend((t[1], t[2]) for t in epoch['trials'])
        qualified = read(f'consolidated-{course}-bank')
        for group in qualified['banks'].values():
            rows.extend((t[1], t[2]) for t in group['trials'])
        labels.append(course)
    return {int(seed) for seed, _ in rows}, dict(stages=labels, points=len(rows), frames=sum(f for _, f in rows))


def score(report, training):
    games = []
    for game in report['games']:
        stats = dict(seed=game['seed'], score=game['score'], win=game['score'][0] > game['score'][1],
                     points=len(game['points']), own_serves=0, legal_serves=0, returns=0, multiple=0,
                     second_ball_opportunities=0, first_legal_serve=None, first_legal_return=None, frames=0)
        for row in game['points']:
            if row['seed'] in training:
                raise ValueError('Scored point seed occurred in training')
            stats['own_serves'] += row['server'] == 1
            player_returns = 0
            further_ball = False
            for event in row['events']:
                key = (event['kind'], event.get('side'))
                if key == ('legal_serve', 1):
                    stats['legal_serves'] += 1
                    if stats['first_legal_serve'] is None:
                        stats['first_legal_serve'] = stats['frames'] + event['step']
                elif key == ('legal_return', 1):
                    player_returns += 1
                    if stats['first_legal_return'] is None:
                        stats['first_legal_return'] = stats['frames'] + event['step']
                elif key == ('legal_return', -1) and player_returns:
                    further_ball = True
            stats['returns'] += player_returns
            stats['multiple'] += player_returns >= 2
            stats['second_ball_opportunities'] += further_ball
            stats['frames'] += len(row['actions'])
        if [stats['legal_serves'],stats['returns'],stats['multiple']] != [game['serves'],game['returns'],game['multiple']]:
            raise ValueError('Point events disagree with the game totals')
        games.append(stats)
    return games


def run(folder):
    results = {}
    for tag in ('center-frozen','full-frozen','center-rgb','full-rgb','full-rgb-confirm','full-from-wide','gradual-confirm'):
        path = folder / f'match-{tag}.json.gz'
        report = _read(path)
        if len(report['seeds']) != len(report['games']):
            raise ValueError(f'Incomplete suite: {tag}')
        training, cost = practice(folder, report['bank_source'])
        results[tag] = dict(practice=cost, games=score(report, training), replayed_points=replay(report), sha256=_hash(path))
    restored = _read(folder / 'match-full-reloaded.json.gz')['games'][0]
    originals = _read(folder / 'match-full-rgb.json.gz')['games']
    if restored != next(game for game in originals if game['seed'] == restored['seed']):
        raise ValueError('Reloaded optical actions, scores, plans or RGB hashes changed')
    retention = {}
    for tag, name in [('full-match', 'full-rgb'), ('rollout', 'model-rollout-native'), ('gradual', 'full-from-wide')]:
        checkpoint = folder / f'match-{name}-player.json.gz'
        digest = _hash(checkpoint)
        retention[tag] = dict(checkpoint_sha256=digest)
        for kind, compare in [('gantry', compare_gantry), ('pong', compare_pong)]:
            before = _read(folder / f'retention-{kind}-before.json.gz')
            after = _read(folder / f'retention-{kind}-after-{tag}.json.gz')
            if after['checkpoint_sha256'] != digest:
                raise ValueError(f'{kind} retention used a different player')
            retention[tag][kind] = compare(before, after)
    _write(folder / 'match-audit.json.gz', dict(results=results, retention=retention,
           reloaded_game_seed=restored['seed'], reloaded_game_points=len(restored['points'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    run(parser.parse_args().folder)
