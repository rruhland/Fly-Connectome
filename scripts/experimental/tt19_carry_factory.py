"""Carry the factory-serve player in the same endpoint as its gantry and Pong policies."""
import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_factory_serve_trial import ARENA, load_checkpoint, play_game
from .tt19_stroke_trial import DONOR
from .tt19_retention import compare_gantry, compare_pong


def package(folder):
    source = folder / 'factory-serve-distinct-player.json.gz'
    practice = folder / 'factory-serve-distinct-practice.json.gz'
    proposals = folder / 'factory-serve-proposals.json.gz'
    carried = _read(DONOR)
    saved = _read(source)
    trials = _read(practice)
    if (trials['corrected_pick'] != saved['corrected_pick'] or len({row['seed'] for row in trials['trials']}) != 12
            or _read(proposals)['programs'] != saved['programs'] or _hash(proposals) != trials['proposal_sha256']):
        raise ValueError('Factory policies must match the chronological twelve-serve experiment')
    carried['tt19_factory_serve'] = saved
    carried['tt19_factory_lineage'] = dict(donor_sha256=_hash(DONOR), arm_sha256=_hash(source),
        practice_sha256=_hash(practice), proposal_sha256=_hash(proposals),
        target_practice_frames=trials['target_practice_frames'])
    output = folder / 'continual-so101-player.json.gz'
    _write(output, carried)
    return output


def check(folder, output):
    cache = folder / 'factory-dry-catalog.npz'
    report = _read(folder / 'factory-serve-held.json.gz')
    audit = []
    for game in report['games']:
        policy = load_checkpoint(output, cache, game['mode'])
        try:
            for role in (1, -1):
                row = next(p for p in game['points'] if p['server'] == role)
                policy.reset(role)
                actions = [policy.command(i, ball).tolist() for i, ball in enumerate(row['optical'])]
                if actions != row['actions'] or policy.player.plans != row['plans']:
                    raise ValueError('The bundled player no longer reproduces its served or receiving point')
                audit.append(dict(game=game['seed'], mode=game['mode'], point=row['seed'], role=role,
                                  actions=len(actions)))
        finally:
            policy.player.close()
    _write(folder / 'continual-so101-replay.json.gz', dict(checkpoint_sha256=_hash(output), arm=audit))


def play(folder, checkpoint, seed, mode):
    # The held runner also uses this constructor for the published comparisons.
    # Never open a source donor or training archive when resuming a player.
    cache = folder / 'factory-dry-catalog.npz'
    data = _read(checkpoint)
    raw = data['tt19_factory_serve']
    player = load_checkpoint(checkpoint, cache, mode)
    practice = dict(factory_pick=raw['factory_pick'], corrected_pick=raw['corrected_pick'], seeds=[],
                    target_practice_points=12, target_practice_frames=data['tt19_factory_lineage']['target_practice_frames'])
    game = play_game(player, practice, seed, mode)
    path = folder / f'continual-game-{mode}-{seed}.json.gz'
    _write(path, dict(seeds=[seed], games=[game], native=True, strict_endline=True, course='full', arena=ARENA,
                      checkpoint_sha256=_hash(checkpoint)))
    print(path, game['score'], flush=True)


def audit(folder):
    checkpoint = folder / 'continual-so101-player.json.gz'
    digest = _hash(checkpoint)
    checks = dict(checkpoint_sha256=digest, native_games=[])
    for seed, name in [(930477119, 'factory-serve-held'), (1529880313, 'factory-serve-1529880313-held')]:
        original = _read(folder / f'{name}.json.gz')
        restored = _read(folder / f'continual-game-factory-{seed}.json.gz')
        factory = next(g for g in original['games'] if g['mode'] == 'factory')
        if restored['checkpoint_sha256'] != digest or restored['games'] != [factory]:
            raise ValueError('Restoring the combined endpoint changed the native game')
        checks['native_games'].append(dict(seed=seed, points=len(factory['points']), score=factory['score'],
                                           returns=factory['returns'], serves=factory['serves']))
    for kind, compare in [('gantry', compare_gantry), ('pong', compare_pong)]:
        before = _read(folder / f'retention-{kind}-before.json.gz')
        after = _read(folder / f'retention-{kind}-after-factory.json.gz')
        if after['checkpoint_sha256'] != digest:
            raise ValueError('The old-task evaluation used a different endpoint')
        checks[kind] = compare(before, after)
    _write(folder / 'continual-so101-audit.json.gz', checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--play', type=int, metavar='GAME_SEED')
    parser.add_argument('--mode', choices=('factory', 'corrected'), default='factory')
    parser.add_argument('--audit', action='store_true', help='Verify saved games and old-task receipts')
    args = parser.parse_args()
    if args.audit:
        audit(args.folder)
    elif args.play is None:
        check(args.folder, package(args.folder))
    else:
        play(args.folder, args.folder / 'continual-so101-player.json.gz', args.play, args.mode)
