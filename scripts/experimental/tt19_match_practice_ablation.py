"""Replace the practiced SO-101 opening bank while keeping the match player fixed."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from .tt3_pong import state_sha256
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_contextual import StrokeMemory
from .tt19_match_player import evaluate
from .tt19_match_report import practice, score
from .tt19_model_rollout_trial import replay


TT19 = ROOT / 'artifacts/tt19'
GAME_SEED = 36325598
GENERIC_SEED = 654382795
COURSES = ('center', 'full')


def generic_bank():
    bank = {}
    for role in (-1, 1):
        source = StrokeMemory(GENERIC_SEED + (role + 1), role)
        programs = [source.choose([0., .32], learning=True).tolist() for _ in range(16)]
        # Identical neutral entries give every untested proposal the same prior.
        # The existing bank's deterministic tie break selects the first draw.
        trials = [[index, 0, 0, 0., .32, .5] for index in range(len(programs))]
        bank[str(role)] = dict(programs=programs, trials=trials)
    return bank


def match_prefix(trained, generic):
    rows = []
    for first, second in zip(trained['points'], generic['points']):
        if (first['seed'], first['server']) != (second['seed'], second['server']):
            break
        rows.append(dict(seed=first['seed'], server=first['server'],
                         same_action_tape=first['actions'] == second['actions'],
                         trained_reward=first['result']['points_for'], generic_reward=second['result']['points_for'],
                         trained_return=first['result']['legal_returns'], generic_return=second['result']['legal_returns'],
                         trained_serve=first['result']['legal_serve'], generic_serve=second['result']['legal_serve']))
    return rows


def run(output):
    catalog = TT19 / 'factory-dry-catalog.npz'
    bank = generic_bank()
    result = dict(schema='tt19-opening-practice-removal-v1', game_seed=GAME_SEED, generic_seed=GENERIC_SEED,
                  catalog_sha256=_hash(catalog), new_motor_learning_points=0, generic_bank=bank, courses={})
    with TemporaryDirectory(prefix='tt19-opening-removal-') as folder:
        folder = Path(folder)
        for course in COURSES:
            trained_checkpoint = TT19 / f'match-{course}-rgb-player.json.gz'
            trained_receipt = TT19 / f'match-{course}-rgb.json.gz'
            trained = _read(trained_receipt)
            choice = next(game for game in trained['games'] if game['seed'] == GAME_SEED)
            trained = dict(trained, seeds=[GAME_SEED], games=[choice])
            state = _read(trained_checkpoint)
            policy = state['tt19_match']
            if (policy['practice_sha256'] != trained['practice_sha256'] or policy['receiver'] != 'memory'
                    or policy['catalog_sha256'] != result['catalog_sha256'] or not trained['native']):
                raise ValueError('trained endpoint differs from the matched native game')
            source_seeds, charges = practice(TT19, policy['bank_source'])
            altered = dict(policy, bank=bank, bank_source='unpracticed-generic', practice_sha256=state_sha256(bank))
            checkpoint = folder / f'generic-{course}-player.json.gz'
            _write(checkpoint, dict(state, tt19_match=altered))
            cold = evaluate(folder, catalog, course, [GAME_SEED], True, 'memory', policy['source'],
                            'unpracticed-generic', subset=course, endpoints=False, resume=checkpoint)
            if not (choice['finished'] and cold['games'][0]['finished']):
                raise ValueError('a matched game did not finish')
            prefix = match_prefix(choice, cold['games'][0])
            if len(prefix) < 11:
                raise ValueError('the two games did not share their opening point streams')
            joined = result['courses'][course] = dict(trained_checkpoint=trained_checkpoint.name,
                                                     trained_checkpoint_sha256=_hash(trained_checkpoint),
                                                     trained_receipt=trained_receipt.name, trained_practice=charges,
                                                     trained=score(trained, source_seeds)[0],
                                                     generic=score(cold, set())[0], matched_prefix=prefix,
                                                     trained_tapes_replayed=replay(trained),
                                                     generic_tapes_replayed=replay(cold), generic_game=cold['games'][0])
            print(course, joined['trained']['score'], joined['generic']['score'], len(prefix), flush=True)
            _write(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=TT19 / 'opening-practice-removal.json.gz')
    args = parser.parse_args()
    run(args.output)
