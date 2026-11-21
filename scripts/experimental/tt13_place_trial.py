"""Qualify stereo-trained return depth on matched incoming balls and full games."""
import argparse
from copy import copy, deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_trial import check_tapes
from .tt9_precontact_trial import _play_point
from .tt10_series_trial import PHASES, _restore, _save
from .tt10_window_trial import CARRIED
from .tt13_place import DEPTHS, PlacementMemory, PlacementPlayer, visible_landing


SOURCE = ROOT / 'artifacts/tt12/motor-gantry-recall-player.json.gz'
STAGES = ('practice', 'pair', 'game')


def restore(path, arena, memory, policy):
    state, library, player = _restore(path, arena)
    # The base restore already reconstructed the RNG and portable per-installation state.
    # This subclass adds hooks without a second constructor/reset consuming a draw.
    player.__class__ = PlacementPlayer
    player.placement = memory
    player.placement_policy = policy
    return state, library, player


def openings(row):
    incoming = [event for event in row['events']
                if (event['kind'], event.get('side')) in (('legal_serve', -1), ('legal_return', -1))]
    if row['server'] == 1:
        incoming = [event for event in incoming if event['kind'] == 'legal_return']
    return incoming


def summary(rows):
    own = [row for row in rows if row['server'] == 1]
    playable = [row for row in rows if openings(row)]
    later = [row for row in playable if len(openings(row)) >= 2]
    return dict(points=len(rows), won=sum(row['points_for'] for row in rows),
                own_serves=len(own), legal_serves=sum(row['evaluator']['legal_serve'] for row in own),
                playable=len(playable), playable_won=sum(row['points_for'] for row in playable),
                first_returns=sum(row['player_returns'] > 0 for row in playable),
                later_opportunities=sum(len(openings(row)) - 1 for row in playable),
                later_returns=sum(max(0, row['player_returns'] - 1) for row in playable),
                sustained=len([row for row in playable if row['player_returns'] >= 2]),
                later_playable_points=len(later), later_playable_won=sum(row['points_for'] for row in later),
                total_returns=sum(row['player_returns'] for row in rows),
                choices=[sum(shot['action'] == action for row in rows
                             for shot in row['return_placement']['choices']) for action in range(len(DEPTHS))],
                visible_landings=sum(shot['landing'] is not None for row in rows
                                     for shot in row['return_placement']['choices']))


def play(player, arena, wiring, run_seed, case, server):
    seed, phase = case
    row = _play_point(player, arena, wiring, run_seed, seed, phase, server)
    row['ball_views'] = [{key: frame[key] for key in ('tick', 'ball')} for frame in player.trace]
    return row


def check_pair(left, right):
    if (left['seed'], left['phase'], left['server']) != (right['seed'], right['phase'], right['server']):
        raise ValueError('paired openings must describe the same point')
    if not openings(left) or openings(left)[0] != openings(right)[0]:
        raise ValueError('playable first incoming ball changed in the pair')
    # First changed *command*, following all identical prior public input.
    choices = [shot['tick'] for shot in right['return_placement']['choices'] if shot['action']]
    if not choices:
        if left['action_sha256'] != right['action_sha256']:
            raise ValueError('actions changed without placement intervention')
        return left['steps']
    stop = choices[0]
    if (left['actions'][:stop] != right['actions'][:stop] or left['applied'][:stop] != right['applied'][:stop]
            or left['ball_views'][:stop + 1] != right['ball_views'][:stop + 1]):
        raise ValueError('paired traces diverged before return placement')
    if left['server'] == 1:
        contact = next(event for event in left['events'] if event['kind'] == 'contact' and event['side'] == -1)
        if contact['step'] > stop:
            raise ValueError('placement changed actions before the rival hit the reply')
    return stop


def audit_optics(rows):
    associated = 0
    misses = []
    visible = []
    chosen_actual = [[] for _ in DEPTHS]
    unassociated = 0
    for row in rows:
        bounces = [event for event in row['events'] if event['kind'] == 'legal_return' and event['side'] == 1]
        contacts = [event for event in row['events'] if event['kind'] == 'contact' and event['side'] == 1
                    and (row['server'] != 1 or event['step'] > (openings(row)[0]['step'] if openings(row) else row['steps']))]
        for shot in row['return_placement']['choices']:
            landing = shot['landing']
            contact = min(contacts, key=lambda event: abs(event['step'] - shot['contact']), default=None)
            matched = contact is not None and abs(contact['step'] - shot['contact']) <= 6
            if matched:
                following = next((event['step'] for event in contacts if event['step'] > contact['step']), float('inf'))
                scorer = next((e for e in bounces if contact['step'] < e['step'] < following), None)
            else:
                scorer = None
            if scorer is not None:
                associated += 1
                chosen_actual[shot['action']].append(scorer['location'][1])
            if landing is None:
                continue
            if scorer is None or abs(scorer['step'] - landing['bounce']['tick']) > 4:
                misses.append(dict(seed=row['seed'], optical=landing['bounce']['tick'],
                                   scorer=None if scorer is None else scorer['step']))
                unassociated += 1
            else:
                visible.append(dict(seed=row['seed'], action=shot['action'], optical=landing['bounce']['ball'][1],
                                    scorer=scorer['location'][1]))
    return dict(legal_associated=associated, visible=visible, unmatched_optics=unassociated,
                unmatched_examples=misses, physical_by_action=chosen_actual)


def audit(report):
    count = 0
    for name, section in report['sections'].items():
        rows = section['points']
        game = report['stage'] == 'game'
        if game and not section['finished']:
            raise ValueError('incomplete game')
        check_tapes(dict(report, **section, practice=not game))
        memory = PlacementMemory(deepcopy(report['memory_before']['observations']))
        for row in rows:
            digest = sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest()
            if digest != row['action_sha256']:
                raise ValueError('stored command digest did not replay')
            choices = row['return_placement']['choices']
            expected = memory.choose(name)
            for shot in choices:
                if (shot['action'] != expected or shot['landing'] != visible_landing(row['ball_views'], shot['contact'])):
                    raise ValueError('stored choice or measurement did not replay')
            if name == 'online':
                memory.observe(choices, row['public_reward'])
        if section['memory_after'] != memory.state():
            raise ValueError('checkpoint memory disagrees with chronological updates')
        if summary(rows) != section['metrics'] or audit_optics(rows) != section['optics_audit']:
            raise ValueError('stored evaluation disagrees with point traces')
        count += len(rows)
    if report['stage'] == 'pair':
        for branch, section in report['sections'].items():
            if branch == 'fixed':
                continue
            for left, right in zip(filter(openings, report['sections']['fixed']['points']),
                                   section['points'], strict=True):
                if check_pair(left, right) != right['verified_prefix']:
                    raise ValueError('stored pair prefix did not replay')
    return count


def schedule(seed, limit):
    return {str(server): list(zip(_seeds(seed, 4731 + i, 0, limit)['tests'],
                                  match_phases(PHASES, limit, [seed, 4731 + i]), strict=True))
            for i, server in enumerate((-1, 1))}


def run(args):
    entry = _read(args.resume)['series']
    memory_state = (dict(observations=[]) if args.memory is None else _read(args.memory)['return_placement'])
    source_hash = _hash(args.resume)
    report = dict(schema='tt13-return-placement-v1', stage=args.stage, seed=args.seed, arena=entry['arena'],
                  installation_seed=entry['arena']['seed'], source_sha256=source_hash,
                  placement_source_sha256=None if args.memory is None else _hash(args.memory),
                  implementation_sha256={name: _hash(Path(__file__).with_name(name))
                                         for name in ('tt13_place.py', 'tt13_place_trial.py')},
                  depths=DEPTHS, memory_before=memory_state, schedule=schedule(args.seed, args.limit), sections={})
    arena = TT8ServiceArena(**entry['arena'])
    wiring = TT6Wiring(arena, entry['arena']['seed'])
    try:
        cache_hall_stereo(arena, reflections=True)
        policy = {'practice': 'online', 'pair': 'fixed', 'game': args.policy}[args.stage]
        memory = PlacementMemory(deepcopy(memory_state['observations']))
        state, library, player = restore(args.resume, arena, memory, policy)
        sections = report['sections']
        sections[policy] = dict(points=[])
        if args.stage == 'pair':
            sections.update(frozen=dict(points=[]), reward=dict(points=[]))
        index = {'-1': 0, '1': 0}
        score = ServiceScore(first_server=args.first_server)
        for i in range(args.limit):
            server = score.server if args.stage == 'game' else (-1, 1)[i % 2]
            side = str(server)
            case = report['schedule'][side][index[side]]
            index[side] += 1
            before = {key: deepcopy(getattr(player, key)) for key in CARRIED} if args.stage == 'pair' else None
            point = play(player, arena, wiring, args.seed, case, server)
            sections[policy]['points'].append(point)
            if args.stage == 'pair' and openings(point):
                for branch in ('frozen', 'reward'):
                    active = copy(player)
                    for key, value in before.items():
                        setattr(active, key, deepcopy(value))
                    active.placement_policy = branch
                    row = play(active, arena, wiring, args.seed, case, server)
                    row['verified_prefix'] = check_pair(point, row)
                    sections[branch]['points'].append(row)
            if args.stage == 'game':
                score.record(point['evaluator'])
                sections[policy].update(score=score.score, finished=score.finished, race_to=11,
                                        first_server=score.first_server)
            print(args.stage, policy, i + 1, server, point['points_for'], point['player_returns'],
                  [s['action'] for s in point['return_placement']['choices']],
                  sum(s['landing'] is not None for s in point['return_placement']['choices']),
                  score.score if args.stage == 'game' else None, flush=True)
            if score.finished and args.stage == 'game':
                break
        for name, section in sections.items():
            section['memory_after'] = memory.state()
            section['metrics'] = summary(section['points'])
            section['metrics_by_server'] = {str(side): summary([row for row in section['points'] if row['server'] == side])
                                            for side in (-1, 1)}
            section['optics_audit'] = audit_optics(section['points'])
        report['values'] = dict(optical=memory.values(), reward=memory.values(False))
        if args.checkpoint:
            portable = dict(state, return_placement=memory.state())
            series = {key: entry[key] for key in ('arena', 'model', 'optics', 'calibration', 'probes', 'attempts', 'donor_sha256')}
            _save(args.checkpoint, portable, library, player, series, entry['completed'] + int(args.stage == 'game'))
            if _read(args.checkpoint)['return_placement'] != memory.state():
                raise ValueError('placement state did not survive checkpoint save')
            report['checkpoint_sha256'] = _hash(args.checkpoint)
        if _hash(args.resume) != source_hash:
            raise ValueError('source checkpoint changed')
    finally:
        arena.close()
    _write(args.output, report)
    report['replayed'] = audit(_read(args.output))
    _write(args.output, report)
    print(report['replayed'], report['values'], {name: section['metrics'] for name, section in sections.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=(*STAGES, 'verify'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--resume', type=Path, default=SOURCE)
    parser.add_argument('--memory', type=Path, help='optional return-placement memory to import from a checkpoint')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--limit', type=int, default=65)
    parser.add_argument('--first-server', type=int, choices=(-1, 1), default=-1)
    parser.add_argument('--policy', choices=('fixed', 'online', 'frozen', 'reward'), default='online')
    args = parser.parse_args()
    if args.stage == 'verify':
        print(audit(_read(args.output)), 'physical tapes replayed')
    elif args.seed is None or args.limit < 1:
        parser.error('supply a seed and positive limit')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        run(args)


if __name__ == '__main__':
    main()
