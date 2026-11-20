"""Play scored first-to-seven matches with a frozen air-hockey deployment checkpoint."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random

from air_hockey import AirHockeyCamera, HEIGHT, HockeyServe
from air_hockey_learning import GoalMemory, HockeyPlayer
from arcade_wall_learn import WallMemory


START = 67133001
ROUND_LIMIT = 32
POINTS_TO_WIN = 7


def center_serve(seed):
    """Give both sides center-line faceoffs without aiming the serve at either goal."""
    rng = random.Random(f'neutral faceoff {seed}')
    angle = rng.uniform(-.95, .95)
    direction = rng.choice((-1, 1))
    speed = rng.uniform(.61, .80)
    return HockeyServe(.5, rng.uniform(.20, HEIGHT - .20),
                       direction * speed * math.cos(angle), speed * math.sin(angle))


def play_point(seed, state, trained, *, opponent, fixed_center=False, record=False):
    camera = AirHockeyCamera(seed, serve=center_serve(seed), max_frames=540, opponent=opponent)
    goals = GoalMemory.from_state(state['goal']) if trained else GoalMemory()
    agent = HockeyPlayer(WallMemory.from_state(state['wall']), goals, seed=seed, fixed_center=fixed_center)
    frame = camera.frame()
    agent.observe(frame)
    frames = [frame] if record else []
    actions = bytearray()
    while not (camera.terminated or camera.truncated):
        action = agent.action()
        actions.append(action)
        obs = camera.step(action)
        agent.observe(*obs)
        if record and (camera.frames % 4 == 0 or camera.terminated or camera.truncated):
            frames.append(obs[0])
    touches = [e for e in camera.events if e['kind'] == 'contact']
    own = sum(c['side'] == 'player' for c in touches)
    result = dict(seed=seed, opponent=opponent, serve=asdict(camera.serve), score=camera.reward, frames=camera.frames,
                  own_touches=own, rival_touches=len(touches)-own,
                  goal_after_touch=camera.reward == 1 and own > 0,
                  action_sha256=sha256(actions).hexdigest())
    return result, frames


def play_match(index, state, condition, opponent):
    score = [0, 0]
    rows = []
    for point in range(ROUND_LIMIT):
        row, _ = play_point(START + index * 100 + point, state, condition == 'after_hockey',
                            opponent=opponent, fixed_center=condition == 'fixed_center')
        rows.append(row)
        if row['score']:
            score[0 if row['score'] > 0 else 1] += 1
        if max(score) == POINTS_TO_WIN:
            break
    return dict(index=index, player=score[0], opponent=score[1], won=score[0]==POINTS_TO_WIN,
                lost=score[1]==POINTS_TO_WIN, capped=max(score)<POINTS_TO_WIN,
                goals_after_touch=sum(r['goal_after_touch'] for r in rows), points=rows)


def run(state, matches, opponent):
    all_runs = {}
    for label in ('after_hockey', 'before_hockey', 'fixed_center'):
        runs = [play_match(i, state, label, opponent) for i in range(matches)]
        all_runs[label] = dict(matches=runs, wins=sum(m['won'] for m in runs),
                               losses=sum(m['lost'] for m in runs), capped=sum(m['capped'] for m in runs),
                               scores=[sum(m[k] for m in runs) for k in ('player','opponent')],
                               goals_after_touch=sum(m['goals_after_touch'] for m in runs))
    return all_runs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--matches', type=int, default=16)
    parser.add_argument('--opponent', choices=('guard', 'laneguard'), default='guard')
    args = parser.parse_args()
    data = run(json.loads(args.checkpoint.read_bytes()), args.matches, args.opponent)
    files = [args.checkpoint, Path(__file__)]
    files.extend(Path(__file__).with_name(s) for s in
                 ('air_hockey.py','air_hockey_learning.py','arcade_wall_learn.py'))
    data['source_sha256'] = {f.name:sha256(f.read_bytes()).hexdigest() for f in files}
    with args.output.open('x') as out:
        json.dump(data, out, separators=(',', ':'), allow_nan=False)
    for label in ('after_hockey', 'before_hockey', 'fixed_center'):
        print(label,{k:v for k,v in data[label].items() if k!='matches'})
