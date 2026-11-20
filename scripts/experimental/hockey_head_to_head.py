"""Evaluate hockey shot selection against an equally mobile copy of the learner."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from air_hockey import ACTIONS, AirHockeyCamera, HEIGHT, OPPONENT_COLOR, PLAYER_COLOR
from air_hockey_learning import GoalMemory, HockeyPlayer
from arcade_wall_learn import WallMemory
from run_hockey_matches import center_serve


class TwoPlayers(AirHockeyCamera):
    def __init__(self, seed):
        super().__init__(seed, serve=center_serve(seed), max_frames=540)
        self.rival = [1 - self.player[0], HEIGHT / 2]
        self.right_action = 0

    def _rival_command(self):
        x, y = ACTIONS[self.right_action]
        return (-x, y), .98


def opposite_view(frame):
    image = frame[:, ::-1].copy()
    # Disjoint masks avoid recoloring an already swapped object.
    player = image[:, :, 0] == PLAYER_COLOR[0]
    rival = image[:, :, 0] == OPPONENT_COLOR[0]
    image[player] = OPPONENT_COLOR
    image[rival] = PLAYER_COLOR
    return image


def point(seed, state, arms):
    arena = TwoPlayers(seed)
    agents = []
    for label in arms:
        goals = GoalMemory.from_state(state['goal']) if label=='trained' else GoalMemory()
        agents.append(HockeyPlayer(WallMemory.from_state(state['wall']),goals, seed=seed,
                                  fixed_center=label=='fixed_center'))
    image = arena.frame()
    agents[0].observe(image)
    agents[1].observe(opposite_view(image))
    actions = bytearray()
    while not (arena.terminated or arena.truncated):
        left, right = (agent.action() for agent in agents)
        actions.extend((left,right))
        arena.right_action = right
        image, reward, done, capped = arena.step(left)
        agents[0].observe(image,reward,done,capped)
        agents[1].observe(opposite_view(image),-reward,done,capped)
    touches = [e['side'] for e in arena.events if e['kind']=='contact']
    return dict(left=arms[0],right=arms[1], score=arena.reward, frames=arena.frames,
                left_contacts=touches.count('player'),right_contacts=touches.count('rival'),
                actions_sha256=sha256(actions).hexdigest())


def run(state, first, pairs):
    rng = np.random.default_rng(310398)
    seeds = rng.integers(10**8,2**31-1,size=first+pairs).tolist()[first:]
    return [dict(seed=seed, opponents={
        other:[point(seed,state,('trained',other)),point(seed,state,(other,'trained'))]
        for other in ('untrained','fixed_center')}) for seed in seeds]


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--pairs',type=int,default=48)
    parser.add_argument('--first',type=int,default=0)
    args=parser.parse_args()
    rows=run(json.loads(args.checkpoint.read_bytes()),args.first,args.pairs)
    summary={}
    for other in ('untrained','fixed_center'):
        scores=[(row['opponents'][other][0]['score'],-row['opponents'][other][1]['score']) for row in rows]
        means=np.mean(scores,axis=1)
        ix=np.random.default_rng(88304).integers(len(rows),size=(4000,len(rows)))
        scores=np.asarray(scores).ravel()
        summary[other]=dict(goals=int(sum(scores>0)),conceded=int(sum(scores<0)),timeouts=int(sum(scores==0)),
                            net_per_point=float(means.mean()),ci95=np.quantile(means[ix].mean(axis=1),[.025,.975]).tolist())
    with args.output.open('x') as f:
        paths=[Path(__file__), Path(__file__).with_name('air_hockey.py'),
               Path(__file__).with_name('air_hockey_learning.py'), args.checkpoint]
        json.dump(dict(summary=summary,rows=rows,source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in paths}),
                  f,allow_nan=False)
    print(summary)
