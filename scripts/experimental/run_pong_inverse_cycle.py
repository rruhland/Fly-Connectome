"""Native breadth check using the public tracker and encoder's entity-only path."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from fly_connectome.control.environment import PongCamera
from fly_connectome.control.session import model_bytes
from fly_connectome.vision import VisualStateEncoder, load_default
from pong_categorical_motor_fir import CategoricalMotorController, CategoricalMotorFIR
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
from public_entity_pong_native import snapshot
from return_metrics import ReturnCounter


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, allow_nan=False))
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule():
    rng = np.random.default_rng(72491632)
    starts = []
    while len(starts) < 32:
        candidate = int(rng.integers(1, 2147483647))
        directed = (candidate * 48271 + 1) % 2147483647
        if directed % 2 == len(starts) % 2 and candidate not in starts:
            starts.append(candidate)
    return starts


def play(learner, policy, seed, mapping):
    camera = PongCamera(seed, max_episode_ticks=2400, opponent_speed=.65)
    vision, encoder = load_default(), VisualStateEncoder(64, 64, .02)
    controller = CategoricalMotorController(policy, learner)
    counts = ReturnCounter()
    step = camera.game.step
    camera.game.step = lambda drive, **kw: counts.step(camera.game, step, drive, **kw)
    frame = camera.game.render(64, 64)[0].clone()
    previous = torch.zeros_like(frame)

    def observe(frame, previous):
        events = torch.stack((previous & ~frame, ~previous & frame)).float()
        return snapshot(encoder.encode_entities(*vision.track_entities(events, frame.float())))

    public = observe(frame, previous)
    requests = []
    events = []
    while True:
        action, _ = controller.choose(public, 1)
        opportunities = counts.opportunities
        result = camera.step(mapping[action])
        public = observe(result.frame, frame)
        frame = result.frame
        controller.observe(own_hits=result.hits, reward=result.reward,
                           terminated=result.terminated, truncated=result.truncated,
                           elapsed_seconds=result.elapsed_seconds, public_entities=public)
        requests.append(action)
        if counts.opportunities > opportunities:
            events.append(dict(camera=len(requests), hit=bool(result.hits)))
        if result.terminated or result.truncated:
            break
    assert len(events) == counts.opportunities and sum(e['hit'] for e in events) == counts.returns
    return dict(seed=seed, requests_sha=hashlib.sha256(bytes(requests)).hexdigest(),
                transitions=len(requests), initial=1, returns=counts.returns,
                opportunities=counts.opportunities, events=events,
                seconds=camera.total_ticks / 120, updates=learner.updates)


def run(root):
    root.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(1)
    model_bytes()
    starts = schedule()
    normal, inverse = [0, 1, 2], [2, 0, 1]
    plan = dict(seeds=starts, learner_seed=5180973, schedule=['normal8','inverse16','normal8'],
                matched_frozen='all 16 inverse games', max_games=48, max_transitions=48000,
                phase_outcomes='all first/later and fixed final 8 B games; exploratory generality test',
                sources={str(p):sha(p) for p in (Path(__file__),Path('src/fly_connectome/vision/state.py'),
                    Path('src/fly_connectome/vision/interface.py'),Path('scripts/experimental/pong_categorical_motor_fir.py'))})
    save(root/'plan.json', plan)
    model = CategoricalMotorFIR(plan['learner_seed'])
    policy = PersistentRelationalPolicy(RelationalPolicy(3,0,1,0,-1,np.zeros(5),np.ones(4)),False)
    rows = []
    started = time.monotonic()
    for label, mapping, indices in [('A',normal,range(8)),('B',inverse,range(8,24)),
                                    ('restored',normal,range(24,32)),('frozen',inverse,range(8,24))]:
        if label == 'B':
            save(root/'acquired.json',model.state())
        if label == 'frozen':
            model = CategoricalMotorFIR.from_state(json.loads((root/'acquired.json').read_text())).frozen_branch()
        for i in indices:
            row = dict(phase=label, index=i, **play(model, policy, starts[i], mapping))
            rows.append(row)
            save(root/'games.json',rows)
        save(root/(label+'-end.json'),model.state())
        print(label,sum(r['returns'] for r in rows if r['phase']==label),
              sum(r['opportunities'] for r in rows if r['phase']==label),flush=True)
    save(root/'cost.json', dict(games=len(rows), transitions=sum(r['transitions'] for r in rows),
                                 wall_seconds=time.monotonic()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    run(parser.parse_args().directory)
