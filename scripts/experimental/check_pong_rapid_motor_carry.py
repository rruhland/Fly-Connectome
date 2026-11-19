"""Verify uninterrupted native playback with serialized action belief at point boundaries."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fly_connectome.control.environment import PongCamera
from fly_connectome.control.session import model_bytes
from fly_connectome.vision import VisualStateEncoder, load_default
from pong_rapid_motor_memory import RapidMotorMemory
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
from pong_visual_button_effects import VisualButtonEffects
from public_entity_pong_native import snapshot
from run_pong_inverse_cycle import save
from run_pong_surprise_recovery import POLICY


SEED = 1491879208
NORMAL = (0, 1, 2)
CYCLE = (1, 2, 0)
POINTS = ((516, 480, NORMAL, CYCLE), (480, 0, CYCLE, CYCLE), (480, 0, CYCLE, NORMAL))


def run(directory, checkpoint):
    torch.set_num_threads(1)
    model_bytes()
    directory.mkdir(exist_ok=False, parents=True)
    learner = VisualButtonEffects.from_state(json.loads(checkpoint.read_text()))
    memory = RapidMotorMemory.from_learner(learner)
    policy = PersistentRelationalPolicy(RelationalPolicy(*POLICY['choices'],
        POLICY['coefficients'], POLICY['scales']), POLICY['require_first_hit'])
    camera = PongCamera(SEED, opponent_speed=.65)
    vision, encoder = load_default(), VisualStateEncoder(64, 64, .02)

    def encode(frame, previous):
        changes = torch.stack((previous & ~frame, ~previous & frame)).float()
        return snapshot(encoder.encode_entities(*vision.track_entities(changes, frame.float())))

    points = []
    for index, (max_ticks, switch_tick, normal, changed) in enumerate(POINTS):
        camera.max_episode_ticks = max_ticks
        frame = camera.reset() if index else camera.game.render(64, 64)[0].clone()
        vision.reset_state()
        encoder.reset_state()
        public = encode(frame, torch.zeros_like(frame))
        memory_path = directory / f'point-{index + 1}-memory.json'
        save(memory_path, memory.state())
        restored = RapidMotorMemory.from_state(json.loads(memory_path.read_text()))
        frozen = restored.frozen_branch()
        live, replay, static = (entry.new_controller(policy) for entry in (memory, restored, frozen))
        assert memory == restored
        for controller in (live, replay, static):
            assert controller.belief.states is None and controller.motor_id is None
            assert tuple(controller.belief.probabilities) == memory.probabilities

        start_tick = camera.total_ticks
        transitions = hits = 0
        while True:
            request, decision = live.choose(public, 1)
            assert (request, decision) == replay.choose(public, 1)
            static.choose(public, 1)
            execution = changed if camera.episode_ticks >= switch_tick else normal
            result = camera.step(execution[request])
            public = encode(result.frame, frame)
            frame = result.frame
            for controller in (live, replay, static):
                controller.observe(public_entities=public, own_hits=result.hits, reward=result.reward,
                    terminated=result.terminated, truncated=result.truncated,
                    elapsed_seconds=result.elapsed_seconds)
            for field in ('states', 'covariances', 'probabilities'):
                assert np.array_equal(getattr(live.belief, field), getattr(replay.belief, field))
            assert live.effective_requests == replay.effective_requests
            assert frozen.capture(static) == frozen
            assert tuple(static.belief.probabilities) == frozen.probabilities
            transitions += 1
            hits += result.hits
            if result.terminated or result.truncated:
                break
        after = memory.capture(live)
        assert restored.capture(replay) == after
        points.append(dict(index=index + 1, start_tick=start_tick, end_tick=camera.total_ticks,
            transitions=transitions, hits=hits, initial=memory.state(), final=after.state(),
            terminated=result.terminated, truncated=result.truncated))
        memory = after
    assert points[0]['final']['order'] == list(NORMAL)
    assert max(points[0]['final']['probabilities']) < .85
    assert points[1]['final']['order'] == [CYCLE.index(intent) for intent in NORMAL]
    assert points[2]['final']['order'] == list(NORMAL)
    report = dict(seed=SEED, points=points, replay_matched_transitions=sum(p['transitions'] for p in points),
                  initial_cameras=len(points), serialized_partial_posterior=True,
                  preserved_frozen_branches=True)
    save(directory / 'summary.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.directory, args.checkpoint), sort_keys=True))
