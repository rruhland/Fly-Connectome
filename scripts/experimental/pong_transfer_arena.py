"""Native visual Pong episode with optional environment-side execution changes."""
from collections import deque
from dataclasses import replace
import hashlib

import torch

from fly_connectome.control.environment import PongCamera
from fly_connectome.pong import Pong
from fly_connectome.vision import VisualStateEncoder, load_default
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
from public_entity_pong_native import snapshot
from return_metrics import ReturnCounter
from run_pong_surprise_recovery import POLICY


def interception_policy():
    return PersistentRelationalPolicy(RelationalPolicy(*POLICY['choices'],
        POLICY['coefficients'], POLICY['scales']), POLICY['require_first_hit'])


def play(controller, seed, *, opponent_speed=.65, duration=20., delay_frames=0,
         activation_seconds=None, mapping=(0, 1, 2), switch_seconds=0.):
    """The controller receives entities, requested-action feedback and public time only."""
    camera = PongCamera(seed, max_episode_ticks=round(duration * 120), opponent_speed=opponent_speed)
    if activation_seconds is not None:
        camera.game = Pong([seed], replace(camera.game.config, tau_activation=activation_seconds))
    counts = ReturnCounter()
    game = camera.game
    native_step = game.step
    game.step = lambda drive, **kwargs: counts.step(game, native_step, drive, **kwargs)
    vision, encoder = load_default(), VisualStateEncoder(64, 64, .02)

    def observe(frame, previous):
        events = torch.stack((previous & ~frame, ~previous & frame)).float()
        return snapshot(encoder.encode_entities(*vision.track_entities(events, frame.float())))

    frame = game.render(64, 64)[0].clone()
    public = observe(frame, torch.zeros_like(frame))
    history = deque([1] * delay_frames)
    requests = bytearray()
    contacts = []
    elapsed = 0.
    while True:
        requested, _ = controller.choose(public, 1)
        requests.append(requested)
        history.append(requested)
        executed = history.popleft()
        before = counts.opportunities
        result = camera.step(mapping[executed] if elapsed >= switch_seconds else executed)
        public = observe(result.frame, frame)
        frame = result.frame
        controller.observe(public_entities=public, own_hits=result.hits, reward=result.reward,
                           terminated=result.terminated, truncated=result.truncated,
                           elapsed_seconds=result.elapsed_seconds)
        elapsed = camera.total_ticks/120
        if counts.opportunities != before:
            contacts.append((elapsed, bool(result.hits)))
        if result.terminated or result.truncated:
            return dict(seed=seed, hits=counts.returns, opportunities=counts.opportunities,
                        own_hit_win=int(result.reward>0 and counts.returns>0), score=int(result.reward),
                        contacts=contacts, elapsed=elapsed, cameras=len(requests),
                        action_sha256=hashlib.sha256(requests).hexdigest())
