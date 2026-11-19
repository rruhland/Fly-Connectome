"""An inference fast path must preserve the controller's public observations."""
import torch

from fly_connectome.control.environment import PongCamera
from fly_connectome.vision import load_default, VisualStateEncoder


def test_entity_only_stream_matches_full_vision_through_scene_resets():
    torch.set_num_threads(1)
    vision, light_vision = load_default(), load_default()
    encoder, light_encoder = [VisualStateEncoder(64, 64, .02) for _ in range(2)]
    missing_seen = False
    populated = 0
    for seed in (17030, 16991):
        for obj in (vision, light_vision, encoder, light_encoder):
            obj.reset_state()
        camera = PongCamera(seed, max_episode_ticks=2400, opponent_speed=.65)
        frame = camera.game.render(64, 64)[0].clone()
        previous = torch.zeros_like(frame)
        for i in range(160):
            events = torch.stack((previous & ~frame, ~previous & frame)).float()
            image = frame.float()
            expected = encoder.encode(vision.step(events, image), image)['entities']
            observed = light_encoder.encode_entities(*light_vision.track_entities(events, image))
            assert len(expected) == len(observed)
            populated += bool(expected)
            for reference, actual in zip(expected, observed):
                assert actual.keys() == reference.keys()
                for key, value in reference.items():
                    if isinstance(value, torch.Tensor):
                        assert torch.equal(actual[key], value)
                    else:
                        assert actual[key] == value
                missing_seen |= not actual['observed']
            previous = frame
            result = camera.step((i // 19) % 3)
            if result.terminated or result.truncated:
                break
            frame = result.frame
    assert populated > 100 and missing_seen
