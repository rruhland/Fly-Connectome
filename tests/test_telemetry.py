import time
import torch

from test_training import trainer
from fly_connectome.telemetry import Telemetry, latest_snapshot


def test_no_subscriber_does_not_capture_and_slow_reader_drops_frames(tmp_path):
    telemetry = Telemetry(tmp_path)
    calls = []
    telemetry.publish(lambda: calls.append(1) or {'step': 1})
    assert calls == []
    telemetry.subscribe()
    telemetry.publish(lambda: {'step': 1})
    telemetry.publish(lambda: {'step': 2})
    assert latest_snapshot(tmp_path)['step'] == 2


def test_preview_is_actual_body_state_and_does_not_change_learning(tmp_path):
    a, b = trainer(), trainer()
    telemetry = Telemetry(tmp_path)
    telemetry.subscribe()
    for _ in range(4):
        a.step(); b.step()
        telemetry.publish(lambda: a.snapshot())
    snapshot = latest_snapshot(tmp_path)
    assert snapshot['player_y'] == a.environment.body.position[0].item()
    assert snapshot['ball'] == a.environment.ball[0].tolist()
    torch.testing.assert_close(a.network.magnitudes, b.network.magnitudes, rtol=0, atol=0)
    torch.testing.assert_close(a.network.voltage, b.network.voltage, rtol=0, atol=0)


def test_expired_subscription_disables_preview(tmp_path):
    telemetry = Telemetry(tmp_path)
    telemetry.subscribe(now=0)
    assert not telemetry.enabled(now=100)


def test_headless_and_telemetry_checkpoints_are_byte_identical(tmp_path):
    a, b = trainer(), trainer()
    telemetry = Telemetry(tmp_path/'ui')
    telemetry.subscribe()
    for _ in range(3):
        a.step(); b.step()
        telemetry.publish(b.snapshot)
    a.save(tmp_path/'headless.pt'); b.save(tmp_path/'ui.pt')
    assert (tmp_path/'headless.pt').read_bytes() == (tmp_path/'ui.pt').read_bytes()
    telemetry.close()
