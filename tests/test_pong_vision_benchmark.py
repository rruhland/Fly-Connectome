"""The Pong camera adapter preserves the existing event and time semantics."""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from fly_connectome.sensor import EventCamera

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_pong_vision import dense_events, physics_ticks_for_sample, score_counts


def test_dense_events_match_existing_event_camera():
    camera = EventCamera(1, 64, 64)
    previous = torch.zeros(64, 64, dtype=torch.bool)
    frames = []
    for left in (4, 7, 7, 11):
        frame = torch.zeros(64, 64, dtype=torch.bool)
        frame[10:13, left:left+3] = True
        frames.append(frame)
    for frame in frames:
        sparse = camera.observe(frame[None])
        reference = torch.zeros(2, 64, 64)
        for pixel, on in zip(sparse.pixels.tolist(), sparse.on.tolist()):
            reference[int(on), pixel//64, pixel%64] = 1
        assert torch.equal(dense_events(previous, frame), reference)
        previous.copy_(frame)


def test_physics_ticks_preserve_120_to_50_hz_cadence():
    assert [physics_ticks_for_sample(sample) for sample in range(5)] == [2, 2, 3, 2, 3]
    assert sum(physics_ticks_for_sample(sample) for sample in range(50)) == 120


def test_score_counts_keep_opposite_outcomes_separate():
    assert score_counts(torch.tensor([1., -1., 0., -1.])) == (1, 2)


def test_benchmark_selects_declared_source_outside_checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / 'scripts' / 'benchmark_pong_vision.py'
    output = tmp_path / 'result.json'
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    completed = subprocess.run([sys.executable, str(script), '--source-root', str(root),
                                '--samples', '3', '--warmup', '1', '--output', str(output)],
                               cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert Path(result['source_package']).resolve() == root / 'src' / 'fly_connectome'
    assert result['mean_total_ms'] > 0
    assert result['compute_samples_per_second'] > 0
    assert 0 <= result['deadline_misses_20ms'] <= result['timed_samples']
    assert result['long_context_forecasts'] >= 0


def test_parity_runner_records_online_and_learned_evaluation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'trajectory.json'
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    completed = subprocess.run([sys.executable, str(root / 'scripts' / 'check_pong_vision_parity.py'),
                                '--source-root', str(root), '--samples', '3', str(output)],
                               cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert len(result['online']['samples']) == 3
    assert len(result['evaluation']['samples']) == 3
    assert result['online']['checkpoint'] == result['evaluation']['checkpoint']
