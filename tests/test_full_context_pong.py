import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from full_context_pong import OpenLoopContextRun


def test_online_score_pairs_previous_issue_with_current_event_frame():
    run = object.__new__(OpenLoopContextRun)
    run.frame = 1
    run.previous_prediction = torch.tensor([-.8, .7, .4, 0.])
    run.previous_target = torch.tensor([0., 0., 0., 1.])
    run.metrics = {name: dict(samples=0, squared_error=0., persistence_error=0.,
                              anticipation=0., false_alarms=0)
                   for name in ('all', 'on', 'off', 'quiet')}
    run.trace = []
    run._score(torch.tensor([-1., 1., 0., 0.]))
    assert run.metrics['on']['samples'] == 1
    assert run.metrics['off']['samples'] == 1
    assert run.metrics['quiet']['samples'] == 2
    assert run.metrics['on']['anticipation'] == torch.tensor(.8).item()
    assert run.metrics['off']['anticipation'] == torch.tensor(.7).item()
    assert run.metrics['quiet']['false_alarms'] == 1
    assert run.metrics['off']['persistence_error'] == 1
    assert run.trace[0]['frame'] == 1
