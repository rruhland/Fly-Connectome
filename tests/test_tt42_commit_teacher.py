from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.experimental.tt42_commit_teacher import run_point
from scripts.experimental.tt42_commit_teacher_export import score_clips
from scripts.experimental.tt42_commit_teacher_verify import metrics


def test_new_solver_does_not_interrupt_committed_motion():
    rates = np.zeros((5, 6))
    rates[:3, 0] = [0.22, 0.31, -0.22]
    hit = {"step": 2, "kind": "contact", "side": 1, "time": 0.025}
    landing = {"step": 3, "kind": "legal_return", "side": 1, "time": 0.049}

    class Court:
        def __init__(self):
            self.events = [
                {
                    "kind": "separation",
                    "collider": "rival_racket",
                    "step": 0,
                    "time": 0.0,
                }
            ]
            self.actions = []
            self.legal_returns = self.rival_returns = 0
            self.ball = np.array([0.0, -1.34, 1.04])
            self.velocity = np.array([0.0, -3.6, -0.3])
            self.terminated = self.truncated = False
            self.data = SimpleNamespace(time=0.0)

        def exchange(self, u, far):
            self.actions.append(u.tolist())
            n = len(self.actions)
            self.data.time = n / 60
            actual = [hit] if n == 2 else [landing] if n == 3 else []
            self.events.extend(actual)
            self.legal_returns = int(n == 3)
            return None, 0, False, False, {"events": actual}

        def point_placements(self):
            return {"delivered": []}

        def motor_audit(self):
            return {}

        def clearance_result(self):
            return {}

    class MockTeacher:
        n = 0

        def search(self):
            self.n += 1
            return {"rates": rates}, {
                "virtual_legal": 1,
                "cpu_s": 1.0,
                "rounds": [],
                "selection": {
                    "native": [0.0, 0.025, 0.049],
                    "family": "unit-test",
                    "departure": [0.0, 4.0, 2.2],
                },
            }

        def recover(self):
            return np.ones(6)

    a = Court()
    teacher = MockTeacher()
    with patch(
        "scripts.experimental.tt42_commit_teacher.audit",
        side_effect=lambda events, **options: {
            "learner": sum(
                e["kind"] == "legal_return" and e.get("side") == 1 for e in events
            )
        },
    ):
        result = run_point(a, teacher, {}, {"id": 1, "seed": 7}, [], 1, 1, 5, 0)
    assert teacher.n == 1
    assert result["actions"] == rates[:3].tolist()
    assert result["teacher_searches"][0]["physical_legal"]
    assert metrics(result)["actual_legal"] == 1
    assert list(score_clips(result["events"], {1: result["teacher_searches"][0]})) == [
        (1, landing)
    ]
