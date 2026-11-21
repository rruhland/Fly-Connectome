import json

import numpy as np

from scripts.experimental.tt21_online_strokes import OnlineBank


def test_resume_carries_rng_and_reward_history_through_creation_of_new_strokes():
    programs = np.zeros((4, 13)).tolist()
    stream = OnlineBank(programs)
    for t in range(18):
        stream.choose([.01, .09], None)
        stream.finish(1 if t % 3 else -1)

    restored = OnlineBank(programs, state=json.loads(json.dumps(stream.state())))
    for t in range(8):
        cue = [.02, .1 + .003 * t]
        assert stream.choose(cue, None) == restored.choose(cue, None)
        stream.finish(-1)
        restored.finish(-1)
    assert stream.state() == restored.state()
    assert len(stream.programs) > len(programs)
