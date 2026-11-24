import numpy as np

from scripts.experimental import tt34_latency


def test_clock_learns_changes_and_carries_estimate_through_stillness(monkeypatch):
    rng = np.random.default_rng(34011)
    reference = rng.normal(size=(90, 5))
    bank = {"q": reference, "xy": reference[:, :4] * 20}
    clock = tt34_latency.Clock(bank)
    # Supply ideal observed pixel centers to isolate the estimator from segmentation.
    monkeypatch.setattr(tt34_latency, "pixels", lambda centers: centers)
    q = rng.normal(size=(64, 5))
    for delay in (3, 1, 0):
        clock.begin()
        positions = [clock.body(row) for row in q]
        for i, row in enumerate(q):
            clock.update(row, positions[i - delay] if i >= delay else None)
        assert clock.lag == delay
        clock.begin()
        for _ in range(40):
            clock.update(q[-1], positions[-1])
        assert clock.lag == delay
