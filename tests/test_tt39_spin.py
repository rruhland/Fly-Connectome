"""Check the RGB observer's kinematics and evidence boundaries."""

import numpy as np

from scripts.experimental.tt39_spin import Observer, TORQUE, kernels, model, solve
from scripts.experimental.tt39_spin_diagnostic import delayed


SETTINGS = dict(height=.769, restitution=.9, spin_prior=70., friction=.065)


def test_magnus_kernel_has_physical_sign_and_derivative():
    t = np.array([-.15, .09, .30])
    v = np.array([.2, -4., .15])
    a, d, p, dv, basis, dbasis = kernels(t, v)
    eps = 1e-5
    earlier, later = kernels(t-eps, v), kernels(t+eps, v)
    np.testing.assert_allclose((later[0]-earlier[0])/(2*eps), d, atol=1e-7)
    np.testing.assert_allclose((later[2]-earlier[2])/(2*eps), dv, atol=1e-6)
    np.testing.assert_allclose((later[4]-earlier[4])/(2*eps), dbasis, atol=1e-8)
    # Positive spin around x accelerates an inbound ball down, not up.
    assert np.all(basis[:, 2, 0] < 0.)


def test_spin_impulse_uses_outgoing_minus_incoming_not_absolute_spin():
    np.testing.assert_allclose(TORQUE @ [.10, -.40, 5.], [-50., -12.5, 0.])


def test_single_arc_can_constrain_transverse_spin_more_than_travel_axis():
    time = np.linspace(-.40, 0., 25)
    v = np.array([.10, -4.40, -1.28])
    omega = np.array([54., 0., -26.])
    p0 = np.array([.06, -.82, 1.11])
    travel, _, p, _, j, _ = kernels(time, v)
    rgb = p0 + travel[:, None]*v + p + j @ omega
    m = model(time, rgb)
    estimate = solve(time, rgb, m, SETTINGS, 'arc', 0., None)
    assert estimate.omega[0] > 12. and estimate.omega[2] < -15.
    assert estimate.posterior[1] > 1.5 * min(estimate.posterior[0], estimate.posterior[2])


def test_bounce_requires_seen_ascending_pixels_and_gaps_clear_history():
    observer = Observer(1/60, SETTINGS)
    time = np.arange(-12, 0)/60.
    ground = np.array([.04, -.74, .768])
    pre = np.array([.05, -4., -2.9])
    post = np.array([.06, -3.66, 2.6])
    for t in time:
        a, _, p, _, _, _ = kernels(t, pre)
        observer.ingest(t, ground + a[0]*pre + p[0])
    x = observer.fit('coupled')
    assert x is not None and x.post is None
    for t in range(4):
        stamp = t/60
        a, _, p, _, _, _ = kernels(stamp, post)
        observer.ingest(stamp, ground + a[0]*post + p[0])
    x = observer.fit('coupled')
    assert x is not None and x.post is not None
    assert x.post[2] > .8
    observer.ingest(1., ground)
    assert observer.fit('coupled') is None


def test_delayed_label_rejects_other_point_and_reversal():
    t = np.arange(7)/60.
    p0, v = np.array([.02, -.3, 1.04]), np.array([.05, -3.8, .90])
    a, _, p, _, _, _ = kernels(t, v)
    r = np.c_[t, p0 + a[:, None]*v + p]
    groups = np.ones((len(t), 3))
    assert delayed(r, groups, 0, 1/60) is not None
    opposed = r.copy()
    opposed[-2:, 2] = opposed[-3, 2] + .06
    assert delayed(opposed, groups, 0, 1/60) is None
    groups[-1, 1] = 2
    assert delayed(r, groups, 0, 1/60) is None
