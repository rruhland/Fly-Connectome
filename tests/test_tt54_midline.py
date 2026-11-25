import numpy as np
import pytest

from scripts.experimental.tt54_midline import line
from scripts.experimental.tt54_midline_course import cue_prototype


def test_midline_band_uses_forward_signed_time_and_gravity():
    x,z=line(dict(xyz=[.01,-1.10,.98],velocity=[.35,-3.7,1.2]))
    t=.43/(.98*3.7)
    assert x==pytest.approx(.01+.35*t)
    assert z==pytest.approx(.98+1.2*t-4.905*t*t)


def test_optical_context_uses_only_inbound_camera_before_earned_phase():
    rows=[dict(tick=28,x=[.015,-1.06,.97,.2,-3.5,.62],bounce=True),
          dict(tick=30,x=[-.44,-1.10,.8,2.8,4.,5.],bounce=False)]
    t=.04/(.98*3.5)
    np.testing.assert_allclose(cue_prototype(rows,dict(source_phase_tick=29)),
        [.015+.2*t,.97+.62*t-4.905*t*t,.2,-3.5,.62-9.81*t])
    with pytest.raises(ValueError):
        cue_prototype(rows,dict(source_phase_tick=27))
