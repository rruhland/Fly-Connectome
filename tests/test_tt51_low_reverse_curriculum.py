from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from scripts.experimental.tt51_low_reverse_curriculum import seeded_rows,plain


class PublicBody:
    model=SimpleNamespace(geom=lambda name:SimpleNamespace(size=np.array([.076,.005,0.])))
    def pose(self,q):return np.array([0.,-1.2,1.]),np.array([0.,1.,0.])


def packet():
    rows=[dict(tick=i,x=[0.,-.7-.08*i,1.,0.,-3. if i<8 else 3.,0.],
        q=[0.]*6,ref=[.01*i]*6,next_ref=[.01*(i+1)]*6) for i in range(11)]
    return dict(rows=[dict(reward='physical legal and answered',name=str(i),mode_start=0,
        observations=deepcopy(rows),native_touch=[99.]*3,x_at_receive_line=[999.]*5) for i in range(2)])


def test_teacher_readout_ignores_privileged_ball_and_contact_metadata():
    original=packet();changed=deepcopy(original)
    for row in changed['rows']:
        row['native_touch']=[-999.]*3
        row['x_at_receive_line']=[-999.]*5
    assert plain(seeded_rows(original,PublicBody()))==plain(seeded_rows(changed,PublicBody()))


def test_no_oracle_fill_when_source_rgb_is_absent():
    data=packet()
    for observation in data['rows'][0]['observations']:observation['x']=None
    with pytest.raises(ValueError):seeded_rows(data,PublicBody())


def test_unscored_source_is_not_admitted_as_a_rewarded_template():
    data=packet();data['rows'][1]['reward']='miss'
    with pytest.raises(ValueError):seeded_rows(data,PublicBody())
