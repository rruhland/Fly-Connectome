import sys
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('scipy')
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from local_information import motion_phase
from temporal_visual import oscillation, recurrent_boundaries
from timing_transfer import padded, noisy_currents


@pytest.mark.parametrize('dwell',[2,3,4,6])
def test_timing_changes_keep_next_frame_horizon_and_exclude_one_cycle(dwell):
    blank=12
    frames=oscillation(blank,dwell=dwell)
    assert len(frames)==blank+8*dwell+1
    on=frames[:,0,30,39].numpy().astype(int)
    events=np.diff(on,prepend=0)
    issue=recurrent_boundaries([blank],dwell=dwell)
    future=(issue+8)//8
    assert (events[future]==1).sum()==3
    assert (events[future]==-1).sum()==3
    assert motion_phase(blank,blank,dwell)=='on'
    assert motion_phase(blank+dwell,blank,dwell)=='off'
    assert motion_phase(blank+2*dwell-1,blank,dwell)==f'after_off_{dwell-1}'
    assert motion_phase(blank+8*dwell,blank,dwell)=='blank'


def test_margin_padding_is_symmetric_without_mutating_original():
    b=np.array([[[1.,.2],[2.,.4]],[[3.,.5],[4.,.8]]])
    before=b.copy()
    result=padded(b,[.01,.02],np.array([10.,.5]))
    np.testing.assert_allclose(result[:,0],b[:,0]-[.1,.01])
    np.testing.assert_allclose(result[:,1],b[:,1]+[.1,.01])
    np.testing.assert_array_equal(b,before)
    x=np.ones((10,4));noisy=noisy_currents(x,np.array([1.,2.]),9050)
    np.testing.assert_array_equal(x,np.ones((10,4)))
    np.testing.assert_array_equal(noisy[:,[0,3]],x[:,[0,3]])
    np.testing.assert_array_equal(noisy,noisy_currents(x,np.array([1.,2.]),9050))
