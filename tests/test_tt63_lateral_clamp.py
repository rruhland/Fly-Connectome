import numpy as np

from scripts.experimental import tt63_lateral_clamp as lateral


def test_full_clamp_and_base_slide_together_without_moving_fixed_cameras_or_net():
    tracked=('learner_base','tt20_table_end_clamp','tt29_partner_carriage')
    models=[]
    for dx in (0.,.12,0.):
        with lateral.slid(dx):
            a=lateral.BASE.arena(width=64,height=48)
            try:
                m=a.model
                models.append(([m.body(k).pos.copy() for k in tracked],m.cam_pos.copy(),m.geom('net').pos.copy()))
            finally:a.close()
    p,observed,restored=models
    np.testing.assert_allclose(np.array(observed[0])-p[0],[[.12,0,0],[.12,0,0],[0,0,0]],atol=1e-9)
    np.testing.assert_array_equal(observed[1],p[1]);np.testing.assert_array_equal(observed[2],p[2])
    np.testing.assert_array_equal(restored[0],p[0])


def test_lateral_selection_does_not_use_best_low_existence_at_cost_of_ordinary_coverage():
    d=dict(rows=[dict(dx=0.,central=3,ordinary_covered=25,all_interior=27),
                 dict(dx=.12,central=13,ordinary_covered=24,all_interior=38),
                 dict(dx=.18,central=20,ordinary_covered=20,all_interior=45)])
    assert lateral.decision(d)==[0.,.12]


def test_stock_local_readout_gets_offset_and_world_measurement_isnt_mutated():
    class Raw:
        def get(self,x,q):return x,q
    x=np.array([.08,-.70,.95,-.1,-4.,1.])
    local,encoder=lateral.BodyRelativeRaw(Raw(),.12).get(x,[0]*6)
    np.testing.assert_allclose(local,[x[0]-.12,*x[1:]])
    assert x[0]==.08
    assert encoder==[0]*6
