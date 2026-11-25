from scripts.experimental.tt56_central_training import changed, choose, deep


def test_held_selection_uses_actual_camera_deviation_not_requested_shift():
    origin={'optical_x':0.,'optical_z':1.012}
    scouts=[dict(seed=4,target=[0,0,0,0],visible_deep=[.012,1.001,3.66,.4]),
            dict(seed=2,target=[.11,0,0,0],visible_deep=[.012,1.001,3.66,.4]),
            dict(seed=5,target=[0,0,0,0],visible_deep=[-.04,1.011,3.83,.6]),
            dict(seed=8,target=[0,0,0,0],visible_deep=[.136,1.001,3.66,.4])]
    assert [r['seed'] for r in choose(scouts,origin,2)]==[4,5]


def test_random_goal_seed_override_preserves_the_prior_role():
    c={'seed':12,'reply_seed':98,'id':9}
    assert changed(c,0,77)==dict(c,seed=77)
    assert changed(c,1,77)==dict(c,reply_seed=77)


def test_downcourt_projection_has_correct_velocity_sign():
    p=deep([.014,-1.10,.990,.20,-4.0,.21])
    assert p[0]>.014 and p[1]<.990 and p[2]==4.
