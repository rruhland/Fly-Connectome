import numpy as np

from scripts.experimental.tt38_contact_id import Correction, HYPOTHESES, outcome_cost


def test_observed_signed_touch_and_plain_visual_miss_differ():
    feedback = np.zeros((3,9))
    feedback[0,0] = .20
    feedback[1:,1] = 1.
    feedback[1,2:5] = [0., 2.1, -1.]
    feedback[2,2:5] = [0., -1.1, 1.]
    actual = dict(outgoing=[0., -1.1, 1.], center=[0.,0.,0.])
    assert outcome_cost(feedback,actual,None).argmin() == 2
    assert outcome_cost(feedback,None,.20).argmin() == 0


def test_local_selector_abstains_without_sensor_support():
    x=np.zeros(15)
    loss=np.full(len(HYPOTHESES),2.)
    loss[4]=.1
    selector=Correction([dict(context=x,loss=loss)])
    assert selector.choose(x)==4
    x[0]+=4.
    assert selector.choose(x)==0
