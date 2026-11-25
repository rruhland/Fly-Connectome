import numpy as np

from scripts.experimental import tt81_preparation_play as play


def test_uncertainty_mixes_nearby_earned_preparation_and_abstains_high():
    high = np.array([-.04, 1.14, .07, -3.05, .32, .20])
    low = high.copy(); low[1] = .980
    keys=[dict(id=i,leg=0,key=(low[:5]+[dx,0.,0.,0.,0.]).tolist(),pre=[q]*6,
               landing=[1.]*6,radius=.026) for i,q,dx in [(2,.10,-.015),(3,.22,.018),(17,.92,.023)]]
    data=dict(center=low.tolist(),scale=[.11,.08,.5,.8,.6,.4],weights=[0.]*7,
              spread=.024,keys=keys,strength=.68,actuator_limit=[.4]*5)
    p=play.Policy(data)
    got,metadata=p.get(low,np.zeros(6))
    assert set(metadata['supports'])=={2,3}
    assert .1<float(got[0])<.22
    assert p.get(high,np.zeros(6)) is None
