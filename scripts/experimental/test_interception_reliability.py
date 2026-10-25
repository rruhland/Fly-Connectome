import numpy as np
from run_interception_reliability import paired,interval


def test_paired_counts_denominators_and_absolute_uncertainty():
    r=paired([0,0,1,1],[1,0,1,0],71)
    assert r['episodes']==4 and r['gains']==1 and r['losses']==1 and r['paired_difference']==0
    assert r['original_hits']==r['current_hits']==2 and r['paired_bootstrap95'][0]<=0<=r['paired_bootstrap95'][1]
    a=interval(116,128);assert 0<a[0]<116/128<a[1]<1
    r=paired(np.zeros(128,dtype=int),np.ones(128,dtype=int),72)
    assert r['gains']==128 and r['paired_bootstrap95']==[1.,1.]
