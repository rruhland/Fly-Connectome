"""Generic spatial coherence prior within the existing raw visual grids.

No new policy inputs or nonlinear inference. The prior penalizes neighboring
residual coefficients differing within each current/future/relation/context
channel. Metadata/outside entries have no invented spatial adjacency.
"""
import numpy as np
import scipy.sparse as sp


def spatial_laplacian():
    i=[];j=[]
    for offset,channels,side in ((0,7,16),(1792,8,16),(3848,5,9),(4253,2,8)):
        for channel in range(channels):
            start=offset+channel*side*side
            for y in range(side):
                for x in range(side):
                    a=start+y*side+x
                    if x+1<side:i.append(a);j.append(a+1)
                    if y+1<side:i.append(a);j.append(a+side)
    i=np.array(i);j=np.array(j);values=np.full(len(i),.25)
    return sp.coo_matrix((np.r_[values,values,-values,-values],(np.r_[i,j,i,j],np.r_[i,j,j,i])),shape=(4389,4389)).tocsc()


def parameter_metric(feature_scale,input_rms):
    scale=np.asarray(feature_scale,dtype=np.float64);assert scale.shape==(4389,) and (scale>0).all()
    inverse=sp.diags(1/scale,format='csc');single=sp.eye(4389,format='csc')+input_rms**2*(inverse@spatial_laplacian()@inverse)
    return sp.block_diag((single,single),format='csc')
