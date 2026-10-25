"""Score-free finer spatial optimizer coordinates in the existing linear head."""
import numpy as np
from policy_search_math import spatial_basis


def extra_transform(raw_calibration,layout,dimension,*,old_frequencies=3,frequencies=5,ridge=.01):
    assert 1<=old_frequencies<frequencies<=8 and ridge>0
    basis,bias=spatial_basis(layout,dimension,frequencies)
    selected=[];offset=0
    for name,(start,end) in layout.items():
        if name in ('current','future','relations','context'):
            side={'current':16,'future':16,'relations':9,'context':8}[name]
            for _ in range((end-start)//(side*side)):
                selected.extend(offset+v*frequencies+h for v in range(frequencies) for h in range(frequencies)
                                if v>=old_frequencies or h>=old_frequencies)
                offset+=frequencies**2
        else:offset+=end-start
    extra=basis[selected];values=np.asarray(raw_calibration@extra.T)
    centered=extra-values.mean(0)[:,None]*basis[bias]
    values=np.asarray(raw_calibration@centered.T);gram=values.T@values/len(values)
    eigenvalues,vectors=np.linalg.eigh(gram);threshold=1e-8*float(eigenvalues.max());supported=eigenvalues>threshold
    whitening=(vectors[:,supported]/np.sqrt(eigenvalues[supported]+ridge))@vectors[:,supported].T
    result=whitening@centered
    return result,dict(frequencies=frequencies,old_frequencies=old_frequencies,ridge=ridge,extra_rows=len(result),
        supported_rank=int(supported.sum()),rank_threshold=threshold,calibration_rows=len(values),
        recipe='new DCT modes only; generic-cache mean subtraction via frame-present bias; supported second-moment ridge whitening')


def map_split(theta,old_transform,extra_transform):
    theta=np.asarray(theta);n=len(old_transform)
    assert theta.shape==(3,n+len(extra_transform))
    return theta[:,:n]@old_transform+theta[:,n:]@extra_transform
