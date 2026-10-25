"""Small action-contrast policy subspace; all candidate reward is measured in games."""
import numpy as np


def build_basis(current,direct,original,layout,raw_calibration):
    directions=[direct-current,original-current];names=['learned_DIRECT_difference','learned_original_RIGHT_difference']
    for name,(start,end) in layout.items():
        d=np.zeros_like(current);d[:,start:end]=current[:,start:end];directions.append(d);names.append('gain_'+name)
    raw=np.asarray(directions);raw-=raw.mean(axis=1,keepdims=True)
    outputs=np.asarray(raw_calibration@raw.reshape(-1,raw.shape[-1]).T).reshape(raw_calibration.shape[0],len(raw),3)
    scales=np.sqrt(np.mean(outputs*outputs,axis=(0,2)));valid=scales>1e-12;safe=np.where(valid,scales,1.)
    normalized=outputs/safe[None,:,None];normalized[:,~valid]=0
    gram=np.einsum('nja,nka->jk',normalized,normalized)/(3*len(outputs));values,vectors=np.linalg.eigh(gram)
    supported=values>max(1e-12,1e-8*float(values.max()));assert supported.any()
    mapping=(vectors[:,supported]/np.sqrt(values[supported]+.01))/safe[:,None];mapping[~valid]=0
    basis=np.einsum('jk,jai->kai',mapping,raw);basis-=basis.mean(axis=1,keepdims=True)
    return basis,mapping,dict(names=names,generic_direction_rms=scales.tolist(),supported_dimension=int(supported.sum()),eigenvalues=values.tolist(),
        ridge=.01,common_action_logit_mode_removed=True,raw_coefficient_bounds=[-1.,1.])


def map_weights(base,basis,coefficients):
    coefficients=np.asarray(coefficients)
    if not np.any(coefficients):return base.copy()
    return base+np.einsum('k,kai->ai',coefficients,basis)


def legal_rms(outputs,weights,step):
    q=np.einsum('nka,k->na',outputs,step);q-=q.mean(1,keepdims=True)
    return float(np.sqrt(np.sum(weights[:,None]*q*q)/3))


def population(rng,incumbent,mapping,outputs,weights,base_logits,count=16,dose=.15):
    assert count%2==0;result=[];decoded=mapping@incumbent;q0=base_logits+np.einsum('nka,k->na',outputs,incumbent)
    for direction in range(count//2):
        for attempt in range(100):
            vector=rng.standard_normal(len(incumbent));norm=legal_rms(outputs,weights,vector)
            if norm>1e-12:break
        else:raise ValueError('no supported legal policy direction')
        for sign in (1.,-1.):
            step=sign*dose*vector/norm;raw_step=mapping@step;limits=np.full(len(raw_step),np.inf)
            pos=raw_step>0;neg=raw_step<0;limits[pos]=(1-decoded[pos])/raw_step[pos];limits[neg]=(-1-decoded[neg])/raw_step[neg]
            alpha=max(0.,min(1.,float(limits.min())));coefficients=incumbent+alpha*step;new=q0+np.einsum('nka,k->na',outputs,alpha*step)
            result.append(dict(coefficients=coefficients,raw_coefficients=mapping@coefficients,direction=direction,sign=sign,box_alpha=alpha,
                legal_output_rms=legal_rms(outputs,weights,alpha*step),legal_action_disagreement=float(weights@(np.argmax(new,1)!=np.argmax(q0,1)))))
    return result


def acceptable(score,incumbent,anchor,maximum_regression=.03):
    return score['key'][0]>incumbent['key'][0]+1e-15 and all(score['rates'][p]>=anchor['rates'][p]-maximum_regression-1e-15 for p in ('left','right'))
