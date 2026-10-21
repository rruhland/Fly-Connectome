"""Score-free smooth linear policy coordinates and mirrored episodic search.

No oracle inputs, automatic differentiation, critic, or reward-derived transform.
"""
from dataclasses import dataclass
import copy
import numpy as np

SEARCH_CONFIG=dict(version='dct3-centered-supported-ridge-v1',frequencies=3,
                   ridge=.01,rank_relative_tolerance=1e-8,directions=8,top=4,
                   noise_scale=.25,step_size=.10,initial_scale=.10)


def spatial_basis(layout,dimension,frequencies=3):
    rows=[];bias=None
    for name,(start,end) in layout.items():
        size=end-start
        if name in ('current','future','relations','context'):
            side={'current':16,'future':16,'relations':9,'context':8}[name]
            assert size%(side*side)==0 and 1<=frequencies<=side
            axis=np.arange(side)+.5
            dct=np.cos(np.pi*np.arange(frequencies)[:,None]*axis[None,:]/side)*np.sqrt(2/side)
            dct[0]/=np.sqrt(2)
            for plane in range(size//(side*side)):
                for vertical in range(frequencies):
                    for horizontal in range(frequencies):
                        row=np.zeros(dimension)
                        offset=start+plane*side*side
                        row[offset:offset+side*side]=np.outer(dct[vertical],dct[horizontal]).ravel()
                        rows.append(row)
        else:
            for coordinate in range(start,end):
                row=np.zeros(dimension);row[coordinate]=1.
                if name=='metadata' and coordinate==start+3:bias=len(rows)
                rows.append(row)
    assert bias is not None
    return np.array(rows),bias


@dataclass
class SearchTransform:
    matrix: np.ndarray
    diagnostics: dict

    def mapped_weights(self,theta):
        theta=np.asarray(theta,dtype=np.float64)
        if theta.shape!=(3,self.matrix.shape[0]) or not np.isfinite(theta).all():
            raise ValueError('finite matching three-head search parameters required')
        result=theta@self.matrix
        if not np.isfinite(result).all():raise FloatingPointError('nonfinite mapped policy')
        return result


def fit_transform(raw_calibration,layout,dimension):
    basis,bias=spatial_basis(layout,dimension)
    projected=np.asarray(raw_calibration@basis.T)
    present=projected[:,bias]
    if not np.array_equal(present,np.ones(len(present))):raise ValueError('calibration frame-present bias must be one')
    means=projected.mean(axis=0);means[bias]=0.
    centered=basis-means[:,None]*basis[bias][None,:]
    values=np.asarray(raw_calibration@centered.T)
    # SECOND MOMENTS, not np.cov: retain the explicit frame-present bias.
    gram=values.T@values/len(values)
    eigenvalues,eigenvectors=np.linalg.eigh(gram)
    tolerance=SEARCH_CONFIG['rank_relative_tolerance']*float(eigenvalues.max())
    supported=eigenvalues>tolerance
    vectors=eigenvectors[:,supported]
    whitening=(vectors/np.sqrt(eigenvalues[supported]+SEARCH_CONFIG['ridge']))@vectors.T
    matrix=whitening@centered
    transformed=np.asarray(raw_calibration@matrix.T)
    info=dict(config=dict(SEARCH_CONFIG),calibration_rows=len(values),basis_dimension=len(basis),
        supported_rank=int(supported.sum()),free_action_contrasts=2*int(supported.sum()),
        retained_frame_bias_basis_index=bias,raw_frame_bias_coordinate=layout['metadata'][0]+3,
        eigenvalues=eigenvalues.tolist(),rank_threshold=tolerance,
        maximum_whitening_gain=float(1/np.sqrt(eigenvalues[supported].min()+SEARCH_CONFIG['ridge'])),
        calibration_feature_rms=float(np.sqrt(np.mean(transformed**2))),
        calibration_energy_quantiles=np.quantile(np.sum(transformed**2,axis=1),[0,.5,.99,1]).tolist(),
        projected_bias_energy=float(gram[bias,bias]),
        missingness='all-zero maps to zero; frame-present mean subtraction can fill individually missing coordinates',
        policy_class='linear low-frequency214-coordinate subspace, further restricted to score-free calibration-supported modes')
    return SearchTransform(matrix,info),transformed


def function_rms(parameters,calibration):
    return float(np.sqrt(np.mean((calibration@parameters.T)**2)))


def sample_directions(rng,calibration,count=8):
    directions=[]
    for _ in range(count):
        delta=rng.standard_normal((3,calibration.shape[1]));delta-=delta.mean(axis=0,keepdims=True)
        norm=function_rms(delta,calibration)
        if not np.isfinite(norm) or norm<=0:raise ValueError('unsupported perturbation')
        directions.append(delta/norm)
    return np.array(directions)


def mirrored_update(theta,directions,returns,calibration,*,top=4,step_size=.10):
    returns=np.asarray(returns,dtype=np.float64)
    if returns.shape!=(len(directions),2) or not np.isfinite(returns).all():
        raise ValueError('a full finite mirrored generation is required')
    if not 1<=top<=len(directions):raise ValueError('valid top direction count required')
    # Stable index order breaks reward ties; no opportunistic selection.
    selected=sorted(range(len(directions)),key=lambda i:(-float(returns[i].max()),i))[:top]
    spread=float(np.std(returns[selected]))
    update=np.zeros_like(theta)
    if spread>0:
        update=step_size/(top*spread)*np.sum((returns[selected,0]-returns[selected,1])[:,None,None]*directions[selected],axis=0)
        update-=update.mean(axis=0,keepdims=True)
        norm=function_rms(update,calibration)
        if norm>step_size:update*=step_size/norm
    proposed=theta+update
    if not np.isfinite(proposed).all():raise FloatingPointError('nonfinite policy-search parameters')
    return proposed,dict(selected_directions=selected,selected_return_std=spread,
        update_function_rms=function_rms(update,calibration),zero_update=bool(np.array_equal(theta,proposed)))


def generation_spec(theta,rng_state,generation,calibration,*,training_seed=172900127,
                    serves_per_generation=4,excluded_seeds=()):
    if type(serves_per_generation) is not int or serves_per_generation<1:
        raise ValueError('positive integer training serve count required')
    rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(rng_state)
    directions=sample_directions(rng,calibration,SEARCH_CONFIG['directions'])
    candidates=theta[None,None,:,:]+SEARCH_CONFIG['noise_scale']*directions[:,None,:,:]*np.array([1.,-1.])[None,:,None,None]
    # Independent declared training seeds; each is a fresh environment/camera phase.
    serve_rng=np.random.default_rng(np.random.SeedSequence([training_seed,872631,generation]))
    excluded={904271833,308713321,1203078761,1983461927,849375169}
    excluded.update(excluded_seeds)
    serves=[]
    while len(serves)<serves_per_generation:
        value=int(serve_rng.integers(1,2147483647))
        if value not in excluded and value not in serves:serves.append(value)
    return directions,candidates,serves,copy.deepcopy(rng.bit_generator.state)
