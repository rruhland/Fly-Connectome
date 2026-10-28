"""Fixed anonymous continuous entity/pair RFF; frozen original40069 score path."""
import numpy as np
from fly_connectome.control.scaling import policy_features
from pong_fixed_pairwise import FixedPairwise,FrozenPairwiseReadout

SCHEMA='fixed-continuous-entity-pair-rff-v1'


def canonical_scene(encoded):
    rows=[]
    for e in encoded['entities']:
        p=np.asarray(e['normalized_position'],dtype=float);v=np.asarray(e['observed_velocity'],dtype=float);assert p.shape==v.shape==(2,);rows.append([*p,*v,float(e['observed']),float(e['velocity_known'])])
    X=np.asarray(rows,dtype=float).reshape(-1,6);assert np.isfinite(X).all() and np.isin(X[:,4:],[0,1]).all();return X[np.lexsort(X[:,::-1].T)] if len(X) else X


def descriptors(scene):
    X=np.asarray(scene,dtype=float).reshape(-1,6);X=X[np.lexsort(X[:,::-1].T)] if len(X) else X
    n=len(X)
    if n<2:return X,np.empty((0,12))
    i,j=np.where(~np.eye(n,dtype=bool));pair=np.column_stack((X[i,:4],X[j,:4]-X[i,:4],X[i,4:],X[j,4:]));return X,pair[np.lexsort(pair[:,::-1].T)]


def freeze(a):
    a=np.asarray(a,dtype=float).copy();assert np.isfinite(a).all();a.flags.writeable=False;return a


class ContinuousBasis:
    def __init__(self,matrices,phases,input_means,input_scales,mean,rms):
        self.matrices=tuple(freeze(a) for a in matrices);self.phases=tuple(freeze(a) for a in phases);self.input_means=tuple(freeze(a) for a in input_means);self.input_scales=tuple(freeze(a) for a in input_scales);self.mean=freeze(mean);self.rms=freeze(rms);self.output_dim=256
        for j,d in enumerate((6,12)):assert self.matrices[j].shape==(128,d) and self.phases[j].shape==(128,) and self.input_means[j].shape==self.input_scales[j].shape==(d,) and (self.input_scales[j]>0).all()
        assert self.mean.shape==self.rms.shape==(256,) and (self.rms>0).all()

    def pooled(self,scene):
        blocks=descriptors(scene);out=[];available=[]
        for j,X in enumerate(blocks):
            total=np.zeros(128)
            for start in range(0,len(X),1024):
                z=(X[start:start+1024]-self.input_means[j])/self.input_scales[j];total+=np.cos(z@self.matrices[j].T+self.phases[j]).sum(0)
            out.extend(np.sqrt(2/128)*total/len(X) if len(X) else total);available.extend([bool(len(X))]*128)
        return np.asarray(out),np.asarray(available)

    def features(self,scene):
        pooled,available=self.pooled(scene);return np.where(available,(pooled-self.mean)/self.rms,0.)

    def parameters(self):return self.matrices,self.phases,self.input_means,self.input_scales,self.mean,self.rms


def fit_basis(episodes):
    scenes=[s for e in episodes for s in e];weights=np.concatenate([np.full(len(e),1/len(episodes)/len(e)) for e in episodes]);families=[descriptors(s) for s in scenes];means=[];scales=[];report={}
    for j,d in enumerate((6,12)):
        k=4 if j==0 else 8;mass=0.;mean=np.zeros(k);second=np.zeros(k)
        for w,blocks in zip(weights,families):
            X=blocks[j]
            if len(X):mass+=w;mean+=w*X[:,:k].mean(0);second+=w*np.mean(X[:,:k]**2,axis=0)
        assert mass>0;mean/=mass;sd=np.sqrt(np.maximum(0,second/mass-mean*mean));positive=sd[sd>0];assert len(positive),'constant continuous descriptor family';floor=.1*float(np.median(positive));means.append(np.r_[mean,np.zeros(d-k)]);scales.append(np.r_[np.maximum(sd,floor),np.ones(d-k)]);report[str(j)]=dict(standard_deviations=sd.tolist(),floor=floor,floored=int(np.count_nonzero(sd<floor)))
    rng=np.random.default_rng(831496043);matrices=[rng.normal(size=(128,d))/np.sqrt(d) for d in (6,12)];phases=[rng.uniform(0,2*np.pi,128) for _ in range(2)];basis=ContinuousBasis(matrices,phases,means,scales,np.zeros(256),np.ones(256));pooled=[];available=[]
    for scene in scenes:
        p,a=basis.pooled(scene);pooled.append(p);available.append(a)
    pooled=np.asarray(pooled);available=np.asarray(available);mass=weights@available;mu=np.divide(weights@pooled,mass,out=np.zeros(256),where=mass>0);centered=np.where(available,pooled-mu,0.);rms=np.sqrt(weights@(centered**2));positive=rms[rms>0];assert len(positive),'constant pooled features';floor=.1*float(np.median(positive));basis=ContinuousBasis(matrices,phases,means,scales,mu,np.maximum(rms,floor));F=centered/basis.rms;G=F.T@(weights[:,None]*F);diagonal=float(np.trace(G)/256);assert diagonal>0;G/=diagonal;G.flat[::257]+=.01;centered_F=F-weights@F;eigen=np.linalg.eigvalsh(centered_F.T@(weights[:,None]*centered_F));rank=float(eigen.sum()**2/np.sum(eigen*eigen)) if np.any(eigen) else 0
    report.update(rows=len(scenes),episodes=len(episodes),feature_RMS_min=float(rms.min()),feature_RMS_max=float(rms.max()),feature_RMS_floor=floor,feature_RMS_floored=int(np.count_nonzero(rms<floor)),centered_effective_rank=rank,metric_mean_diagonal_before_normalization=diagonal,floor_fraction=.1,entity_count_max=max(map(len,scenes)),ordered_pair_count_max=max(len(s)*(len(s)-1) for s in scenes));return basis,F,G,weights,report


class ContinuousReadout:
    schema=SCHEMA
    def __init__(self,original,basis,output):
        self.original=original;self.base=original.base;self.basis=original.basis;self.continuous=basis;self.output=freeze(output);assert self.output.shape==(3,256);self.requires_structured=bool(np.any(self.output))

    def choose(self,indices,values,legal_actions=None):
        if self.requires_structured:raise ValueError('continuous residual requires exact structured visual observations')
        return self.original.choose(indices,values,legal_actions)

    def choose_features(self,indices,values,features,legal_actions=None):
        action,q=self.original.choose(indices,values,legal_actions)
        if not self.requires_structured:return action,q
        q=q+self.output@features;legal=np.arange(3) if legal_actions is None else np.asarray(legal_actions);assert np.isfinite(q).all();return int(legal[np.argmax(q[legal])]),q

    def choose_encoded(self,encoded,scaler):return self.choose_features(*policy_features(encoded,scaler),self.continuous.features(canonical_scene(encoded))) if self.requires_structured else self.original.choose(*policy_features(encoded,scaler))
    def frozen_parameters(self):return SCHEMA,self.original.frozen_parameters(),self.continuous.parameters(),self.output


def save_readout(path,readout):
    b=readout.original.basis;c=readout.continuous
    with open(path,'wb') as stream:np.savez(stream,kind=SCHEMA,base=readout.base.weights,hashes=b.hashes,signs=b.signs,scale=b.scale,mean=b.mean,rms=b.rms,residual=readout.original.residual,entity_A=c.matrices[0],pair_A=c.matrices[1],entity_phase=c.phases[0],pair_phase=c.phases[1],entity_mean=c.input_means[0],pair_mean=c.input_means[1],entity_scale=c.input_scales[0],pair_scale=c.input_scales[1],continuous_mean=c.mean,continuous_rms=c.rms,continuous_output=readout.output)


def from_archive(f):
    original=FrozenPairwiseReadout(f['base'],FixedPairwise(f['hashes'],f['signs'],f['scale'],f['mean'],f['rms']),f['residual']);basis=ContinuousBasis([f['entity_A'],f['pair_A']],[f['entity_phase'],f['pair_phase']],[f['entity_mean'],f['pair_mean']],[f['entity_scale'],f['pair_scale']],f['continuous_mean'],f['continuous_rms']);return ContinuousReadout(original,basis,f['continuous_output'])


def save_scenes(path,scenes):
    lengths=np.array([len(s) for s in scenes]);np.savez_compressed(path,entities=np.concatenate(scenes) if scenes else np.empty((0,6)),indptr=np.r_[0,np.cumsum(lengths)])


def load_scenes(path):
    with np.load(path) as f:return [f['entities'][a:b].copy() for a,b in zip(f['indptr'][:-1],f['indptr'][1:])]
