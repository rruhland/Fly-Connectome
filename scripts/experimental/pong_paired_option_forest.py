"""Direct paired own-reward option advantages with anonymous whole-scene features."""
import numpy as np
from fly_connectome.control.scaling import policy_features
from pong_continuous_visual import canonical_scene
from pong_fixed_rbf import FreshRBFEvaluator

SCHEMA='horizontal-anonymous-paired-option-forest-v1'
ORDER=(1,0,2,3,4,5)  # original rows retain [y,x,vy,vx,observed,velocity_known]
HOLD=16


def horizontal_scene(scene):
    rows=np.asarray(scene,dtype=np.float64).reshape(-1,6)
    assert np.isfinite(rows).all() and np.isin(rows[:,4:],[0,1]).all()
    return np.asarray(sorted(rows.tolist(),key=lambda r:tuple(r[i] for i in ORDER)),dtype=np.float64).reshape(-1,6)


def features(scene,baseline_q,capacity):
    rows=horizontal_scene(scene);q=np.asarray(baseline_q,dtype=np.float64)
    assert q.shape==(3,) and np.isfinite(q).all()
    if len(rows)>capacity:return None
    padded=np.zeros((capacity,6));padded[:len(rows)]=rows;present=np.zeros(capacity);present[:len(rows)]=1.
    return np.r_[padded.ravel(),present,float(len(rows)),q]


class OptionForest:
    requires_stateful_reward_history=True
    def __init__(self,capacity,feature,threshold,left,right,value):
        self.capacity=int(capacity)
        self.feature=np.array(feature,dtype=np.int32,copy=True);self.threshold=np.array(threshold,dtype=np.float64,copy=True)
        self.left=np.array(left,dtype=np.int32,copy=True);self.right=np.array(right,dtype=np.int32,copy=True);self.value=np.array(value,dtype=np.float64,copy=True)
        assert self.feature.shape==self.threshold.shape==self.left.shape==self.right.shape and self.feature.shape[0]==64
        assert self.value.shape==(*self.feature.shape,3) and np.isfinite(self.value).all()
        for a in (self.feature,self.threshold,self.left,self.right,self.value):a.flags.writeable=False

    @classmethod
    def export(cls,regressor,capacity):
        size=max(t.tree_.node_count for t in regressor.estimators_);shape=(64,size)
        feature=np.full(shape,-2,dtype=np.int32);threshold=np.zeros(shape);left=np.full(shape,-1,dtype=np.int32);right=left.copy();value=np.zeros((*shape,3))
        for i,t in enumerate(regressor.estimators_):
            n=t.tree_.node_count;feature[i,:n]=t.tree_.feature;threshold[i,:n]=t.tree_.threshold;left[i,:n]=t.tree_.children_left;right[i,:n]=t.tree_.children_right;value[i,:n]=t.tree_.value[:,:,0]
        return cls(capacity,feature,threshold,left,right,value)

    def predict(self,F):
        # sklearn trees first cast input to float32, then compare with double thresholds.
        F=np.asarray(F,dtype=np.float32).reshape(-1,7*self.capacity+4);nodes=np.zeros((len(F),64),dtype=np.int32);trees=np.arange(64)[None,:]
        for _ in range(6):
            coordinate=self.feature[trees,nodes];active=coordinate>=0
            if not active.any():break
            rows,cols=np.where(active);cur=nodes[rows,cols];right=F[rows,coordinate[rows,cols]]>self.threshold[cols,cur]
            nodes[rows,cols]=np.where(right,self.right[cols,cur],self.left[cols,cur])
        assert (self.feature[trees,nodes]<0).all()
        return self.value[trees,nodes].cumsum(axis=1)[:,-1]/64.

    def inspect(self,scene,q):
        f=features(scene,q,self.capacity)
        if f is None:return dict(option=3,advantages=[0.,0.,0.],representation_supported=False,entity_count=len(scene))
        advantage=self.predict(f)[0];option=int(np.argmax(advantage)) if advantage.max()>0 else 3
        return dict(option=option,advantages=advantage.tolist(),representation_supported=True,entity_count=len(scene))

    def frozen_parameters(self):return SCHEMA,self.capacity,ORDER,HOLD,self.feature,self.threshold,self.left,self.right,self.value


def save_forest(path,forest):
    np.savez_compressed(path,kind=SCHEMA,capacity=forest.capacity,order=ORDER,hold=HOLD,feature=forest.feature,threshold=forest.threshold,left=forest.left,right=forest.right,value=forest.value)


def load_forest(path):
    with np.load(path,allow_pickle=False) as f:
        assert str(f['kind'])==SCHEMA and tuple(f['order'])==ORDER and int(f['hold'])==HOLD
        return OptionForest(int(f['capacity']),*[f[n] for n in ('feature','threshold','left','right','value')])


class ForestController:
    def __init__(self,forest):self.forest=forest;self.earned=False;self.remaining=0;self.held_action=None
    def observe_reward(self,hits):self.earned=self.earned or bool(hits)
    def choose(self,scene,baseline_action,baseline_q):
        before=self.remaining
        if not self.earned:
            action=int(baseline_action);d=dict(option=3,mode='baseline_before_hit')
        elif before:
            action=self.held_action;self.remaining-=1;d=dict(option=action,mode='earned_hold')
        else:
            d=self.forest.inspect(scene,baseline_q);action=int(baseline_action) if d['option']==3 else d['option'];self.held_action=None if d['option']==3 else action;self.remaining=0 if d['option']==3 else HOLD-1;d['mode']='baseline_camera' if d['option']==3 else 'earned_start'
        d.update(action=int(action),baseline_action=int(baseline_action),past_hit=self.earned,hold_before=before,hold_after=self.remaining)
        return int(action),d


class ForestEvaluator(FreshRBFEvaluator):
    def __init__(self,original,forest,config):
        self.option_forest=forest;self.controller=ForestController(forest);self.option_choices=[];super().__init__(original,config);self.initial_learned_hash=self.learned_hash()
    def learned_hash(self):
        from fly_connectome.control.durable import fingerprint
        return fingerprint((super().learned_hash(),self.option_forest.frozen_parameters()))
    def _begin_episode(self):self.controller=ForestController(self.option_forest);super()._begin_episode()
    def _choose_policy(self):
        baseline,q=self.rbf_readout.choose(*policy_features(self.latest_encoded,self.scaler));action,choice=self.controller.choose(canonical_scene(self.latest_encoded),baseline,q);self.option_choices.append(choice);return action,q
    def step(self,**kwargs):
        r=super().step(**kwargs);self.controller.observe_reward(r['hits']);return r


def window_end(record,length,cap=160):
    if record['hits']:return 'own_hit',True,1.
    if record['terminated']:return 'natural_termination',True,0.
    if length>=cap:return 'full_window_no_hit',True,0.
    if record['truncated']:return 'environment_timeout_censored',False,None
    return None
