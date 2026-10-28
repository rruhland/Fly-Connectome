"""Forward-dose parameter proposals and exact earned-prefix projection; no BP."""
import time
import numpy as np
import scipy.sparse as sp
import osqp
from pong_constrained_residual import CONTRAST
from pong_fixed_pairwise import FrozenPairwiseReadout


def crossing_scale(q,actions,direction_scores,weights,dose=.1):
    """One shared +/- scale from weighted first action-crossing thresholds."""
    q=np.asarray(q);a=np.asarray(actions);n=len(a);w=np.asarray(weights,dtype=float);w=w/w.sum();thresholds=[]
    for sign in (1,-1):
        d=sign*direction_scores;gap=q[np.arange(n),a,None]-q;slope=d-d[np.arange(n),a,None];ratio=np.full_like(q,np.inf);np.divide(np.maximum(gap,0),slope,out=ratio,where=slope>0);ratio[np.arange(n),a]=np.inf;thresholds.append(np.min(ratio,axis=1))
    radii=np.concatenate(thresholds);ww=np.tile(w/2,2);order=np.argsort(radii,kind='stable');i=np.searchsorted(np.cumsum(ww[order]),dose);radius=float(radii[order[min(i,len(order)-1)]]);scale=float(np.nextafter(radius,np.inf))
    if not np.isfinite(scale) or scale<=0:raise ValueError('degenerate fixed forward-dose direction; no scale retry')
    return scale


class GuardProjector:
    def __init__(self,carried,X,actions,q,G,kappa):
        self.carried=carried;self.X=sp.csr_matrix(X);self.actions=np.asarray(actions);self.G=np.asarray(G);self.m=carried.basis.output_dim;self.P=sp.block_diag((2*sp.csc_matrix(G),2*sp.csc_matrix(G)),format='csc');self.active=[];self.last=np.zeros(2*self.m);self.duals={};q=np.asarray(q);other=np.array([[b for b in range(3) if b!=a] for a in self.actions]);row=np.arange(len(self.actions));old=q[row,self.actions,None]-q[row[:,None],other];self.required=np.minimum(kappa*1e-4,.5*np.maximum(old,0));Phi=carried.basis.batch(X);contrast=CONTRAST[self.actions,None,:]-CONTRAST[other];v=np.repeat(Phi,2,axis=0);direction=contrast.reshape(-1,2);self.A=np.hstack((v*direction[:,0,None],v*direction[:,1,None]));self.lower=(self.required-old).ravel();norm=np.linalg.norm(self.A,axis=1);assert (norm>0).all();self.An=self.A/norm[:,None];self.lowern=self.lower/norm
        for i,a in enumerate(self.actions):
            lo,hi=self.X.indptr[i:i+2];got,actual=carried.choose(self.X.indices[lo:hi],self.X.data[lo:hi]);assert got==a and np.array_equal(actual,q[i])

    def project(self,center,deadline):
        began=time.monotonic();center=np.asarray(center).reshape(2,self.m);linear=-2*np.concatenate([self.G@r for r in center]);flat=center.ravel();violation=self.lower-self.A@flat;fresh=np.flatnonzero(violation>1e-9);working=sorted(set(self.active)|set(map(int,fresh[np.argsort(-violation[fresh],kind='stable')[:256]])));x=self.last.copy();duals=dict(self.duals);history=[];result=None;status='bounded_projection_no_verified_proposal'
        for iteration in range(20):
            remaining=deadline-time.monotonic()
            if remaining<=3.5:break
            solver=osqp.OSQP();solver.setup(P=self.P,q=linear,A=sp.csc_matrix(self.An[working]),l=self.lowern[working],u=np.full(len(working),np.inf),verbose=False,polishing=True,eps_abs=1e-8,eps_rel=1e-8,max_iter=20000,time_limit=max(.01,min(remaining-3.5,2.)));solver.warm_start(x=x,y=np.array([duals.get(i,0.) for i in working]));answer=solver.solve(raise_error=False)
            if answer.x is None or not np.isfinite(answer.x).all() or 'infeasible' in answer.info.status:status='unverified_solver_'+answer.info.status;break
            x=np.asarray(answer.x);duals={i:float(v) for i,v in zip(working,answer.y)} if answer.y is not None else {};bad=self.lower-self.A@x;maximum=max(0.,float(bad.max()));history.append(dict(round=iteration+1,working=len(working),solver_status=answer.info.status,max_original_violation=maximum))
            if maximum<=1e-9:
                R=self.carried.residual+CONTRAST@x.reshape(2,self.m);policy=FrozenPairwiseReadout(self.carried.base.weights,self.carried.basis,R);wrong=[];maximum_actual=0.;checked=0
                for i,a in enumerate(self.actions):
                    if i%128==0 and time.monotonic()>=deadline:status='certificate_budget_partial';break
                    lo,hi=self.X.indptr[i:i+2];got,q=policy.choose(self.X.indices[lo:hi],self.X.data[lo:hi]);maximum_actual=max(maximum_actual,float(np.max(self.required[i]-np.array([q[a]-q[b] for b in range(3) if b!=a]))));checked+=1
                    if got!=a:wrong.append(i)
                if checked==len(self.actions) and not wrong and maximum_actual<=1e-9:result=R;status='verified_exact_deployed_guard_choices'
                elif checked==len(self.actions):status='deployed_guard_certificate_failed'
                break
            existing=set(working);fresh=np.array([i for i in np.flatnonzero(bad>1e-9) if int(i) not in existing],dtype=int)
            if not len(fresh):status='bounded_numerical_projection_residual';break
            working=sorted(existing|set(map(int,fresh[np.argsort(-bad[fresh],kind='stable')[:256]])))
        if result is not None:self.active=[i for i in working if abs(duals.get(i,0))>1e-9];self.last=x.copy();self.duals={i:duals[i] for i in self.active}
        return result,dict(status=status,verified=result is not None,wall_seconds=time.monotonic()-began,history=history,original_guard_rows=len(self.actions),working_inequalities=len(working),original_deployed_rows_certified=checked if result is not None else 0,maximum_actual_logit_violation=maximum_actual if result is not None else None,projection_objective=float(np.sum(((x-flat).reshape(2,self.m)@self.G)*(x-flat).reshape(2,self.m))) if result is not None else None,no_donor_targets=True,no_reward_feedback=True,new_observations=0)
