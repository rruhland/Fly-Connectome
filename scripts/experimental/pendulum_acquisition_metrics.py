"""Paired frozen survival and retained-Pong effects; no model selection."""
import numpy as np

CONTROLS=('negative','noop','positive','random1','random10')


def contrast(a,b,index):
    delta=a[index].mean(1)-b[index].mean(1)
    return dict(point=float(a.mean()-b.mean()),ci95=np.quantile(delta,[.025,.975]).tolist())


def survival_evidence(groups,seed,draws=10000):
    names=('experienced_endpoint','fresh_endpoint',*CONTROLS)
    if any(len(groups.get(name,[]))!=64 for name in names):return dict(complete=False)
    reference=[r['seed'] for r in groups[names[0]]]
    for name in names:
        if ([r['serve_index'] for r in groups[name]]!=list(range(64))
                or [r['seed'] for r in groups[name]]!=reference):
            raise ValueError('survival bootstrap requires explicit identical cohort ordering')
    index=np.random.default_rng(seed).integers(64,size=(draws,64))
    values={name:np.array([r['survival_seconds'] for r in groups[name]]) for name in names}
    control=np.stack([values[name] for name in CONTROLS]);strongest=float(control.mean(1).max())
    sampled_max=control[:,index].mean(2).max(0);report={}
    for name in names[:2]:
        v=values[name];d=v[index].mean(1)-sampled_max
        mean=float(v.mean());cap=float(np.mean([r['time_limit'] for r in groups[name]]))
        lower,upper=np.quantile(d,[.025,.975]).tolist();lift=mean-strongest
        report[name]=dict(mean_survival_seconds=mean,time_limit_fraction=cap,
            mean_lift_over_strongest_control_seconds=lift,ci95=[lower,upper],
            competence_screen=bool(mean>=3 and cap>=.5 and lift>=1 and lower>0),
            own_initial_control='negative: zero-Q greedy tie is action0; physics/action identity')
    curves={}
    for arm in ('experienced','fresh'):
        curves[arm]={}
        for dose in (5000,10000):
            name=f'{arm}_at{dose}'
            if len(groups.get(name,[]))==32:
                a=np.array([r['survival_seconds'] for r in groups[name]])
                curves[arm][str(dose)]=dict(mean_survival_seconds=float(a.mean()),
                    time_limit_fraction=float(np.mean([r['time_limit'] for r in groups[name]])))
    return dict(complete=True,serve_clusters=64,bootstrap_seed=seed,bootstrap_draws=draws,
        arms=report,controls={n:dict(mean_survival_seconds=float(values[n].mean()),
            time_limit_fraction=float(np.mean([r['time_limit'] for r in groups[n]]))) for n in CONTROLS},
        experienced_minus_fresh=contrast(values[names[0]],values[names[1]],index),
        fixed_midpoint_curves_descriptive=curves,
        scope='One paired training seed, fixed two visual starting conditions; no population inference')


def retention_evidence(before,after,seed,draws=10000):
    """Shared reused development serves, with A policy fixed and no A retraining."""
    if len(before)!=64 or len(after)!=64:return dict(complete=False)
    for a,b in zip(before,after):
        a=a['first_incoming'];b=b['first_incoming']
        sig=lambda x:None if x is None else {k:v for k,v in x.items() if k!='success'}
        if sig(a)!=sig(b):raise ValueError('A retention first-encounter geometry changed')
    index=np.random.default_rng(seed).integers(64,size=(draws,64));points={};sampled={}
    for name,data in [('before',before),('after',after)]:
        first=np.array([bool(r['first_incoming'] and r['first_incoming']['success']) for r in data])
        present=np.array([r['first_incoming'] is not None for r in data])
        hits=np.array([r['own_returns'] for r in data]);opps=np.array([r['own_opportunities'] for r in data])
        points[name]=dict(first=float(first.sum()/present.sum()),all=float(hits.sum()/opps.sum()))
        with np.errstate(divide='ignore',invalid='ignore'):
            sampled[name]=dict(first=first[index].sum(1)/present[index].sum(1),all=hits[index].sum(1)/opps[index].sum(1))
    result={}
    for metric in ('first','all'):
        delta=sampled['after'][metric]-sampled['before'][metric];valid=np.isfinite(delta)
        result[metric]=dict(point=points['after'][metric]-points['before'][metric],
            ci95=np.quantile(delta[valid],[.025,.975]).tolist() if valid.any() else None,
            invalid_bootstrap_draws=int((~valid).sum()))
    return dict(complete=True,before=points['before'],after=points['after'],effects=result,
        serve_clusters=64,bootstrap_seed=seed,bootstrap_draws=draws,
        scope='Reused online-drift development cohort; no formal noninferiority criterion')
