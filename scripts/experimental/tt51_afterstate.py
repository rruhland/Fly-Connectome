"""Fit and audit the conditional distribution of a genuinely achieved outgoing stroke."""
import argparse
from pathlib import Path

import numpy as np

from . import tt45_tactical_coverage as tactical
from . import tt46_continuous_season as season
from . import tt50_opponent_vision as video
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json

ROOT=Path("artifacts/tt51_afterstate")
INPUTS=Path("artifacts/tt49_sustain/fit.json")
SOURCE=ROOT/"snapshot.json"
LIMIT=np.array([.97,.99,1.02])


def projected(x):
    x=np.asarray(x)
    t=(-1.53-x[...,1])/(.98*x[...,4])
    return np.stack((x[...,2]+t*x[...,5]-4.905*t*t, x[...,0]+t*x[...,3],
                     -x[...,4], x[...,5]-9.81*t),axis=-1)


def episode(r, leg):
    incoming=r["placements"]["delivered"][leg-1]
    if not incoming.get("near_contact") or not incoming["near_landed"]:
        raise ValueError("Receiver was not preceded by an actual legal arm return")
    strike=incoming["near_contact"]
    depart=next(e for e in r["events"] if e.get("collider")=="learner_racket"
                and strike["time"]<e["time"]<strike["time"]+.087)
    p,v=depart["ball"],depart["velocity"]
    shape=tactical.arc_shape(p,v)
    accepted=[s for s in r.get("searches",[]) if s.get("target_errors") and s.get("contact_delay_s") is not None
              and 0<strike["step"]-s["tick"]<9]
    cand=None
    if accepted and abs(accepted[-1]["tick"]+accepted[-1]["contact_delay_s"]*60.-strike["step"])<4.1:
        val=accepted[-1]["target_errors"]
        if len(val)==7 and np.isfinite(val).all():
            cand=val
    ray=incoming.get("receive_line")
    context=projected(np.r_[ray["xyz"],ray["velocity"]]).tolist() if ray else None
    return depart, shape, cand, context


def archive(output=SOURCE):
    schedule=_read(season.PLAN)
    prohibited={c["id"] for set_ in ("practice","assessment") for c in schedule[set_]}
    docs={}; rows=[]; inputs={}
    for name in ("observations","difficult"):
        files=sorted((video.ROOT/name).glob("case-*.npz"))
        for path in files:
            with np.load(path) as z:
                i=int(z["index"]); ident=int(z["id"])
                if ident in prohibited:
                    continue
                leg=int(z["leg"]) if "leg" in z else 1
                p=Path(str(z["source"])) if "source" in z else video.SRC
                if p not in docs:
                    docs[p]=_read(p)
                    inputs[str(p)]=_hash(p)
                doc=docs[p]
                card,r=doc["cards"][i],doc["rows"][i]
                if card["id"]!=ident:
                    raise ValueError("The camera packet and original physical source differ")
                dep, shape, candidate, context=episode(r,leg)
                opt, ts, stream=z["ball"],z["tick"],z["stream"]
                near=z["contact"]
                good=np.isfinite(near)
                cutoff=min(near[good])-3 if good.any() else -np.inf
                # Only use the first stream for input. The chosen exposures
                # precede all far hits and contain no pixels of the opponent.
                window=(ts>=int(z["anchor"])+5)&(ts<=int(z["anchor"])+21)&(ts<cutoff)
                window&=(stream==0)&np.isfinite(opt).all(axis=1)&(opt[:,4]>.9)
                features=None
                if window.sum()>=3:
                    view=tactical.arc_shape(opt[window,:3],opt[window,3:])
                    view=view[np.isfinite(view).all(axis=1)]
                    if len(view)>=3:
                        features=np.median(view,axis=0).tolist()
                begin=1 if name=="difficult" else 0
                labs=[]
                for j,label in enumerate(z["label"]):
                    if j<begin:
                        continue
                    if np.isfinite(label).all() and label[4]<-1.:
                        labs.append(projected(label).tolist())
                    else:
                        labs.append(None)
                old=z["label"][0]
                rows.append(dict(id=ident, cohort=name, source=str(p), index=i, leg=leg,
                                 archive_path=str(path), packet_sha256=_hash(path),
                                 stroke_sha256=digest_json(dep),
                                 shape=shape.tolist() if np.isfinite(shape).all() else None,
                                 physical_dep=dep["ball"]+dep["velocity"],spin=dep["spin"],
                                 optical_shape=features,native_candidate=candidate,incoming_context=context,
                                 external_hindsight_height=(float(projected(old)[0]) if
                                      name=="difficult" and np.isfinite(old).all() and old[4]<-1 else None),
                                 replies=labs, seeds=z["seeds"][begin:].tolist()))
    _write(output,dict(rows=rows, prohibited=sorted(prohibited),source_hashes=inputs,
                       source_roles="difficult stream zero was the original conditioning result and was removed"))
    print("read",[(m,sum(r["cohort"]==m for r in rows)) for m in ("observations","difficult")],flush=True)


def receiver_corpus(excluded):
    files=_read(INPUTS)["input_files"]
    rows={}
    for f,sha in files.items():
        p=Path(f)
        if _hash(p)!=sha:
            raise ValueError("Physical outcome corpus changed")
        d=_read(p)
        for c,r in zip(d["cards"],d["rows"],strict=True):
            if c["id"] in excluded or c.get("profile")!="reach":
                continue
            for o in r["placements"]["delivered"]:
                line=o["receive_line"]
                if line is None:
                    continue
                fp=digest_json((o["far_contact"],o["near_bounce"],o["near_contact"],o["near_landed"]))
                rows.setdefault((c["id"],fp),
                    (projected(np.r_[line["xyz"],line["velocity"]]),o["near_landed"]))
    return (np.array([r for r,_ in rows.values()]),np.array([y for _,y in rows.values()],dtype=float),
            np.array([g for g,_ in rows]))


def recv_feature(x):
    # Height and transverse arrival define much of the reach envelope, but
    # speed and vertical motion carry separate observable action demands.
    a=(x-[1.041,0.,3.7,.25])/[.062,.20,.65,.64]
    a=np.clip(a,-4.,4.)
    return np.c_[a,np.abs(a),np.maximum(0., (1.01-x[:,0])/.03),
                 np.maximum(0., abs(x[:,1])-.26)/.10, a[:,0]*abs(a[:,1]),
                 a[:,0]*a[:,2],a[:,0]*a[:,3]]


def fit(x,y,l2):
    mu=x.mean(0)
    s=np.maximum(x.std(0),.035)
    u=np.clip((x-mu)/s,-4.,4.)
    cy=y.mean(0)
    w=np.linalg.solve(u.T@u+l2*np.eye(u.shape[1]),u.T@(y-cy))
    return mu,s,cy,w


def predict(fit,x):
    mu,s,cy,w=fit
    return cy + np.clip((x-mu)/s,-4.,4.)@w


def returnability(doc):
    rejected={r["id"] for r in doc["rows"]} | set(doc["prohibited"])
    x,y,g=receiver_corpus(rejected)
    f=recv_feature(x)
    pred=np.empty(len(x))
    ref=np.empty(len(x))
    releases=np.random.default_rng(510680).permutation(np.unique(g))
    for group in np.array_split(releases,6):
        test=np.isin(g,group)
        pred[test]=np.clip(predict(fit(f[~test],y[~test],70.),f[test]),0.,1.)
        ref[test]=y[~test].mean()
    fitted=fit(f,y,70.)
    for row in doc["rows"]:
        q=np.array([w for w in row["replies"] if w is not None])
        yb=np.clip(predict(fitted,recv_feature(q)),0.,1.) if len(q) else []
        row["outcomes"]=np.c_[q[:,0], q[:,0]<.99,1-yb].tolist() if len(q) else []
    return dict(corpus=len(x), releases=len(releases), real_success=float(y.mean()),
                brier_rx=float(np.mean((y-pred)**2)),brier_prevalence=float(np.mean((y-ref)**2)),
                separate_training_releases=True, fit=[v.tolist() for v in fitted])


def variance(outcomes):
    stable=[np.asarray(r,dtype=float) for r in outcomes if len(r)>=4]
    means=np.array([v.mean(0) for v in stable])
    n=np.array([len(v) for v in stable])
    within=np.array([v.var(0,ddof=1) for v in stable])
    latent=np.maximum(0.,means.var(0,ddof=1)-np.mean(within/n[:,None],axis=0))
    w=within.mean(0)
    a=np.array([r[::2].mean(0) for r in stable])
    b=np.array([r[1::2].mean(0) for r in stable])
    corr=[float(np.corrcoef(x,y)[0,1]) if min(x.std(),y.std())>0 else None for x,y in zip(a.T,b.T,strict=True)]
    h=np.concatenate(stable,axis=0)[:,0]
    cdf=np.mean([np.mean(v[:,0,None]<LIMIT[None,:],axis=0) for v in stable],axis=0)
    return dict(n=len(stable),replies=int(n.sum()),mean=means.mean(0).tolist(),
                within_std=np.sqrt(w).tolist(),latent_between_std=np.sqrt(latent).tolist(),
                draw_intraclass=(latent/np.maximum(w+latent,1e-9)).tolist(),split_half=corr,
                receiver_height_p10_p50_p90_m=np.quantile(h,[.10,.50,.90]).tolist(),
                cdf_height_cutoffs_m=LIMIT.tolist(),context_weighted_cdf=cdf.tolist())


def bootstrap_difference(a,b,ids):
    ids=np.asarray(ids)
    rows=np.array([np.mean((a-b)[ids==i],axis=0) for i in np.unique(ids)])
    grid=np.random.default_rng(510050).integers(0,len(rows),(1000,len(rows)))
    return np.quantile(rows[grid].mean(axis=1),[.1,.9],axis=0).tolist()


def evaluate(original):
    # Each entry excludes the entire held release. Compare feature groups on
    # the same support, not on physically easier cases with complete features.
    scopes={}
    for name in ("observations","difficult","native","ordinary_pool","combined"):
        use=("observations","native") if name=="ordinary_pool" else (name,)
        all_rows=[r for r in original if len(r["outcomes"])>=4 and (name=="combined" or r["cohort"] in use)]
        unique={}
        for r in all_rows:
            context=(r["id"],r["leg"],r["stroke_sha256"] if "stroke_sha256" in r else digest_json(r["native_departure"]))
            unique.setdefault(context,r)
        all_rows=list(unique.values())
        if len({r["id"] for r in all_rows})<5:
            continue
        overall=variance([r["outcomes"] for r in all_rows])
        oldh=[r["external_hindsight_height"] for r in all_rows if r["external_hindsight_height"] is not None]
        masks={}
        for family in (("shape","incoming_context"),("native_candidate","incoming_context"),
                       ("optical_shape","incoming_context"),("physical_dep","incoming_context"),
                       ("physical_with_spin","incoming_context")):
            rows=[r for r in all_rows if all(r[f] is not None for f in family)]
            groups=np.array(sorted({r["id"] for r in rows}))
            if len(groups)<8:
                continue
            g=np.array([r["id"] for r in rows])
            draws=[np.c_[z, z[:,0,None]<LIMIT[None,:]]
                   for z in (np.asarray(r["outcomes"]) for r in rows)]
            y=np.array([v.mean(0) for v in draws])
            confidence={}
            for xname in ("prior","incoming","outbound","joint"):
                ext=np.array([r[family[0]] if xname=="outbound" else r[family[1]] if xname=="incoming"
                              else r[family[0]]+r[family[1]] if xname=="joint" else [0.]
                              for r in rows])
                predictions=np.empty_like(y)
                one=np.empty((4,*y.shape))
                for group in np.array_split(np.random.default_rng(51493).permutation(groups),5):
                    test=np.isin(g,group)
                    predictions[test]=predict(fit(ext[~test],y[~test],45.),ext[test])
                    for i in range(4):
                        single=np.array([v[i] for v, held in zip(draws,test,strict=True) if not held])
                        one[i,test]=predict(fit(ext[~test],single,45.),ext[test])
                predictions[:,1:]=np.clip(predictions[:,1:],0.,1.)
                one[:,:,1:]=np.clip(one[:,:,1:],0.,1.)
                errors=(y-predictions)**2
                if xname=="prior":
                    baseline=errors
                dispersion=np.array([np.mean((v-p)**2,axis=0)
                                     for v,p in zip(draws,predictions,strict=True)])
                ordinary=np.mean((one-y[None])**2,axis=1)
                confidence[xname]=dict(context_mse=errors.mean(0).tolist(), draw_mse=dispersion.mean(0).tolist(),
                                      single_draw_context_mse=ordinary.tolist(),
                                      single_draw_mean_context_mse=ordinary.mean(0).tolist(),
                                      delta_from_prior_release_bootstrap_80pct=bootstrap_difference(errors,baseline,g),
                                      delta_from_single_release_bootstrap_80pct=bootstrap_difference(errors,
                                                    ((one-y[None])**2).mean(0),g))
            masks[family[0]]=dict(points=len(rows),releases=len(groups),models=confidence)
        scopes[name]=dict(variance=overall,original_selected_count=len(oldh),
                          original_selected_low=sum(x<.99 for x in oldh),validation=masks)
    return scopes


def transfer(rows):
    result={}
    for k in ("shape","physical_dep","native_candidate"):
        result[k]={}
        for train,test in (("observations","native"),("native","observations"),
                           ("difficult","native"),("difficult","observations")):
            a,b=([r for r in rows if r["cohort"]==x and len(r["outcomes"])>=4 and r.get(k) is not None]
                 for x in (train,test))
            held_ids={r["id"] for r in b}
            a=[r for r in a if r["id"] not in held_ids]
            if min(len({r["id"] for r in a}),len(held_ids))<8:
                continue
            x,z=(np.array([r[k] for r in dataset]) for dataset in (a,b))
            label, truth=(np.array([np.mean(r["outcomes"],axis=0) for r in dataset]) for dataset in (a,b))
            p=predict(fit(x,label,45.),z)
            p[:,1:]=np.clip(p[:,1:],0.,1.)
            result[k][train+"_to_"+test]=dict(n=[len(a),len(b)], prediction_mse=np.mean((p-truth)**2,axis=0).tolist(),
                  train_test_releases=[len({r["id"] for r in a}),len(held_ids)],
                  training_prior_mse=np.mean((label.mean(0)-truth)**2,axis=0).tolist())
    return result


def earlier(rows):
    """Fixed old sources, tested only where their fit never saw this release."""
    from .tt43_outgoing_coverage import Coverage, ROOT as RISK, estimate
    from .tt45_tactical_goal import Envelope
    from .tt49_sustain_value import predict as old_predict

    envsrc=Path("artifacts/tt49_tactical_retrieval/disjoint-practice.json.gz")
    env=Envelope(_read(envsrc.with_name("envelope.json")))
    excluded=set(_read(envsrc)["fit_releases"])
    regs={}
    for name in ("difficult","ordinary_pool"):
        b=[[],[]]
        for r in rows:
            if r["id"] in excluded or not r["outcomes"] or r["shape"] is None:
                continue
            if (name=="difficult") != (r["cohort"]=="difficult"):
                continue
            b[env.contains(np.asarray(r["shape"]))].append(r)
        regs[name]={}
        for key,g in zip(("outside","inside"),b,strict=True):
            regs[name][key]=dict(contexts=len(g), distinct_releases=len({r["id"] for r in g}),
                      fresh_draws=sum(len(t["outcomes"]) for t in g),
                      realized=np.array([np.mean(t["outcomes"],axis=0) for t in g]).mean(0).tolist() if g else None)
    old=_read(RISK/"outgoing-risk.json")
    good=Coverage(old)
    fitted=_read(INPUTS)
    past_ids=set(fitted["source_ids"])
    prior={r["id"] for r in old["rows"]}
    cands=[r for r in rows if r["native_candidate"] is not None and len(r["outcomes"])>=4]
    a={}
    for name in ("difficult","ordinary_pool","combined"):
        risk=[r for r in cands if r["id"] not in prior and
                (name=="combined" or (name=="difficult")== (r["cohort"]=="difficult"))]
        q=estimate(good.fit,np.array([r["native_candidate"] for r in risk])) if risk else []
        g=np.array([r["id"] for r in risk])
        yt=np.array([np.mean(r["outcomes"],axis=0)[1:] for r in risk])
        a[name]=dict(contexts=len(risk), release_count=len(set(g)))
        if len(risk) and len(set(g))>1:
            local=np.array([yt[g!=i].mean(axis=0) for i in g])
            a[name].update(mean_squared_error=np.mean((q-yt)**2,axis=0).tolist(),
                           old_empirical_prevalence_mse=np.mean((good.fit[1]-yt)**2,axis=0).tolist(),
                           local_release_leaveout_prevalence_mse=np.mean((local-yt)**2,axis=0).tolist())
    y,p=[],[]
    sources={}
    for r in cands:
        if r["id"] in past_ids:
            continue
        name=r["source"]
        if name not in sources:
            sources[name]=_read(Path(name))
        row=sources[name]["rows"][r["index"]]
        end=row["placements"]["delivered"][r["leg"]-1]["near_contact"]["step"]
        s=[z for z in row["searches"] if 0<end-z["tick"]<9
                 and z.get("target_errors")==r["native_candidate"]]
        if not s:
            continue
        last=s[-1]
        feat=np.r_[r["native_candidate"],last["p"],last.get("robust",0.),last.get("score",0.)]
        one,two=old_predict(tuple(map(np.asarray,fitted["fit"])),feat)[0]
        p.append(1.-min(1.,two/max(.01,one)))
        y.append(float(np.mean(r["outcomes"],axis=0)[2]))
    return dict(older_qualified_envelope=regs, late_tactical_risk=a,
                sustained_value_conditional=dict(contexts=len(p),
                   mean_squared_error=float(np.mean((np.asarray(y)-p)**2)) if p else None,
                   qualification="Future/current success probability conditioned on legal current return. Repeated receiver proxy is not the original actual reward."),
                note="All comparisons remove releases present in the corresponding earlier fit.")


def request_analysis_only(rows):
    """How much changing the independent requested values moves a fixed stroke's outcomes."""
    from .tt37_opponent import draw

    archive={}
    scopes={}
    for cohort in ("observations","difficult","native"):
        u,y,release=[],[],[]
        for r in rows:
            if r["cohort"]!=cohort or len(r["outcomes"])<4:
                continue
            path=r["source"]
            if path not in archive:
                archive[path]=_read(Path(path))
            card=archive[path]["cards"][r["index"]]
            x=np.array([draw(dict(card,reply_seed=s),r["leg"],"random",None)[0]
                        for s,v in zip(r["seeds"],r["replies"],strict=True) if v is not None])
            x=(x-.5*(tactical.TARGET_LO+tactical.TARGET_HI))/(tactical.TARGET_HI-tactical.TARGET_LO)
            features=np.c_[x,np.abs(x)]
            u.extend(features-features.mean(axis=0))
            p=np.array(r["outcomes"])
            y.extend(p-p.mean(axis=0))
            release.extend([r["id"]]*len(p))
        groups=np.array(sorted(set(release)))
        if len(groups)<8:
            continue
        x,v,g=map(np.asarray,(u,y,release))
        pred=np.empty_like(v)
        for block in np.array_split(np.random.default_rng(510913).permutation(groups),5):
            kept=~np.isin(g,block)
            pred[~kept]=predict(fit(x[kept],v[kept],65.),x[~kept])
        raw=np.mean(v*v,axis=0)
        change=np.mean((pred-v)**2,axis=0)
        scopes[cohort]=dict(release_count=len(groups),within_draw_mse_zero=raw.tolist(),
                            within_draw_mse_requests=change.tolist())
    return scopes


def analyze(snapshot=SOURCE, supplemental=None, output=ROOT/"analysis.json"):
    doc=_read(snapshot)
    native = _read(supplemental) if supplemental else None
    if native:
        if not native["complete"]:
            raise ValueError("Native extension still incomplete")
        doc["rows"].extend(native["rows"])
    for r in doc["rows"]:
        r["physical_with_spin"]=r["physical_dep"]+r["spin"]
    rx=returnability(doc)
    counts={k:dict(contexts=len(v),distinct_releases=len({r["id"] for r in v}),
                   assigned=sum(len(r["replies"]) for r in v),
                   tracked=sum(len(r["outcomes"]) for r in v),
                   answered_native=sum(b["answered"] for r in v for b in r["branch"]) if k=="native" else None)
            for k,v in ((k,[r for r in doc["rows"] if r["cohort"]==k])
                        for k in ("observations","difficult","native"))}
    result=dict(snapshot_sha256=_hash(snapshot),sample=doc["rows"],receiver=rx,
                native_sha256=None if supplemental is None else _hash(supplemental),
                observed_counts=counts,
                modality_transfer=transfer(doc["rows"]),
                previous_fixed_models=earlier(doc["rows"]),
                request_evaluator_only=request_analysis_only(doc["rows"]),
                outputs=["height_m","low_indicator","estimated_receiving_failure"],
                model_outputs=["height_m","low_indicator","estimated_receiving_failure",*[f"cdf_{i}" for i in LIMIT]],
                evaluation=evaluate(doc["rows"]),
                scope="Original difficult result excluded. Receiving failure is a proxy fitted on disjoint prior real scores, not a physically tested next own return for these alternate draws.")
    _write(output,result)
    print({k:v["variance"] for k,v in result["evaluation"].items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("archive","analyze"))
    p.add_argument("--snapshot",type=Path,default=SOURCE)
    p.add_argument("--native",type=Path)
    p.add_argument("--output",type=Path,default=ROOT/"analysis.json")
    a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    archive(a.snapshot) if a.phase=="archive" else analyze(a.snapshot,a.native,a.output)
