"""Learn which observed late-stroke phase leaves a playable next exchange."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_ready import centered
from .tt34_results import paired
from .tt35_error import Observations, passage
from .tt36_online import outcome
from .tt36_reach import Body, Reach, Relation, smooth
from .tt37_online import MODEL, OUT, PUPIL, RELATION, ROOT, SCALE, sensory_context
from .tt37_opponent import arena as make_arena

ANCHORS = (5530, 14832, 3025, 3180, 8129)
PHASE_AND_FACE = np.array([[0., 0.], [-.067, .165], [.067, -.165]])
PRIOR = np.array([.63, .55, .55])
CAP = 40
GUIDES = (
    ROOT/"tt35"/"confirmation-plan.json.gz",
    ROOT/"tt34"/"body-novel-plan.json.gz",
    ROOT/"tt32"/"fresh-plan.json.gz",
    ROOT/"tt36"/"course-new.json.gz",
    ROOT/"tt36"/"course-confirm.json.gz",
)


def prepare():
    cases = {c["id"]: c for source in GUIDES for c in _read(source)["cards"]}
    practice, held = [], []
    for block in range(4):
        for index, sid in enumerate(ANCHORS):
            item = cases[sid]
            point = block * len(ANCHORS) + index
            practice.append(dict(item, opening="anchor", reply_seed=372957+13*point,
                                 key=f"pace-practice-{point}-{sid}"))
            held.append(dict(item, opening="anchor", reply_seed=375561+17*point,
                              key=f"pace-held-{point}-{sid}"))
    shared = _read(OUT/"opponent-plan.json")
    used = set(ANCHORS)
    if any(c["id"] in used for c in shared["held_openings"]):
        raise ValueError("This direct practice set overlaps novel physical evaluation")
    _write(OUT/"online-pace-plan.json", dict(practice=practice, later=held,
           new=shared["held_openings"], feed_sha256=shared["feed_sha256"]))


class Preference:
    def __init__(self, rows=(), label="continuation"):
        self.rows = list(rows)
        self.label = label

    def select(self, x, rng, epsilon):
        estimate = PRIOR.copy()
        if self.rows:
            context = np.array([r["context"] for r in self.rows])
            k = np.exp(-.68*np.sum(((context-x)/SCALE)**2, axis=1))
            arm = np.array([r["primitive"] for r in self.rows])
            y = np.array([r[self.label] for r in self.rows])
            n = np.bincount(arm, weights=k, minlength=len(PRIOR))
            estimate = (1.3*PRIOR + np.bincount(arm, weights=k*y, minlength=len(PRIOR)))/(1.3+n)
            estimate[1:][n[1:] < .36] = -1
        greedy = int(np.argmax(estimate))
        search = rng.random() < epsilon
        return int(rng.integers(len(PRIOR))) if search else greedy, greedy, search

    def record(self, row):
        self.rows.append(row)
        self.rows = self.rows[-160:]


class Pace:
    """Credit the outgoing stroke only after the ensuing observable exchange."""

    def __init__(self, body, policy, memory, rng, epsilon, learn):
        self.body, self.policy, self.memory = body, policy, memory
        self.rng, self.epsilon, self.learn = rng, epsilon, learn
        self.stroke = self.pending_legal = self.old_y = None
        self.records, self.outgoing = [], []

    def command(self, t, optical, q, frames, reach, camera):
        bias = reach.bias(t, optical, q, frames)
        if optical is None:
            return self.policy.command(optical, q, offset=bias)
        y = optical[1]
        if self.stroke is not None and self.old_y is not None and self.old_y < -.58 and y > .25:
            self.finish(t, camera, False)
        self.old_y = y
        if self.stroke is None and -1.04 < y < -.25:
            context = sensory_context(optical, q, self.body)
            arm, exploit, probe = self.memory.select(context, self.rng, self.epsilon)
            self.stroke = dict(onset=t, context=context.tolist(), primitive=arm, greedy=exploit,
                               explored=probe, eligible_memory=len(self.memory.rows), max_edit=0.)
            self.outgoing = []
        if self.stroke is None or not self.stroke["primitive"] or not -1.53 < y < -.61:
            return self.policy.command(optical, q, offset=bias)

        blade, jac, support = self.body.local(q[:5])
        lead = y-blade[1]
        edit = smooth((.43-lead)/.29)*smooth((lead+.16)/.16)
        edit *= 1-smooth((support-.85)/.65)
        if edit <= .02:
            return self.policy.command(optical, q, offset=bias)

        phase, face = PHASE_AND_FACE[self.stroke["primitive"]]
        state = optical.copy()
        # Reparameterize the current observation in the learned readout's depth
        # coordinate. No future ball location, intercept, or strike time is computed.
        state[1] += edit*phase
        change = np.array([0., 0., 0., .44*face, face])
        change -= jac.T@np.linalg.solve(jac@jac.T+.0014*np.eye(3),jac@change)
        bias = edit*np.clip(change,-.17,.17)+(0. if bias is None else bias)
        self.stroke["max_edit"] = max(self.stroke["max_edit"], edit)
        return self.policy.command(state, q, offset=bias)

    def emit(self, row, credit_delay):
        row["delay_expired_at"] = credit_delay
        self.records.append(row)
        if self.learn:
            self.memory.record(row)

    def finish(self, tick, camera, legal):
        if self.stroke is None:
            return
        keep = self.stroke
        self.stroke = None
        cross = passage([(t, b, racket) for t, b, racket in camera if keep["onset"] <= t <= tick],keep["onset"])
        close = float(np.linalg.norm(cross["gap_xz_m"])) if cross else None
        forward = max(self.outgoing, default=None)
        grade = outcome(legal, forward, close)
        keep.update(end=tick, measured_gap_m=close, forward_mps=forward, legal=legal, immediate=float(grade or 0.),
                    visible=grade is not None)

        if self.pending_legal is not None:
            former = self.pending_legal
            former.update(next_observed=True, next_legal=legal, continuation=former["immediate"]+.14+.60*keep["immediate"])
            self.emit(former,tick)
            self.pending_legal = None
        if legal:
            self.pending_legal = keep
        elif grade is not None:
            self.emit(dict(keep,next_observed=False,next_legal=False,continuation=keep["immediate"]),tick)
        else:
            # Missing images do not create a fictitious joint-action outcome.
            self.records.append(dict(keep,censored=True))

    def close(self, tick, camera):
        self.finish(tick,camera,False)
        if self.pending_legal is not None:
            row = self.pending_legal
            self.pending_legal = None
            self.emit(dict(row,next_observed=False,next_legal=False,continuation=row["immediate"]),tick)


def point(arena, view, rig, model, stride, low, high, body, relation, shot, card, memory, rng, eps, learn):
    arena.launch(shot, card=card)
    arena.partner.reset(seed=card["seed"])
    view.begin()
    joint_policy = joint.JointPolicy(model, stride, low, high)
    reach = Reach(body, relation, joint_policy, rig, .38, 1.)
    eye = joint.Vision(rig, arena.control_dt, raw=True)
    policy = Pace(body,joint_policy,memory,rng,eps,learn)
    commands=[]
    start=process_time()
    for tick in range(arena.max_steps):
        q=arena.joints
        frames=view.capture() if tick%2 == 0 else None
        optical=eye.update(tick,frames,q)
        action=policy.command(tick,optical,q,frames,reach,view.rows)
        f=eye.eye.latest
        if (policy.stroke is not None and f is not None and f.time==tick*arena.control_dt
                and f.velocity[1]>.9 and -1.57<f.position[1]<-.48 and f.rms<2.4 and f.depth_sigma<.16):
            policy.outgoing.append(float(f.velocity[1]))
        _,_,done,cap,info=arena.exchange(action,joint.FAR_COMMAND)
        commands.append(action.tolist())
        if any(e["kind"]=="legal_return" and e["side"]==1 for e in info["events"]):
            policy.finish(tick,view.rows,True)
        if done or cap or arena.legal_returns+arena.rival_returns-1>=CAP:
            break
    policy.close(tick,view.rows)
    return dict(shot=shot, opponent_seed=card["seed"], actions=commands, events=arena.events.copy(),
                score=audit(arena.events,opening_side=1), placements=arena.point_placements(),
                outcomes=policy.records, runtime_cpu_s=process_time()-start)


def play(args):
    source=OUT/"online-pace-plan.json"
    plan=_read(source)
    if plan["feed_sha256"]!=_hash(args.feeds):
        raise ValueError("Wrong launch source")
    cards=plan[args.phase][:args.limit]
    keep=OUT/"online-pace-checkpoint.json"
    holding=args.mode in ("carry","immediate")
    state=_read(keep) if holding else None
    label="immediate" if args.mode=="immediate" else "continuation"
    memory=Preference(state["rows"] if state else (),label)
    args.learn=args.mode=="learn"
    base=_read(PUPIL)
    with np.load(MODEL) as data:
        body=Body(data)
    with np.load(RELATION) as data:
        relation=Relation(data)
    archived=args.output or OUT/f"online-pace-{args.phase}-{args.mode}.json.gz"
    doc=dict(mode=args.mode, phase=args.phase, cards=cards, ready=base["ready"], protocol="random",
             opening_side=1, rows=[], cap=CAP, plan_sha256=_hash(source), checkpoint_sha256=_hash(keep) if holding else None)
    launches=measured_launches(args.feeds)
    if state and set(state["seeds"])&{x["reply_seed"] for x in cards}:
        raise ValueError("The held practice seeds are not fresh")
    with centered(base["ready"]):
        arena=make_arena(protocol="random")
        camera,rig,stride,low,high=joint.resources(arena)
        view=Observations(camera,rig)
        try:
            for i,c in enumerate(cards):
                eps=.63 if args.learn or args.mode=="explore" else 0.
                before=len(memory.rows)
                row=point(arena,view,rig,base,stride,low,high,body,relation,launches[c["id"]],c,memory,
                          np.random.default_rng(372811+i),eps,args.learn)
                row.update(initial_memory=before, final_memory=len(memory.rows))
                doc["rows"].append(row)
                _write(archived,doc)
                if args.learn:
                    _write(keep,dict(rows=memory.rows, seeds=[c["reply_seed"] for c in cards[:i+1]],
                                     source_sha256=_hash(archived)))
                print(args.phase,args.mode,i+1,c["id"],row["score"]["learner"],row["score"]["returns"],
                      [(r["primitive"],r["legal"],r.get("next_legal")) for r in row["outcomes"]],flush=True)
        finally:
            camera.close()
            arena.close()


def scores(rows):
    counts = [r["score"]["learner"] for r in rows]
    all_outcomes = [sample for r in rows for sample in r["outcomes"]]
    return dict(points=len(rows), own=sum(counts), openings=sum(i>0 for i in counts),
                later=sum(max(i-1,0) for i in counts), longest=max(counts,default=0),
                alternating=sum(r["score"]["returns"] for r in rows),
                cpu=sum(r["runtime_cpu_s"] for r in rows),
                samples=len(all_outcomes), used_edits=sum(row["max_edit"]>.02 for row in all_outcomes),
                observed_next=sum(row.get("next_observed", False) for row in all_outcomes),
                next_landed=sum(row.get("next_legal", False) for row in all_outcomes))


def summarize():
    doc = {}
    for phase, modes in (("practice", ("explore","learn")), ("later", ("fixed","immediate","carry")),
                         ("new", ("fixed","immediate","carry"))):
        paths = {m: OUT/f"online-pace-{phase}-{m}.json.gz" for m in modes}
        logs = {m:_read(p) for m,p in paths.items()}
        baseline = logs[modes[0]]
        selected = [i for i,r in enumerate(baseline["rows"]) if r["placements"]["delivered"]]
        fixed = dict(baseline,cards=[baseline["cards"][i] for i in selected],
                     rows=[baseline["rows"][i] for i in selected])
        doc[phase] = {}
        for m,data in logs.items():
            rows = data["rows"]
            if (data["cards"]!=baseline["cards"] or len(rows)!=len(data["cards"])
                    or data["plan_sha256"]!=baseline["plan_sha256"]):
                raise ValueError("Incomplete or unmatched protocol")
            if selected != [i for i,r in enumerate(rows) if r["placements"]["delivered"]]:
                raise ValueError("Learner and control differ in first legal opposing delivery")
            if any(sum(o["legal"] for o in r["outcomes"])!=r["score"]["learner"] for r in rows):
                raise ValueError("A legal arm return was not uniquely credited")
            if m!="learn" and any(r["initial_memory"]!=r["final_memory"] for r in rows):
                raise ValueError("A frozen condition changed its checkpoint")
            subset = dict(data,cards=[data["cards"][i] for i in selected],rows=[rows[i] for i in selected])
            first = [r["outcomes"][0] for r in rows if r["outcomes"]]
            measured = {}
            for arm in range(len(PRIOR)):
                arms=[a for a in first if a["primitive"]==arm]
                speed=[a["forward_mps"] for a in arms if a["forward_mps"] is not None]
                measured[arm]=dict(issued=len(arms), legal=sum(a["legal"] for a in arms),
                                   following_approaches=sum(a.get("next_observed",False) for a in arms),
                                   following_legal=sum(a.get("next_legal",False) for a in arms),
                                   observed_speed_mps=speed)
            doc[phase][m] = dict(stats=scores(rows), by_first_primitive=measured,
                                  paired=paired(fixed,subset), blocks=[scores(rows[i:i+5]) for i in range(0,len(rows),5)],
                                  sha256=_hash(paths[m]))
    _write(OUT/"online-pace-results.json",doc)


def audit_changes():
    audited={}
    with TemporaryDirectory(prefix="tt37-paced-direct-") as directory:
        subset,check=(Path(directory)/p for p in ("game.json.gz","cad.json"))
        for phase in ("later","new"):
            baseline=_read(OUT/f"online-pace-{phase}-fixed.json.gz")
            path=OUT/f"online-pace-{phase}-carry.json.gz"
            trained=_read(path)
            differences=[(i,b["score"]["learner"]-a["score"]["learner"])
                         for i,(a,b) in enumerate(zip(baseline["rows"],trained["rows"],strict=True))]
            positions=[i for i,d in differences if d]
            if not positions:
                positions=[max(range(len(trained["rows"])),key=lambda i: trained["rows"][i]["score"]["learner"])]
            doc=dict(trained,cards=[trained["cards"][i] for i in positions],rows=[trained["rows"][i] for i in positions])
            _write(subset,doc)
            with patch.object(joint,"replay_arena",lambda cards,**options:make_arena(cards,protocol="random",**options)):
                joint.verify(subset,check,cad=True)
            receipts=_read(check)["points"]
            if any(r["clearance"]["penetration_samples"] for r in receipts):
                raise ValueError("Changed rally failed physical clearance replay")
            audited[phase]=dict(source_sha256=_hash(path), replay=receipts, indices=positions)
    _write(OUT/"online-pace-audit.json",audited)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase",choices=("plan","practice","later","new","summary","audit"))
    ap.add_argument("mode",choices=("learn","explore","fixed","carry","immediate"),nargs="?")
    ap.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    ap.add_argument("--limit",type=int)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    operator={"plan":prepare,"summary":summarize,"audit":audit_changes}.get(args.phase)
    operator() if operator else play(args)
