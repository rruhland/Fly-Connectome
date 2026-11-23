"""Use sensed launch conditions to blend feedback weights learned on other arms."""
import gzip
import json

import numpy as np

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_bounce_ident import CleanState
from .tt24_broad_donors import FamilyWorld, FamilyArena, TRAIN, NOVEL, fixed_seeds
from .tt24_donor_adapt import historical_basis
from .tt24_donor_assess import CHECKPOINT, ShiftedBounce
from .tt24_donor_trial import FUSED_OUTPUT, ROOT, Native, _live, load_public

COORDINATES = np.array([(a,b) for a in (-1.,0.,1.) for b in (-1.,0.,1.)])
WARM_TICKS = 3


def write(file, obj):
    with gzip.open(ROOT/file, "wt") as out:
        json.dump(obj, out, separators=(",", ":"))


def draw(native, state, weights, issued):
    state = np.ascontiguousarray(state, dtype=np.float64)
    action = np.zeros(5)
    native.command(state.ctypes.data, weights.ctypes.data, issued.ctypes.data, action.ctypes.data)
    return action


def context(states):
    # Four frames are available at the point where the head first changes.
    p = np.array([s[:3]-s[6:9] for s in states])
    return np.r_[p[0], (p[-1]-p[0])*60/WARM_TICKS]


def donor_bank(native, anchor, basis):
    bank = anchor + np.einsum("nk,kij->nij", COORDINATES, basis)
    features, rewards = [], []
    for i, spec in enumerate(TRAIN):
        world = FamilyWorld(native, spec)
        try:
            seeds = fixed_seeds(64, 1104101+i)
            batch = world.prepare(seeds, "held")
            # Source exploration evaluates each stationary feedback law from release.
            scores = world.play(bank, batch, cap=260)
            rewards.extend(scores[:,:,1].T)
            arena = world.arena
            for seed in seeds:
                obs = arena.reset(seed=seed, split="held", regime="changed", render=False)
                sensor = CleanState(arena)
                issued = np.zeros(5)
                states = []
                for tick in range(WARM_TICKS+1):
                    state = sensor.observe(obs)
                    states.append(state)
                    if tick < WARM_TICKS:
                        obs, _, done = arena.step(draw(native,state,anchor,issued), render=False)
                        if done:
                            raise ValueError("source ended in the launch prefix")
                features.append(context(states))
        finally:
            world.close()
    return np.array(features), np.array(rewards)


def interpolation(query, inputs, scores, mode):
    # Scales, candidate policies and returns come solely from donor bodies.
    scale = np.maximum(inputs.std(axis=0), .001)
    distances = np.sum(((inputs-query)/scale)**2, axis=1)
    affinity = np.exp(-.5*distances/2.25)
    affinity /= max(affinity.sum(), 1e-9)
    if mode == "pooled":
        affinity = np.full(len(scores), 1/len(scores))
    expectation = affinity @ scores
    # Reward-scaled soft selection keeps the resulting readout continuous.
    probability = np.exp((expectation-expectation.max())/1.15)
    return probability @ COORDINATES / probability.sum()


def evaluate(arena, native, seed, regime, anchor, basis, inputs, scores, mode):
    observer = load_public(FUSED_OUTPUT)
    obs = arena.reset(seed=seed, split="held", regime=regime, render=True)
    issued = np.zeros(5)
    weights = anchor
    states, tape = [], []
    coeff = np.zeros(2)
    for tick in range(arena.max_steps):
        state = observer.observe(obs)
        if tick <= WARM_TICKS:
            states.append(state.copy())
        if tick == WARM_TICKS and mode != "frozen":
            coeff = interpolation(context(states), inputs, scores, mode)
            weights = np.ascontiguousarray(anchor+np.tensordot(coeff,basis,axes=1))
        action = draw(native,state,weights,issued)
        tape.append(action.tolist())
        obs, _, done = arena.step(action,render=True)
        if done:
            break
    metrics = arena.judge_metrics()
    checked = _live(native,arena,anchor,seed,regime,"replay",tape=tape)
    if digest_json(metrics) != digest_json(checked["metrics"]):
        raise ValueError("physical outcome changed on replay")
    return dict(seed=seed, coefficients=coeff.tolist(), query=context(states).tolist(),
                run=metrics["longest_consecutive_rebounds"], score=metrics["rebounds"], actions=tape)


def main():
    native = Native()
    try:
        with np.load(CHECKPOINT,allow_pickle=False) as data:
            anchor = np.ascontiguousarray(data["weights"])
        basis, _ = historical_basis(anchor)
        inputs, scores = donor_bank(native,anchor,basis)
        perm = np.random.default_rng(9205165).permutation(len(scores))
        np.savez_compressed(ROOT/"launch-context.npz",inputs=inputs,rewards=scores,basis=basis,anchor=anchor,permutation=perm)
        for i,(name,factory) in enumerate((("stock",lambda: BounceArena(view="wrist",width=192,height=144,max_steps=300)),
                ("tilted",ShiftedBounce),("novel",lambda: FamilyArena(NOVEL,view="wrist")))):
            arena = factory()
            try:
                seeds = fixed_seeds(16, 1623112089+i)
                groups = {}
                for mode in ("context","shuffled","pooled","frozen"):
                    groups[mode] = [evaluate(arena,native,s,"standard" if i==0 else "changed",
                                             anchor,basis,inputs,scores[perm] if mode=="shuffled" else scores,mode)
                                    for s in seeds]
                    print(name,mode,sum(row["run"] for row in groups[mode]),flush=True)
                write("launch-context-"+name+".json.gz",dict(groups=groups,seeds=seeds,exact_replays=True))
            finally:
                arena.close()
    finally:
        native.working.cleanup()


if __name__ == "__main__":
    main()
