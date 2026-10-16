# Pong scoring feasibility before further learning

This is privileged physics-only environment validation. It supplies no training
observations, actions, features, labels or updates to a learner. Original Pong
physics, score reward, policy observations and completed failures are preserved.
Final holdout seeds remain unused.

All seven observed winning training rallies (six original controls and the
larger-alpha screen) had zero own paddle contacts. The opponent missed the
initial serve after about 1.04 seconds. No learner has experienced a controlled
return-to-win trajectory. Reactive privileged tracking and fixed shot offsets
tested separately by the parent also produced no return-to-win trajectories.

There is a concrete potential ceiling in the original opponent. After a hit,
maximum vertical ball speed is `.55 * sin(.9) = .43083`. Its proportional tracking
gain is four, giving a steady tracking error of `.10771`, inside the legal
collision reach `.10 + .0125 = .1125`. Opponent speed `.45` exceeds the maximum
vertical ball speed. Its acceleration bound is two. This is an explanatory
calculation; it is not a formal proof covering all discrete reachable states,
clamps, wall reversals and initial serves.

An optimistic upper-bound diagnostic sampled 512 dispersed initial serve phases.
It guaranteed own returns by instantaneous paddle placement, then replaced the
outgoing ball direction with each of 65 angles uniformly spanning the complete
legal range `[-.9,.9]`. Speed, wall bounces and opponent updates used the original
Pong implementation. This relaxes actual body constraints. Twelve serves scored
naturally before any own contact. The remaining 500 states supplied 32,500
candidate first-return shots. **All 32,500 were returned by the opponent.**
Maximum right-plane error was `.10768253`, below `.1125`. Batched execution took
.77 seconds. Failure of this relaxed controller is stronger environment evidence
than a particular learned policy failing, but the finite grid is not exhaustive.

A separate fixed later-return diagnostic used 256 dispersed serve phases, each
with 64 independent uniform legal-angle schedules, for 100 physical seconds per
environment. It deliberately removed the 20-second rally cap for this feasibility
test. Among 16,384 relaxed rollouts there were 429,614 guaranteed own contacts and
429,390 opponent contacts, **zero return-to-score wins**, zero losses, 256 natural
serve wins and 16,128 survivors at 100 seconds. Maximum right-plane error after
any own contact was `.10767144`; the larger overall error `.119965` occurred only
before own contact. It took 18.62 seconds. No vision or learner ran.

The same first-return grid then changed only the opponent speed cap:

| Speed cap | Natural serve wins / 512 | First-contact states | Winning legal-angle shots | States with a winning angle |
| --- | ---: | ---: | ---: | ---: |
| .45 original | 12 | 500 | 0 / 32,500 | 0 |
| .35 | 70 | 442 | 2,917 / 28,730 (10.15%) | 442 |
| .25 | 122 | 390 | 6,836 / 25,350 (26.97%) | 390 |
| .15 | 156 | 356 | 9,876 / 23,140 (42.68%) | 356 |

These counts describe a relaxed privileged action space, not a learned win rate.
They justify investigating a separately labeled winnable-environment curriculum
with the same score-only reward. They do not demonstrate standard Pong success.
Further learner spending against the original opponent is paused pending new
feasibility evidence. Any curriculum result must report its exact opponent and
retain a distinct original-opponent comparison.

Sources, seed schedules, original physics hash and raw per-serve reports live in
`/workspace/m1b-experiments/pong-feasibility-angle-grid-20261015/`. The journal-only
training audit is
`/workspace/m1b-experiments/training-win-ownhit-audit-20261015.json`. No external
artifact is needed to reconstruct these diagnostics from the saved scripts.
