# Carried-learning execution evidence — 2026-10-30

The fresh qualification completed all 544 planned games and passed its pinned
last32 criteria and independent final audit. Whole live learning streams,
including acquisition and change recovery, returned 1,719/1,815 (94.71%); the pass
does not apply to every acquisition game. No native process remains. This note
records execution, state and feedback-credit evidence; the main outcome report
owns performance interpretation.

Implementation commits: `c70a9b8` (learner/pilot) and `ef6fb7e` (qualification).
The latter adds an explicit closed end-A barrier and freezes the structure used
in the actual final acquisition game, rather than recomputing an optimistic index.
The 11 learner and eight qualification tests passed; surrounding compact-policy
and persistent-binding tests also passed.

## Fixed qualification and exact camera cost

Artifact directory:
`/workspace/m1b-experiments/pong-carried-qualification-20261030-r1`.
Spec SHA256:
`fb058e6c8b71ee83e711b4c2bec766d5e1e423ff4227e04c25707cf8d578be4e`.
Summary SHA256:
`ab24c7548faed71645cead27e5bd67387add26a9e9e921302cb414ef6a371f48`.

| Stream | Completed games | Transition cameras | Initial cameras | Reported stream wall seconds |
|---|---:|---:|---:|---:|
| Replica 0 live | 192 | 155,411 | 192 | 1,884.148 |
| Replica 1 live | 192 | 156,432 | 192 | 1,871.588 |
| Replica 0 reset control | 96 | 23,585 | 96 | 305.630 |
| Replica 0 frozen control | 64 | 34,371 | 64 | 428.508 |
| Total | 544 | 369,799 | 544 | Streams overlap |

The prospective maximum was 544,000 transition cameras. Independent live starts
were 384 unique canonical seeds, disjoint from historical and all declared pilot
starts. Controls repeat only their explicitly matched live tail starts. Only
replica 0 has matched controls. All planned streams completed without extension,
and the final coordinator source/assets and streaming audit checks passed.
The independent final audit passed all 61 source/asset checks and tape,
checkpoint and control-pair checks. The 32 matched live/frozen A-return tapes
were byte-identical, as were their public-command digests and annotations.

## Carried state and branch evidence

Each live learner consumes its actual game before choosing the next structure.
It receives only own hits, task reward, termination/truncation and elapsed public
time. Each own hit updates success evidence; only a public negative terminal
score updates failure evidence. Positive terminal scores and time caps censor
the unfinished return. No opportunity annotation enters the learner.

The primary learner was restored from JSON after every completed game while an
independent shadow stayed live. Their states matched exactly. Sensory tracking
and controller bindings reset between games; learned reward evidence did not.

Final live checkpoints are `replica0-live/checkpoint-latest.json` and
`replica1-live/checkpoint-latest.json` under the artifact directory. Their learner
state hashes are respectively
`dbca04b851844c630befd91269e2723512d5a225f0744748d70d6451af36c131` and
`81cde5060858d1852c0f96e4fa268b045c5b61da8b656a0f9caf81474671405b`.
Both replicas retain sufficient statistics accumulated through A/B/A. Their
discounted working evidence and lifetime counts are distinct from transient
per-game state; receipts and event tapes preserve the chronological history.

`replica0-frozen/acquired-prefix.json` records the exact end-A checkpoint, final
acquisition receipt, ready-barrier hashes and copied arm. Frozen feedback cannot
reach the live learner. The reset control clears reward evidence each game and
uses separate rotating uniform draws; its skipped prefix preserves the cold
policy distribution, not a hypothetical identical random-draw trajectory.

## Why terminal loss alone was a poor change signal

In the interrupted pilot, capped-selector B games 32/33 (zero-based;
`capped_beta/game-080.json` and `game-081.json`) made one and three returns before
losing. The second loss queued another full 16-arm sweep; `game-082.json` starts
that sweep. The rule treated those partial successes like a motor reversal's
zero-return failures.

Fresh discounted-UCB replica 1 A-return games 36/37
(`replica1-live/game-164.json` and `game-165.json`) each made two returns before
losing. Per-hit and terminal evidence retained that distinction: no sweep or
reset occurred, and `game-166.json` kept arm 4 and made six returns. This explains
an observed mechanism difference; the pilot algorithms used different starts,
so it is not a paired causal comparison of their performance.

## Interrupted pilot accounting

The environment restart left 142 capped and 137 UCB completed receipts:
178,913 known transition cameras plus 279 initials. No native process survived;
there are no orphan event tapes. One possibly active unreceipted grant per stream is charged,
giving an upper bound of 180,913 transitions plus 281 initials. The missing games
were not replayed and the pilot was not declared passed. Evidence is preserved
in `pong-carried-learning-20261030-r2/environment-interruption.json` and
`interrupted-independent-accounting.json` under the experiment root.

Across the interrupted pilot and qualification, known cost is 548,712 transition
cameras plus 823 initials; the conservative upper bound is 550,712 plus 825.
No further native experiment is selected. This remains structural online
learning over inherited frozen vision and a fixed controller family, not full
M1B completion or learning a visual controller from scratch.
