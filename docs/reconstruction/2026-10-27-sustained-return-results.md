# Sustained-return continuation

## Starting evidence

The immutable October25 final head is the carry. Re-reading its existing32 full-game receipts (no new episodes) gives28/32 first returns and14/42 later returns. All32 ended in an agent miss. The matched r4 baseline had28/32 first and17/45 later returns. This reinforces that the prior milestone did not improve sustained play; the opponent was not ending those games early.

## Fresh TRAIN coverage, allocation1

Root: `/workspace/m1b-experiments/pong-rally-baseline-20261027-r1`. Same frozen carry, native.65 opponent, K1,20-second full-game cap,64 new unfiltered TRAIN seeds. No fit or candidate model.

| Incoming contact | Returns | Opportunities |
|---|---:|---:|
| First |54|64|
| Second |19|54|
| Third |9|19|
| Fourth |4|9|
| Fifth |1|4|
| All later contacts |33|86|

63 agent losses, zero opponent-miss endings, one timeout. The opponent returned all121 opportunities, including87 following our own hits. Maximum own-hit streak5. Allocation cost22,384 TRAIN transition cameras +649 explicitly counted parity replay cameras;64+1 initial cameras;71.74 wall seconds. The diagnostic logger reproduced the reference episode's records and terminal graph exactly, and weights/core sources remained unchanged.

Root independently analyzed the saved TRAIN traces, with **no additional camera executions**. A privileged diagnostic estimated reflected ball impact and reachable paddle bounds under constant extreme drive. At40 cameras before the53 later misses,32 remained approximately reachable; at20 cameras only13 did, and at5 cameras only1 did. Among the47 cases already incoming80 cameras before the miss, all47 were approximately reachable. These are diagnostic approximations, not an oracle policy, learning labels or proof of maximal reachability.

The nearest tracked visual entity was within3 pixels of the true ball in all53 later misses at each of the5/10/20/40-camera offsets; velocity was available in52–53 cases. This rules against wholesale ball disappearance at those sampled times, not against all visual errors. Some trajectories show the paddle pinned at the top for over a second while the policy continues to command up and the ball approaches below it. Others start moving correctly too late.

Current decision: test earlier and longer reward-verified action sequences, because single-frame interventions near terminal cannot correct most of these positions. Successful exploratory trajectories must subsequently be learned by the ordinary policy; finding a good scripted branch is only discovery, not deployed learning. Preserve current successful segments, not the failed suffixes. Root's diagnostic script/report are in `/workspace/scratch/analyze_rally_baseline.py` and `/workspace/scratch/rally-diagnostic-summary.json`; neither is a learner input.

## Recovery discovery, allocation2

Root: `/workspace/m1b-experiments/pong-rally-discovery-20261027-r1`. Twelve later misses selected from TRAIN, balanced by initial serve direction. Frozen prefixes were replayed exactly. At fixed earlier points, test one-camera alternatives in six contexts, then held commands of10/20/40/80 cameras, all three commands, stopping search after a successful next return per case. No hidden geometric target supplied actions or labels.

All12 cases had a reward-successful burst witness: four with20 cameras, six with40, two with80. The36 one-camera alternative trials in six contexts yielded zero returns;216 burst trials yielded12 successful witnesses. Exposure to duration was conditional on earlier failures, so this is not an unbiased duration comparison. Successful witnesses include the legal roll-in, override and released-policy continuation; they are whole reward-earned behaviors, not proofs that each forced action is individually optimal.

Cost24,959 actual transition cameras (4,240 prefix +20,719 suffix, including the exact null replay),12initial images,92.47wall seconds. No censored branches. A null later-miss replay retained previous hits but correctly received a next-return outcome of0 and matched the full native records/terminal graph. At this stage no learned candidate had been evaluated. Root approved one same566-coordinate analytic fit with broad successful-segment preservation, followed by64 native TRAIN full games. A separate untouched32-start paired full-game screen is reserved at `pong-rally-fresh-screen-20261027-r1`; it will not supply fitting labels.

## First learned update, allocation3

Root: `/workspace/m1b-experiments/pong-rally-imitation-20261027-r1`. Same566-coordinate analytic reward-earned sequence fit; no backpropagation, no new architecture, no held actions at deployment. Twelve successful witnesses supplied418 contrasting actions;87 baseline successful segments supplied soft preservation. Fit took0.263s and reproduced44.85% of weighted contrasting actions. This low absorption can reflect a regression objective that shrinks below action-choice boundaries; it is not yet evidence of insufficient representation.

On the same64 TRAIN full games, first returns54→53, later returns33/86→49/100 (38.37%→49%), total hits87→102. Games reaching2/3/4/5 returns changed19/9/4/1→27/15/5/2. The opponent returned136/136 opportunities; no opponent miss explains the improvement. Two timeouts versus one baseline. Only2/12 original witness contexts gained an extra deployed return. Thus there is partial closed-loop learning, but weak absorption and unresolved generalization.

Cost25,043 transition cameras +64initials,78.005s. Immutable October25 carry retained. Selected next: reserved32 paired fresh screen, then improve reward-earned action absorption or state coverage according to the result. Candidate SHA256 `979b986422b8ebcbd83abd53099c8fe70185ee644b2e523ef79689fa9f2f806a`. No global promotion or reliability claim.

## Fresh transfer screen, allocation4

Root: `/workspace/m1b-experiments/pong-rally-fresh-screen-20261027-r1`. Thirty-two reserved new starts, paired immutable carry versus first learned candidate, no fitting or retries. First returns29/32→31/32; later19/48→37/67 (39.58%→55.22%); total hits48→68. Games reaching2/3/4/5 returns14/5/0/0→20/8/6/3. All opponent opportunities returned; carry32 own misses, candidate31 own misses and1timeout. Paired bootstrap95% interval for extra mean hits/game was[0.125,1.125]; later-return-rate difference[0.0138,0.2994], noting policy-dependent later exposure. These are useful transfer gains, not reliable match-long control.

Cost28,142 transition cameras +64initials,86.677s. Root accepts r1 as a provisional useful sustained-return checkpoint, while retaining the immutable October25 control. Fresh data remains outside fitting. Next authorized allocation: one soft action-margin candidate using exactly the same TRAIN witnesses and successful segments, centered two-contrast coordinates, soft positive margins and soft preservation of earned action choices, analytic convex solves without backprop. This tests objective mismatch rather than expanding data and changing the objective simultaneously. Native64TRAIN reward determines whether it improves on r1; no parameter grid.

Sources informing this direction: [classification-based policy iteration](https://ijcai.org/proceedings/2003/226) motivates learning action preferences rather than numerical value targets; [Self-Imitation Learning](https://proceedings.mlr.press/v80/oh18b.html) motivates reuse of the learner's own successful experience. Our analytic classifier is not the paper's actor-critic implementation, and reward-successful sequences do not provide exact per-frame advantages.

## Action-margin learning, allocation5

Root: `/workspace/m1b-experiments/pong-rally-margin-20261027-r1`. Same original TRAIN data, same policy/vision, same normalization and ridge. Replaced numerical residual regression with soft pairwise action margins, solved by analytic active-set Newton steps (no autodiff/backprop). Two focused numerical derivative/fit tests passed. Fit converged in1.193s, reducing its convex objective1.63359→0.032897; weighted recovery-action agreement44.85%→83.69%. Soft preservation allowed16.56% changes on prior successful-segment actions, so deployed performance remains decisive.

On64 TRAIN games, first returns51/64 versus original54 and ridge53; later84/127 (66.14%) versus original33/86 and ridge49/100; total hits135 versus87/102. Games reaching2/3/4/5/6 returns34/24/15/8/3 versus ridge27/15/5/2/0. Six of12 witness contexts gained an extra deployed return. The opponent returned164/164 opportunities;56 own misses and8 timeouts. Cost31,298 transition cameras +64initials,94.367s. Head SHA256 `fe132e0dd2466494142ef4578b187e6856341dc1bb4a3803456ca16e7703b625`.

Root selected this as the next experimental TRAIN carry; the first ridge checkpoint remains the fresh-supported fallback, and the original export stays immutable. No fresh qualification claim for the margin head. Next authorized learning cycle: add32 new TRAIN starts, discover approximately24 rewarded recoveries spanning opening and later misses, aggregate successful experience, fit one same-margin update and evaluate on the expanded TRAIN bank. This follows learner-induced state coverage rather than repeating objective variants on the original12 witnesses.
