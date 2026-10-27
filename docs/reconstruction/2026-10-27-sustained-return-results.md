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
