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
