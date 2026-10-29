# Second SMDP feedback cycle: no further learning gain

The fixed C2 endpoint failed its declared eight-game progress gate. C1 returned
16 balls; C2 returned 13. Both retained seven first returns. Games with at least
two returns fell five→two; later returns fell 9/15→6/13. Frozen40069 remains
23total and16later on these starts. No remaining24comparison games or fresh63265
qualification ran, and no model was promoted. Further data-only SMDP cycles stop.

C2 was produced by the unchanged observed-return MC initialization plus12completed
FQI updates under the180s fit allocation, finishing in168.054s. All100,286training
predictions and2048serialized option queries matched exactly; the optimized and
original prediction/support calculations also matched. This verifies implementation
and fitting consistency, not control quality or convergence.

Its actual feedback data comprised all56newly admitted comparison trajectories
(21,664previously paid cameras) and64new phase-balanced TRAIN games under frozenC1
with the unchangedε=.25macro exploration contract. Collection2 completed in62.941s:
19,998new transition cameras+64initials,73ownhits,2354post-hit macro boundaries and
445non-greedy explored options. Every trace retained actualN+1float64scenes and
baseline scores; actual action/reward/timing/termination and RNG/hold replay passed.
No fresh outcomes were admitted.

The native comparison completed in16.508s, costing3270new transition cameras+
eightinitials. All3270stateful rows replayed exactly. Source, model and baseline
hashes stayed fixed. C0, C1, C2 and all success/failure trajectories remain intact.
The first correction had improved matched TRAIN32 from44to50returns; this second
correction does not establish monotone or reliable learning. Fit counts differed
under the same fixed compute budget: C0/C1/C2 completed26/16/12FQI updates.

- C2SHA: `785d35d48814633f253073f2e6f7804cb98b4293e99dadc632375ba2f4c7b43d`.
- Collection2: `/workspace/m1b-experiments/pong-smdp-onpolicy-collection-20261029-r2`.
- Refit2: `/workspace/m1b-experiments/pong-smdp-feedback2-refit-20261029-r1`.
- Comparison: `/workspace/m1b-experiments/pong-smdp-cycle2-comparison-20261029-r1`.
- Comparison specSHA: `92ae13da91aaf38de4a6d58e20ec7a85212492ee4f678ed99a93c7560c4f1481`.

Keep40069as the practical baseline and761as the previously preserved secondary
candidate. The user's sustained-return milestone remains unmet. The next proposed
mechanism is a genuinely learned dynamics model with receding-horizon control;
this note authorizes no additional fit, collection or native evaluation.
