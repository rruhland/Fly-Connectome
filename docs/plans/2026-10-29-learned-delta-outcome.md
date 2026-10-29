# Learned visual delta planner outcome

The single frozen learned-dynamics pilot failed its first eight actual TRAIN games. Baseline40069 returned 23 balls; the planner returned 11. First returns stayed 7/8, later returns fell from 16/21 to 4/11, and maximum consecutive returns fell from 6 to 2. Four games still reached a second return, but none reached a third. The planner changed 1,346 actual commands. Baseline40069 remains the retained endpoint. No horizon, beam, tail, threshold, or data variants were run.

The fixed model learned anonymous one-camera scene deltas and event masses from 114,229 saved physical transitions across 206 starts. Training took 39.131 seconds. Exported world/event queries matched sklearn exactly on 2,048 rows each, and 32 serialized beam searches matched exactly. Inference averaged 9.145 ms over the fixed saved probes, expanding all 156 nodes of the declared horizon-eight, width-eight search. The controller used baseline40069 until an actual own-hit reward, then executed one primitive command before observing and replanning. The C1 option-three value tail was frozen.

Evaluation consumed 2,958 transition cameras and eight initial cameras in 21.307 seconds of the authorized single 600-second ceiling. All 2,958 recorded baseline-Q/history rows were checked; 402 planner queries were replayed, including one complete chronological game and at most nine fixed reconstructed-history queries per other game. This is bounded planner replay coverage, not exhaustive replay of all policy decisions. Source and asset hashes remained unchanged. The eight-game rejection stopped the run before 32-game or reserved fresh-tag63281 evaluation. No endpoint was promoted.

- Model: `/workspace/m1b-experiments/pong-learned-delta-planner-20261029-r1/planner-model.npz`, SHA256 `171b9acabd3c4abc4b446b9f8c83a373bd36816aedfc8f96cebda80a075a189d`.
- Evaluation spec: `/workspace/m1b-experiments/pong-delta-native-20261029-r1/spec.json`, SHA256 `f08ff375b2434fefbe4d2e784e50e9521691eccc3e09c45f8a5e60af3ecac5b7`.
- Actual receipts and outcome: `/workspace/m1b-experiments/pong-delta-native-20261029-r1/screen8/` and adjacent `summary.json`.
- Dataset: `/workspace/m1b-experiments/pong-delta-dynamics-data-20261029-r1/transitions.npz`, SHA256 `69a126e0385f75c45617813cacf859e53cdf7108e08e17e20a2536b0f8d9fd20`.
- Implementation: `438a70f`, `4a21ab3`; learner/model source: `18f09cb`. Five focused extraction/integration tests passed.

These actual returns reject this controller. Offline prediction/export checks establish implementation consistency and do not establish useful closed-loop planning.
