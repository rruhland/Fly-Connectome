# Explicit .25-opponent score-only acquisition screen

The original .45-opponent results and task are preserved. Privileged relaxed and
legal-controller feasibility checks support a scoring ceiling after own returns
against that opponent, whereas opponent speed .25 permits return-to-score wins
with actual legal controls. This is a separately labeled winnable-environment
development experiment. It makes no claim of beating the original opponent.

Change only the training/evaluation opponent speed cap **.45 -> .25** relative to
the completed raw alpha screen. Keep actual body dynamics, frozen prior, raw
structured transport, zero initial readout, true-online Sarsa, alpha **.001**,
lambda **.995**, gamma_per_second **.95**, epsilon **.2**, score-only terminal ±1
reward, legal three actions, exact camera cadence and explicit reset behavior.
The new spatial-sharing prototype remains disabled. No game-state labels or
oracle diagnostic trajectories enter training, policy features or calibration.

Train original development seed **172900127** for exactly **6000 new camera
transitions**; save both 2000/6000 weights. Freshly evaluate both checkpoints on
the same first **12 rallies** of development serve stream **904271833**, with
the same .25 opponent and **2400-tick/20-second** rally limit. Recompute paired
initial/constant-down, constant-up, no-op and uniform-random baselines on that
same serve schedule (random action seed **447201113**). They may skip perception
after the existing full-versus-skipped exact action/physics proof, while keeping
the camera decision schedule. Checkpoint zero is the initial/down policy.

Report wins per attempted rally, completed-rally score differential, completion
fraction, losses, truncations, return per actual physical second, hits and rally
durations. Separate every positive score with **zero own hits** (natural serve
win) from every positive score after **one or more own hits**. This split is
required in both training and all baselines/checkpoints. The latter is a physical
trajectory observation; state-dependent acquisition still needs baseline lift,
replication and a visual-input check.

Freeze the staged rule before starting: a complete checkpoint is promising if
its total wins are **at least three above the strongest simple baseline** and
its own-hit-followed wins are **at least three above the largest corresponding
baseline count**, out of the same 12 attempted rallies. Truncations are nonwins;
they do **not** independently disqualify acquisition. Preserve all truncation
counts and durations. If neither endpoint passes, stop this arm without extension
or replication. If both pass, select most total wins, tie to earlier checkpoint.
Persist the selected exact training budget before replication.

For a promising checkpoint, run the same selected training budget (2000 or 6000)
on original independent training seeds **736187003** and **1597334677**; run both
replications regardless of their outcomes. Freshly evaluate each selected endpoint
on the same .25 development serves. In parallel, evaluate the first candidate
with a blank policy camera, leaving actual physics, acquisition timing, rewards
and reset schedule unchanged. A developmental score lift is not final acceptance.
No final seeds may be used, and no transfer to .35 or further standard-opponent
training follows automatically. Genuine .25 acquisition must be established with
a separately frozen fresh evaluation before such a transfer is proposed.

Maximum allocation: **18,000 learning transitions**, **108 attempted evaluation
rallies** (four baselines, two initial endpoints, one blank-camera candidate and
two replicated endpoints), conservative **129,600 evaluation transitions**, and
**30 aggregate wall minutes**. The common stop case is 6000 learning transitions
plus 72 attempted evaluation rallies. Each independently durable chunk contains
at most 200 transitions; up to five chunks reuse one interpreter within a shared
160-second work/180-second subprocess bound. Preserve complete source/runtime
snapshots, exact configs, weight hashes, all durable journals, partial allocations,
costs and numerical diagnostics. Existing reserved final seeds remain untouched.
