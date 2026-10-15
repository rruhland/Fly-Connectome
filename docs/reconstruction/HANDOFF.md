# Fly-Connectome cloud handoff — paused before registered pilot

Ryan requested a safe stopping point so he can provide his own orchestration.
This supersedes earlier permission to automatically launch the toy pilot.
Do not launch training/evaluation from this handoff without new direction.

## Exact checkout and checkpoint

- Working repository: `/workspace/Fly-Connectome-m1b`
- Branch: `reconstruction/m1b-foundation`
- Code checkpoint: `b8449d5b6103f8ae531575222ac793d982ad2799`
- This handoff is committed in a following documentation-only commit. `git rev-parse
  HEAD` identifies that final handoff commit; source is unchanged from the code checkpoint.
- Original checkout: `/workspace/Fly-Connectome`, clean on `work` at
  `b52f06e9a4cf1664d8429e7098a6371eafb77bc7`.
- No public push or merge occurred. The working branch exists locally; Library
  review packets are evidence/source excerpts, not a guaranteed complete Git backup.

## Stop state and actual experiments

No registered toy training, competence evaluation, or Pong competence experiment
started. **Pilot counts: 0 learning + 0 evaluation = 0/4560 new transitions.**
There is no pilot manifest, pilot database, partial pilot allocation or pilot result.
Inspection of `/proc` found no running pytest, recovery-proof or pilot processes.
No background task/scheduler was installed. A transient exec transport disconnect
recovered on a read-only retry; it did not launch a job or leave a partial script.

Controlled unit/recovery diagnostics DID update synthetic policy state on separate
seeds. These are correctness checks, not competence evidence. Selected proof counts:

- Stage one: 48 vs 24+24 frozen-policy continuation, 96 transitions in successful
  proof, 193 including preceding diagnostics. It proved no learner recovery.
- Stage two: 32 vs 16+16 learning continuation, 64 new transitions, 12.37 seconds;
  nonzero weights, traces/q_old/pending action and RNG/perception/cadence matched.
- Storage proof: 16 vs 9 + crash + replay 2 + 7, 32 new + 2 replay transitions,
  7.84 seconds in final run. A prior same-sized diagnostic also passed.
- Toy-specific proof: 18 vs 10 + crash + replay 3 + 8 on diagnostic seed 99031,
  36 new + 3 replay transitions, 7.80 seconds. The interruption was inside delayed
  episode 2 at delay step 2. Side permutation/index, latched choice, pending action,
  all RNGs and the complete state/audit matched exactly after recovery.

These counts describe named proofs, not a sum of all repeatedly executed unit
fixtures. Proof JSON and test logs live under `docs/reconstruction/` and raw runtime
artifacts under `/workspace/m1b-verification/`.

## Implemented and validated

- True-online Sarsa(lambda), Dutch correction and q_old, physical-time discount
  indexing, pre-update next-action sampling/reuse, terminal/truncation distinction.
  Independent online forward-view tests compare each update prefix with overlapping
  nonbinary features, variable durations and action blocks.
- Pong boundary can suppress auto-reset, stops immediately at a score, and captures
  the genuine final image. Existing default Pong auto-reset behavior remains intact.
- Frozen nonzero Q evaluation. Fresh toy evaluation accepts only copied policy
  weights and registered task settings, reloads pinned visual prior, and creates
  fresh camera/perception/encoder histories and environment RNG. It does not copy
  the training trajectory or call learner updates. Diagnostic tests verify exact
  initialization, repeated same-seed results after unrelated ending scenes, and
  unchanged training/global RNG state, weights and traces.
- Immediate and eight-transition delayed visible-choice task implementation,
  first-action latching, ignored later commands, persistent target/consequence,
  zero intermediate rewards, and true .02-second cadence. Balanced evaluation
  schedules and identical schedules between lambda arms are tested.
- SQLite WAL + FULL durability, atomic sample/checkpoint updates, sequence/hash
  checks, one immutable frozen bank, periodic active snapshots (100 transitions or
  10 seconds, plus stops), fail-closed committed-tail replay, poisoned failed writers.
  Replay-v2 explicitly dispatches known Pong/toy session types and checks source,
  model/configuration and runtime identity. Toy code/settings are included in checks.
- Timing records separate active process prefixes from restart gaps. Crash gaps
  include unobserved execution after the last heartbeat; they are not pure downtime.

Pinned model: `src/fly_connectome/vision/data/default.pt`, 680428 bytes,
SHA256 `a899338c94278854dc9a790c21e3cba746658abb7ac91fc3f85c84754f253173`.
Load configuration: legacy prior upgraded by `load_default` to consensus-v3 64x64.
Learned vision remains frozen. No backprop was introduced.

## Validation and remaining failure

Latest full preflight: **790 passed, 4 skipped, 1 failed in 54.60 seconds**.
Focused evaluator/toy/durability: **21 passed in 8.20 seconds**.
After moving the unrun supervisor into the experimental directory, its manifest
allocation test was rerun: **1 passed in 0.93 seconds**. No competence run occurred.

The sole full-suite failure is the independently reproduced baseline
`test_native_neural_step_matches_reference[False]`, a voltage difference of at most
`1.1920928955078125e-07` at tick 1 under an exact-equality assertion. The assertion
was not relaxed. Two missing Linux -fPIC flags were fixed earlier. The baseline
rounding issue is distinct from reconstructed learner/evaluator correctness.

## Unvalidated work kept separately

`scripts/experimental/run_registered_toy_pilot.py` is explicitly UNVALIDATED as a
supervisor: its 36-job manifest and exact allocation are unit-tested, but the
worker/supervisor path has never run end to end. It was parked rather than launched.
Ryan should replace or review it before using it. It must not be treated as a
validated production scheduler. All other pending code edits were committed;
no uncommitted source changes are intended to remain.

## Registration and pending hypotheses

The authoritative proposal is `docs/reconstruction/2026-10-15-toy-preregistration.md`,
including the final review amendment. The previous authorization is now paused.

Budget if explicitly reauthorized: 2400 learning + 2160 evaluation = 4560 new
transitions across 36 jobs. Two tasks, lambda .8 and 0, three paired seeds. Fixed
alpha=.0001, gamma_per_second=.95, epsilon=.2; zero initial readout weights, fixed
features, no tuning/extensions/success-stopping. Learning seeds: immediate
2101–2103, delayed 2201–2203. Fresh evaluation: 3101–3103 / 3201–3203; random baseline
RNG: 4101–4103 / 4201–4203. At most 200 transitions and 180 seconds per chunk, saving
reserve and timeout backstop, aggregate 60 minutes including replay/startup/I/O.
Initial/reset camera observations must be counted separately from transitions.

This measures acquisition of TWO FIXED VISUAL TEMPLATES, not visual generalization.
Report left/right counts separately; repeated templates are not independent
visual-generalization trials. Screens are 16/20 immediate and 14/20 delayed per
seed. Immediate arms must match complete training trajectories AND final weights.
The lambda comparison is exploratory: persistent consequence features permit
lambda=0 bootstrapping, and hidden countdown prevents a full-Markov-value claim.
Pong learning remains unproven. Forecast availability is not evidence of benefit.
No rebuilt toy acquisition result exists yet; performance at the small fixed alpha
and budget is an open question, not a reason to silently tune.

## Runtime and exact safe commands

CPU environment: `/workspace/venvs/fly-m1b-cpu/bin/python`, Python 3.12.14,
NumPy 2.0.1, Torch 2.5.0+cpu, SciPy 1.18.1, PyArrow 26.0.0, pytest 9.1.1,
matplotlib 3.11.2. Debian 13.6 x86_64, 5 logical CPUs, about 17 GiB RAM.
The editable package points at `/workspace/Fly-Connectome-m1b`. Dependency freeze:
`docs/reconstruction/cloud-requirements.txt`. GPU capability is not established;
this environment and the proved paths use CPU only.

Inspection and tests (these do not start registered competence experiments):

```bash
cd /workspace/Fly-Connectome-m1b
git status --short --branch
git rev-parse HEAD
/workspace/venvs/fly-m1b-cpu/bin/python -m pytest -q tests/test_fresh_evaluation.py tests/test_toy_session.py tests/test_m1b_durable.py
/workspace/venvs/fly-m1b-cpu/bin/python -m pytest -q
```

The second command retains the known baseline failure. Recovery proofs are
controlled learning diagnostics; run only deliberately, with a NEW output path:

```bash
/workspace/venvs/fly-m1b-cpu/bin/python scripts/prove_toy_recovery.py --proof /workspace/m1b-verification/NEW-toy-proof
```

For an existing compatible durable run, the new orchestrator should open
`DurableRun(exact_db_path)` without supplying a fresh session. Inspect
`run.session.spec`, `run.session.transitions`, `run.replayed`, and actual committed
sample counts before advancing. `run.step()` commits one new transition;
`run.stop()` writes a stop checkpoint and closes the connection. After an error,
never reuse the advanced in-memory writer: reopen the database and replay its
committed tail. Never count replay as new learning; charge it to compute/time.
Use one writer per DB. Do not overwrite an existing run with a fresh session.
Do not bypass source/model/runtime mismatch errors or resume old schemas by guess.

There is no live pilot to resume today. Historical diagnostic databases may be
incompatible with the latest source/schema and are retained as evidence; return to
their exact source checkpoint for reproduction instead of bypassing validation.
Trusted same-platform pickle remains required; Windows or arbitrary runtime
portability is unproved. Local filesystem survival across replacement cloud tasks
is not guaranteed. There is no automatic background continuation or scheduler.
FULL durability was tested against process exits, not machine power loss.

## Historical claims versus current evidence; roles/model assumptions

The parent supplied historical M1B toy successes, Windows 96 vs 48+48 recovery,
and historical 925/928 test counts. Those commits/weights were not imported.
They guided requirements but are not evidence for this new reconstruction.
Current evidence is only the checked-in tests and named Linux diagnostics above.

Ryan selected Sol6.1 ExtraHigh and requested the existing two-thread arrangement
remain until he restructures orchestration. No model, reasoning or thread setting
was changed here. Actual executing model ID/reasoning is not independently exposed
by the tools, so the user selection is an assumption, not a verified runtime fact.
This cloud thread implemented code; the parent coordinated; Astra reviewed via
Library because its filesystem is separate. Ryan is now taking over orchestration.
