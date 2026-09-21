# Milestone 1 implementation and continuation ledger

Target: https://github.com/rruhland/Fly-Connectome
Spec: `2026-09-16-milestone-1-design.md` (read in full before continuing).
Branch: `codex/milestone-1`; work remains in the shared project checkout.

The approved architecture is authoritative. Implement sequentially; each chunk
ends with relevant tests, then the full existing suite, and a ledger update.
Use the executing-plans and test-driven-development workflows. No additional
approval is required except for major scientific ambiguities or design changes.

## Invariants

- No autograd, backpropagation, artificial learned encoder/readout, or invented edges.
- Fixed directed measured graph; linear contact-count initialization; immutable signs.
- Only ON/OFF events enter sensory-facing L1-L3; never direct L4 injection.
- Reward changes eligible central/motor weights only; it never becomes input.
- Per-environment neural state survives Pong scores and automatic resets.
- CPU reference and CUDA share tensor semantics; B=1 is canonical.
- Training telemetry must be optional, bounded, nonblocking, and truthful.
- Fixture graphs test software only and never support MaleCNS learning claims.

## Dependency order and verification gates

- [x] 1. Package and immutable graph (`graph.py`, `tests/test_graph.py`).
  Aggregate ordered contacts; retain neuron roster under T1/T3/T5; hash canonical
  representation. Test counts [2,3] on 10->20 aggregate to 5 while 20->10 stays 1.
- [x] 2. Reproducible data and anatomy (`data.py`, `anatomy.py`, `extraction.py`, corresponding tests).
  Pin official release URLs/checksums, preserve metadata, require explicit selection
  rules, induced edges, reachability and SCC report, anatomy-only threshold choice.
- [x] 3. Events and retinotopy (`sensor.py`, `tests/test_sensor.py`).
  Packed polarity events, persistent frame reference, fixed full-field hex mapping,
  registered L1-L3 projection only. Verify flashes, motion, empty and batched frames.
- [x] 4. Continuous body and Pong (`pong.py`, `tests/test_pong.py`).
  Vector physics, deterministic independent reset RNG, acceleration-limited opponent,
  visible resets, signed rate drive, bounded curriculum. Verify collisions, resets,
  braking/reversal/rest and absence of privileged state in observation interface.
- [x] 5. Sparse adaptive LIF (`dynamics.py`, `tests/test_dynamics.py`).
  Fixed delays, current and membrane leaks, refractory/adaptation, sparse delivery.
  Verify hand-calculated spikes, quiet input, batch independence and CPU/CUDA parity.
- [x] 6. Local plasticity (`plasticity.py`, `tests/test_plasticity.py`).
  Resolve prediction/observation definition before implementation. Active-edge
  eligibility, natural decay, separated regional rules, slow homeostasis, sign bounds.
  Verify confirmed/expired/unexpected predictions, reward isolation, and that
  eligibility has no forward effect. No dense batch-by-all-edges eligibility.
- [x] 7. Training/checkpoints (`training.py`, integration tests).
  M1A->M1B warm expansion by body-ID edge pairs, atomic exact-resume checkpoints,
  normalized windowed proposals, seed/config/manifests, read-only frozen evaluation.
  Verify interrupted/resumed trajectories and duplicate/permuted batch reductions.
- [ ] 8. Evaluation and diagnostics (`evaluation.py`, `diagnostics.py`).
  Held-out random/frozen/persistence comparisons, score-only learning, stability,
  deterministic probe/silencing bundles. Report measurements without invented success.
- [ ] 9. Optional local UI and process isolation (`ui/`, optional server/worker).
  Safe-boundary commands, latest-only telemetry, actual sampled body preview,
  independent frozen evaluation/diagnostics, anatomical inspection. Verify slow or
  disconnected UI cannot block training, trajectory equivalence and overhead budget.
- [ ] 10. Real-data acceptance runs and profiling. Preregister body IDs, masks and
  hyperparameters before training. Record M1A/M1B baseline results, final score-only
  behavior, bounded backend differences and headless/UI throughput measurements.

## Implementation history (latest checkpoint at the end)

Initial repository contained design documents only; no baseline tests existed.
Origin was added and `git ls-remote origin` succeeded with no refs (empty remote).
Git requires per-command `-c safe.directory=C:/Users/rruhl/OneDrive/Documents/Fly-Connectome`
because the checkout owner differs from this process user. No global Git trust changed.
Python 3.11 has PyTorch 2.5.0+cpu; CUDA unavailable. `.venv` uses that installation.

User approved anatomy-defined feedforward/recurrent current separation on 2026-09-17.
Use only existing edges, record the partition in the manifest, and add no error network.

Chunk 1: `python -m pytest -q`: 5 passed. Contact direction/counts, immutable arrays,
nested masks, isolated roster entries, invalid anatomy and deterministic hashes verified.

Data/anatomy utilities: source checksum checks, explicit sign resolution, bounded path
selection/reciprocal expansion/SCC inclusion, coverage/reachability and threshold choice
verified on fixtures. Actual MaleCNS extraction remains open: the downloaded pair table
has no neuropil metadata. Preserve that limitation; do not invent ROI coverage.

Sensor/body chunks: 19 tests pass across the suite. Persistent event reference,
ON/OFF polarity, full-field projection and no L4 injection; continuous force,
independent per-environment resets, collisions, visible resets and reward fade verified.
Default sensor routing is supplied explicitly by the caller, never inferred from type.

Neural/plasticity chunks: 27 passed, 1 CUDA skip on CPU-only installation. Sparse
arrival expansion uses existing outgoing CSR ranges and fixed delay history; no
individual-neuron/edge/spike Python loop. Visual prediction uses previous-tick
recurrent current, threshold-normalized and clipped to [0,1], compared with current
feedforward sensory current. Signed causal eligibility adjusts magnitude in the
direction that reduces local current error. Behavioral pair eligibility and scalar
reward are separate. Eligibility is stored only for active environment/edge keys;
weight proposals are batch-normalized and applied at explicit boundaries.

Training/checkpoint chunk: 30 passed, 1 CUDA skip. Exact resume includes physics RNG,
event reference, membrane/current/history, adaptation, eligibility, pending proposals,
motor rates, counters and curriculum position. Checkpoint writes use atomic replace.
Frozen evaluation has no Plasticity instance, cannot save, and uses score reward only.
Warm expansion matches ordered body-ID pairs and rejects sign changes. Dataset-scale
throughput and actual M1A/M1B training acceptance remain unmeasured.

Review fixes verified (45 passed, 1 CUDA skip): sparse behavioral arrival traces
survive until a later postsynaptic spike; trace time is measured from synaptic arrival,
not presynaptic emission. Homeostatic decay integrates simulated time. Feedforward
observation edges are fixed under prediction-error learning (homeostasis still applies);
only recurrent predictive edges receive local prediction-error updates. This keeps the
anatomically separated observation current from learning to inflate its own target.

Evaluation/probe and optional UI infrastructure implemented and unit/integration tested.
Remaining acceptance work includes browser inspection, richer evaluation views,
throughput measurement, real-data extraction, initial artifact creation, and learning runs.
The independent reviewer supplied concrete timing/homeostasis findings before its
session hit a usage limit; it did not finish a complete review of every module.

Preregistered extraction defaults are in `extraction.Selection` and executed by
`scripts/prepare_malecns.py`: right optic lobe, 8-hop measured paths at T3,
one reciprocal expansion at 20 contacts each way, SCC cap 128, confidence >= .5,
gain .005. Central candidates are official `cb_intrinsic` neurons with recorded
AOTU/PVLP/PLP innervation. Signs use curated ground truth, then confident individual
prediction, then confident cell-type prediction. Glutamate/GABA/histamine are modeled
as inhibitory and acetylcholine as excitatory (fixed model convention, not a receptor
claim). Four official raw files are pinned by local SHA-256 and stay uncommitted.

Real extraction completed with fixed rules: 60,366 neurons in every threshold roster.
T1: 8,617,335 edges / 38,536,533 contacts. T3: 3,587,714 edges / 31,879,824 contacts.
T5: 2,092,069 edges / 26,809,439 contacts; four isolated roster entries retained.
Both DNa02 bodies (10360 R, 523769 L) are reachable and all required ROIs are present,
so T5 is canonical. Visual-only M1A: 47,413 neurons / 1,377,103 T5 edges.
Graph/canonical manifest: `data/cache/milestone-1/`; initial M1A checkpoint:
`checkpoints/visual-initial.pt`. These generated files remain uncommitted.

Browser QA of software-fixture training: start, actual Pong preview, changing metrics
and safe stop exercised in the local browser. Found and fixed missing final stopped
telemetry, covered by a worker integration test. Fixture UI process on port 8765 was
used only for software QA; its scores do not support connectome learning claims.

CLI supports pinned download, init, headless train/resume, held-out comparisons and
offline diagnostics. M1A->M1B artifact-level warm expansion verified. Real CPU smoke
and performance checks are in progress; no learning-success claim has been established.

## Latest verified continuation point

57 tests pass; two CUDA tests skip on CPU-only PyTorch. JavaScript syntax passes.
Checkpoint ZIP serialization now uses a stream so identical headless/UI states
produce byte-identical files regardless of temporary save filename. Shared-weight
proposal reduction uses canonical sorted segment reduction rather than CUDA atomic
index-add. Actual CUDA execution still needs hardware verification.

Versioned extraction summary/body IDs are in `data/manifests/m1-v1.json`.
Real M1A ran 3,000 environment steps with four CPU threads; checkpoint is
`checkpoints/visual-trained.pt`. Held-out two-seed, 1,000-step scripted comparison
is `runs/m1a-heldout.json`; compact committed evidence is
`docs/experiments/2026-09-17-m1a-pilot.json`.
Only L1 (812), L2 (810), L3 (802) spiked in either condition. Local-current MSE
slightly worsened (5.80124e-6 initial vs 5.80133e-6 trained); sensory-event MSE
barely improved (0.000453268 vs 0.000453250). Persistence was 0.000887747.
Do not infer learning success from beating persistence with almost identical,
mostly inactive networks. The ON moving-edge gain sweep .005 through 1.0 produced
no downstream spikes. User requested investigation and a proposal for grounded
resting-current/cell-class dynamics while keeping topology, signs and learning fixed.
Do not silently change the canonical model before that proposal is reviewed.

Real M1B warm initialization succeeded: `checkpoints/pong-warm-initial.pt`,
60,366 neurons / 2,092,069 T5 edges. This is pipeline verification, not trained Pong.

UI diagnostics preserve independently named bundles, compare response A/B from
configured read-only checkpoints, replay stimuli and spikes by body ID, and report
latency/rate/sparsity. Browser QA verified paused OFF-flash intact vs L2-silenced
fixture traces and replay. Raw/accumulated event selector, sampled edge detail,
rally/memory/error metrics implemented. Worker periodic checkpoints added.
Remaining UI work includes tuning-sweep summaries and full-process overhead evidence.
Producer-only fixture benchmark: three alternating 3,000-step pairs, median overhead
-5.2% (timing noise), identical checkpoint bytes. This does not establish full UI <=1%.

Next: finish and review the neuron-model proposal; validate approved changes using
non-Pong visual probes, retain failed-pilot artifacts, then repeat M1A before spending
on long M1B runs. CPU/CUDA acceptance and post-fade learning remain open.

## Approved operating-point correction

User approved the resting-dynamics proposal. Implemented fixed class-specific
rest currents/membrane/synaptic time constants, signed ON-negative/OFF-positive
L1-L3 transduction, and decaying sensory-current state. Local learning is unchanged.
Schema 2 records these settings/state; schema 1 retains legacy semantics, tested
against exact continued trajectories. Warm expansion rejects mismatched profiles.
`init --profile` is explicit, so old artifacts are not silently converted.

64 tests passed, 2 CUDA skips after this chunk. Fixed no-event neural warm-up does
not step Pong or plasticity. Diagnostic baselines use its second half; UI offers
absolute and baseline-subtracted response plots. Held-out evaluation adds a zero
event predictor and labels signed versus legacy binary event target encoding.

`configs/resting-calibration-v1.json` preregisters six profiles. Full frozen results:
`docs/experiments/2026-09-17-resting-calibration-v1.json`. Selection code is in
`scripts/calibrate_resting.py`. Lowest total-current qualifying profile: lamina
1.5, selected medulla 1.2, saved as `configs/resting-v1-provisional.json`.
Mi1/Tm3 ON and Tm1/Tm2 OFF responses exceeded matched no-event activity. Motion and
projection populations remained silent. No parameter was selected using Pong.

New initial checkpoint: `checkpoints/resting-v1-visual-initial.pt`. The 500-step
pilot completed in `checkpoints/resting-v1-visual-trained.pt`; the final saved step
was verified after the process session ended. Do not overwrite old pilot artifacts.
Profiling found sorting consumed 33% of the first ten training steps. B=1 proposals
are unique by edge, so their redundant reduction sorts were removed; multi-environment
deterministic reduction is unchanged. Batching/resume/equivalence tests pass.

Expanded frozen validation completed: ON/OFF edges in two directions at .04/.16
pixels per neural tick, local flashes, matched no-event control, and recovery traces.
Reproduce using `scripts/validate_visual_profile.py`; report is
`docs/experiments/2026-09-18-resting-v1-probe-suite.json`. No T4/T5, LPi, LC10 or L4
spikes occurred in these conditions. Mi1/Tm3/Tm1/Tm2 responses depend on stimulus.
An independent read-only reviewer found no concrete implementation defects; its
provisional-calibration coverage concern motivated this expanded validation.

Held-out seeds 1001/1002, 500 scripted steps each:
`docs/experiments/2026-09-18-resting-v1-heldout.json`. Local prediction MSE improved
0.00439549 -> 0.00234941, still worse than persistence 0.000350311. Sensory-event MSE
worsened 0.000995708 -> 0.001023488; zero-event baseline 0.000928278, persistence
0.001936870. Mean population rate 1.115 -> 1.119 Hz. No motion-population spikes.
Do not claim predictive acceptance, useful motion processing, or Pong learning.

67 tests pass; three CUDA cases skip on unavailable hardware. JavaScript syntax,
Python script compilation and package wheel build verified. The probe onset bug
(moving stimuli appearing before their configured start) is fixed and tested.

Next decision: `2026-09-18-signed-prediction-proposal.md`. Signed ON transduction
exposed that the unchanged rule clips all negative observations/predictions to zero.
User was asked to approve preserving these signs via bounded [-1,1] encoding for
new experiments, leaving the local equation and all other invariants unchanged.
The user approved this representation change on 2026-09-18 and requested diagnosis
and repair of T4/T5 silence. Scientific acceptance, CUDA evidence, full-process UI
overhead and M1B learning remain open.

## Signed-current prediction and motion-pathway investigation

Implemented explicit `signed-current-v1` encoding in local updates and metrics.
Absent configuration retains `rectified-current-v1` for old checkpoints. Warm
expansion and baseline comparisons reject differing encodings. No eligibility,
topology, sign, reward, or homeostasis changes. `configs/resting-v1-signed-prediction-v1.json`
is the separate experiment profile. Tests: 76 passed, 3 CUDA skips, including both
signs of unexpected/confirmed/expired predictions, duplicated batches, signed metric
state, exact resume and legacy loading. Separate initial/trained artifacts use the
`checkpoints/signed-v1-visual-` prefix. The 500-step training run completed. Held-out
seeds 1001/1002, 500 steps each: event MSE 0.00115624 -> 0.00103756 (persistence
0.00193687, zero 0.000928278); local MSE 0.00536651 -> 0.00288364 (trained persistence
0.000444486). Thus signs are preserved and error improves, but the stronger zero
event control still wins. Report: `docs/experiments/2026-09-18-signed-v1-heldout.json`.

Frozen current tracing found measured Mi1/Tm3 and Tm1/Tm2 input reaches T4/T5.
Peak voltage was only 0.114 / 0.123 versus threshold 1; even peak excitation <0.48.
T5 inhibition was zero, ruling out inhibitory cancellation as its immediate cause.
T4/T5 have zero intrinsic current in the first resting profile. A preregistered
subthreshold class-current experiment is now testing this operating-point mismatch.
No biological direction-selectivity or M1 acceptance claim follows from waking cells.

Motion calibration completed: the smallest qualifying tested current is 0.95 for
all T4a-d/T5a-d, recorded in `configs/resting-v2-motion-provisional.json`. All other
parameters remain fixed. Eleven-condition independent probes restore T4/T5 activity;
silencing measured non-motion afferents eliminates all T4/T5 spikes without silencing
motion neurons themselves. LPi shows a small response; L4/LC10 remain quiet. See
`docs/experiments/2026-09-18-motion-operating-point.md` and its companion JSON reports.
T4 polarity tuning remains imperfect. This resolves the silence, not full physiology.

77 tests pass, 3 CUDA skips. The disconnected-afferent unit test verifies the
subthreshold profile cannot generate spikes without input. Independent read-only
review found no actionable defects and confirmed the bounds on empirical claims.
`checkpoints/motion-v2-visual-initial.pt` and `checkpoints/motion-v2-visual-trained.pt`
are complete (500 training steps, periodic saves every 100 steps). The paired frozen
evaluation is `docs/experiments/2026-09-18-motion-v2-heldout.json`, including checkpoint
hashes and profile provenance. Held-out seeds 1001/1002, 500 steps each: event MSE
0.00115611 -> 0.00103601 (persistence 0.00193687, zero 0.000928278); local MSE
0.00538999 -> 0.00289317 (trained persistence 0.000445576). Population mean rate
1.134 -> 1.131 Hz. T4 counts 5,817 -> 3,471; T5 counts 1,710 -> 1,421. Motion
populations remain active after learning. The original current audit was rerun with
an E/I reconstruction assertion and produced byte-identical results. Wheel build passed.

Both approved corrections are implemented and verified. No approval question or
experiment process remains pending. The stronger zero-event control still wins;
do not claim complete M1A learning, correct T4 polarity/direction tuning, or Pong
learning. Scientific acceptance, CUDA evidence, full UI overhead and M1B remain open.
Continue from the versioned motion profile and preserve the prior experiments;
any further change to the learning objective requires a concrete design explanation.

## Approved event-aligned learning revision (2026-09-18)

The user approved `2026-09-18-event-aligned-learning-proposal.md`. Separate versioned
options now expose raw local input increments and causal visual eligibility matched
to postsynaptic current decay. Forward dynamics, measured edges, transmitter signs,
behavioral R-STDP and no-backprop remain fixed. Old checkpoints retain old defaults;
new objective metrics are separate from the historical current-proxy metrics.
Delayed-association fixtures demonstrate learning for both signs, but are software
tests, not evidence of effective prediction on the measured connectome. Independent
code review found no actionable defects.

The old-rule 2,000-step control completed and still fails zero-event prediction.
See `docs/experiments/2026-09-18-learning-effectiveness-audit.md`. Frozen feedback
calibration selected C2/C3 rest current 1.0, L4/Lawf1 0.95, with other parameters
fixed. This is the smallest qualifying preregistered candidate; it was selected
using stimulus responsiveness and stability, not Pong scores. Profile:
`configs/resting-v3-feedback-provisional.json`. These are model operating points,
not physiological measurements.

`configs/event-learning-ablation-v1.json` preregisters six conditions with identical
initial graph/dynamics, training seed 1, 500 training steps, and development seeds
1101/1102 for 200 steps. Run/resume each with
`.venv/Scripts/python scripts/run_visual_ablation.py CONDITION` where CONDITION is
legacy, target-only, trace-only, combined, trace-rate, or combined-rate. Each saves
every 100 steps to `checkpoints/event-v1-CONDITION-trained.pt`; audit reports go to
`runs/event-v1-CONDITION-audit.json`. The first three completed conditions
(target-only, trace-rate, combined-rate) still fail zero-event prediction. The
remaining comparisons and independent feedback-silencing probes are in progress.
Reserved final seeds 1201-1204 remain untouched. Do not claim M1A acceptance or
advance M1B on the basis of reduced false positives alone.

Follow-up verification: 96 tests pass, four CUDA skips. The authentic legacy
fixture now removes new state tensors as well as config fields. Independent
eleven-condition feedback probes completed: C2/C3/L4/Lawf1 activity disappears
when their measured external afferents are silenced; targets remain unsilenced.
See `docs/experiments/2026-09-18-event-aligned-learning.md` for evidence and limits.
The approved implementation and preregistered grid are pushed as `1fb7003`.

The fixed-parameter combined-rate continuation is recorded separately in
`configs/event-learning-continuation-v1.json`. Command from the intact 500-step
source: `.venv/Scripts/python -m fly_connectome train checkpoints/event-v1-combined-rate-trained.pt --steps 1500 --output checkpoints/event-v1-combined-rate-2000.pt --threads 1 --checkpoint-every 100`.
If interrupted, resume from the output checkpoint for only the steps remaining
to 2,000 (the CLI `--steps` is additional steps). No final seeds have been used.
Audit the final checkpoint on development seeds 1101/1102 for 500 steps with
`scripts/audit_prediction.py`, which now includes shuffled and shifted-time
diagnostics. Positive lag is a later response and must never count as prediction.

All six 500-step ablations have now completed and all fail the zero-event baseline.
Full paired reports and the combined-rate 500-step timing audit are recorded in
`docs/experiments/2026-09-18-event-aligned-learning.md`. The extra timing audit
uses a longer development sequence, not final seeds. Its aligned/shuffled errors
are nearly identical; no useful anticipation claim is supported. The 2,000-step
continuation is still running. Core tests and feedback probes remain passing.

Windows checkpoint-sharing fix: a progress read overlapped an atomic replacement
and raised WinError 5 during the continuation. The last complete checkpoint was
step 1,600, intact and loadable. Checkpoint writes now retry Windows errors 5/32/33
for at most 30 seconds; unrelated errors fail immediately, and exhausted retries
preserve the previous checkpoint. A fault-injection test failed before the fix;
transient and persistent sharing tests pass. Full suite: 99 passed, four CUDA skips.
The continuation resumed from step 1,600 for 400 additional steps. Do not inspect
the live checkpoint while it is being replaced; wait for the training process exit.
Wheel packaging passed via `pip wheel . --no-deps --no-build-isolation --wheel-dir dist`.
The optional `python -m build` command is unavailable in this environment.

## Current continuation point (2026-09-19)

All experiment processes have finished. The 2,000-step combined-rate continuation
and its frozen development evaluation are complete. Source checkpoint:
`checkpoints/event-v1-combined-rate-2000.pt`; SHA-256
`77299d26385d2e9f6750681ecddb314505fffa9364d7d34eba0f7f48418e7117`.
See the completed section of `docs/experiments/2026-09-18-event-aligned-learning.md`
and `2026-09-19-event-v1-combined-rate-2000-{audit,integrity}.json`.

On matched 500-step development evaluations, pooled event MSE improves
0.001161700 -> 0.001028283 -> 0.000983738 at 0/500/2,000 training steps. Zero
prediction is still better (0.000938364); event-conditioned error is 1.000102
versus zero's 1. Stable activity and active T4/T5 are preserved, and graph/sign/
delay/pathway invariants pass. All six original ablations also failed zero.

The approved bounded revision and verification are complete; M1A effectiveness,
M1B score-only learning, threshold comparisons, CUDA acceptance, and remaining UI
acceptance are not complete. No model change or approval question is pending.
Reserved final seeds remain untouched. The next scientific task is to distinguish
insufficient experience from weak/mistimed local predictive signals with explicit
diagnostics, preserving the current objectives and invariant constraints. Do not
claim acceptance from error reduction alone. Final software suite: 99 passed,
four CUDA skips; wheel build and native Windows sharing-conflict smoke check pass.

## Local-signal and exposure diagnosis (2026-09-19)

Added frozen edge-local moment/current-reconstruction audit and a fast replay of
the exact scripted camera/Pong stream. Tests verify anticipation versus tonic
moments, current reconstruction, checkpoint immutability and exposure replay
against actual Trainer observations. Full suite: 102 passed, four CUDA skips.
CLI training now prints progress only after saved synchronization-boundary
checkpoints, so monitoring does not require opening a live checkpoint file.

Reports: `docs/experiments/2026-09-19-local-signals-and-exposure.md` and accompanying
JSONs. At 2,000 steps, 563/892 L1 neurons received no nonzero projected event and
865/892 received fewer than ten. Median exposure is zero. Local current-trace
moments mostly favor depression; positive correlations are descriptive and not
acceptance evidence. Read-only review confirmed calculations and noted the
decaying warm-up difference from initial plasticity traces; the report states it.

The unchanged 2,000-step checkpoint is being extended to 10,000 under
`configs/event-learning-exposure-continuation-v1.json`. Command:
`.venv/Scripts/python -m fly_connectome train checkpoints/event-v1-combined-rate-2000.pt --steps 8000 --output checkpoints/event-v1-combined-rate-10000.pt --threads 1 --checkpoint-every 100`.
If interrupted, resume the output for the remaining steps to 10,000, not another
8,000. Read progress from stdout; do not open the file while the run is saving.
Then audit the completed output for 500 steps on development seeds 1101/1102.
Final seeds remain untouched unless development criteria pass. No architecture,
learning-rule, sign, topology or physics changes are part of this continuation.

## Current continuation point (2026-09-20)

The 10,000-frame run is COMPLETE. Do not restart or extend it by default.
Checkpoint `checkpoints/event-v1-combined-rate-10000.pt`, SHA-256
`6358107d489ee859f1644105eb274c29b226e6cdc71ae7e58d65a8bf6c925e58`.
Frozen development evaluation on 1101/1102, 500 frames, is in
`docs/experiments/2026-09-20-exposure-generalization-10000.json`.
MSE 0.00095385753 still loses to zero 0.00093836384; event-conditioned MSE
1.0000173. T4/T5 remain active (7,178/1,930 spikes). M1A is NOT passed, M1B has
not advanced, and final seeds remain untouched. Exposure groups are fixed from
first 2,000 training frames for comparison, not recomputed at 10,000.

User prioritized online training efficiency. Profiling found ~92% active time in
plasticity, ~4% in neural dynamics; simulation never waits for live camera events.
10,000 frames are 83.33 simulated seconds, not 10,000 individual events. Removed
three redundant sparse-key binary searches by using unique's inverse map. Balanced
reference/optimized timing improved 2.38 -> 3.02 frames/s (~27% throughput), with
all recorded spikes and final tensor states bit-identical over twenty-frame trials.
Still ~40x slower than real time. All104 tests pass; four CUDA tests skipped.
Read-only review found no actionable issues. Checkpoint format unchanged.

Reproducible tools: `scripts/profile_training.py`, `scripts/benchmark_sparse_merge.py`.
Evidence and next work: `docs/experiments/2026-09-20-training-throughput.md`.
Next engineering chunk should attack measured sparse-learning bookkeeping,
potentially a compiled implementation, without altering local-rule semantics.
Next scientific chunk must diagnose ineffective event anticipation rather than
assume another long unchanged continuation will fix it. No architecture approval
is currently pending, and no training process is running.

Bounded native merge experiment also completed: installed g++13.2 compiled an
explicit opt-in library in ignored runs/. Hand-derived/randomized equality,
input guards, and complete twenty-frame spike/state comparisons pass. Native
merge improves canonical ~3.02 to~3.72frames/s, still not real-time. The production
backend remains PyTorch; no checkpoint or package dependency changed. See
scripts/native_sparse_merge.{cpp,py}, scripts/verify_native_merge.py and the
--native-library experiment flag on scripts/benchmark_sparse_merge.py. Native
trace/error/update fusion is the next measured engineering direction. Do not
repeat the merge experiment as if it had not been done. No training is running.

## Training speed priority — 2026-09-20

User explicitly prioritized near-real-time online learning before more milestone
experiments. Exact opt-in B=1 CPU kernels now cover sparse learning, neuron updates,
arrival extraction, observation preparation and synchronization, with bounded
parallel execution. No topology/sign/dynamics/local-equation changes were made.
An events-only logging mode omits dense error logs but preserves model updates.

Latest100-frame balanced tensor/native test:2.97 versus49–50frames/s, identical
spike/model-state hashes and event scores. Earlier1000-frame run:41.7fps. Realtime
120fps is still unmet. Scientific continuation remains paused and M1A learning
acceptance is still unmet. Existing checkpoints were only read by benchmarks.

Active execution ledger:docs/plans/2026-09-20-realtime-training.md. Next proposal
(needs user approval, not implemented):docs/plans/2026-09-20-deferred-local-updates-proposal.md.
It evaluates repeated quiet local updates with geometric sums within unchanged
synchronization boundaries. It changes floating-point order, so do not silently
adopt it. Relative normalization has not been introduced.

Final1000-frame profile of ABI8:20.545s,48.673fps (about3.4 minutes per10,000
frames projected, excluding startup/checkpoint writes). This supersedes the
earlier41.7fps sample for the final implementation. Realtime120fps remains open.

## Approved deferred execution experiments completed

User approved bounded deferred visual updates and requested lazy neuron analysis.
Implementation and exact fallback are in deferred_cpu.py/native_cpu.cpp; online
behavioral learning and all neural ticks remain unchanged. Geometric regrouping
reaches~52fps (~9.5% above paired native runs), with zero spike-time differences
over1000frames and maxweight delta4.47e-8. Separate manifest identifiers prevent
silent adoption. Exact deferred replay showed no reliablegain.

Lazy-neuron probe found16.3% complete fixed points and95% of spikes without a
same-tick input. Exact sleeping was tested and was5.24% slower; not adopted.
120fps remains unmet. Scientific learning remains paused and M1A still fails
acceptance. Read docs/experiments/2026-09-20-deferred-execution-results.md and
docs/plans/2026-09-20-event-driven-neuron-proposal.md to continue. The latter
requires approval before approximate/regrouped neuron propagation is implemented.
153tests pass,4CUDA skips; no background training process remains.

## Approved long-gap neuron experiment (2026-09-20)

User approved the event-driven proposal and prioritized preserving the biological
model. The wake diagnostic supports long gaps (median neuron mean 30.23 ticks).
The 128-tick prototype deferred 95.99% of neuron updates but changed 930 spike
entries across 8000 ticks, first at tick 250, identically in two repetitions.
It is not adopted. Timings 46-51fps remain below 120 and paired reference timings
were variable. See docs/experiments/2026-09-20-event-neuron-results.md.
Backend switching now materializes old private state before replacing callbacks.
164 tests pass, four CUDA skips. Next model-preserving work: reduce measured
learning/wrapper and fixed retinal projection overhead. Scientific training
remains paused; this result does not establish that biological shortcuts are needed.

Exact projection optimization: immutable retinal mask selections are now cached.
Four balanced 1000-frame native runs preserve every spike and full state hash;
pooled throughput 48.22fps vs46.30fps (~4.1% observed gain). 165 tests pass,
four CUDA skips. Re-enabling a previously used event backend now discards stale
scheduler caches after materialization. See retinal-cache experiment report.

Staged SIMD sparse eligibility experiment is exact but rejected:32.0-32.7fps
versus47.5-51.0fps native over balanced1000-frame runs. Compiler confirms SIMD,
but extra staging/compaction traffic outweighs arithmetic gains. Current native
backend retained.171 tests pass,4 CUDA skips; wheel built and contents verified.
120fps remains unmet. No biological shortcut or changed spike schedule adopted.
The broader persistent compiled-frame runtime proposal is in
2026-09-20-compiled-frame-proposal.md; user direction requested because it changes
runtime/state ownership beyond the completed bounded experiments. Until that
answer, do not implement the proposed runner or resume scientific training.
Final review strengthened the subnormal regression to require actual nonzero
subnormal proposals; final suite173 passed,4CUDA skipped. No benchmark or
training process remains running. Repository is ready to resume from this ledger.

## Bounded optimization pass closed; return to M1A

User approved the compiled-frame proposal but narrowed scope to two exact
software/memory experiments, then stop. Ruling: test scratch reuse and native
arrival filtering as small prerequisites; defer the full runner to honor that
limit. Cost: larger possible runtime gains remain unmeasured.
Seven overhead regressions pass; all six1000-frame trials have identical spike
and complete state hashes. Timing is inconclusive: host variability exceeds2x,
and the first filtered sample overlapped a short test process. Neither candidate
is adopted. No additional throughput experiment is authorized by this pass.
See docs/experiments/2026-09-20-overhead-results.md. Current native backend remains
available;120fps remains unmet. Scientific work is now resumed, not paused.

M1A continuation: frozen predictive-sign support audit on1101/1102,500 frames.
Anatomical sign support is an optimistic limit, not a fitted predictor. Initial
counts find762/2512 signed events structurally unsupported. This is a partial
constraint, not an explanation for failure on the supported remainder. Final
review required explicit event-conditioned versus all-sample MSE denominators;
regression failed then passed after the label/denominator correction.
Next science work: compare initial/trained forecasts separately on sign-supported
ON/OFF events, then examine temporal alignment of their local eligibility signals.
Do not blindly extend the10,000-frame run or change feedback signs/topology.
Final evaluation seeds and M1A/M1B acceptance gates remain untouched.
Final checks:182 tests passed,4CUDA skips. Wheel built and native Python/C++
contents verified. Fresh review found one important reporting issue (MSE
denominators), fixed with a failing-then-passing regression; no remaining
important findings. Scientific checkpoint bytes remain unchanged. No background
benchmark/training process remains. Stop optimization; next work is M1A science.

## M1A supported-forecast and timing diagnosis (2026-09-20)

Preregistered frozen comparison: event-v1-combined-rate-initial.pt and the
unchanged10000-frame checkpoint,500 frames on development seeds1101/1102.
Separate supported/unsupported ON/OFF events and quiet false alarms. Store neural
prediction timelines; lag0 is the preceding-tick forecast, positive lags use
post-event activity and cannot count as anticipation. Use identical target frames
across the -16..+16-tick lag sweep and a fixed frame-shuffle control(seed421).
No fitting, parameter search, learning, new seeds or weight changes. Test the
strict score against ordinary evaluation and hand-check support/lag conventions
before the measured runs. Optimization remains stopped.

Supported-forecast audit completed on both checkpoints with matching graph and
configuration hashes. 99.734% of total error improvement is quiet-period
suppression. Supported OFF forecasts still lose to zero at every tested temporal
offset; positive offsets remain post-event diagnostics, never acceptance scores.
Frozen local signals at10000 frames reconstruct current within2.98e-8; active
L3 predictive traces fall755->74. No acceptance claim or training extension.
Fresh audit review found no actionable issues; comparisons distinguish strict
all-frame shuffle scores from the common-interior lag scores. Strengthened the
integration fixture to exercise nonzero predictive current. Final code tests:
185 passed,4CUDA skipped. No production dynamics/plasticity changes.
Homeostasis-only bound retains>=0.4684465 of starting strength over83.3333s;
2219/3310 L1 predictive edges retain<1%, excluding homeostasis alone as cause.
L3 median incoming predictive weight ratio is1 despite active traces755->74,
so next diagnosis should attribute lost traces to presynaptic classes and their
incoming FF/predictive activity in frozen initial/trained runs. Separate missing
arrivals, subthreshold activity and lost feedback weights. No model revision,
new training run or optimization is implied. Results and reproduction details:
docs/experiments/2026-09-20-supported-forecasts.md. Source hashes unchanged.

## L3 feedback source investigation (2026-09-20)

User requested tracing upstream activity, with longer30-100ms and next-meaningful-
event horizons considered afterward if source tracing is inconclusive. Existing
lag evidence covers only about17ms and does not exclude those longer horizons.
Frozen source audit uses same500frames/seeds1101,1102 and unchanged initial/10000
checkpoints. Reconstruct physical edge traces including warmup, exclude first
frame, classify source spikes and FF/predictive/sensory current by body/cell type.
Post-reset voltage margins describe silent sources, not spike overshoot.
Fixture verifies activity attribution and exact unchanged network tensors.
No model changes, extra training, final-seed use or optimization experiments.
Source attribution completed: C2 accounts for674/681 lost active L3 input traces;
Lawf1 accounts for7. C2 spiking sources669->1 and spikes2350->4; mean recurrent
current0.002239868->-7.5855e-8, while FF current changes<0.3%. C2 rest current1.0.
Preregistered causal test: frozen trained network, replace only existing predictive
magnitudes entering C2 with initial values before warmup. All other weights,
configuration, signs and topology fixed. No checkpoint writes. Hybrid activity
is diagnostic, not trained performance. Selective restoration fixture passes.
C2 restoration completed: existing3091 incoming predictive edges reset to initial
magnitudes in a frozen hybrid, all other parameters/weights fixed. C2 spiking
sources1->671, active outgoing traces1->677, all L3 active traces74->750.
Sensory MSE worsens0.000953858->0.000967914 (zero0.000938364): activity rescue is
not effective learning. Both source checkpoints unchanged. Source tracing yields
a meaningful mechanism; longer30/50/100ms and next-meaningful-event hypotheses
remain open and should be diagnosed before changing target/dynamics, not inferred
from the earlier~17ms sweep. See feedback-source-investigation report.
Fresh review confirms instrumentation/restoration; corrected first-frame wording
(source statistics exclude it, event metrics include it).187 tests pass,4CUDA
skips. No model changes, new training or optimization. All experiment processes
finished. Next bounded scientific question is the local objective/time horizon,
not blind restoration or increased tonic firing.

## Longer-horizon frozen diagnosis (2026-09-20)

User requested next investigations. Preregister30/50/100ms lead, rounded to neural
ticks (lead includes preceding-tick interval), common target frames and fixed
shuffle421. Persistence uses only latest camera frame known at forecast time.
Separately score next nonzero signed input at same sensory neuron within100ms,
from every frame end; omit incomplete tail windows, retain no-event windows,
compare last polarity/opposite polarity/frame persistence and shuffled forecasts.
Overlapping windows are descriptive and not independent evidence. Use unchanged
initial/10000 checkpoints,500frames,seeds1101/1102. No fitted readout, training,
biological changes or final seeds. Tests establish future-only event selection,
tail censoring, and exact earlier-tick indexing before measured runs.
Completed both frozen horizons and bounded next-event diagnostics. Trained all-
sample MSE remains above zero at30.208/50/100ms and next-event window. Supported
OFF event errors remain>1. Weak shuffled advantages are descriptive only.
This does not test retraining with longer-horizon objectives or latent information
in other states. Next evidence should examine existing edge-local trace alignment
with future targets before proposing a causal local rule revision. No blind
extension, new readout or current/excitability fix. Model/checkpoints unchanged.
190 tests passed,4CUDA skips. Fresh review found no blockers; report explicitly
separates100ms frame window from actual sampled tick leads. All measured runs
finished; final seeds untouched. See2026-09-20-long-horizon-investigation.md.

## Edge-local long-horizon signal investigation (2026-09-20)

User authorized a few bounded follow-ups, stopping on meaningful evidence or a
needed change of architecture/goals. First test physical signed predictive-edge
traces onto L1-L3 at one-tick/30/50/100ms and next-event horizons. Extend frozen
observation to2000frames on same development seeds for repeated local events.
Select top10 edges per population/trial on1101; check identical edges on1102.
Require >=5 distinct underlying target events, trace power>1e-10, correlation>=.05,
positive target-trace moment, covariance above shuffled timing421 in each seed.
Seek >=3 distinct validation posts in a trial as actionable exploratory evidence;
these multiple comparisons are not statistical significance or M1A acceptance.
Use common frames, complete next-event windows, ring of exact old physical traces,
and validate replay against actual injections plus current reconstruction.
Initial/trained weights remain frozen; no readout fitting or final seeds.
Review caught next-event repetition counts including unreferenced tail events.
Stopped the preliminary runs before interpreting output. Regression fails then
passes after counting unique frame+wait event IDs actually referenced by retained
windows, separately per seed/neuron. Also strengthened tests for seed separation
(opposite relationships) and ring wraparound. Restart measured runs after fix.
Meaningful exploratory stopping criterion met: trained next-event trial validates
three L2->L1 edges at three posts across1101/1102; one-tick validates none. Two of
these edges also validate initially, and their weights shrink strongly during
training. Initial/trained current reconstruction5.96e-8/2.98e-8. Correlations are
modest, multiple comparisons and reused development seeds prohibit acceptance
claims. Stop expanding experiments per user instruction. Next work is a bounded
proposal for causal local temporal credit, including event-history controls,
quiet false alarms and C2 activity, not an unapproved implementation or blind run.
A secondary read-only C2 sign inventory is in runs/c2-sign-audit.json (909 FF
edges,907 inhibitory); it was not used as a causal result or model-change basis.
194 tests passed,4CUDA skips. Source checkpoints unchanged; final seeds untouched.
See docs/experiments/2026-09-20-local-trace-horizon-findings.md. No remaining
experiment processes; no architecture, target, signs or topology changes.

## Proposed local confirmation window (2026-09-20)

User asked to continue from the longer-horizon trace finding. Wrote concrete
proposal2026-09-20-local-confirmation-window-proposal.md; no rule implementation.
GateA: fixed three pairs, new development1103/1104,2000frames, local-history-
stratified shuffles; stop if no incremental information or inadequate evidence.
If approved and GateA passes: sensory predictive edges only,100ms first-event/
expiry confirmation, stored issuing prediction/eligibility, fixed eta/H scaling,
unchanged other visual rules/R-STDP/dynamics/signs/topology. Exact pending-state
resume and one B1 CPU arm to10000frames, no automatic parameter search. Original
next-tick metrics and final-seed/M1A gates remain intact. Written spec self-reviewed
for causal ordering, deadline ties, zero-weight eligibility, no-event scoring,
private state, schema compatibility, overhead and distinct acceptance semantics.
Await approval under user's model-change boundary before implementing new rule.

## Approved confirmation-window Gate A execution

User approved written proposal. Implementation plan recorded; executing inline
without repeat chunk approvals. Added fixed-candidate history gate, no production
learning changes. Tests cover causal history, within-stratum label preservation,
constant strata, both-seed pair aggregation, frozen collector and reconstruction.
Run exactly2000frames on1103/1104 against10000 checkpoint, candidates unchanged,
10conditional shuffles421..430. Stop if fewer than two pairs pass on both seeds
or evidence is inadequate, as approved. Do not substitute other candidates.
Gate A completed and FAILED:0/3 fixed pairs pass both1103/1104;2 required.
All comparisons have>=5 distinct events, nonzero power and98.58-100% label-
permutable windows. Two pairs fail history-controlled shuffles on both seeds;
third beats shuffles on1103 but correlation .044<.05, and fails shuffles on1104.
Not a missing-data inconclusive outcome. Stop approved proposal here: no rule
implementation, training, alternative candidates, threshold changes or sweeps.
Prior raw correlations did not establish incremental information beyond history.
198 tests passed,4CUDA skips; independent review no findings; reconstruction4.66e-10.
Source/checkpoint/model unchanged; final seeds untouched. All processes finished.
Next action requires a new discussion/proposal about local target and circuit
representation, not continuation of conditional GatesB/C. See GateA report.


## Approved controlled moving-dot test (2026-09-20)

User approved replacing full-Pong failure investigation with a bounded small
controlled test. Added scripts/controlled_visual.py and seven regression tests.
No production sensor, neuron or learning code changed. Fixed anatomy-only crop:
nine retinal columns and measured two-hop return intermediates, induced graph
112neurons/509edges. Original weights/signs/delays/pathways/dynamics/retinotopy.
Bright dot, fixed rightward motion, random blank intervals; 200train/50eval
traversals, seeds9001/9002, eight-tick primary horizon, original one-tick rule.

First execution112.11sec. Review found missing continuous raw-state guards and
misleading zero frozen eligibility arrays; regression tests resolve both. Same
verified replay67.52sec, combined179.63sec<600sec. Forecasts/targets/spikes/arrivals/
final weights bit-identical. No sweep. Raw states finite, weights bounded.
205tests pass,4CUDA skips. Source unchanged; full traces saved locally at
runs/controlled-visual-v1-verified; checksums and edge mapping in committed results.

FAIL: trainedMSE.05772970, frozen.05782128, zero.05770341; stimulus control perfect.
No useful event anticipation. Major test limitation: L3 has only inhibitory
prediction inputs (all silent here), making positive OFF prediction impossible
at this cell. L1's only inhibitory sourceC2 is silent. This crop is not a clean
learning-capability test; sign support should have been checked before training.
70neurons spike, so not whole-network silence. Target incoming proposal sums:
ON-.12377/OFF+.11799/quiet-.45127. See controlled-visual-findings report.

Stop bounded attempt. Next small test needs a measured motif with verified
sign-compatible output and actual precursor response before target, checked
with a short frozen preflight before any more training. No model changes,
full-Pong runs or M1B advancement. M1A remains unmet. No experiments running.


## Frozen compatible-motif preflight (2026-09-21)

User asked to continue. No additional training or model changes. Read-only
anatomical inventory:136/137 predictiveL2->L1 edges share mapped column; sole
cross-column pair22312->24823; none spans tested rightward2-4pixel interior gaps.
Fixed previousL1target38366 and trajectory. Retain all3predictive sources plus
ALL their direct parents and trajectory sensory neurons:63neurons/274edges.
Mixed measured output signs; exact retained parameters, no renormalization.

Ten frozen paired probes(seed9011), full dot vs separately blanked pre-ON/pre-OFF
images from matched warmed state. Read8ticks before actual target; physical
trace reconstructed from recorded spikes/delays.6.76sec execution; gateFAIL
ON0/10 OFF1/10,8/10 each required. Reconstructionerror8.38e-9<1e-6. C2supplies
zero spikes in all10 full probes despite all5direct inputs retained; positive
arrival impulse totals+.700 versus inhibitory-13.515. These sums do not prove
voltage causality. Most other positive forecast residuals equal blank controls.
OneOFFtrial has0.122628 full vs0.111344 blank, insufficient consistency.

207tests pass,4CUDA skips. Focused review no code blockers; clarify correctly
signed frozen current is not necessary for eventual learning because weights
could unmask opposed signals. No global learning-impossibility claim. Source
checkpoint unchanged; weights remain frozen; no final seeds/full-Pong/M1B work.
See docs/experiments/2026-09-21-visual-preflight-findings.md and local probe data
in runs/visual-preflight-v1. No processes running.

Next recommendation: specify a repeating two-position dot around an actual
same-column L2->L1 motif so a directly stimulated precursor can support temporal
prediction. Retain positive feedback if both polarities are scored. Frozen
preflight before training, no automatic architecture/timing/excitability sweep.


### 2026-09-21: temporal visual motif and extended training

Implemented scripts/temporal_visual.py with 200/1000-trial options and three focused tests. Anatomy inventory selected measured L3 82450 with directly driven L1 inhibitory and L2 excitatory predictive inputs. Frozen preflight passed 30/30 ON and OFF availability probes. The 250-neuron/1664-edge induced circuit preserves retained topology, signs, dynamics and local rule. At 200 trials local ON anticipation improves; causal target-input weight swaps reproduce/remove the gain. OFF remains wrong-sign. Conditional fixed-spike feature analysis distinguishes timing-objective mismatch from insufficient phase discrimination without changing model weights.

User requested 1000 training steps, interpreted explicitly as 1000 total trials matching the prior unit. First 200 weights and all recorded traces match exactly. 48,905 camera frames / 391,240 ticks took 512.4 seconds (95.4 FPS); total including storage and frozen evaluation 524.46 seconds. Same 50 evaluation trajectories: MSE 0.134747 (200) -> 0.133457 (1000), ON 0.928360 -> 0.906754, OFF 1.002372 -> 1.003115. Gate still fails, M1A unmet. No full-Pong/M1B advancement or architecture changes. Diagnostic overhead was not separately profiled.

See docs/experiments/2026-09-21-temporal-visual-findings.md, protocol, numerical results, causal swaps, feature analysis, learning curve and artifact hashes. Raw arrays remain under runs/temporal-visual-v1 and runs/temporal-visual-1000. Next: discuss discriminating predictive traces and horizon mismatch; do not assume more repetitions alone will fix OFF. Fresh validation: 210 passed, 4 skipped.


### 2026-09-21: user-authorized matched frame horizon

Implemented experimental frame-horizon-v1 in scripts/frame_prediction.py and an opt-in controlled/temporal runner schedule. Eight neural ticks/frame remain; visual forecasts issue at boundaries and are confirmed +8 ticks with retained local issue-time eligibility. Intermediate visual penalties removed; quiet camera targets retained. Production rule/default, topology, signs, dynamics and behavioral learning unchanged. TDD: missing-module red, then timing/quiet/behavior tests green; diagnostic integration verified. Full suite 215 passed, 4 skipped. Review found no blocker; pre-observe saved eligibility indexing documented.

Same 200-trial inputs/initial weights and exact frozen controls: frame supervision MSE 0.121254 vs 0.134747, ON 0.543302 vs 0.928360, ON anticipation 0.270484 vs 0.036593. OFF still wrong-sign (MSE 1.017638), quiet false alarms 16.30%; fixed gate fails, M1A unmet. Exact learned paired precursor probes confirm stimulus-dependent ON improvement (signed full-minus-blank 0.277513 vs 0.037152). Runtime 85.53s + 10.74s paired probes. See docs/experiments/2026-09-21-frame-horizon-findings.md and protocol/results. No further training sweep. Next architectural proposal should address temporal discrimination across existing signed pathways, requiring approval before changing synaptic dynamics.


### 2026-09-21: OFF sign audit and signed predictive kinetics

User authorized narrow OFF sign trace followed by bounded distinct-time-course experiment if no correctable sign error. scripts/off_sign_audit.py audits three recurring OFF cases each on initial/learned copies, with tickwise source sensory/current/spike state, signed contributions, issue eligibility and update proposals. All six: OFF target +1, excitatory positive proposal, inhibitory negative proposal, both correcting toward positive. No accidental flip. OFF eligibility small and nearly equal/opposite across L1/L2; prior ON proposal ~26x stronger in the illustrated case. Source artifacts unchanged.

Implemented opt-in slow-excitation-v1 (excitatory predictive decay20ms, inhibitory5ms; matching local eligibility) with small default-preserving Network hooks. Preserves topology/signs/no-backprop and other dynamics. Experimental state not supported by native engines/production checkpoints. Initial run stopped before training due to old reconstruction omitting warmup tails; regression red->green, include warmup spikes, unchanged1e-6 tolerance. Successful reconstruction error7.63e-9.

Same200 training trials and50 frozen eval trials: learned ON anticipation0.290590, OFF0.159964, all150 events of EACH polarity correct-sign. MSE0.107658 vs owninitial0.137102; ON0.522855, OFF0.707032. All error/anticipation gates pass; quiet false alarms37.08% still fail5% limit. Paired learned-minus-blank effects ON0.258903, OFF0.150228, showing stimulus dependence. Runtime98.18s +14.20s paired probes. Gain confound documented: fixed-amplitude20ms excitation increases impulse area. No parameter sweep.

Report: docs/experiments/2026-09-21-off-sign-and-kinetics-findings.md; protocol, tick CSV/JSON, raw result JSON/hashes, and plot alongside. Full suite218 passed4skipped; independent review no blockers. M1A remains unmet. Next proposal should address quiet-frame temporal selectivity and separate impulse area from decay shape, preserving two-polarity gains; no silent promotion of candidate.


### 2026-09-21: exact-area control retains OFF learning

User requested20ms excitation with impulse area matched to5ms, then temporal-selectivity diagnosis. Added opt-in area-matched-excitation-v1: gain(1-exp(-dt/.020))/(1-exp(-dt/.005))=0.2698567 atdt1/960. Gain consistently applied to current and local eligibility; inhibition/defaults unchanged. TDD missing-class red, then impulse integral and current/eligibility regression green. Full suite219 passed4skipped, review no blocker. Native/checkpoint support not extended.

Same200trial training: ON anticipation0.281664, OFF0.110316, all150 events per polarity correct-sign. MSE0.108329, OFFerror0.792185, quietfalsealarms25.69% vs37.08% unscaled. Ownfrozen MSE improvement19.927% narrowly misses20%; quietgate alsofails; M1A unmet. Magnitudes partiallycompensate (L2raw0.604755,effectiveimpulse0.163197 versusold0.231282). Frozen identicaloldlearned weights: uniform5ms OFFanticipation-0.014790, area20ms+0.028026, unscaled20ms+0.159964. Supports temporalshape contribution to sign, additionaldrive contribution to strength; coupled spikes allowed to change. Paired learned OFF-minusblank0.093959 confirms stimulus dependence.

Quiet diagnosis: transitionphase errors and first6blankframes dominate; only6/926falsealarms fromblankframe7onward. No weakeningOFF or additionalarchitecture bundled into experiment. Next bounded proposal should improve transient localization while retaining signedmemory. See docs/experiments/2026-09-21-area-matched-kinetics-findings.md and protocol/results/controls/quietcounts/hashes. Runs under runs/temporal-area-matched-excitation-v1 and runs/area-matched-frozen-control-v2.


### 2026-09-21: rise/decay candidate rejected for localization

User approved sign-preserving5msrise/20msdecay excitatory response atcontrolledarea. Added scripts/rise_decay.py localcurrent andmatchingeligibility components, opt-in runner andindependent warmup-inclusive reconstruction. Eightticks/frame andframe-horizon learning unchanged, inhibition/topology/signs/no-backprop preserved. Unitimpulse startszero, peaks~9ms, has5msreferencearea; matchingissue-timeeligibility verified. Unsupportedone-tickmode rejected. Fulltests221passed4skipped; independentreview noblockers.

Same200training/50frozen eval: rise/decay ONanticipation0.308595, OFF0.169321, all150correctsign each. MSE0.103932 vspriorarea0.108329; howeverquietfalsealarms38.69% vs25.69% (717vs476/1853). EarlyOFF142/150, immediatelyafterOFF92/150, lateblank53/926 vs6/926 previously. Rejectcandidate aslocalizationfix; preservepriorarea modelasreference. 20ms tail remains despite delayedpeak. No parameter sweep.

Current reconstruction5.38e-9, preflight30/30both, inputs/initialweightsidentical, source/frozenweightsunchanged, statesfiniteweightsbounded. Runtime107.95s. See docs/experiments/2026-09-21-rise-decay-findings.md, protocol/results/phasecounts/hashes. M1Aunmet. Nextarchitecture discussion should distinguish retainedmemory frombrief eventforecast ratherthan onlyreshape long-livedcurrent; notimplemented orassumedapproved here.


### 2026-09-21: separate local contrast forecast rejected at preflight

User authorized continuing localmemory/brief-eventforecast proposal. Implemented event-contrast-v1: physicalarea-matched20/5ms dynamics unchanged, forecastrawI(t)-I(t-8), matching capturedlocaleligibilitydifference. Eighttickframeconfirmation, signs/topology/no-backprop preserved. No derivativefeedback tomembrane. All20 frozenprecursor spike/target/arrivalarrays exactlymatchpriorarea control; unitphysicalstateequality verified.

Preflight ON10/30 OFF30/30 fails24/30minimum. Allten cases firstscoredONcyclepass, second/thirdfail: sourceL2/L1 contrastfeatures bothpositive (example+0.101629,+0.276630 atissue272,targetON-1at280), because negativeinhibitorycurrentdecays towardzero. Correctphysicalsigns but unsuitableforecastencoding. Reconstruction6.39e-9. Stoppedbeforetraining after4.50s; no gatebypass orfurtherparametersearch.

Tests223passed4skipped; independentreview noblockers. Docs/experiments/2026-09-21-event-contrast-findings.md plusprotocol/results/hashes recordfailure. Existingmemorymodel remainsreference, defaultsunmodified, M1Aunmet. A futureconcretedesign shouldpreservesignedmemory whilecontrollingexpressiontime (e.g.nonnegative localtiminggate); gatemechanismnotyetdesigned/implemented.


### 2026-09-21: existing local timing information confirmed

User requested information audit before another mechanism. Frozen area-matched model, forecast-issue local features only, +8tick labels; old prediction/target/spike arrays exactly reproduced. Fixed15-neighbor offline probe, train/calibration/test split25/10/15 trials, fresh50 trials without retuning. No neural learning or architecture change. Current components alone: fresh ON144/150(96%), OFF146/150(97.33%), quiet82/1922(4.27%); only6/600 adjacentquiet false positives. Synaptic features:96.67%/100%/3.75%, zero adjacentquiet errors. Labelshuffle AUC0.459; 1% feature noise preserves result. Trialbootstrap quiet95%interval3.49-5.20%, so no guarantee of gatepass.

Post-result scalar-only ablation:75.33%ON/89.33%OFF/3.49%quiet, supports separate excitation/inhibition access. Positive evidence justifies designing local timing expression using existing state; no new temporal cue currently required for this fixed periodic task. Offline classifier is not a biological gate or local-plasticity success; M1A remains unmet. No new gate implemented. Fulltests225passed4CUDA skipped, review no blockers, diagnostic33.20s. See docs/experiments/2026-09-21-local-information-findings.md and protocol/results/bootstrap/scalar-ablation/artifacts/plot. Reproduce scripts/local_information.py with data extra; local raw features in runs/local-information-v1.


### 2026-09-21: simple E/I boundary found, margins fragile

User requested inspect simple local boundaries before an elaborate learned gate. Diagnostic-only nine fixed families, two polarity-associated windows/rectangles, training quantile bounds and calibration quiet cap5%. Choices saved before fresh seed9025 frozen50trial collection. Selected total current S=E+H and inhibitory fraction R=H/S windows; expressible as eight linear comparisons, no division needed. Fresh ON139/150 OFF139/150(92.67%both), quiet36/1869(1.93%), only3/600 adjacentquiet errors. Existing neighbor comparator96.67%/99.33%/4.23% on same new batch. No network change or gate implementation.

Critical limitation:1% training-SD independent current noise drops recall59.33%/78.67%; contracting bounds1% coordinateSD gives17.33%/67.33%. Expanding gives94%/96%/4.55%quiet, not tested with noise and not promoted. Post-result margin audit113/139 detectedON and38/139 OFF within1%SD of a boundary. Quantile thresholds sit on repeated event clusters. Next recommendation: margin-focused calibration of same simple windows before mechanism design, no new temporal cue yet, no mini-model. M1A unmet. Fullsuite227passed4skipped; source/weights/topology/signs/locallearning unchanged. See docs/experiments/2026-09-21-simple-ei-boundary-findings.md and associated protocol/results/choices/frontier/margins/plot. Script scripts/simple_ei_boundary.py; raw runs/simple-ei-boundary-v1.


### 2026-09-21: frozen E/I boundary fails tempo transfer; padding does not repair it

User required timing transfer before margins, then margins regardless. Frozen area-matched weights, boundary and original current-only neighbor probe; dwells2/4/6,25 matched-blank trials each,8ticks/frame,+8tick forecast fixed,first complete cycle excluded. Original boundary ON/OFF recall0/0%,40/0%,1.33/0%; quiet8.75%,14.88%,8.21%. Neighbor also transfers poorly. Six-frame signedON only10/75 correct; OFF75/75. Regions describe trained tempo, not demonstrated general event timing. No hardcoded gate added.

Then original train/calibration-only25-candidate padding search with five1%currentSD noise seeds selects2%coordinateSD total/ratio expansion. Choices frozen before new50trial seed9027 control. Padded clean94%ON/96.67%OFF/5.08%quiet; noisy93.33-94.67%ON/89.33-93.33%OFF/4.86-5.35%quiet. Stronger noise tolerance but robustness criterion fails. Padded tempo-transfer OFF remains0% all newtempos; quiet increases. No further tuning or gate promotion.

Recommend varied predictable-tempo local-learning experiment with frozen held-out-tempo evaluation and separate online adaptation, before further gate design. This failure does not prove absent local information at other tempos or architectural incapacity; model trained one tempo only. Default collector regression exactly matches old firsttrial features/targets/phases/predictions/spikes. Fullsuite232passed4skipped. No production neural changes; no-backprop/topology/signs/localrule preserved, M1A unmet. See docs/experiments/2026-09-21-timing-transfer-findings.md, protocol/results/frozen choices/plot; scripts/timing_transfer.py transfer then margin; runs/timing-transfer-v1 raw arrays.


### 2026-09-21: mixed-tempo learning fails; specialization partially restores forecasts

User authorized next meaningful learning/transfer/adaptation investigations. Added reproducible staged runner (98ff1ae):200 interleaved trials dwells2/4/6 (67/67/66), heldout3, actual neural predictions only, then100 adaptationtrials3. Architecture/topology/signs/no-backprop/frame-localrule unchanged, no offline gate. Mixed training11158frames164.62s; adaptation4913frames73.95s. Both finite/bounded, checkpoints/traces saved. Evaluation15 matched-blank trials/tempo45ON45OFF, identical targets asserted and weights frozen.

Mixed ONanticipation2/3/4/6=.03544/.08145/.07702/.03887; OFF=-.01279/-.00009/.00133/.00054; quiet6.98/9.24/6.58/9.96%. All controlled-task gates fail. L2->L3 magnitude.025initial to.012119mixed versus.604755prior single3. Actual mixed L2 proposed updates ON-2.08765,OFF+.82444,quiet+1.01080: not simply quiet suppressing excitation. L1 ON+2.30352,OFF-.29282,quiet-1.94497. Opposing class updates describe this run, not proof of irreducible conflict.

Adaptation3 lifts L2 to.391117,L1to.519809. Heldout3 anticipationON.24850/OFF.06941, all45signs correct each, MSE.12842to.11239, but quiet15.99% andOFF<.1. Neighbor4 event forecasts also improve; tempo6 MSE.09569to.11448 worsens19.6%, quiet17.53%. All adapted gates fail. Local learning can retune predictions, but does not yield robust cross-tempo prediction. No blind extension/threshold gate promoted; M1A unmet,M1Bblocked. Next discriminating audit: whether local states/available history distinguish incompatible future targets across tempos before architecture change. One training seed/batch, fixed two positions; no general vision claim.

Fulltests233passed4skipped; source checksum verified. Findings/results/weights/updates/artifact hashes/plot at docs/experiments/2026-09-21-multitempo-*. Raw stages runs/multitempo-v1. Reproduction scripts/multitempo.py train,evaluate,adapt,evaluate-adapted; completed evaluations resumable.


### 2026-09-21: local context differs despite failed cross-tempo decoding

User requested whether existing membrane/adaptation/synaptic history distinguishes same-E/I different-future cases, before adding context. Frozen mixed model30trials each2/3/4/6 seed9070. Fit1-15 andcal16-20 at2/4/6,test21-30; all3 heldout. Fixed15NN fourgroups, pertempoquietcap5%, no metadata features, signed confusion andnoise/shuffle controls. Heldout3 all-local detects1/90ON17/90OFF, quiet4.37%, AUC.64; EI1/90ON0/90OFF,quiet9.90%. Synaptic history no useful improvement. Probes fail; this is not proof of missing information.

Cross-tempo nearest EI matching withoutlabelselection identifies275 close conflicting pairs (139atheldout3,79ofthoseeventqueries). Post-result audit: existing adaptation differs >1%trainingSD in ALL275. Example unseenONversusquiet EI RMS.00148SD but adaptation.93523vs.51263(2.26SD). Full-state L-infinity matches at1%SD21 and5%SD74; none conflicting, but matches rare, so not sufficiency proof. Existing context can distinguish these examples; general predictive mapping remains unproven. Single shuffled control heldoutAUC.655, noeventrecall; no significance claim fromAUC.

Recommendation: first discuss explicitlocal coupling of existing adaptation into prediction/plasticity, not addingmemory solelyfromprobe failure. Findings compare later short-term synaptic dynamics, localintervaltrace, recurrentcontext costs/risks with primaryresearch references; no architecture change/thresholdgate/extra state implemented. No-backprop/topology/signs/localrule preserved,M1Aunmet. Fulltests235passed4skipped. Docs/experiments/2026-09-21-context-audit-findings.md plusprotocol/results/proximity/pairs/plot; scripts/context_audit.py; raw runs/context-audit-v1.
