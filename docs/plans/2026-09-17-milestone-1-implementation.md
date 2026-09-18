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
`checkpoints/signed-v1-visual-` prefix; the 500-step experiment is in progress.

Frozen current tracing found measured Mi1/Tm3 and Tm1/Tm2 input reaches T4/T5.
Peak voltage was only 0.114 / 0.123 versus threshold 1; even peak excitation <0.48.
T5 inhibition was zero, ruling out inhibitory cancellation as its immediate cause.
T4/T5 have zero intrinsic current in the first resting profile. A preregistered
subthreshold class-current experiment is now testing this operating-point mismatch.
No biological direction-selectivity or M1 acceptance claim follows from waking cells.
