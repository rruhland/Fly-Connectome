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
- [ ] 2. Reproducible data and anatomy (`data.py`, `anatomy.py`, corresponding tests).
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

## Current state

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
