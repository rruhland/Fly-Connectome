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
- [ ] 3. Events and retinotopy (`sensor.py`, `tests/test_sensor.py`).
  Packed polarity events, persistent frame reference, fixed full-field hex mapping,
  registered L1-L3 projection only. Verify flashes, motion, empty and batched frames.
- [ ] 4. Continuous body and Pong (`pong.py`, `tests/test_pong.py`).
  Vector physics, deterministic independent reset RNG, acceleration-limited opponent,
  visible resets, signed rate drive, bounded curriculum. Verify collisions, resets,
  braking/reversal/rest and absence of privileged state in observation interface.
- [ ] 5. Sparse adaptive LIF (`dynamics.py`, `tests/test_dynamics.py`).
  Fixed delays, current and membrane leaks, refractory/adaptation, sparse delivery.
  Verify hand-calculated spikes, quiet input, batch independence and CPU/CUDA parity.
- [ ] 6. Local plasticity (`plasticity.py`, `tests/test_plasticity.py`).
  Resolve prediction/observation definition before implementation. Active-edge
  eligibility, natural decay, separated regional rules, slow homeostasis, sign bounds.
  Verify confirmed/expired/unexpected predictions, reward isolation, and that
  eligibility has no forward effect. No dense batch-by-all-edges eligibility.
- [ ] 7. Training/checkpoints (`training.py`, `checkpoint.py`, integration tests).
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
