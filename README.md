# Fly-Connectome

Milestone 1: event-camera Pong through a measured MaleCNS subgraph, adaptive spiking
neurons, local predictive plasticity and reward-modulated STDP. No backpropagation,
learned encoder/readout, or added internal connections.

Repository: https://github.com/rruhland/Fly-Connectome

**Status:** Implementation and verification in progress. Passing software tests are
not evidence that the connectome has learned Pong. See the
[approved architecture](docs/plans/2026-09-16-milestone-1-design.md) and
[continuation ledger](docs/plans/2026-09-17-milestone-1-implementation.md).

## Setup

Python 3.11+ and PyTorch 2.5+ are required. Install the appropriate PyTorch CPU or CUDA
distribution for your machine, then install this package and its data/test extras:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[data,test]"
.venv/Scripts/python -m pytest -q
```

CUDA tests skip explicitly if CUDA is unavailable. This development checkout currently
uses CPU-only PyTorch; CUDA correctness and performance are not yet established.

## Reproduce the measured graph

The pinned inputs are official [MaleCNS v1.0 downloads](https://male-cns.janelia.org/download/).
The source manifest includes URLs, byte counts and SHA-256 checksums. Downloads total
approximately 5.8 GB; extraction requires additional RAM and disk space. Raw inputs,
derived caches, checkpoints and runs are ignored by Git. Existing inputs must match
the recorded hashes; a modified input fails verification rather than being overwritten.

```powershell
.venv/Scripts/python -m fly_connectome download
.venv/Scripts/python scripts/prepare_malecns.py
```

Selection rules are fixed in `extraction.Selection` before any training: right optic
lobe, named visual seeds, measured paths toward bilateral DNa02, explicit reciprocal
expansion and bounded SCC inclusion. The final roster has every induced directed
edge; T1/T3/T5 retain the same roster. T5 is selected only if anatomy passes, otherwise
T3 is considered. Failure of both stops initialization. No threshold is chosen by score.

Each ordered body pair becomes one edge, initialized linearly from contact count.
Signs use curated ground truth or confident individual/cell-type transmitter evidence.
The fixed current-model convention treats acetylcholine as excitatory and
glutamate/GABA/histamine as inhibitory; it is not a claim about every receptor.
Unresolved signs are excluded before extraction and missing required populations fail.
The manifest preserves sign evidence, body IDs, source checksums, ROI metadata,
recorded soma positions, pathway partitions and the default fixed delay model.

## Train in stages

```powershell
.venv/Scripts/python -m fly_connectome init --stage M1A --output checkpoints/visual-initial.pt
.venv/Scripts/python -m fly_connectome train checkpoints/visual-initial.pt --steps 1000 --output checkpoints/visual-trained.pt
.venv/Scripts/python -m fly_connectome init --stage M1B --output checkpoints/pong-initial.pt
.venv/Scripts/python -m fly_connectome init --stage M1B --warm checkpoints/visual-trained.pt --output checkpoints/pong-warm.pt
.venv/Scripts/python -m fly_connectome train checkpoints/pong-warm.pt --steps 31000 --output checkpoints/pong-trained.pt
```

These are example run lengths, not a claimed sufficient training budget. M1A runs an
open-loop scripted Pong stream on the visual-only induced graph. M1B preserves matching
visual weights and adds measured central/motor pathways. `--threshold 1`, `3` or `5`
changes only the measured edge mask. `--seeds 1,2,3` enables shared-weight batching;
B=1 is canonical. `--device cuda` selects the same tensor implementation on CUDA.

The default reward schedule bootstraps for 10,000 environment steps, fades over the
next 20,000, then uses score only. There is one miss penalty. Reward never enters the
observation path. Scores reset body/actuator state but preserve neural dynamics,
eligibility and event-camera reference. Checkpoints include pending weight proposals
and all private state for exact resume. Ctrl+C requests a save at a synchronization
boundary. The CLI prints coarse metrics; no full spike histories are logged in training.
Use `--threads 4` to set the CPU thread count and `--checkpoint-every 1000` for
periodic saves. The worker also saves periodically and on safe stop.

Predictive edges use their local signed causal eligibility and the difference between
next-tick feedforward observed current and the preceding recurrent predicted current,
normalized by the fixed firing threshold. Feedforward observation edges do not receive
prediction-error updates. Behavioral edges use arrival-timed pair STDP eligibility.
Homeostasis integrates elapsed simulated time and does not force minimum firing.
No rule creates edges or changes fixed signs.

## Evaluate and diagnose

```powershell
.venv/Scripts/python -m fly_connectome evaluate checkpoints/pong-trained.pt --initial checkpoints/pong-initial.pt --seeds 1001,1002,1003 --steps 10000 --output runs/evaluation.json
.venv/Scripts/python -m fly_connectome diagnose checkpoints/visual-trained.pt --output runs/flash.pt
.venv/Scripts/python -m fly_connectome diagnose checkpoints/visual-trained.pt --silence L1,L2 --output runs/flash-silenced.pt
```

Evaluation creates fresh private state on held-out seeds, loads weights read-only and
has no plasticity object. It compares learned, frozen, random-motor and scripted
controllers under fixed physics, with hit shaping zero. Reports distinguish local
current prediction from sensory-event prediction, each against persistence. Acceptance
still requires measured post-fade learning, baseline improvements, stable activity and
CPU/CUDA parity; the software does not manufacture a success verdict.
For M1A checkpoints, `evaluate` compares trained and initial weights on the same
scripted held-out sequences and reports per-cell-type spike counts. Comparison
rejects different neuron dynamics, sensor mappings or delay/pathway assignments.

Probe JSON can specify `kind` (`flash`, `moving_edge`, `target`, `neighbor_sequence`),
`steps`, `start`, `duration`, `x`, `y`, `radius`, `direction`, `speed`, and `polarity`.
Diagnostic bundles include source checkpoint hash, graph manifest, stimulus, seed,
backend, spikes and population responses. They never overwrite the checkpoint.

## Optional local browser interface

```powershell
.venv/Scripts/python -m fly_connectome.ui.server checkpoints/pong-warm.pt --directory runs/training-ui
# In another process, for frozen evaluation:
.venv/Scripts/python -m fly_connectome.ui.server checkpoints/pong-trained.pt --evaluation --port 8766 --directory runs/evaluation-ui
```

Open the printed loopback URL. Start, pause, resume, safe stop, save and reset commands
are handled by an independent worker at synchronization boundaries. Restarting from
initial weights requires explicit confirmation. Closing/reloading the browser or
server leaves the worker running. The sampled preview uses actual environment state,
is labeled delayed, and uses bounded latest-value shared memory. Without a subscriber,
capture/serialization is disabled. Training never streams weights, eligibility or spikes.

Evaluation offers slower playback, single-step, control comparisons, visual probes,
recorded anatomical positions and measured-connection inspection. Missing positions are
not fabricated. UI completeness and the <=1% throughput target remain verification
items in the continuation ledger.

Pass `--comparison-checkpoint checkpoints/visual-initial.pt` (repeatable) to an
evaluation server to make additional read-only checkpoints available to probes.
Each UI probe saves a separate bundle under the run's `diagnostics/` directory.
Choose Response A/B and a cell type to compare traces, latency, mean rate and
sparsity; scrub Replay tick to show its stimulus and recorded anatomical activity.
Anatomical overlays match body IDs across different graph sizes. Event viewing
supports raw latest samples and accumulated frames. Diagnostic results remain
exploratory and separate from training.

## Current empirical result

The first real T5 M1A pilot (3,000 training steps, seed 1) did not establish learned
visual dynamics. On held-out seeds 1001/1002, only L1-L3 spiked; T4/T5 and projection
populations were silent. Initial and trained predictions both beat persistence,
but differed negligibly from each other. This is not milestone success. The
zero-background LIF model and positive-only L1 ON injection motivated the approved
operating-point correction below. See `docs/experiments/2026-09-17-m1a-pilot.json`.

The fixture producer benchmark produced byte-identical headless/preview checkpoints.
Its noisy median timing showed no measurable overhead, but excludes browser/server
CPU contention; the full <=1% UI acceptance check remains open. CUDA tests are skipped
on the current CPU-only PyTorch installation.

## Approved resting-current experiment

`configs/resting-v1-provisional.json` supplies fixed cell-class currents/timescales
and signed lamina contrast transduction (ON hyperpolarizes, OFF depolarizes).
It was selected on frozen non-Pong edge probes from a six-candidate preregistered
grid. Parameters are explicit spiking-model approximations, not measured firing
rates. Topology, signs and plasticity rules are unchanged. There is no firing floor.

```powershell
.venv/Scripts/python scripts/calibrate_resting.py
.venv/Scripts/python -m fly_connectome init --stage M1A --profile configs/resting-v1-provisional.json --output checkpoints/resting-v1-visual-initial.pt
.venv/Scripts/python -m fly_connectome train checkpoints/resting-v1-visual-initial.pt --steps 500 --threads 4 --checkpoint-every 100 --output checkpoints/resting-v1-visual-trained.pt
.venv/Scripts/python -m fly_connectome evaluate checkpoints/resting-v1-visual-trained.pt --initial checkpoints/resting-v1-visual-initial.pt --seeds 1001,1002 --steps 500 --threads 4 --output runs/resting-v1-heldout.json
```

Omitting `--profile` preserves legacy dynamics. Existing checkpoints are not
converted; schema 2 records the profile and sensory-current state for exact resume.
Use the same profile for later warm expansion. Fresh evaluation uses a fixed
no-event neural warm-up, while exact training resume restores the saved state.
Diagnostics report baseline firing and responses above baseline; signed contrast
evaluation labels its target encoding and includes a zero-event predictor.

The correction restored first-order medulla modulation in controlled probes, but
T4/T5 and projection populations remained silent. The profile is provisional;
useful learned prediction and score-only Pong behavior are not established.
