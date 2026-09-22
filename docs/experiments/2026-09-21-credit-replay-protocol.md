# Credit context during learning and nonalternating column events

User authorized continuation of the diagnostic proposed in the credit-context
findings. No model mechanism is changed.

1. Verify a passive FramePrediction recorder on baseline and facilitation toy
   networks: identical weights/proposals/state, correct issue/confirmation timing.
2. Replay exactly the first 24 trials of the existing 200-trial seed-9060 training
   sequence from original weights, separately for baseline and F. Require bitwise
   agreement with saved predictions, targets, spikes and issued eligibility.
   Check each selected-edge proposal against the actual accumulator and each
   synchronized weight against clipping/homeostasis. Record existing state at
   issue and confirmation, actual proposals and weights; do not route learning.
3. From each existing 200-trial weight set with fresh standard warmup/state, run
   three matched 16-trial diagnostic learning sequences (seed9091, four trials
   each dwell2/3/4/6): ordinary alternation, omission of one ON/OFF pair, and two
   successive pixel activations then deactivations inside the same retinal column.
   Only binary images enter the existing camera. At one binary pixel polarity
   necessarily alternates; merely skipping moves cannot test repeated polarity.
4. Apply the previously selected zero sensory-state boundary without refitting.
   Report ON/OFF/quiet directions and update magnitudes separately, early/late
   and per tempo. This is not an event detector or an M1A acceptance experiment.
   Repeated-column events deliberately test generalization beyond one-pixel
   occupancy; failure does not imply that unpredictable omissions were learnable.
5. Save per-condition artifacts and hashes, verify results, report the next
   evidence-based decision. Stop without adding channels, gates or synaptic
   components. Any model change requires a separate proposal.

Recorder tests: 3 passed before real-data runs. Source and runner checksums gate
reuse of completed saved conditions. Raw results live in runs/credit-replay-v1.

Preflight correction: the actual scored retinal column has only one rendered
pixel at32x64. A second-pixel repeated-polarity challenge cannot be constructed
without changing the input mapping. Do not change the mapping. Record the exact
column evidence and run only replay, ordinary alternation and omission for this
column. The generic stimulus fixture remains tested, but cannot establish a
real-data result on this target. This reinforces the binary-occupancy caveat.
