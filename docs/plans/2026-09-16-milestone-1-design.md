# Fly-Connectome Milestone 1 Design

**Status:** Approved through iterative design review on 2026-09-16  
**Target repository:** `rruhland/Fly-Connectome`  
**Dataset:** MaleCNS v1.0

## 1. Objective

Milestone 1 will test whether a fixed, biologically measured fly-connectome subgraph can learn closed-loop Pong from sparse visual events using only local spiking plasticity.

The resulting system must:

- Receive no Pong coordinates, velocities, object labels, collision flags, or action labels as neural input.
- Use discrete spiking neurons and no backpropagation, BPTT, surrogate gradients, dense learned encoders, or learned artificial readouts.
- Preserve the selected MaleCNS directed topology: existing edges may strengthen, weaken, or become functionally silent, but no internal edges may be created.
- Learn general visual dynamics through local predictive plasticity.
- Learn behavior through reward-modulated STDP in central and motor pathways.
- Control a paddle through continuous motor force rather than symbolic `UP`, `DOWN`, and `STAY` actions.
- Run on a CPU reference backend and a sparse CUDA backend.

Milestone 1 deliberately targets one fixed version of Pong. Generalization to changed physics, other games, active vision, attention, experts, counterfactual imagination, and expanded connectome regions are later milestones.

## 2. Success criteria

After the privileged paddle-hit shaping reward has faded to zero, the learner must:

1. Continue improving or maintain learned performance using score reward only.
2. Beat random-motor and frozen-connectome baselines on unseen initial trajectories under the same Pong physics.
3. Produce visual predictions that beat a persistence baseline which simply repeats the previous event state.
4. Remain stable without runaway activity or global silence.
5. Produce equivalent qualitative behavior and numerically bounded differences on the CPU and CUDA backends.

No arbitrary Pong score target will be chosen before observing the environment's natural difficulty.

## 3. End-to-end architecture

```text
fixed Pong renderer and physics
        |
        v
sparse ON/OFF event generation
        |
        v
fixed full-field retinotopic projection onto one optic lobe
        |
        v
L1-L4 -> medulla -> T4/T5 -> lobula/lobula-plate partners
        |
        v
selected visual projection neurons and central intermediates
        |
        v
opponent DNa02 populations
        |
        v
continuous actuator force -> paddle motion
        |
        +---------------- visual reafference ----------------+
```

The event-camera stream is the only observation received by the SNN. Simulator state may be used for environment logic, reward generation, and offline diagnostics, but never as network input.

## 4. Connectome data and graph representation

### 4.1 Dataset

Use MaleCNS v1.0 from the official Janelia release. Raw downloads, derived caches, and checkpoints are not committed. A manifest will record dataset version, source URLs, checksums, extraction rules, and body IDs.

### 4.2 Directed connectivity

Chemical connections are directed from the recorded presynaptic neuron to the recorded postsynaptic neuron. Reciprocal relationships are independent; `i -> j` and `j -> i` may have different contact counts and signs.

### 4.3 Contact aggregation

All retained contacts for an ordered neuron pair are represented by one computational edge:

```text
n_ij biological contacts -> one directed i -> j edge
```

The original contact count and relevant neuropil metadata remain available offline. Milestone 1 does not model independently plastic contact sites, dendritic compartments, or contact-specific release.

### 4.4 Initial strength and sign

For retained edges, initial magnitude is linear in contact count:

```text
abs(w_ij) = g * n_ij
```

`g` is a global or small cell-class-specific conversion from anatomical contact count to model current. The presynaptic neurotransmitter prediction fixes the sign where confidence is sufficient. Plasticity changes magnitude only: signs cannot flip, edges cannot be created, and magnitudes may approach zero.

Square-root/logarithmic count transforms, fan-in normalization, and contact-level simulation are follow-up ablations rather than the canonical model.

### 4.5 Threshold variants

Milestone 1 supports three nested edge masks over the same fixed neuron roster:

```text
T1: at least 1 detected contact
T3: at least 3 detected contacts
T5: at least 5 detected contacts

E5 is a subset of E3, which is a subset of E1.
```

All variants use identical neuron, environment, plasticity, reward, and evaluation hyperparameters in the primary comparison. A neuron remains in the roster even if thresholding isolates it.

Threshold 5 is the reliability starting point. Before training, a lightweight anatomy report compares retained contact mass, reachability from sensory populations to DNa02, required region coverage, and recurrent connectivity. If T5 breaks the preregistered circuit, T3 becomes the canonical graph. T1 remains available to test whether weak measured edges form a useful plastic reservoir. Threshold choice is never tuned on Pong score.

## 5. Milestone-one subnetwork

The seed populations are:

- One optic lobe's L1, L2, L3, and L4 pathways.
- Their selected medulla partners.
- T4 and T5 motion populations.
- Relevant LPi and selected lobula plate tangential partners.
- An LC10-centered small-object pathway.
- Selected visual projection neurons and AOTU/PVLP/PLP intermediates that lie on measured paths toward the motor population.
- Bilateral opponent DNa02 populations.

L5, LC11, additional visual projection populations, the full central complex, VNC motor circuitry, ascending proprioception, and broader descending populations are deferred or predefined ablations.

Recurrent partners are selected by preregistered graph rules rather than subjective picking:

1. Begin from the named seed populations.
2. Retain actual directed paths satisfying fixed confidence and threshold rules.
3. Add reciprocal partners only when they exceed a fixed input/output fraction or contact-count criterion.
4. Preserve strongly connected components subject to a predetermined size cap.
5. Preserve all induced directed edges among the final selected neurons for the chosen threshold.

Body IDs and extraction rules are versioned so later milestones can expand the graph while preserving compatible learned weights.

## 6. Visual sensor and retinotopic interface

### 6.1 Event stream

The renderer produces changes in log intensity or the binary equivalent needed by the simple Pong scene. The sensor emits explicit, separate ON and OFF events. Only the polarity that occurred emits a spike; two channels do not double activity for a single transition.

### 6.2 Full-field mapping

The full Pong arena is projected through a fixed aspect-ratio-preserving mapping onto the selected optic lobe's hexagonal retinotopic extent. Multiple render pixels may contribute to one ommatidial bin, so render resolution does not determine SNN size.

This is effectively a fixed full-screen field of view. Milestone 1 has no crop, zoom, gaze, saccade, foveation, learned encoder, or attention-controlled sensor.

### 6.3 Optic-lobe injection

External ON/OFF events stimulate only preregistered plausible L1-L3 sensory-facing pathways. L4 receives no direct sensor injection and is activated through measured connectome paths such as L2-to-L4 connectivity. Downstream neurons receive activity only through retained connectome edges.

### 6.4 Visible resets

After a point, Pong resets immediately. The event-camera reference is not silently cleared or disabled. Repositioning the ball and paddles therefore produces ordinary ON/OFF events, allowing the network to observe that a new rally began.

## 7. Neuron and synapse dynamics

Milestone 1 uses current-based adaptive leaky integrate-and-fire point neurons with:

- Discrete spikes.
- Fixed propagation delays.
- Leaky membrane voltage.
- Leaky postsynaptic current.
- Fixed refractory intervals.
- Slow spike-triggered threshold adaptation.

Conductance-based, graded-potential, multicompartment, and hybrid neuron models are later milestones.

Fixed delay values use anatomical/path-length information where practical and a small, documented non-Pong-specific default model otherwise. Delays are not plastic in milestone 1. Eligibility assigns delayed credit; it does not schedule spikes or act as a second connection.

## 8. Plasticity

### 8.1 Regional separation

Three mechanisms have separate roles:

- **Visual predictive plasticity:** optic-lobe and visual recurrent pathways.
- **Reward-modulated STDP:** VPN-to-central, central recurrent, and descending motor pathways.
- **Homeostasis:** throughout the selected network.

Rules do not compete on the same edge by default. Weak reward modulation of higher visual pathways may be studied later as attention, but is not canonical milestone-one behavior.

### 8.2 Local predictive plasticity

Existing neurons and connectome dynamics generate the prediction; there is no artificial prediction head or dedicated error network.

For a relevant neuron `j`:

```text
p_j: transient expected near-future sensory activity
delta_j: locally observed sensory activity minus p_j
e_ij: temporary eligibility showing that i -> j contributed recently
```

The conceptual update is:

```text
delta_w_ij = eta_prediction * delta_j * e_ij
```

Underprediction strengthens eligible pathways, a prediction that expires unconfirmed weakens them, and an accurate prediction produces little further change. Only active predictions and observed events require work; silent neurons are not scanned merely to confirm silence.

Prediction timing comes from fixed propagation delays, membrane/synaptic dynamics, and multi-hop recurrent paths. Eligibility is a temporary causal tag, not a probability, learned delay, or second weight.

Milestone 1 learns a predictive sensory-dynamics model: local motion continuation, direction and speed structure, collision-related transitions, and distributed implicit separation of ball and paddle dynamics. Explicit object symbols, long counterfactual rollouts, and action-conditioned imagination are deferred.

### 8.3 Behavioral R-STDP

Central and motor synapses maintain naturally decaying behavioral eligibility. Scalar reward changes eligible weights without resetting membrane or recurrent state:

```text
delta_w_ij = eta_reward * reward(t) * e_ij
```

Reward pulses are bounded. Eligibility decays naturally across play and is not wiped after every teaching signal. A full neural-state reset is never triggered by reward.

### 8.4 Homeostasis

Homeostasis does not teach the task. It prevents persistent runaway excitation or silence through:

- Spike-triggered adaptive thresholds that decay toward a baseline.
- Sign-preserving weight bounds.
- Slow multiplicative downscaling for chronically overactive neurons.
- No forced minimum firing rate, preserving sparse quiet periods.

Homeostatic changes operate much more slowly than spike, prediction, and behavioral eligibility dynamics.

## 9. Reward curriculum

The reward is separated into permanent score reward and temporary hit shaping:

```text
reward(t) = beta * score_reward(t) + alpha(k) * hit_reward(t)
```

Where:

```text
score_reward = +1 when the scripted opponent misses
score_reward = -1 when the player misses
score_reward =  0 otherwise

hit_reward   = +1 when the player paddle contacts the ball
hit_reward   =  0 otherwise
```

There is no duplicated miss penalty. `beta` is moderately larger than the initial hit-shaping scale. `alpha(k)` follows a configurable step-based curriculum:

1. **Bootstrap:** hit shaping and score reward are active.
2. **Fade:** hit shaping gradually approaches zero.
3. **Score-only:** hit shaping remains zero through final training and evaluation.

Reward is never neural input. Local visual prediction remains reward-independent. Final claims are based on performance after privileged hit shaping has ended.

## 10. Pong environment and motor embodiment

### 10.1 Environment

Milestone 1 uses one fixed Pong implementation and fixed physics. Environments vary only in initial conditions and random seeds. Physics randomization and cross-game transfer are later milestones.

The opposing paddle is a fixed, deterministic, non-learning, speed/acceleration-limited scripted controller. It may use simulator ball state internally, but that state is isolated from the SNN.

After a point, the game resets and begins the next rally immediately. Learned weights, membrane voltage, recurrent state, and naturally decaying eligibility persist. The game body and actuator reset according to the fixed Pong rules.

### 10.2 Continuous motor interface

There is no categorical action decoder and no explicit `STAY` action. Opponent DNa02 population activity supplies signed motor drive:

```text
u(t) = rate_up(t) - rate_down(t)
tau_a * da/dt = -a + g_motor * u(t)
dv/dt = k_motor * a - drag * v
dy/dt = v
```

Balanced or absent drive allows activation and velocity to decay, producing rest. Sustained activity produces smooth acceleration; opposing activity brakes and reverses the paddle. Gain, inertia, drag, boundaries, and maximum velocity belong to the fixed artificial body rather than the learned policy.

The paddle's movement is rendered normally, creating visual reafference. No paddle position, velocity, actuator state, artificial efference copy, or invented motor-to-visual synapse enters the network. Existing connectome recurrence may retain recent motor-generating state.

## 11. Training stages

### 11.1 M1A: visual learning validation

Train the open-loop visual subnetwork on event-camera Pong sequences and verify:

- Decreasing local next-event prediction error.
- Prediction better than persistence on held-out sequences.
- Direction/velocity-related anticipatory activity.
- Stable sparse firing.

The learned visual weights continue into M1B rather than being discarded.

### 11.2 M1B: closed-loop Pong

Attach the central and DNa02 pathways and continuous actuator. Train with the bootstrap, fade, and score-only reward phases. The final learner must operate without hit shaping.

Warm graph expansion preserves compatible learned weights. Same-size training from scratch remains a control; graph expansion never creates edges outside the newly selected measured subgraph.

## 12. CPU, CUDA, and multiple environments

### 12.1 Backends

The CPU backend is the deterministic reference and development target. The CUDA backend uses the same graph, update semantics, and tests. Benchmarking determines the crossover; sparse CPU execution may win for a small active network, while CUDA is expected to benefit from larger graphs or environment batches.

### 12.2 Shared-weight environment batching

Multiple environments are an optional engineering accelerator, not part of the canonical biological claim:

- Topology and weights are shared once.
- Membrane voltage, refractory state, spike history, prediction state, actuator state, and eligibility are private per environment.
- All environments use a shared weight snapshot for a short simulation window.
- Each environment computes local, gradient-free plasticity proposals.
- Proposals are normalized and deterministically reduced at a synchronization boundary.
- No environment may leak neural or eligibility state into another.

`B=1` remains the canonical online biological model. Batched runs are compared against it. Dense per-environment edge eligibility is avoided; lazy active-edge or factorized trace storage is required for GPU scalability.

### 12.3 Environment implementation

PufferLib is an engineering reference, not a milestone-one dependency. The custom vector environment uses contiguous state arrays, deterministic per-environment seeds, packed sparse event buffers with offsets, fixed-shape motor interfaces, and independent automatic resets. Pong simulation remains lightweight compared with the SNN.

## 13. Lightweight telemetry

Runtime instrumentation must not materially reduce training throughput. Record only:

- Episode reward, points, hits, and misses.
- Rally duration.
- Environment steps and wall-clock throughput.
- Coarse population firing rates already required by homeostasis.
- Occasional aggregate weight statistics.
- Configuration, random seeds, dataset manifest, and periodic checkpoints.

Do not continuously record full spike rasters, per-edge histories, eligibility tensors, pathway usage, or lesion statistics. Saved checkpoints permit deeper offline analysis later.

## 14. Verification strategy

Before long training runs, verify:

### Graph and data

- Direction, contact aggregation, sign, and threshold masks on hand-checkable fixtures.
- `E5 subset E3 subset E1` with an identical neuron roster.
- Deterministic extraction from the pinned MaleCNS manifest.
- Required sensory-to-DNa02 reachability and selected-region coverage.

### Sensor and environment

- Correct ON/OFF polarity and retinotopic mapping for controlled moving patterns.
- Reset-induced motion remains visible to the event stream.
- No simulator state reaches the SNN observation path.
- Scripted opponent and physics are deterministic under a seed.

### Neural dynamics and learning

- Fixed signs never flip and absent edges never appear.
- Eligibility alone does not alter forward activity.
- Confirmed, absent, and unexpected sensory events produce the intended local predictive updates.
- Reward affects only eligible central/motor connections.
- Homeostasis bounds sustained activity without forcing spikes during quiet input.
- Continuous DNa02 drive produces acceleration, braking, reversal, and rest.

### Backend and batching

- `B=1` CPU and CUDA spike/update traces agree within explicit tolerances.
- Identical duplicated trajectories yield the same normalized update as one trajectory.
- Environment-order permutation does not change reduced updates.
- Resetting one environment cannot affect another's state.

### Behavioral baselines

- Random motor drive.
- Frozen connectome weights.
- Persistence visual predictor.
- Trained learner after hit shaping reaches zero.

## 15. Deferred work

Later milestones may add:

- Variable ball physics and cross-game training.
- Larger optic-lobe and central-brain subnetworks.
- L5, LC11, additional VPNs and descending populations.
- Central complex context and multi-skill routing.
- Attention and mixture-of-expert mechanisms grounded in measured circuitry.
- VNC motor networks, ascending proprioception, and explicit biological efference copy.
- Active gaze, foveation, binocular vision, and sensor motion.
- Conductance-based, graded, multicompartment, or mixed neuron models.
- Plastic conduction delays and contact-level synapse simulation.
- Internally generated replay and counterfactual imagined environments.
- Removal of score reward in favor of visual/intrinsic persistence objectives.

## 16. Sources

- MaleCNS project and downloads: <https://male-cns.janelia.org/> and <https://male-cns.janelia.org/download/>
- MaleCNS publication: <https://doi.org/10.1016/j.cell.2026.08.015>
- Male CNS visual pathways: <https://doi.org/10.1016/j.cell.2026.08.014>
- FlyVis: <https://www.nature.com/articles/s41586-024-07939-3>
- Fly whole-brain graph statistics: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11446825/>
- Fly effectome modeling: <https://www.nature.com/articles/s41586-024-07982-0>
- Predictive local spiking plasticity: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10442404/>
- Predictive Coding Light: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12500994/>
- DNa02 steering: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12279373/>
- NeuroMechFly v2: <https://gizemozd.github.io/assets/pdf/2024_neuromechflyv2.pdf>
- PufferLib documentation: <https://puffer.ai/docs.html>

