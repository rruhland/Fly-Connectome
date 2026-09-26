# M1A representation-first family audit

**Status:** Registered opt-in research protocol. It changes no production M1A path. Run each of seven substantially different local/no-backprop principles once, without tuning against held-out cases. A failed family is reported rather than patched with a sequence of variants.

## Shared sensory and data contract

Freeze the existing locally learned observation model after its 64 original generic training scenes. Every family receives the same six retinotopic input planes: credible ON/OFF events, signed appearance from the observer's contrast state, and a causal two-axis local coincidence map from recent credible events. The latter two are sensory primitives, not the final representation. A no-learning sensory control receives precisely these planes. The model never receives object masks, direction, speed, identity, Pong, or future sensory targets. All plasticity uses pre/post activity, local sensory error or temporal correlation, local homeostasis, and no backpropagation.

Use 64 unlabeled training episodes: seeds 0–15 at motion strides 1, 2, 4, and 8, 80 frames each. Refresh observer intensity every eight frames as in the previous frozen-observer audits. Evaluate on the existing calibration, changed-position, changed-speed, changed-shape, separated-entity, and crossing cases, plus clean/corrupted held-out generic scenes and a three-frame missing-evidence interruption. The case generator may supply labels and masks **only to offline probes and scores**. Train each family once with fixed parameters, and compare its trained state to an identical frozen-initialization control where applicable. Report wall time and state activity.

## Seven independent principles

1. **Slow feature:** a sparse local convolutional population receives a temporal-difference Hebbian/anti-Hebbian update that penalizes changes in postsynaptic response while retaining input variance. Its state is evaluated without any next-event loss.
2. **Common fate:** local links strengthen when neighboring sensory changes co-vary in time; linked evidence is grouped into an online spatial state. No entity labels or preassigned mover slots enter learning.
3. **Sparse reconstruction:** a convolutional dictionary uses local matching competition and residual-driven Hebbian updates to explain the *current* input with few active causes. No temporal transition is trained.
4. **Attractor:** local sensory prototypes and coactive recurrent connections learn distributed assemblies; partial evidence is allowed to complete an assembly. It is evaluated for false persistence as well as recovery.
5. **Synchrony:** locally active sites learn phase coupling from temporally coherent activation. The state encodes amplitude and relative phase; test whether distinct movers avoid global phase collapse.
6. **Relational graph:** spatially anchored sensory nodes learn nearby coactivity/continuity edges and exchange messages. The graph has no object, position, or velocity variable; fixed node sites are a retinotopic routing primitive.
7. **Reservoir:** a fixed sparse recurrent circuit with local homeostasis supplies temporal state; only its local sensory association/readout learns. This tests whether learning recurrent connectivity is necessary.

## Frozen representation audit

The primary measures are: (a) full-frame occupancy F1 from one fixed local linear probe calibrated only on the separate calibration scenes, across position/speed/shape/separated/crossing; (b) direction decodability from a frozen centroid probe across the same splits, with oracle object regions explicitly labeled as a *capacity* measure rather than autonomous discovery; (c) two-entity spatial separation and crossing identity consistency; (d) similarity of clean versus corrupted latent state relative to unrelated scenes; and (e) continuity and recovery after three missing camera frames. Report activity density and variance to detect collapsed, all-on, or all-off codes. No downstream next-event predictor is attached in this audit.

Treat a family as **promising, not production-ready**, only if it beats both the sensory and its frozen-learning controls by at least .05 absolute on a hard representation measure (multi-entity separation, crossing, noise robustness, or missing-evidence recovery), while retaining direction transfer and occupancy F1 within .10 of the sensory control on position/speed/shape. A single stronger easy-scene score or a persistent but uninformative state is insufficient. Report all measures even if no family passes. No held-out-driven parameter sweep follows this run; return for M1A reassessment if none passes.

The learned observer's every-frame rendered-intensity training teacher remains an explicit experimental assumption. This protocol does not promote a production architecture or claim that a linear probe proves autonomous entity discovery.
