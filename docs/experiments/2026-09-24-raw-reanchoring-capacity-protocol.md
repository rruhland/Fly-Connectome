# Frozen-code capacity of raw-event re-anchoring

The raw ON/OFF + coincidence learner detects first sightings but forecasts poorly. This evaluation asks whether its frozen learned latent still makes future events linearly readable. It changes no model, training data, local rule, or production code.

Train the same 18-channel learned and random-dictionary populations with `train_raw_models()` (96 generic episodes). Freeze both. Fit the existing local 5×5 linear future-event decoder on the same training episodes. This offline least-squares fit is a **diagnostic upper bound**, not M1A learning and never replaces the local predictive weights. Evaluate each frozen code on training episodes, 48 unseen single-pattern cases, and the five existing robust families. Also report the learned model's original local readout F1 on training episodes, readout coverage, and the held-out threshold profile. Use the same seeds and evaluation window as the previous 16-channel capacity test.

Reference values from the previous 16-channel experiment: learned-code offline capacity 0.963 train / 0.918 unseen singles; random-code capacity 0.528 / 0.283; local learned readout 0.671 train. These are contextual references, not a matched ablation, because the new input adds raw channels.

Interpretation registered before the run:

- Learned-code offline F1 at least 0.8 on unseen singles and at least 0.1 above the random code: the representation retains useful future-event information; investigate the local target/readout and why it fails on interruptions.
- Learned-code train F1 at most 0.5: the dictionary/spatial code itself loses necessary information; investigate separating raw first-sighting access from correlation competition.
- High train but weak held-out capacity: learned code or offline readout overfits; avoid blaming the local update alone.

For robust scenes, report family F1 and coverage rather than selecting a threshold after viewing results. If the evidence is mixed, keep both diagnoses open. Do not promote any architecture change from this offline diagnostic alone.
