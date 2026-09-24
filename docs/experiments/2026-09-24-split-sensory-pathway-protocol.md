# Separate first-sighting sensory pathway: frozen-code test

The 18-channel raw+coincidence dictionary saw reappearance but damaged ordinary motion coding. This opt-in test keeps the successful 24-unit coincidence population intact and adds a distinct eight-unit locally learned raw ON/OFF population. Both branches use the existing 5×5 competitive Hebbian dictionary. The raw branch receives no motion, object, direction, or game label. There is no recurrent memory or architectural production change in this first step.

Train the coincidence branch with the existing 96-episode procedure. Train the raw branch on those same episodes. Freeze both dictionaries. Fit the existing evaluation-only local linear next-event decoder on their concatenated source activity, using the same training episodes. Compare with (1) the unchanged coincidence-only branch and (2) an eight-unit random raw branch beside the unchanged coincidence branch. The offline decoder is a representation-capacity probe, not the online M1A learning rule. Evaluate unseen single shapes, all five robust families, spatial coverage, and the exact t=10→11 reappearance transition. The readout threshold remains 0.5.

The split model has eight more units than the coincidence-only baseline; compare the learned and random raw branches to isolate the effect of raw dictionary learning. This is a prerequisite test, not proof that the extra units or this readout are an efficient final architecture.

Registered decisions:

- The split learned code should stay within 0.03 F1 of the unchanged coincidence-only code on unseen single patterns. Otherwise the extra branch/readout still disrupts ordinary transfer.
- Its raw branch must activate on t=10 first-sighting evidence, while the coincidence branch remains silent there. This confirms access without interference.
- If exact t=10→11 F1 improves by at least 0.10 over coincidence-only and by at least 0.05 over the random raw branch, the raw learned pathway offers predictive re-anchoring. Otherwise sensory separation solves access but is insufficient to learn the missing temporal context. Do not tune a threshold after seeing results.

No variant here feeds its forecast back into the latent or claims a working online predictor. Continue to a bounded local-state-memory experiment only if this prerequisite preserves clean motion capacity and the reappearance question remains open.
