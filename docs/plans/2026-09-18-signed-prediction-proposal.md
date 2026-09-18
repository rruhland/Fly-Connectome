# Signed local prediction representation: proposed follow-up

Status: proposed, not implemented. The resting-current correction is implemented;
this document identifies a separate representation issue exposed by that correction.

## Concrete issue

The approved sensory transduction now maps ON to negative current and OFF to
positive current at L1-L3. However, the unchanged prediction rule still clips
observed and predicted normalized currents to [0, 1]. Consequently, currents -0.8,
-0.4 and 0 all become the identical value 0. The local error cannot distinguish
these inhibitory observations, and the event predictor cannot output a negative
value to match an ON event in the new signed target encoding.

This follows directly from `Plasticity.observe` and `Trainer.step`; it is not an
inference that biological L1 should use a particular learning rule. The existing
rule was deliberately left unchanged under the previous approval. More training
cannot recover information removed by this clipping operation.

## Proposed bounded change

Add an explicit, versioned prediction encoding for new experimental checkpoints:

```
observed_j = clip(feedforward_current_j / threshold_j, -1, 1)
predicted_j = clip(recurrent_current_j / threshold_j, -1, 1)
delta_j = observed_j - previous_predicted_j
delta_magnitude_ij = eta_prediction * delta_j * signed_eligibility_ij
```

Keep the update equation, eligibility timing/decay, fixed signs, magnitude bounds,
anatomical pathway partition, fixed feedforward observation edges, behavioral
R-STDP and homeostasis unchanged. Intrinsic current still contributes to neither
the observed nor predicted current. There is no added network, head, signal,
connection or gradient. The change preserves the sign of information already
present in local currents.

For an inhibitory eligible edge (eligibility negative):

| Observed | Prior prediction | Effect on magnitude |
| --- | --- | --- |
| -0.4 | 0 | Strengthen to address underpredicted inhibition |
| -0.4 | -0.4 | No prediction-error update |
| 0 | -0.4 | Weaken the unconfirmed inhibitory prediction |

The analogous excitatory cases retain their existing behavior. This changes the
representation used by learning, even though the local update equation is the
same, and therefore needs explicit approval under the instruction to keep local
learning fixed.

## Verification and experiment

1. Hand-check both signs: unexpected, confirmed and expired predictions; sign
   preservation; no absent edges; no reward effect on visual updates.
2. Apply the same encoding to offline prediction metrics, including the signed
   sensory-event target. Report zero-event and persistence baselines, and keep
   the target encoding visible. Current forecasts remain model-current proxies,
   not calibrated probabilities of biological spikes.
3. Preserve the old rectified encoding as the default for old checkpoints. Store
   the new encoding in configuration; reject incompatible warm expansions. Verify
   exact resume, B=1/duplicated batch equivalence and CPU/CUDA when available.
4. Start a separate initial/trained pair with the same resting profile, measured
   graph, signs and fixed gains. Repeat the held-out M1A protocol. Do not overwrite
   the rectified-rule experiment or change parameters based on Pong score.

This correction does not guarantee motion-population activity or behavioral
learning. The persistent T4/T5 silence is a separate empirical limitation; do not
mask it by injecting sensory current directly downstream or assigning extra edges.
