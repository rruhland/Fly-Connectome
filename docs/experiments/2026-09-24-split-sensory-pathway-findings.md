# Separate raw pathway preserves motion code, but does not recover motion after a gap

The [registered frozen-code test](2026-09-24-split-sensory-pathway-protocol.md) kept the learned 24-unit coincidence branch intact and added a separate eight-unit raw ON/OFF branch. The latter was either locally dictionary-trained or left random. A fitted local linear next-event readout measured code capacity only; it is not the online M1A learning rule. The run took 174 seconds.

| Offline next-event F1 | Coincidence only | Split learned raw | Split random raw |
| --- | ---: | ---: | ---: |
| Training episodes | 0.963 | 0.966 | 0.964 |
| Unseen single patterns | **0.918** | **0.918** | **0.918** |
| Five robust families pooled | 0.810 | 0.811 | 0.811 |
| Exact reappearance t=10→11 | 0.000 | **0.016** | 0.000 |

At t=10 across eight disappearance cases, the unchanged coincidence branch had **zero** source activations and each raw branch had **116**. Its separate pathway therefore restores access without damaging ordinary motion representation. The robust spatial reach of the frozen readout rose from 0.920 to 0.965; at the exact reappearance transition, **all 124** future events fall within the raw branch's 5×5 local fields. Yet the learned raw branch predicted only one of those 124 events at the registered 0.5 threshold. The pooled robust score hides this failure because most scored frames are ordinary visible motion. The random raw branch matched clean and pooled robust capacity, so the learned raw dictionary has no demonstrated predictive advantage here.

There is a causal reason to require memory. A matched test generates leftward and rightward trajectories with the **same raw event image at t=10** after three missing frames and different events at t=11. Both have zero coincidence channels at t=10. The memoryless split population produces identical source activity for the two futures. No readout of that current state, linear or otherwise, can predict both correctly. The prior visible history differs, so a hidden state that preserves useful temporal context could disambiguate them. This counterfactual is generic moving visual structure; no object labels or Pong semantics enter the model.

The next decisive experiment should let a locally plastic hidden transition maintain context across a brief missing interval, while keeping predicted hidden activity separate from visible-event emission. Score the matched opposite-motion pairs, exact reappearance, quiet-frame false alarms, clean transfer, and a state-reset ablation. The split sensory branch is a useful component of that experiment, **not** an effective M1A predictor on its own. Do not promote it to production.
