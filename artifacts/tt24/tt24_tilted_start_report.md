# Donor practice on independently tilted starts

Training on broad initial poses improved the frozen actor's second-rebound rate on both fresh wrist-controlled arms. Direct deployment sacrificed the ordinary-start skill. A separately sealed test of an encoder-conditioned blend is recorded in the context receipt.

## Bounded broad fit

| Held cohort | Actor | Second qualifying rebound | Mean consecutive | Longest |
|---|---|---:|---:|---:|
| Wide donors, precise simulated observations | Warm | 17/48 | 2.10 | 14 |
| Wide donors, precise simulated observations | Cold | 14/48 | 1.48 | 5 |
| Wide donors, precise simulated observations | Original | 2/48 | 0.96 | 2 |
| Wide donors, precise simulated observations | Stationary | 2/48 | 1.00 | 2 |
| Wide stock, wrist | Warm | 6/16 | 1.44 | 4 |
| Wide stock, wrist | Cold | 5/16 | 1.25 | 3 |
| Wide stock, wrist | Original | 0/16 | 0.88 | 1 |
| Wide stock, wrist | Stationary | 2/16 | 1.06 | 2 |
| Wide altered, wrist | Warm | 4/16 | 1.69 | 6 |
| Wide altered, wrist | Cold | 5/16 | 1.56 | 8 |
| Wide altered, wrist | Original | 0/16 | 0.94 | 1 |
| Wide altered, wrist | Stationary | 2/16 | 1.06 | 2 |
| Ordinary stock, wrist | Warm | 10/16 | 2.00 | 6 |
| Ordinary stock, wrist | Cold | 8/16 | 1.50 | 2 |
| Ordinary stock, wrist | Original | 16/16 | 11.38 | 18 |
| Ordinary stock, wrist | Stationary | 15/16 | 2.75 | 4 |

On the held wide stock wrist tapes, median first face-contact tilt fell from 13.09° under the original to 2.76° with the warm actor. The median second descending crossing moved from 173 mm outside the rim to 14 mm outside among drops with a measured crossing. On the altered arm, median first-contact tilt fell from 14.51° to 5.48°. In a separate retrospective test with precise ball and paddle state throughout, the warm actor made a second rebound on 7/16 stock and 6/16 altered drops, with longest runs of four and eight. That diagnostic used the same sealed weights.

The three source bodies used changed actuator and ball parameters and independent perturbations of all five initial joint angles. The actual median paddle tilt was 13.0° on the fixed broad donor validation roster and 1.3° on the ordinary validation roster. Both 5×22 readouts had the original optical and motor features plus two channels for the upward normal at reset. The warm actor started with the existing 5×20 coefficients and two zero columns. The cold actor started at all zeros. Both could change all 22 coefficients per joint.

The search used exact simulated source observations, 30 shared signed motor probes, and the same 3,060 practice drops and 2,196 validation drops per actor. It selected weights using only physical donor rewards, emphasizing second and later consecutive rebounds. Ordinary donor validation had a 15% contribution. Target conditions used only RGB and own encoders to compute actor actions. They used the same frozen observer, with zero target practice. The altered arm changed hinge and payload geometry, drive and ball dynamics. All 192 target tapes and 24 source parity audit tapes replayed exactly, as did the 96 target diagnostic tapes.

Reproduce with `MUJOCO_GL=egl python -m scripts.experimental.tt24_tilted_start_trial`. The `--held` and `--clean-diagnostic` options rerun the sealed tests.
