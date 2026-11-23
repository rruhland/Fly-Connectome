# Selecting retained motor skills from wrist observations

Two separately reward-trained five-joint feedback heads share the same physical center donor arm and camera. The solo head learned repeated paddle rebounds. The overhead head learned exchanges with a physically moving overhead partner paddle. The partner follows lateral ball motion at finite speed, and actual collision impulses return the ball. The two source heads were trained with simulated metric observations and no motor demonstrations. Before this comparison the solo head had 18,000 reward trials and the wide overhead head had 6,048. Public deployment uses the existing encoder and wrist calibration from four generic optical probe drops. [Source solo training](tt24_donor-refined.json.gz) · [Source overhead training](overhead-source-training.json.gz) · [Public observer](public-morph.md)

## Optical feasibility

The original public observer produced zero overhead pairs on either partner profile. In an isolated diagnostic the ball was visible at the initial overhead release, roughly 75 mm above the blade. The observer rejected that measurement relative to its ordinary-bounce release prior and emitted a fallback near 261 mm. It then reacquired just before the first learner-paddle encounter. On the same wide seeds a clean-state diagnostic produced 48 physical pairs. The clean branch served only as a measurement diagnostic.

A shared startup correction accepts the first geometrically admissible RGB detection and uses the first measured position difference on the second frame. Every motor head and both games use the same correction. On an independent wide preflight the overhead head made 12 pairs on four of 12 starts. The solo head and stationary control made zero. On the same body's ordinary course the solo and overhead heads averaged 7.67 and 1.33 consecutive rebounds. The narrower partner profile made four pairs on three starts and did not preserve a comparably strong solo reference on its corresponding cohort.

The wide profile missed the predeclared stricter reliability threshold of six nonzero partner starts in 12. Because it made actual repeated contacts and the specialized heads had different physical strengths, the routing comparison proceeded as exploratory with fresh schedules. The two preflight passes each evaluated 144 treatment episodes. Their scores were used for feasibility rather than to fit the router. [Original preflight](dual-game-gate.json.gz) · [Optical diagnostic](dual-game-optical-diagnostic.json.gz) · [Startup correction preflight](dual-game-open-start-gate.json.gz)

## Reward-trained selection

The router uses the opening RGB image and joint angles to derive a generic patch descriptor, an optical ball location relative to the blade, and encoder features. It considers five choices on the convex line between the two fixed motor weight matrices. It uses the selected matrix for the rest of that release. The game label, partner-contact channels and per-step rewards are unavailable to that decision. The controller receives only the single scalar terminal score to update the next choice's reward estimate. Twelve calibration trials use randomized mixture assignments, followed by 12 trials with online kernel reward predictions and an uncertainty bonus. On the overhead game this calibration observed seven real completed pairs across three trials. Exactly 24 terminal scores entered the fit. The underlying motor weights did not change.

The held schedule contains 16 new releases from each game, in randomized order with 14 switches. Neither the router nor the motor weights update during qualification. All wrist-controlled trajectories were replayed to the same physical metrics, with a separate implementation replaying the overhead judge. [Code](../../scripts/experimental/tt24_dual_game.py) · [Frozen schedule](dual-game-open-start-plan.json) · [Reward calibration and full tapes](dual-game-open-start.json.gz)

| Mixture selector | Solo rebounds, 16 starts | Overhead pairs, 16 starts |
|---|---:|---:|
| Learned from wrist context and terminal rewards | 85 | 16 |
| Fixed solo head | 85 | 2 |
| Fixed overhead head | 22 | 16 |
| Fixed half blend | 39 | 19 |
| Context-free reward fit | 85 | 2 |
| Shuffled context fit | 54 | 2 |
| Game-identity head choice | 85 | 16 |

The learned selector chose the solo weight endpoint on all 16 solo trials and the overhead endpoint on 13 of 16 overhead trials. The three other overhead trials scored zero with either head, so its physical outcome matched the identity-supplied selection on all 32 releases. On immediate revisits after the other game, it earned 39 solo rebounds across seven switches and 11 overhead pairs across seven. The learned selector gained 0.875 overhead pairs per drop over the fixed solo head, with a paired 95% bootstrap interval of 0.063 to 1.750. The fixed half blend earned three more overhead pairs and 46 fewer solo rebounds. The routing result represents retention and context-dependent recall of fixed skills. It covers an episode-level choice in two visibly different setups, with sparse partner success and no qualification of continuous within-episode readout switching.

## Reproduce

```sh
export PYTHONPATH=src LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
python -m scripts.experimental.tt24_dual_game --gate-only
python -m scripts.experimental.tt24_dual_game --open-start --exploratory-sparse
```
