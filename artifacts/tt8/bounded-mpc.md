# Full-plan motor limits in receiving games

Bounding the predictive plan brought one five-axis game to 10–12 from 6–11. On a second five-axis installation, the bounded player finished 2–11 and the standard player finished 3–11. On the six-joint installation, it gained one point while landing one fewer legal return. The results support keeping the bounded solver optional.

| Installation | Standard game | Bounded game | Shared starts | Standard / bounded legal returns on shared starts | Standard / bounded wins on shared starts |
|---|---:|---:|---:|---:|---:|
| Five-axis offset 386073911 | 6–11 | 10–12 | 17 | 17 / 20 | 6 / 8 |
| Five-axis offset 416707297 | 3–11 | 2–11 | 13 | 6 / 5 | 3 / 2 |
| Six-joint rotary 419081909 | 2–11 | 3–11 | 13 | 4 / 3 | 2 / 3 |

On the 17 shared starts of the first game, the bounded player opened 11 points and had five with a second legal return. The standard player opened 13 and had three with a second return. [First offset standard](bounded-mpc-offset-a-legacy.json.gz) · [First offset bounded](bounded-mpc-offset-a-bounded.json.gz) · [Second offset standard](bounded-mpc-offset-b-legacy.json.gz) · [Second offset bounded](bounded-mpc-offset-b-bounded.json.gz) · [Rotary standard](bounded-mpc-rotary-legacy.json.gz) · [Rotary bounded](bounded-mpc-rotary-bounded.json.gz)

The predictive receiver fits a short motor response from camera probes and produces a three-move plan. The standard player solves its pose objective and clips the first command to the signed rate and cumulative budget. The bounded player applies the same constraints to all three moves while solving the same objective. It uses the earlier active-set solver through the shared predictive controller. A strike-specific objective can use the same solver through `BoundedPlanMixin`. [Earlier rotary experiment](../tt6/bounded-mpc.md)

Both groups start from the same portable 42-point hall player and fit their motor responses using the same public RGB probe commands. One installation and a common schedule of receiving starts determine each comparison. Shuffled four-point blocks mix the stock and softened rackets with intervals of actuator delay and noise. Games use a target of 11 points, a two-point lead, and a 420-frame point limit.

The solver reports how often the unconstrained plan exceeds a normalized motor limit, how often it changes the first move, time spent solving on CPU, and the objective improvement over sequential clipping of the entire command sequence. The same RGB estimates, prior, timing observer, impact adaptation, and trajectory objective feed both players.

| Installation | Plans over a limit | First moves changed | Total fitted objective reduction | CPU per solve |
|---|---:|---:|---:|---:|
| Five-axis offset 386073911 | 374 / 1,998 | 374 | 13.46 | 0.22 ms |
| Five-axis offset 416707297 | 240 / 780 | 238 | 7.49 | 0.24 ms |
| Six-joint rotary 419081909 | 264 / 706 | 264 | 2.50 | 0.32 ms |

There were zero solver warnings. Rates stayed within one normalized unit per frame and cumulative requests within 3.85 normalized units on every point. These limits apply to the player's commands. The evaluator can delay them and add noise before passing them to the actuators. All 93 complete points replayed with matching command hashes, applied rates, events, and referee outcomes.

## Reproduce

Use the [TT6 environment instructions](../tt6/README.md). Change `--motor-plan` to `legacy` for the control. Use seed `416707297` for the second offset installation, or seed `419081909` and course `rotary` for the six-joint comparison.

```sh
python -m scripts.experimental.tt8_bounded_trial /tmp/bounded-offset.json.gz \
  --motor-plan bounded --seed 386073911 --course offset \
  --resume artifacts/tt6/continual-player.json.gz --race-to 11 --max-steps 420 \
  --phases ordinary soft mixed delay --cached-render --reflections \
  --probe-cache /tmp/tt7-calibration
```
