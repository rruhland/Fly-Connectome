# A motor learner that keeps updating as Pong points arrive

The simulated SO-101 collected its own commanded rates and next joint readings at 60 Hz while a reward-trained screen player played Pong. At each point break with enough movement history, it updated a five-axis motor map. It consumed one live stream at a time. The per-frame collector retains only joint and command measurements. The fit revisits a bounded set of these measurements at a break, without storing camera frames. It uses linear regression with robust weights and no backpropagation.

For each of two independently drawn drive matrices, twelve points (59.08 and 68.02 simulated seconds) produced exactly the same learned map as the earlier successful whole-stream analysis. The checkpoint was saved and reloaded after the sixth point. Following the twelfth point it saved the receiving map, then continued on points 13 and 14, retiring the oldest history while fitting both updates. The game evaluator verified every physical command and encoder observation as the native player produced them. [Live runner](../../scripts/experimental/tt24_rolling_motor_trial.py) · [Estimator](../../scripts/experimental/tt24_rolling_motor.py)

A new sixteen-drop wrist test loaded the twelfth-point maps and gave the arm **zero target practice**. On identical starts it reached ten consecutive rebounds on **6/16**, versus **1/16** keeping its initial motor map. Its summed longest streaks were 101 versus 31. On the preceding sixteen-drop test, the identical learned maps had reached ten on 8/16 versus 0/16. Taken across both independent stock cohorts, the counts are 14/32 versus 1/32. The receiving arm already had a reward-trained direct bouncing head and calibrated wrist vision. [Drive A and physical replays](rolling-motor-trial-0.json.gz) · [Drive B](rolling-motor-trial-1.json.gz) · [Earlier stock and different-body tests](learned-tip-transfer.md)

## Time and memory

The estimator keeps twelve completed blocks of up to 480 physical control ticks plus the open block, under 0.5 MB of retained numeric data at capacity. Temporary arrays for fitting and the game policies are additional. Refitting at the twelfth break took 135 and 142 ms in the two streams; all fit calls together took 0.83 and 1.07 seconds respectively across those twelve points. Long uninterrupted games automatically roll the observation buffer at eight seconds. Calling `end()` at a break performs the synchronous update. Regions in which no joint moved supply no fit. Shorter alternatives using 7.6 KB of recursive state or a 240-window reservoir qualified only one drive on these same streams. [Recursive estimator](online-tip-report.md) · [240-window estimator and continuation](online-tip-reservoir-report.md)

```python
from scripts.experimental.tt24_rolling_motor import RollingMotorLearner

learner = RollingMotorLearner.load("artifacts/tt24/rolling-motor-0.npz")
learner.begin(current_encoder_reading)
# Call once per actual device tick after issuing the bounded motor request.
learner.observe(requested_motor_rates, next_encoder_reading)
# At a natural break:
learner.end()
learner.save("continued-motor.npz")
```

The saved estimator carries the source history needed to continue fitting. The combined gameplay checkpoints additionally include the wrist motor head, both optical calibrations, the original drive, the selected source-learned map, and the native Pong readout. Switching games and choosing the same electronics across them were provided by the benchmark. [Resumable motor A](rolling-motor-0.npz) · [Resumable motor B](rolling-motor-1.npz) · [Combined gameplay A](learned-tip-learner-0.npz) · [Combined gameplay B](learned-tip-learner-1.npz)

The same two learned maps were also applied directly to competitive two-arm table tennis on sixteen new serves. Neither produced a legal near return. A separate 1.45-second encoder inspection recovered eleven complete exchanges, versus 21 with the supplied drives. [Matched game evidence](rolling-match.md)

We also let the estimator keep updating through six ordinary receiving games per drive. On eight independent altered-arm drops the mean longest streak fell from 2.875 keeping the Pong estimate to 1.625 after updating. Checking candidate maps against the next game’s encoder readings recovered 2.25; the known-map comparison reached 10.50. The experiment preserves the earlier source checkpoint. [Same-stream adaptation and matched outcomes](rolling-motor-wrist-report.md)
