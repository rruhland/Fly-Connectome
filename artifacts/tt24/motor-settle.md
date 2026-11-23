# Learning a new five-motor interface in 1.45 seconds

Eight previously unseen motor mappings were inserted between the wrist controller's five requests and the physical joints. Four were dense random mixes with independent gains. Four also sheared the command axes. Each map was used in one simulated stock or altered SO-101. The portable wrist policy and all optical parameters stayed frozen. A separate contact-free setup interval let the arm observe five small independent movements after the encoders settled. The first four scored ball drops began immediately after setup. Later the familiar wiring returned, followed by a previously learned unfamiliar map. [Portable wrist controller](carried-wrist.md) · [Physical course](../../scripts/experimental/tt24_motor_settle.py)

The first stock release in the first new-map course to reach ten scored 21 qualified rebounds with the learned mapping and one with the original. Both camera pairs show the exact physical trajectories. [Watch the paired wrist and outside views](motor-interface-paired.mp4) · [Replay renderer](../../scripts/experimental/tt24_motor_settle_movie.py)

The model fitted the mapping using five joint displacements and the five requested changes in servo targets. A new mapping needed 87 actual control ticks at 60 Hz. A familiar mapping could be recognized with two directions in 42 ticks. The design uses known encoder order and nominal rate units. The matrices can have cross-axis mixing and sign changes. The ball, game state, prior action tapes and physical wiring matrix were unavailable to identification. The evaluator provided the setup opportunity before each group of four games. Every gameplay frame used live wrist feedback and its encoders. [NumPy identification and portable memory](../../scripts/experimental/tt24_motor_bank.py)

| First four releases after each unfamiliar interface | Actual rebounds | Starts reaching ten consecutive |
| --- | ---: | ---: |
| Stock, learned in 1.45 s | 258 | 11/16 |
| Stock, familiar decoder left frozen | 14 | 0/16 |
| Stock, true inverse supplied | 251 | 11/16 |
| Altered arm, learned in 1.45 s | 125 | 5/16 |
| Altered arm, familiar decoder left frozen | 17 | 0/16 |
| Altered arm, true inverse supplied | 165 | 6/16 |

When a previously encountered altered mapping returned, recognition took 0.70 s in each of four streams. Those 16 drops earned 231 rebounds, compared with 15 frozen and 264 with the known inverse. All four returns to original wiring selected the original identity and reproduced the original commands. Relative mapping error ranged from 0.002–0.007 on the stock body and 0.027–0.147 on the altered body. A fit from settled joint displacements recovered game control especially accurately on stock. Continuous discovery from ordinary play alone had previously struggled. [In-flight comparison](drive-spontaneous.md) · [Stock stream 1](motor-settle-0.json.gz) · [Stock stream 2](motor-settle-2.json.gz) · [Altered stream 1](motor-settle-1.json.gz) · [Altered stream 2](motor-settle-3.json.gz)

The saved five-map bank contains the identity and the two fitted mappings from each of streams one and two. It contains no true interface matrices or game actions. During use, the application supplies a callback that sends a bounded five-joint command and returns the next five measured joint angles. Use the inspection while the physical game is idle. The checkpoint restores motor correspondences that the arm actually measured. [Saved memory](motor-bank.npz)

```python
from scripts.experimental.tt24_motor_bank import MotorBank

bank = MotorBank.load("artifacts/tt24/motor-bank.npz")
inspection = bank.connect(initial_joint_angles, send_and_read_joint_angles, seed=17)
motor_request = bank.apply(wrist_policy.act(observation))
```
