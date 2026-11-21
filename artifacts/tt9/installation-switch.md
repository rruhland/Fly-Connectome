# What carried when the arm changed

The five-axis player brought 12 optical service corrections learned on installation B to a new geometry, C. Both the carried and the reset learner lost 3–11. On returning to B with new throws, the carried player lost 7–11, compared with 11–13 for the player that went directly to the same B game. Native Pong retained all 180 reference action tapes after the complete B–C–B sequence.

A single player stayed active throughout each game. The registration fit, motor library, common and local reward evidence, actuator delay and racket response passed through the checkpoint. Both learners kept updating after each serve, using player-eye RGB and the public point reward. The standard game runner now resumes a saved registration automatically with the stored forward stroke speed. The installation numbers, server and disturbance labels are recorded by the experiment. They do not select the player's correction. The tested player still has a designed serving stroke, calibrated cameras and a falling-ball model. [Practice checkpoint B](precontact-registration-b-player.json.gz) · [First-person stroke comparison](five-axis-same-toss.mp4)

| Game | Final score | Legal own serves | Points won on own serve | Legal returns after service | Points with at least two returns |
|---|---:|---:|---:|---:|---:|
| New arm C, correction carried from B | 3–11 | 1/8 | 1 | 4 | 1 |
| C, same player with empty correction | 3–11 | 2/8 | 2 | 2 | 0 |
| C, same player with fixed stroke | 4–11 | 1/8 | 1 | 4 | 0 |
| Return to B after C | 7–11 | 8/11 | 3 | 11 | 2 |
| B directly from original practice checkpoint | 11–13 | 9/14 | 8 | 12 | 2 |
| B after C, original B service correction restored by evaluator | 6–11 | 3/9 | 3 | 6 | 1 |

Physical replays confirmed the commands and referee results for all six games. A legal own serve requires both bounces. An own serve is excluded from the count of subsequent returns. All comparisons on a given station used the same ordered opening seeds and disturbance schedule. Game length and lets could change which player served a later start. [C carried](cross-geometry-carried.json.gz) · [C empty](cross-geometry-reset.json.gz) · [C fixed](cross-geometry-fixed.json.gz) · [B after C](revisit-after-switch.json.gz) · [B directly](revisit-direct.json.gz) · [B with original correction restored](revisit-restored-correction.json.gz)

On C, the old sideways correction had the wrong sign on several throws. Four of the eight carried serves missed the toss. On B the preceding game cost match points even though the arm again landed most of its serves. All three B runs commissioned a fresh motor fit from eight probe sequences on the new calibration schedule. Restoring only the original service correction made the B outcome worse. The data support examining changes in the rest of the controller as well as the interaction of serve placement with the opponent's reply.

The final player held 31 observed service corrections, up from 12 at entry. Its 180-point native Pong test recovered 129 of 160 first interceptions, 195 of 206 later interceptions, and 41 earned points. All action tapes matched the qualified reference. [After C](cross-geometry-player.json.gz) · [After return to B](revisit-player.json.gz) · [Pong retention](revisit-pong.json.gz)

## Reproduction

Use the [hall environment](../tt6/README.md). On C, add `--registration reset` or `--registration fixed` to reproduce the controls. For the B direct comparison, start from `precontact-registration-b-player.json.gz`. For the restored-correction diagnostic, start from the C checkpoint and replace only its `service_registration` JSON member with the original B practice value. This diagnostic used the evaluator's knowledge that B had returned.

```bash
python -m scripts.experimental.tt8_service_trial /tmp/c.json.gz \
  --resume artifacts/tt9/precontact-registration-b-player.json.gz --seed 94216817 --installation-seed 867281049 \
  --body offset --phases ordinary mixed soft delay --limit 60 --probe-cache /tmp/tt8-cache --checkpoint /tmp/after-c.json.gz
python -m scripts.experimental.tt8_service_trial /tmp/back.json.gz \
  --resume /tmp/after-c.json.gz --seed 407301821 --installation-seed 317557501 \
  --body offset --phases ordinary mixed soft delay --limit 60 --probe-cache /tmp/tt8-cache --checkpoint /tmp/back-player.json.gz
python -m scripts.experimental.tt8_pong_retention /tmp/back-player.json.gz /tmp/native.json.gz
```
