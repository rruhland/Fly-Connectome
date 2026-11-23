# Does the released wrist controller change course when the ball does?

Twenty of 32 mixed wide starts were still playing at 1.2 seconds. At that moment the evaluator added 0.22 m/s sideways to the ball, in one of four balanced, randomly assigned directions. The carried policy continued responding to the wrist and its encoders. Its paired control executed the same controller's original physical commands recorded on the undisturbed release, then held its last servo targets if the tape ended. Both approaches received the same intervention after identical physical command prefixes. Each issued command sequence was replayed on the intervened simulator and reproduced the judged outcome exactly. [Runner](../../scripts/experimental/tt24_carry_disturbance.py) · [Original carried policy](carried-wrist.md)

| Body | Releases | Still playing at intervention | Consecutive post-kick rebounds, live | Original commands after kick | Live better / equal / worse among eligible |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock | 16 | 10 | 65 | 8 | 6 / 4 / 0 |
| Altered | 16 | 10 | 40 | 15 | 5 / 4 / 1 |

The numbers sum each release's uninterrupted sequence of qualified impacts immediately after the intervention. Early terminal releases earn zero. The controller made 105 such rebounds versus 23 using its original commands. The comparison establishes active correction of novel ball trajectories by the frozen policy. [Stock receipts](carry-disturbance-stock.json.gz) · [Changed-arm receipts](carry-disturbance-alternate.json.gz)
