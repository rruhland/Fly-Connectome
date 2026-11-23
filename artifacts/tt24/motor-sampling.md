# Waiting for the joints to settle improves body identification

A physical inspection using five independent movements recovered control on the first new interfaces, but the same 11-tick wait after each command produced larger errors on the altered body. The new sampler moves among small positions around its initial command setting and observes the joints until three successive encoder changes are small. It limits the wait to 30 ticks per movement. The six displacements form a closed sequence and the fit uses a single linear least-squares estimate. All calibration happens during a contact-free setup opportunity. The subsequent wrist controller recomputes all five outputs during every game. [Initial experiment](motor-settle.md) · [Sampler and motor memory](../../scripts/experimental/tt24_motor_bank.py) · [Fresh physical experiment](../../scripts/experimental/tt24_motor_sampling.py)

| Eight new interfaces, four fresh games per interface | Stock rebounds | Altered rebounds | Ten consecutive on stock | Ten consecutive on altered |
| --- | ---: | ---: | ---: | ---: |
| Encoder-triggered settling | 248 | 159 | 11/16 | 6/16 |
| Original fixed-timing movements | 111 | 119 | 4/16 | 5/16 |
| Frozen decoder | 14 | 18 | 0/16 | 0/16 |
| Supplied true inverse | 222 | 147 | 10/16 | 6/16 |

Estimation error was 0.2–0.4% in relative matrix norm on all eight interfaces using the new sampler. Setup took 82–99 control ticks on stock (1.37–1.65 s) and 119–156 on the altered arm (1.98–2.60 s). Fixed sampling took 1.45 s on both and its error reached 17.8%. The two samplers differ in both the requested positions and their timing. A small error can change a long physical rally in either direction. The score differences above the known inverse on these finite cohorts do not establish that an approximate inverse is preferable.

The learner had no task reward or supplied actuator mapping during calibration. The target game used the established donor-trained motor head, prior optical calibration and a familiar reference learned in native Pong. Ordinary held releases on the two pre-calibrated physical arms were scored from wrist RGB and encoders. All 128 game tapes reproduced the full physical referee output. [Stock 1](motor-sampling-0.json.gz) · [Altered 1](motor-sampling-1.json.gz) · [Stock 2](motor-sampling-2.json.gz) · [Altered 2](motor-sampling-3.json.gz)

To use the encoder-triggered sampler, call `bank.connect(joints, step, seed, measured_settling=True)`. During normal game control, pass each output from the wrist policy to `bank.apply(output)`.
