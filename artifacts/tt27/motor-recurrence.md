# Learning which joints to probe after a new wiring change

The carried learner identified all 12 new full wirings and landed the same 12 legal returns on 36 feeds as the reset learner. On each of ten recurring changes, it used three port probes and 38 ticks. Reset used five probes and 58 ticks. The two unexpected operations contradicted the prior and triggered full five-port acquisition. [Paired outcomes and 72 continuous command replays](motor-recurrence.json.gz)

| Held change | Carried probes | Reset probes | Correct full maps carried and reset | Legal returns carried and reset |
| --- | ---: | ---: | ---: | ---: |
| Ten recurring changes | 3 | 5 | 10/10 | 8/30 |
| Two unexpected changes | 5 | 5 | 2/2 | 4/6 |

The recurring changes saved two probes, four nonzero commands and 20 ticks, or 0.67 simulated seconds per change. Across all 12 changes, including the full fallbacks, preparation fell from 696 to 496 ticks. Paddle contacts tied at 27. [Preparation and referee outcomes](motor-recurrence.json.gz)

The receiver started with the same 20 empirical motor templates in both conditions. It then inferred three previous wiring changes from physical encoder readings. The carried learner stored each change as a relative permutation, sign pattern and delay toggle between its successive inferred maps. Under a new interface, it composed the previous map with those remembered changes and first probed the unmeasured port that gave the most distinct predictions. It required three measured ports before accepting a recurrent candidate. When a prediction contradicted an encoder signature, it used exhaustive acquisition across all five ports. The reset learner used the same templates and encoder checks with no relative-change memory. [Implementation](../../scripts/experimental/tt27_motor_recurrence.py)

The evaluator supplied the signed-permutation and zero- or one-tick delay family. Each of two six-setting held courses used three recurring relative operations and one independently drawn unexpected operation. It did not supply the operation identity to the learner. The 12 held absolute maps were distinct, including across courses. Each switch also varied the lateral launch position and forward speed on fresh recorded-feed perturbations. Exchanging signs or routes solely on unmeasured ports could evade the short audit. This test covers the observed recurring family and two surprise draws. [Schedules, encoder traces and replays](motor-recurrence.json.gz)

Template acquisition cost 360 actuator ticks once. Learning the relative operations used three exhaustive practice changes per course, each 58 ticks. All held preparations had eight warm-up ticks. Each measured port added two nonzero actuator requests at magnitude 0.32 and eight zero-request settling ticks. The three-port preparation used six nonzero requests, 24 settling ticks and 38 total ticks. The five-port preparation used ten nonzero requests, 40 settling ticks and 58 total ticks. The same simulated arm continued from warm-up through release without intermediate resets. [Preparation and receive implementation](../../scripts/experimental/tt27_motor_active_continuous.py) · [Physical commands](motor-recurrence.json.gz)

## A hidden sign that the three-port audit misses

Flipping the sign of unmeasured request port 3 produced byte-identical encoder observations during the probes of ports 0, 1 and 2. The learner accepted the wrong map. Eight full maps were consistent with all three measured responses. The test chose the unmeasured port with the larger command magnitude on one earlier held block to create a visible receiving effect. [Paired sign-flip counterexample](motor-recurrence-boundary.json.gz)

A separate monitor used the earlier match encoder model to compare prediction errors among those eight compatible maps during ordinary play. It flagged the changed sign at tick six on each of three feeds and gave zero flags on the three nominal counterparts. The first flag triggered the full five-port diagnostic before feed two, restoring the exact map. Across the three feeds, the monitored branch landed one return. The wrong map landed zero, and the nominal map landed one. This checks a single constructed failure with subsequent motor commands that expose the hidden port. [Monitor-triggered recovery](motor-recurrence-boundary.json.gz) · [Implementation](../../scripts/experimental/tt27_motor_recurrence_boundary.py)

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_motor_recurrence
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_motor_recurrence_boundary
```
