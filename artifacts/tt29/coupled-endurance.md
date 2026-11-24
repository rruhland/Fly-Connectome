# Camera-controlled rallies with speed-dependent motor torque

Two fixed RGB cameras observed one continuous ball in regulation table tennis. A simulated SO-101 stayed clamped to the near table edge. The powered practice partner selected a new forehand target on each return. The recorded opening launched the ball once. Each counted return required a new paddle contact and a legal landing on the other half. The opening launch was excluded. [Fixture and camera photographs from simulation](fixture.md) · [Native referee](../../scripts/experimental/tt29_endurance.py) · [Measured opening source](../tt25/recorded-secondary-reference.json)

| Measured opening | Consecutive legal returns | Simulated duration | Median departure speed | Median time between hits | Live CPU time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7271 | 50 by the arm + 50 by the partner | 71.12 s | 4.68 m/s | 0.704 s | 81.8 s |
| 4187 | 50 + 50 | 69.18 s | 4.79 m/s | 0.665 s | 122.7 s |

The arm's five stroke joints had a 3.35 N·m stall cap. Available motoring torque fell to zero at the 5.45 rad/s no-load reference. The model retained braking authority. An independent signed-force audit found zero violations in 170,680 and 166,040 joint samples. Full CAD replay reproduced every referee event, sampled 34,137 and 33,209 states, and found 18.6 mm minimum excluded-geometry clearance in both points. The motor curve is a hardware-informed simulation assumption. The source dataset's 10th, 50th and 90th percentile launch speeds were 4.22, 5.50 and 7.18 m/s. These are cooperative rally speeds. [Signed forces and joint speeds](rally-physics-torque-envelope.json) · [4187 referee and geometry replay](coupled-fresh100-audit.json)

The long rallies followed three continuous practice sequences. In the first sequence scores were 52, 8, 46 and **100**. A second sequence started with 120 rewarded movements and scored **95**, 10 and **100** on a different random opponent schedule. On its final point the arm started with 173 examples, retrieved 49 movements and selected two fresh plans. A third schedule carried the resulting 223 examples and scored 10, **95**, 0 and **100**. The two 95s ended after a legal near-side return. On 7271 the arm intercepted the ball across a 6.8 cm lateral span. Its shots landed across 62.5 cm of the opponent's half. Moving the opponent's targets across a wider region is substantially harder. [First course](rally-objective-envelope.json.gz) · [Second course](envelope-carry.json.gz) · [Third course](envelope-fresh.json.gz) · [Physical net-cord recovery](tt29_partner_short.md)

The live arm received camera pixels, its own encoders, issued command history and legal-return feedback. It used a designed ball detector, a calibrated optical model, candidate arm movements and an isolated simulator to check predicted outcomes. It learned a reusable library of rewarded joint movements without backpropagation. The opponent had access to simulator state. The solo warm-up supplied actuator timing in separate delayed games. Its measured contribution was 10 to 60 and 16 to 30 returns. These 100s used stereo and the retained table movement library. [Matched learning comparisons](endurance.md) · [Warm-up and perturbations](adaptation.md)

## Replay the 7271 point

Run the following after obtaining the pinned feeds and installing MuJoCo as described in the [benchmark instructions](endurance.md).

```sh
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1 \
python -m scripts.experimental.tt29_vision_replay /tmp/tt29-feeds.json \
  --source artifacts/tt29/envelope-carry.json.gz --id 7271 \
  --movie /tmp/coupled-camera100.mp4 --receipt /tmp/coupled-camera100.json
```

To run the second course live, let the first two points update the same memory before the final opening.

```sh
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1 \
python -m scripts.experimental.tt29_action_memory /tmp/tt29-feeds.json /tmp/coupled-live.json.gz \
  --ids 7345 4187 7271 --checkpoint artifacts/tt29/envelope-practice-prefix.json.gz \
  --seed-offset 41155 --camera-period 2 --fast-partner --cached-stereo \
  --rated --motor-envelope --rally-objective --recent --blend --robust-contact
```
