# Always-open local forecast and credit does not pass Pong M1A

The approved bounded ablation removed the periodic timing gate from both
forecast expression and issue-time eligibility on the 17,557 existing
predictive edges entering 2,660 directly observed visual cells. The local
sensory-state context still selects the physical synaptic magnitude and the
component that receives eight-tick-later credit. The measured topology,
transmitter signs, delays, signed synaptic currents, two component bounds
[0,10], and no-backprop update remain fixed. Production defaults and the
original gated checkpoints were not changed.

The gate was a real coverage problem, but removing it was insufficient. In a
frozen 500-frame evaluation, ON anticipation rose only from .000212 to .001297
after training; OFF anticipation fell from -.000147 to -.001402. Neither is
close to the predeclared .1 threshold. OFF event MSE worsened. M1A is not
passed, so M1B remains blocked.

## Checks before learning

With learning disabled, the gated and always-open full graphs matched exactly
over 50 Pong frames and 400 neural ticks in spikes, observations, physical
currents, membrane/adaptation and synaptic state, local eligibility state,
context magnitudes, raw forecasts, and camera targets. The only differences
were the intended issue forecast and eligibility multiplications by the old
gate. The check found 284,096 eligible edge issues behind closed old gates.
The always-open frame-9 checkpoint resumed to exactly the frame-19 neural,
rule, camera, Pong, forecast, score, event and stored forecast state of an
uninterrupted run. The gated mode also passed the same expanded resume check.
The full suite passed: 319 tests, 4 skipped.

## Fixed development pilot

The source checkpoint SHA-256 was
`8c4af80ff3904881bac9313b618dd9989bb51174b47606f28eca88865296005c`.
As in the prior bridge, its one out-of-bound source edge was clipped once to
the existing maximum before warmup. Training ran continuously for 500 Pong
camera frames (4,000 neural ticks) on seed 1101 at the previously selected
rate 1.0. Exact-resume checkpoints were saved every 100 frames. Frozen initial
and trained networks then saw identical 500-frame target sequences on seed
1102; the original gated initial/trained arms were replayed against the same
targets and reproduced their saved scores exactly. Final seeds 1201-1204
were untouched.

| Seed-1102 metric | Always-open initial | Always-open trained | Gated trained control |
|---|---:|---:|---:|
| ON anticipation, 612 events | .000212 | **.001297** | -.000001 |
| OFF anticipation, 610 events | -.000147 | **-.001402** | .002919 |
| ON event MSE | .999634 | .998311 | 1.000002 |
| OFF event MSE | 1.000338 | **1.004980** | .996819 |
| Whole-field MSE | .001078 | .001414 | .000922 |
| Quiet MSE | .000158 | .000492 | .0000024 |
| All-quiet alarms at forecast magnitude >=.1 | .226% | .209% | .00038% |
| Event-adjacent quiet alarms | 5/2147 = .233% | 17/2147 = .792% | 2/2147 = .093% |

Event-adjacent quiet means that the same target cell has an event one frame
before or after a quiet target frame. Both neighboring frames must be observed,
so the first and last target frames are excluded from that denominator. The
trained arm satisfies the 5% adjacent-quiet alarm ceiling, but fails both
event anticipation thresholds and the OFF MSE improvement check. The
quiet-dominated whole-field score also worsens. There is no basis to extend
training or sweep another rate under this approved protocol.

The 500-frame training pass changed 9,447 context magnitudes. In a separate
checkpoint audit, 8,785 changed in the nonpositive local sensory context
(8,656 decreased, 129 increased) and 662 in the positive context (636
decreased, 26 increased). This is consistent with strong downward pressure
in an always-open stream, but endpoint weights alone cannot attribute that
pressure to quiet versus event updates. The trained frozen replay changed
per-neuron spike counts in 4,984 cells relative to its initial replay;
total spikes rose from 225,009 to 228,196. The maximum simultaneous spikes
per tick were 839 and mean firing was 1.16 Hz. All states stayed finite,
non-target magnitudes were unchanged, and update reconstruction error was zero.

Training measured 5.23 camera frames/s in this run; frozen always-open arms
measured 5.32 and 5.04 frames/s. The gated control replays measured 6.38 and
5.37 frames/s on the same host, whereas the earlier gated pilot measured
12.38 frames/s. These timings establish that this run was compute-limited,
but do not isolate a reliable always-open overhead from host variation.

## Decision boundary

The controlled repeated-motion motif still supports useful local
context-dependent efficacy. This full-graph result says that a periodic
local timing gate covers too few irregular Pong events, while removing the
gate yields broad but ineffective credit and weak event forecasts. It does
not establish that more training, a different rate, or an unobserved visual
cell class would fix the objective. The next architecture review should
specify a local mechanism that selects *when* a prediction is expressed and
credited under irregular motion, and test its quiet/event update balance.
No further neural or plasticity mechanism was added in this ablation.

Evidence: [pilot and exact control replay](2026-09-22-pong-gate-ablation-results.json),
[zero-learning parity](2026-09-22-pong-gate-ablation-parity.json), and
[exact resume](2026-09-22-pong-gate-ablation-resume.json). Full checkpoints
remain under `runs/full-context-gate-ablation-v1/` and the original gated
checkpoints under `runs/full-context-m1a-v1/`.
