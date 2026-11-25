# Lowering the stock arm behind the end line did not redirect central lows

A deep held paddle could reach a useful-sized part of its face, but its
orientation and attainable joint drive left the ball traveling backward.
All 15 measured contacts across 24 matched physical branches separated
with negative forward velocity. None landed across the net or got a reply.

This test moves the whole original SO101 into a lower route behind the
end line. It leaves the body fixture, stock jaw, camera calibration, rate
and force envelope, continuously randomized reactive opponent, and
referee unchanged. The protocol fixes 4214, an archived central miss with
a separately earned forecourt rim hit, and two new untouched natural
misses, 11508 and 2741. Each branch keeps the original motion until after
the preceding own strike separates. All changed work is an offline
mechanics diagnostic. No actor sees the archived chronology or true ball
state. The actual ball never touches the arm before the measured near
bounce. [Frozen cases and original archives](protocol.json)

## What held it back

The geometry scan took actual post-bounce arcs at eleven depths per case
from about −1.54 to −2.14 m. It tested both face signs, trial outgoing speeds
and landing depths, and distinct initial joint configurations. The directional
drive maximizes the velocity of the face center along its own normal while
leaving room for a backswing and braking at the original joint limits.
No fit simultaneously met the interior disk, useful face direction,
velocity and clearance filters. Near y = −1.72 m on 4214, the best-scoring
interior fits had forward normal components of .35–.39. The local impulse
proxy required 8.15–9.46 m/s along the blade normal for its selected return
arc, against the sampled 1.14–1.20 m/s joint-rate ceiling. On those poses,
even the simple optimistic reflection remains backward. These velocity
figures are explanatory model estimates, not an upper bound on all
nonlinear dynamics. Past about −1.84 m the closest tested fits were
near-horizontal or too far from the incoming ball. [Finite geometry and velocity scan](deep.json.gz)

The follow-up deliberately issued the geometrically closest feasible
faces even without a favorable modeled impulse. Both face signs physically
settled and were driven under three fixed lead times. Each body had
1.20–1.23 seconds to prepare before the post-bounce handoff. Every branch
kept the original actual far contact and near bounce. All 24 were clean
in the audited changed suffix, and three complete independent replays
had zero penetrations over 13,609 CAD samples. [Predeclared branches](limits.protocol.json)
[Actual controls and referee tapes](limits.json.gz)

| Release | Branches | Actual contacts | Aligned contacts ≤52 mm | Fastest actual forward speed |
| --- | ---: | ---: | ---: | ---: |
| 4214 | 8 | 4 | 2 | −1.664 m/s |
| 11508 | 8 | 8 | 0 | −0.125 m/s |
| 2741 | 8 | 3 | 0 | −0.633 m/s |

On 4214 the two aligned 48.19 mm hits, more than 27 mm inside the blade
rim, caught the ball at y ≈ −1.719 m. They sent it up at 2.52–2.78 m/s
and backward at 1.89–2.16 m/s. On 11508, the fastest attempt improved
forward separation but still sent the ball backward and upward at a
55.35 mm radius. Its fitted shoulder and elbow were already near their
negative position limits. This identifies a directional obstruction in
the tested deep family. It does not rule out every earlier movement on
the fixed body, and there is no new legal return to promote.
[Measured departures and complete-tape CAD audit](report.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt64_deep_receive plan
python -m scripts.experimental.tt64_deep_receive run
python -m scripts.experimental.tt64_deep_limits plan
python -m scripts.experimental.tt64_deep_limits run
python -m scripts.experimental.tt64_deep_report --verify
```
