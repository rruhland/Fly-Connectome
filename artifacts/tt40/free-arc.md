# Defensive free-arc return with the unchanged SO-101

A low rebound does not necessarily demand more racket speed. If the strike is only required to clear the net and land on the far half, the robot can choose among ball trajectories with different landing depths and flight times. Koç, Maeda and Peters use this distinction for a defensive player in their free-time controller. Nakashima and colleagues also optimize the redundant launch and racket angles to reduce the racket velocity required by the impact. I tested that freedom in the existing simulation rather than transplanting either paper's controller. [Koç, Maeda & Peters, Section 5](https://is.mpg.de/uploads/publication_attachment/attachment/446/KocMaePet18.pdf) · [Nakashima et al., inverse rebound](https://ieeexplore.ieee.org/document/6181649/)

The existing SO-101 already searches multiple intercept times, alternate kinematic branches and candidate racket velocities. The distinction here is the **outgoing** search. The earlier terminal planner builds its initial direction from a 4.20 m/s outgoing ball aimed at a fixed far-side depth. It then tests five racket-normal speeds, uses a central-depth penalty and screens the resulting body motion. We form a bounded window of legal flight arcs first, then invert a simple normal restitution law to get the minimum useful racket-normal speed. We fit the actual reachable disk orientation, recompute the useful speed for that achieved geometry and send the proposed sequence to the existing finite-difference endpoint shooter. This does not change the hardware, control rate, force or collision rules, the fixed cameras, or the opponent.

For a preselected far-half landing coordinate and flight time, the linear-drag equation uniquely gives a launch velocity. We reject arcs lacking net margin, forward pace or room for a following physical reply. The optical robot twin rechecks any movement with its original contact model and two stereo uncertainty perturbations. A candidate is not an actual return until its real physical continuation lands legally and the unchanged random opponent replies. The analytical normal restitution law is only an inexpensive proposal generator. In particular it does not represent spin, tangential friction or racket compliance, all of which can materially change the actual departure. [Gossard, measured normal and tangential response](https://arxiv.org/abs/2604.11349) · [Conti et al., coupled flight and contact fit](https://arxiv.org/abs/2606.28805)

## First matched practice check

Four previously failed low practice openings were selected from the same full-random tape used by the wrist-pulse study. The camera-only planner saw the observed ball and the encoder history during each exact original command prefix. Across 19 eligible probe times and 126 theoretical contact samples, the median ideal racket-normal speed was 0.836 m/s in the free window versus 1.402 m/s for the fixed-pace formula at the same camera state. That comparison happens **before** body constraints. No screened disk pose could satisfy the checked reach, contact and outgoing-window conditions together. No command changed, and all four complete physical referee histories matched stock exactly. [First matched practice tapes](free-arc-practice.json.gz)

## Matched extension with the actual stock approach

As a second check, take the already reachable moving racket center from the planned stock tape and change only its distance normal to the proposed contact face. This preserves the stock movement's tangent location. A normal-window search seeded six endpoint targets in 23 probe windows. One target was nominally legal, but neither nearby camera perturbation remained legal. The ordinary gate again left all four incoming physical shots unchanged. [Anchored replay](free-arc-anchored.json.gz)

A deliberately weaker practice-only gate authorized the two candidate trajectories that appeared to have plausible lift and forward speed. Neither scored or got a legal opponent answer. More specifically, both variants reached the exact old racket contact and departure before the first changed issued command. Their nominally different joint targets had not translated into new early actuator commands. This observation is diagnostic about the clipped reference interface, not two genuinely different impact outcomes. [Physically issued reference variants](free-arc-issued.json.gz)

Reparameterizing the same endpoint solve in already bounded actuator-rate coordinates eliminated that ambiguity in the private body. Two 4239 candidates changed normalized rates from the first trial tick by up to .125 and .158 on the ±.8 range. Both made a modeled contact. Neither delivered a legal arc in the camera-initialized collision model. The strongest modeled lift had 4.09 m/s vertical speed but only .98 m/s forward. No candidate was issued by the ordinary gate. [Rate-space replay](free-arc-rates.json.gz)

I then physically challenged that one weak high-lift rate sequence on the exact 4239 prefix. The emitted actions first changed at tick 87, seven ticks before the original collision step. The ball did get more vertical speed, yet actual forward speed collapsed. The prior departure was (3.56, 2.12) m/s in the forward and vertical axes, versus (.99, 4.08) m/s with the rate sequence. The modified ball went to the floor and there was no opponent answer. The private model had predicted (.981, 4.088) m/s, so optical-to-physical transfer of this *particular poor departure* was accurate. The missing result was a useful joint capacity to produce lift and forward impulse together, not simply a need to change estimated spin or choose a softer height target. [Single genuinely actuated physical challenge](free-arc-rate-witness.json.gz)

No part of this approach earned a scored and answered rescue. There is no fresh-game claim. The useful engineering finding is to verify actual pre-contact rate changes, not just changes to unclipped position targets, and to rank impacts on the two achieved departure components jointly when trying to clear the lowest balls.

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt40_free_arc artifacts/tt40/free-arc-practice.json.gz
python -m scripts.experimental.tt40_free_arc artifacts/tt40/free-arc-anchored.json.gz --anchored
python -m scripts.experimental.tt40_free_arc artifacts/tt40/free-arc-issued.json.gz --anchored --issue-diagnostic --limit 2
python -m scripts.experimental.tt40_free_arc artifacts/tt40/free-arc-rates.json.gz --anchored --issue-diagnostic --limit 2 --rates
python -m scripts.experimental.tt40_free_arc artifacts/tt40/free-arc-rate-witness.json.gz --anchored --issue-diagnostic --limit 1 --rates --challenge
python -m pytest -q tests/test_tt40_free_arc.py
```
