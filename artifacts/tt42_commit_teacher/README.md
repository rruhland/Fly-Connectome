# Can a native teacher find and finish stronger movements?

On the unchanged court, a training-only teacher made 16 arm returns on practice release 4179, versus four for the matched camera-controlled learner. The opponent's first delivered strike, first legal bounce and shared random aim draws are checked against the ordinary player's tape. The teacher uses exact *current* ball and body state for private trial rolls, then issues the selected rates through the real landing without replacing them. There are no future opponent draws in its selection and no restart inside a point. This is movement capacity evidence, not a camera-play score. [First completed practice tape](pilot-4179.json.gz) · [Independent physical audit](pilot-audit.json)

## Action coverage and commitment

An earlier exact-state teacher made 47 from a long physical prefix, with both a different motion proposal family and a different search budget. A later exact-initialization control kept the same direct search and found fewer returns despite improved state. This new diagnostic changes action coverage and commitment explicitly alongside the state access, rather than treating better state as an intervention that should automatically help. [Earlier dynamic teacher](../tt38/motorcoach.md) · [Exact state without a different search](../tt42_private_init/README.md)

Each consultation starts from the same currently realized physical far separation. The search retrieves short scored actions from four earned motor banks with 25,753 indexed state/action segments. It also fits reachable initial guesses for both blade faces, at different depths and with different impulse, timing and joint-velocity goals. A bounded Sobol sweep retains distinct seed motions and adapts some with the observed native contact and departure effects of their own hypothetical trials. It rewards interior, answerable landings without looking at the far player's next target. Every issued motion is a direct five-joint swing plus the existing fixed grip reference. No changed clamp, jaw, ball, collision geometry, or servo is used. [Teacher implementation](../../scripts/experimental/tt42_commit_teacher.py)

On 4179 the teacher tested 4,608 schedules and found 249 legal virtual outcomes, representing 219 distinct command-rate tapes. The 16 issued candidates all landed and were all answered. Predicted and actual contact time, landing time and departure agreed to solver roundoff. The point took 26.10 physical seconds and 83.01 CPU seconds, and showed zero penetrations over 12,528 CAD samples. The next low delivery hit the camera foot after the teacher found no valid candidate.

Across the first three preassigned practice releases the teacher made **41 arm returns to the matched camera player's 34**. The individual results were 16–4 on 4179, 11–30 on 3361 and 14–0 on 9519. It proposed 644 nominal legal strokes in 13,184 trials, 556 distinct rate tapes across the searched windows. All 41 issued candidates landed and were answered. There were zero CAD penetrations in 32,439 independently replayed samples. The teacher's best individual point was 16, not the 30 achieved on a different release by the camera control. Its hypothesis search is both privileged and much broader. [All three uninterrupted points](live.json.gz) · [Exact replay and matched draws](live-audit.json)

A virtual legality claim is counted separately from an actual return. The teacher starts its exact bounded rates at a candidate's chosen tick and does not replan until the native referee awards that landing or ends the point. If the tape ends unexpectedly early, it commands zero until the claim resolves. The independent replay checks the applied prefix, referee event history, normalized command bounds, every checked CAD state and the original complete random opponent. [Audit implementation](../../scripts/experimental/tt42_commit_teacher_verify.py)

The four earlier hard later-delivery prefixes remain useful as negative controls. With identical original commands through the opponent's separation, the teacher rescued 8519's 0.986 m projected shot and the actual opponent answered it. On 4322, it found 82 virtual contacts over 2,304 tested actions but no legal landing. It did not rescue 8302 or 14843 either. Earlier preparation can change that reachable set, so the finite search does not rule out a legal return from a better arrival posture. [Matched hard forks](low-forks.json.gz) · [Fork replay](low-forks-audit.json) · [Prior early-preparation results](../tt41/deadline.md)

### More than one motion works on the same hard ball

The original 8519 stroke touched the ball but sent it into its own half. From exactly the same physical state after the opponent's separation, a broader screen found 341 legal outcomes in 768 native proposals, spanning 290 distinct rate tapes. Five candidates were selected for distance in departure velocity, impact time, and landing location. All five then scored and were answered when actually driven through the ordinary arm and opponent, with no CAD penetrations. The table reports physical departure and landing, and the time since the same intervention. [Five exact physical tapes](same-offer-diversity.json.gz)

| Face | Impact in s | Lateral, forward, up in m/s | Landing x, y in m |
| --- | ---: | ---: | ---: |
| Plus | .665 | −.94, 3.78, 2.32 | −.257, .650 |
| Minus | .758 | +.32, 2.86, 3.01 | +.431, .152 |
| Plus | .665 | +.11, 3.18, 3.41 | +.293, .796 |
| Plus | .669 | −.15, 3.40, 1.75 | +.152, .231 |
| Plus | .646 | −.93, 2.20, 3.31 | −.395, .245 |

The initial stock posture and current ball therefore do not force a single tightly timed movement. These remedies took contact at different depths, on both faces, and differed by up to 113 ms. A correction based on the actual effects can retain separate contact placements and outgoing impulses. Averaging positive and negative face trajectories would discard useful solutions. This observation applies to one relatively low fork and does not erase the three lower failures. On the colder 4322, stretching local pose and outgoing-velocity corrections across the longer approach instead of only the last 0.4 s did not produce a return either, and reduced trial contacts from 82 to one at the same 2,304-candidate budget. [Negative longer approach control](phase-adjust-cold.json.gz)

## From a teacher action to something a camera policy can use

Only complete actual teacher returns with an actual subsequent opponent answer are eligible for the pilot optical export. No virtual action, hidden object state, or native strike timing is a training input. The exporter replays the physical command streams, uses the original stereo and encoders, learns its contact clock from the intersection of previously observed incoming and outgoing tracks, and pairs supported histories with the rates the arm really issued. The camera-only student's live search can substitute at most two regular-prior slots with nearby, physically earned teacher fragments. Its perception, private nominal state source, search width, actuator safety and compiled opponent remain unchanged. [Observable exporter](../../scripts/experimental/tt42_commit_teacher_export.py) · [Controlled student](../../scripts/experimental/tt42_commit_teacher_student.py)

On the matched low 8519 fork, a separate 60 Hz stereo-based export recovered the incident and departure motion for the failed original as well as all five alternatives, and joined it to the actual encoder, target-reference and normalized-rate traces. It verified a common RGB-and-action prefix through the intervention and the same physical referee event history on replay. The five privately measured departures are used only to verify this sensory fit. Median optical 3D velocity error was 0.139 m/s. The output records all 15 pairwise observed differences, plus each physical landing label, to distinguish interventions that changed timing or direction from interventions that simply hit the ball. [Stereo effect labels](same-offer-camera.json.gz) · [Encoder and rate traces](same-offer-camera.npz)

Relative to the missed original, one visible remedy added 0.47 m/s forward and 0.65 m/s upward. The opposite-face option added 1.42 m/s upward while taking 0.46 m/s off its forward velocity. Both landed. This is why a useful motor memory retains alternative timed reference sequences instead of averaging the signed correction vectors or turning them into a single lift command.

The first held comparison froze both ordinary carried memories. The live solver kept its original budget and only swapped in up to two proposed reference trajectories at a time when a nearby precontact RGB and encoder example existed. This needs no training labels at runtime and no backpropagation. It did not count virtual hits as learning. The 16 transported clips came from 4179, and neither 9550 nor 15665 appears in any of the three stock, carried or teacher banks. [Earned sensory archive](earned-camera.npz) · [Provenance and clip support](earned-camera.json)

| Held release | Stock returns | Pilot teacher returns | First different issued tick | Shared random targets | Teacher candidate admissions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9550 | 5 | 12 | 173 | 6 | 108 |
| 15665 | 12 | 11 | 91 | 12 | 92 |

The total was 23 to 17, at 69.0 versus 53.5 CPU seconds across both complete points. Same first physical far contact and first legal bounce on both, and identical real referee events leading up to the first changed command. Independent CAD replay reproduced the four scores exactly, with zero penetration samples over 33,195 checked states. The 9550 treatment changed a motion on an approach both arms returned and changed the unpredictable rally that followed. The other release lost one return. The number of admitted candidate snippets is not the number of motions issued. [Paired camera-only games](held-camera.json.gz) · [Independent replay](held-audit.json)

### Can more physical teaching complement the pilot?

The next three fixed assessment releases in the parent course, 11884, 3401 and 14760, are disjoint from the two frozen comparisons, the teacher's three practice starts, and the base direct and carried camera archives. Each arm carries its own physically scored, optically aligned hit history across this sequence. A completed miss to a legal incoming ball may retain a bounded set of its observed incoming situations and actually issued nine-frame motor prefixes. In a later situation it can demote a stock proposal or reject a new teacher proposal only when both the observed ball/joint context and the relative command profile are close. An optically unobserved miss is not a motor label. All treatments get the same new hit and miss update rules. The only factor changed is no teacher, the 16 clips from 4179, or the 41 clips from all three points. The budget stays at 28 with up to two substituted candidates per lookup. [Chronological learner](../../scripts/experimental/tt42_commit_teacher_course.py) · [Expanded optical archive](earned-camera-41.npz) · [Expanded provenance](earned-camera-41.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
out=artifacts/tt42_commit_teacher
python -m scripts.experimental.tt42_commit_teacher live "$out/pilot-4179.json.gz" --limit 1 --batch 128 --rounds 4
python -m scripts.experimental.tt42_commit_teacher live "$out/live.json.gz" --limit 3 --batch 128 --rounds 4
python -m scripts.experimental.tt42_commit_teacher fork "$out/low-forks.json.gz" --ids 8302 4322 14843 8519 --batch 192 --rounds 6
python -m scripts.experimental.tt42_commit_teacher_verify "$out/pilot-4179.json.gz" "$out/pilot-audit.json"
python -m scripts.experimental.tt42_commit_teacher_verify "$out/low-forks.json.gz" "$out/low-forks-audit.json"
python -m scripts.experimental.tt42_commit_teacher_verify "$out/live.json.gz" "$out/live-audit.json"
python -m scripts.experimental.tt42_commit_teacher fork "$out/phase-adjust-cold.json.gz" --ids 4322 --batch 192 --rounds 6 --phase-adjust
python -m scripts.experimental.tt42_commit_teacher_diversity "$out/same-offer-diversity.json.gz"
python -m scripts.experimental.tt42_commit_teacher_effects
python -m scripts.experimental.tt42_commit_teacher_export --source "$out/pilot-4179.json.gz"
python -m scripts.experimental.tt42_commit_teacher_export --source "$out/live.json.gz" --output "$out/earned-camera-41.npz"
python -m scripts.experimental.tt42_commit_teacher_student "$out/held-camera.json.gz" --limit 2
python -m scripts.experimental.tt41_bounce_joint_audit verify "$out/held-audit.json" --tape "$out/held-camera.json.gz" --cad
python -m scripts.experimental.tt42_commit_teacher_course stock "$out/course-stock.json.gz"
python -m scripts.experimental.tt42_commit_teacher_course pilot "$out/course-pilot.json.gz"
python -m scripts.experimental.tt42_commit_teacher_course all "$out/course-all.json.gz"
python -m scripts.experimental.tt42_commit_teacher_course_verify --cad
python -m pytest -q tests/test_tt42_commit_teacher.py tests/test_tt42_commit_teacher_course.py tests/test_tt42_private_init.py
```
