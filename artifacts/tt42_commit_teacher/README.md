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

| Assessment release | Stock | Pilot 16 | Full 41 |
| --- | ---: | ---: | ---: |
| 11884 | 4 | 5 | 4 |
| 3401 | 1 | 35 | 1 |
| 14760 | 0 | 0 | 0 |
| **Total own** | **5** | **40** | **5** |

The 35-return point was one uninterrupted game over 57.04 physical seconds and 106.91 CPU seconds, against 36 actual opponent landings. The key repair was on the second offer. The stock and pilot arms received exactly the same first two physical far strikes and near bounces, and issued exactly the same commands up to tick 164. Both touched the second incoming ball near 1.016 m. Stock sent it forward at 3.002 m/s and downward at 2.553 m/s, faulting on its own half. The pilot arm sent it forward at 3.608 m/s and upward at 2.775 m/s, landing legally at x = .092 m, y = .418 m. Its first changed search prepared a credited motion and then authorized one on the next search. Cumulative own practice can contribute to the result, so this does not isolate which donor clip or search stage caused the repair.

The pilot logged 40 optical updates from its own earned hits and 19 retained miss fragments. No proposal closely matched a failed prefix in any condition, so the negative-credit filter was exercised as a recording mechanism but cannot explain the gains. The exact 16 pilot clips are a prefix of the full 41. The full arm admitted 39 candidates from the added 25 clips among its 56 admissions, changed real commands on the first two releases, and scored no gain on either. More teacher examples competed for the same two proposal slots and did not add complementary behavior under this retrieval rule. This supports selective actual-effect memory, not enlarging it indiscriminately or a general solution to all low returns. On 14760 the opponent never delivered a legal offer in any arm. The nine referee replays were exact and checked 43,807 CAD states without penetration. [Stock course](course-stock.json.gz) · [Pilot course](course-pilot.json.gz) · [Expanded course](course-all.json.gz) · [Exact same-offer and CAD verification](course-audit.json)

A retrospective replay with the pilot teacher allowed but own updates frozen ended at one return on 3401. Its whole command tape was identical to the stock point, despite 16 admitted teacher candidates. Its first two delivered balls and commands through tick 164 were the same as the successful chronological pilot point. The teacher demonstration archive alone was therefore insufficient at that physical offer. [Frozen 3401 control](3401-frozen-pilot.json.gz)

A second retrospective control allowed own updates from the first scored stroke of 3401 but excluded the preceding 11884 point. The new learner stored four optical frames from its first hit, yet ended at the same natural own-half fault with an action tape identical to stock and frozen pilot. This control was capped at a possible second own return, but it faults before the cap. Within-point credit alone did not reproduce the repair. In this context it helped to bring the pilot learner's prior own practice, and the same actor continued to acquire effects during the long point. [Cold updating 3401 control](3401-cold-live-repair.json.gz)

On the next two untouched assessment releases, both fixed before seeing either result, the pilot bank gained from the ordinary own-history updates again. Starting from the same carried memory, 7901 improved from 3 to 5. Its first four balls were physically identical across arms. On the fourth, the updating learner added 0.350 m/s forward and 0.303 m/s upward and landed legally where the frozen learner faulted its own half. Carrying the real hits and misses from 7901 into 8772, the updating pilot got 9 versus 1 for its frozen version. The total was 14 versus 4. This tests updating with the same sparse bank, not a new archive or a change to the visual front end. All four exact replays had zero CAD penetrations over 16,749 samples. [Frozen fresh points](fresh-frozen-pilot.json.gz) · [Updating fresh points](fresh-live-pilot.json.gz) · [Exact referee, same-offer and CAD audit](fresh-pilot-audit.json)

### Was a different useful movement crowded out?

Replaying the successful point's camera and encoders gave identical retrieval keys at ticks 159, 164 and 169. Passing those same keys to the older and expanded banks reproduced the pilot ranks and showed that the expanded bank displaced the second pilot fragment at both 164 and 169. The two admitted reference curves' RMS distance shrank from .241/.249 radians to .129/.179 radians over their first 15 frames. At 164 the displaced reference requested an incident face normal with vertical component .477, versus −.057 for the new nearest candidate that displaced it. This is a difference in requested motor behavior, not proof that the replaced fragment was the one actually used after native screening and adaptation. [Exact sensor lookup comparison](3401-recall-ranks.json.gz)

The prospective coverage variant retains the two-slot limit. Its list considers up to 12 nearby archived candidates and penalizes pairs whose first 15 issued-reference frames are almost redundant. The rule uses context distance and commanded curves, with no privileged ball state or release IDs. The fixed pair deviation target is .22 radians with a .20 maximum sensory-distance allowance above the nearest candidate. Its constants were set from the retrospective 3401 contrast, then frozen before the new disjoint 9965 and 3905 comparison against pooled nearest retrieval.

It did **not** help prospectively. The pooled bank scored 23 and 3 on 9965 and 3905. The same bank with curve-diverse retrieval scored 4 and 3, at the same search budget and live own-credit rules. The first four balls on 9965 were physically identical and both arms returned them. After the changed fourth stroke, the regulation opponent's later physical deliveries differed. All four command tapes replayed exactly with zero penetrations in 27,987 CAD states. A better next candidate should preserve or replace a motor pattern based on its observed signed departure and legal outcome in nearby contexts, not on the distance between requested joint curves alone. [Pooled nearest control](coverage-nearest-control.json.gz) · [Fixed coverage treatment](coverage-diverse.json.gz) · [Exact CAD and referee audit](coverage-audit.json)

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
python -m scripts.experimental.tt42_commit_teacher_course pilot "$out/3401-frozen-pilot.json.gz" --freeze --indices 3
python -m scripts.experimental.tt42_commit_teacher_course pilot "$out/3401-cold-live-repair.json.gz" --indices 3 --cap 4
python -m scripts.experimental.tt42_commit_teacher_recalls
python -m scripts.experimental.tt42_commit_teacher_course pilot "$out/fresh-frozen-pilot.json.gz" --freeze --indices 5 6
python -m scripts.experimental.tt42_commit_teacher_course pilot "$out/fresh-live-pilot.json.gz" --indices 5 6
python -m scripts.experimental.tt42_commit_teacher_course_verify --comparison learning --cad
python -m scripts.experimental.tt42_commit_teacher_course all "$out/coverage-nearest-control.json.gz" --indices 7 8
python -m scripts.experimental.tt42_commit_teacher_course coverage "$out/coverage-diverse.json.gz" --indices 7 8
python -m scripts.experimental.tt42_commit_teacher_course_verify --comparison coverage --cad
python -m pytest -q tests/test_tt42_commit_teacher.py tests/test_tt42_commit_teacher_course.py tests/test_tt42_private_init.py
```
