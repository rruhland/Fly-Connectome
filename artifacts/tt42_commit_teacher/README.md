# Can a native teacher find and finish stronger movements?

On the unchanged court, a training-only teacher made 16 arm returns on practice release 4179, versus four for the matched camera-controlled learner. The opponent's first delivered strike, first legal bounce and shared random aim draws are checked against the ordinary player's tape. The teacher uses exact *current* ball and body state for private trial rolls, then issues the selected rates through the real landing without replacing them. There are no future opponent draws in its selection and no restart inside a point. This is movement capacity evidence, not a camera-play score. [First completed practice tape](pilot-4179.json.gz) · [Independent physical audit](pilot-audit.json)

## Action coverage and commitment

An earlier exact-state teacher made 47 from a long physical prefix, with both a different motion proposal family and a different search budget. A later exact-initialization control kept the same direct search and found fewer returns despite improved state. This new diagnostic changes action coverage and commitment explicitly alongside the state access, rather than treating better state as an intervention that should automatically help. [Earlier dynamic teacher](../tt38/motorcoach.md) · [Exact state without a different search](../tt42_private_init/README.md)

Each consultation starts from the same currently realized physical far separation. The search retrieves short scored actions from four earned motor banks with 25,753 indexed state/action segments. It also fits reachable initial guesses for both blade faces, at different depths and with different impulse, timing and joint-velocity goals. A bounded Sobol sweep retains distinct seed motions and adapts some with the observed native contact and departure effects of their own hypothetical trials. It rewards interior, answerable landings without looking at the far player's next target. Every issued motion is a direct five-joint swing plus the existing fixed grip reference. No changed clamp, jaw, ball, collision geometry, or servo is used. [Teacher implementation](../../scripts/experimental/tt42_commit_teacher.py)

On 4179 the teacher tested 4,608 schedules and found 249 legal virtual outcomes, representing 219 distinct command-rate tapes. The 16 issued candidates all landed and were all answered. Predicted and actual contact time, landing time and departure agreed to solver roundoff. The point took 26.10 physical seconds and 83.01 CPU seconds, and showed zero penetrations over 12,528 CAD samples. The next low delivery hit the camera foot after the teacher found no valid candidate.

A virtual legality claim is counted separately from an actual return. The teacher starts its exact bounded rates at a candidate's chosen tick and does not replan until the native referee awards that landing or ends the point. If the tape ends unexpectedly early, it commands zero until the claim resolves. The independent replay checks the applied prefix, referee event history, normalized command bounds, every checked CAD state and the original complete random opponent. [Audit implementation](../../scripts/experimental/tt42_commit_teacher_verify.py)

The four earlier hard later-delivery prefixes remain useful as negative controls. With identical original commands through the opponent's separation, the teacher rescued 8519's 0.986 m projected shot and the actual opponent answered it. On 4322, it found 82 virtual contacts over 2,304 tested actions but no legal landing. It did not rescue 8302 or 14843 either. Earlier preparation can change that reachable set, so the finite search does not rule out a legal return from a better arrival posture. [Matched hard forks](low-forks.json.gz) · [Fork replay](low-forks-audit.json) · [Prior early-preparation results](../tt41/deadline.md)

## From a teacher action to something a camera policy can use

Only complete actual teacher returns with an actual subsequent opponent answer are eligible for the pilot optical export. No virtual action, hidden object state, or native strike timing is a training input. The exporter replays the physical command streams, uses the original stereo and encoders, learns its contact clock from the intersection of previously observed incoming and outgoing tracks, and pairs supported histories with the rates the arm really issued. The camera-only student's live search can substitute at most two regular-prior slots with nearby, physically earned teacher fragments. Its perception, private nominal state source, search width, actuator safety and compiled opponent remain unchanged. [Observable exporter](../../scripts/experimental/tt42_commit_teacher_export.py) · [Controlled student](../../scripts/experimental/tt42_commit_teacher_student.py)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
out=artifacts/tt42_commit_teacher
python -m scripts.experimental.tt42_commit_teacher live "$out/pilot-4179.json.gz" --limit 1 --batch 128 --rounds 4
python -m scripts.experimental.tt42_commit_teacher live "$out/live.json.gz" --limit 3 --batch 128 --rounds 4
python -m scripts.experimental.tt42_commit_teacher fork "$out/low-forks.json.gz" --ids 8302 4322 14843 8519 --batch 192 --rounds 6
python -m scripts.experimental.tt42_commit_teacher_verify "$out/pilot-4179.json.gz" "$out/pilot-audit.json"
python -m scripts.experimental.tt42_commit_teacher_verify "$out/low-forks.json.gz" "$out/low-forks-audit.json"
python -m scripts.experimental.tt42_commit_teacher_diversity "$out/same-offer-diversity.json.gz"
python -m scripts.experimental.tt42_commit_teacher_export --source "$out/pilot-4179.json.gz"
python -m scripts.experimental.tt42_commit_teacher_student "$out/held-camera.json.gz" --limit 2
python -m pytest -q tests/test_tt42_commit_teacher.py tests/test_tt42_private_init.py
```
