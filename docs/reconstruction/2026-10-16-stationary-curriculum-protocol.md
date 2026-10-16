# Prospective fixed stationary-to-.25 score-only curriculum

The original score-only search failed after7updates/59,903training cameras:
12/24DEVwins(4own-hit) versus original initial10/24(2own-hit). Its fixed
119,831camera continuation also failed:14newupdates/21total,10/24(2own-hit).
Preserve both allocations intact. Endpoint calibration noise/policy RMS.624
and25%candidate action disagreement do not indicate exploration collapse.
First study36/128hit-containing rallies won; continuation83/240won. Neither had
training timeouts. This NEW study tests easier return-to-score feedback, not
another target-only exposure extension or a relabeling of negative evidence.

## Algorithm, phases and exact controls

Start FROM ZERO with exactly original optimizer seed1873926119 and training
seed172900127, score-free .10random initial readout and unchanged frozen transform
(rank183/214storedcoordinates/366actioncontrasts). Original initial/candidate
first-population matrices and four training serves must match exactly. Every
candidate uses frozen camera perception and generic raw transport; no oracle
coefficients, physics inputs, labels or backprop enter learning. No new transform
fit or calibration. Cap256,hold1,noise.25,eightmirroredpairs,top4/std-normalized
step.10 with function-RMS cap.10, identical score-only paired-return update.
Fitness ONLY mean terminal score across four matched serves:win+1,loss−1,trunc0.
Hits/natural-win labels are reporting only. No hit reward or timeout shaping.

Fixed phase0: stationary opponent speed EXACTLY0, up to6FULLgenerations,
30,000NEWtraining camera transitions and8minutes wall INCLUDING initial setup.
Fixed phase1: opponent speed.25, up to8FULLgenerations,60,000NEWtraining camera
transitions and16minutes wall from the persisted phase0 endpoint timestamp.
Maximum90kNEWtraining transitions/14updates,30minutes OVERALL, leaving up to
six minutes beyond maximum phase walls for endpoint evaluation. Four persistent
workers, exact pre-dispatch grants, <=200camera chunks, per-camera commits,
full snapshot/recovery and source/frozen-policy checks. Constructor/reset images
are counted separately. No unvalidated batching or new engine infrastructure.

Transition at the FIRST fixed generation/camera/wall boundary, NEVER from reward,
hits or DEV. Freeze LAST full phase0 update and pass only its parameters/RNG;
per-rollout perception/physics is fresh. Partial generations NEVER update. Definitions are created before checking the remaining phase budget, so even a
ZERO-observation unused draw is skipped. If an unused definition was drawn, retain its work/costs, advance to its POSTDRAW
RNG and skip that global index before phase1; otherwise retain checkpoint RNG
and next global index. Derive from state, not a fixed phase count. Phase-specific
directories prevent filename collisions. Input theta/RNG/global-offset and phase
wall anchors persist before work so a crash cannot reset a budget or redraw
noise. A valid endpoint needs at least2FULLupdates in EACHphase; technical failure
or insufficient completed updates fails rather than receiving replacement seeds,
extra exposure or tuned settings. No best-checkpoint retention or early promotion.

The only core-source change accepts finite NONNEGATIVE opponent speed in the
camera/evaluator validators. Physics already implements speed0. Default.45 and
all positive-speed branches remain unchanged. Reject negatives and nonfinite
values. Full four-control24DEV action/physics/reset projection was reexecuted
exactly on11551oldcamera events,0new observations/updates,9.95seconds; a chained
source proof retains previous baseline and original restricted initial reports.

## Endpoint and acceptance

No intermediate DEV or final seed use. Evaluate ONLY the last full .25phase
update on24DEVrallies/seed904271833, .25physics,20second per-rally caps, full
perception and matched constructor/reset/camera semantics. At most24kNEWendpoint
camera transitions and remaining30minuteoverall wall; no extensions and partial
results stay partial. Phase1 policy_at0 is inherited training lineage ONLY.
Acceptance controls are ORIGINAL FROM-ZERO restricted theta0 plus the four proven
simple controls, NEVER the inherited phase1 initial readout.

Total gate=max(14,maximum matched-control totalwins+4). Own-hit gate=
max(6,maximum matched-control own-hitwins+4), taking maxima independently.
Current gates14/6. Timeouts are nonwins. Report all rally outcomes, score per
rally, physical returns/time, own-hit/natural wins, losses/timeouts, hits,
training used/unused cameras and every actual image/replay/runtime cost. Passing
is a DEV screen only; no automatic independent replication or final evaluation.
All claims remain .25transfer, not original.45 success or forecast benefit.
Reserved final streams308713321,1203078761,1983461927 remain untouched.

## Descriptive mechanism checks, never adaptive schedule inputs

Report per phase/generation: natural and own-hit wins, hit-then-losses/timeouts,
conditional own-hit win rate, and camera/physicaltime by outcome. Report paired
score contrasts/selectednonzero directions and own-hit wins per camera, plus
common natural-win dilution. More raw wins consisting of shared natural serves
are not acquisition. Stationary hit-then-losses remaining common, or no increase
in conditional own-hit success/contrast density, would weaken the intended
feedback mechanism. Stationary saturation/zero gradients does not authorize early
transition. Later comparisons are on changing policies and are descriptive;
first-generation matched candidate/serve comparisons are only four shared clusters.

The task intervention also changes observation/trajectory distributions; positive
transfer would not prove terminal-feedback density is the sole causal mechanism.
No weak-task acquisition claim without its own fresh controls. If transfer fails,
retain the failure and compare alternatives prospectively. True-online SARSA on
this exact generic transform remains untested; its denser updates test conditioning,
while the oracle action witness does not establish linear Bellman-Q capacity.
No additional SARSA or representation arm is allocated by this protocol.
