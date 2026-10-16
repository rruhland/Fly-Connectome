# Fixed reward-credit diagnostic with camera-only DCT SARSA

The user explicitly requested informative reward and vision tests. This NEW
exploratory allocation supersedes launching the staged90k three-seed score-only
DCT screen. Preserve that unstarted study and all earlier failures. No backprop,
major architecture change or final-seed use is authorized here. Parent separately
owns a quarantined frozen-vision assay with physics/raster annotations; those
labels and controllers never enter candidate learning or arm selection.

Three concurrent arms, SAME from-zero seed172900127 and EXACTLY12,000 camera
transitions each (36,000aggregate), use the already frozen214×4389generic DCT/
supported/ridge transform, cap256 AFTER projection, zero weights, true-online
SARSA alpha.001/lambda.995/gamma_per_second.95/epsilon.2/hold1, .25 opponent,
2400physics-tick/20s training rally limit. No representation, prior, tracker,
actuator or observation change; empty frozen long banks/four-step fallback remain.
The fixed1024-row observational transform is reused, no fit/new calibration.

The ONE difference is the scalar training reward, fixed before outcomes:

| Arm | Learning reward per camera transition |
| --- | --- |
| score | raw terminal game score |
| hit | raw game score +1×OWN-paddle contacts |
| survival | raw game score +.1×actual physical seconds in this transition |

Every arm has an explicit reward specification, including zero/zero coefficients
for score, and logs `reward` as raw game score plus separate `learning_reward`,
`hit_reward`, `survival_reward`. Report their totals separately. No clipping,
fading, hit-based termination or automatic curriculum. The Pong implementation's
Outcome.hits is the controlled LEFT-paddle collision indicator; opponent returns
do not pay the hit bonus. Both collision cases are tested. Hits enter only the
explicit privileged scalar diagnostic reward, NEVER policy features or forecasts.
Survival pays only actual physics elapsed, including terminal/truncated transition
duration; resets, image processing and wall time pay nothing. At20s it adds2.

Hit reward supplies immediate controllable credit when a later loss yields
negative terminal feedback. Survival tests whether dense duration feedback alone
helps, but can favor timeouts instead of winning. Both deliberately change the
training objective; neither is relabeled original score-only acquisition. Matched
seed/initialization ceases to match later states once policies diverge. This is
one-seed finite-budget mechanism evidence, not independent statistical replication
or an asymptotic test of any learner. Alpha is fixed, not energy-matched.

Every camera updates frozen vision and commits durably. Existing per-camera
snapshots, full hashes (including T), replay/source checks, immutable exports and
reviewed semantic completion pins remain. Only complete12k endpoints get mapped
theta*T weights and exact cap/provenance; no earlier checkpoint selection.
Unpinned complete JSON requires actual zero-new-work recovery. Eligibility checks
exact declared journal rows and consistent theta/W/cap. Failed/partial arms remain
in the report, no replacement, and do not block completed arms' evaluation.
Nonfinite guards reject before commit; retain the existing max(|q|,|q_next|,
q_spread)>1000 numerical resource stop. Recovery discards poisoned live state.

AFTER all training workers settle, freeze and freshly evaluate every complete
endpoint on the SAME24-rally DEV904271833 stream, original .25 physics/20s limit,
hold1, full frozen perception, deterministic GREEDY decisions and RAW GAME SCORE
ONLY. Evaluator construction does not accept a diagnostic reward transform or
update any parameter; weights/vision/scaler/config learned hashes remain fixed.
Reuse verified initial/down,up,no-op,random controls (random seed447201113), after
exact source/action/camera/physics/reset proof; zero-Q initial is down. Reused
controls are counted only once. No final serve seed is expanded or evaluated.

Report all per-rally rows, wins/own-hit wins/natural wins, losses, truncations as
nonwins, completion fraction, actual durations, contacts, score/rally and return
per physical second. Report each shaped-vs-score and shaped-vs-control difference.
The old14total/6own-hit threshold is a descriptive development screen, not an
acquisition acceptance or automatic selection rule. A particularly informative
signal is at least4additional own-hit wins versus score-only plus positive total
win lift; all raw outcomes remain visible irrespective of that descriptor.
Contact/survival improvements without score lift are proxy-control evidence only.
Negative12k results do not prove reward shaping or the policy class incapable.

Bounds:12shared training minutes,25minutes OVERALL, anchored at actual manifest
creation and never reset on resume. Three single-threaded warm workers, ≤200
camera samples/chunk, ≤5chunks/process, ≤160swork/180skill bounds. Maximum36k
learning +86,400fresh-evaluation cameras (conservative24×1200×3). Full timeout-heavy
evaluation may exhaust wall budget; preserve partials, never extend silently.
No best-arm/seed replacement, further training, replication or holdout is automatic.
Pin source/runtime/driver/helper/protocol/transform/fit/baseline bytes; archive
complete tracked source and every reward config, journal, export, outcome and cost.
