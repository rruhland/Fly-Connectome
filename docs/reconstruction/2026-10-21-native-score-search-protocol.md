# Native-score adaptation at one fixed opponent

This protocol is frozen before candidate registration. The pendulum ON/OFF study
and both integrity audits have finished. Passing current-source prelaunch evidence
is required; clock-free staging expands the declared fresh cohort exactly once.
The supervised budget clock starts separately at launch, not during preparation.

## Question and fixed starting policy

Can an acquired visual return policy learn additional **wins** from native score
feedback at opponent speed .35? Use the first qualified return-learning lineage,
`pong-first-return-curriculum-20261021-r1`, selected before the frozen score-transfer
result. Its exact final theta, original frozen generic visual prior, DCT/support
transform and radial cap are retained. Neither the better recent point estimate
of the second lineage nor pendulum visual memory selects the starting condition.
The original **199,676 observed** return-training cameras remain inherited cost:
178,890 used fitness rows and 20,786 unused partial rows, not free initialization.

This is warm-start score adaptation. Changing from .65 to .35 also changes opponent
contact and returned trajectories, so it does not isolate reward alignment alone.
No backprop, new policy inputs, teacher labels, privileged action targets, hit reward
or first-hit cutoff is used. The 214 stored coordinates, 183 supported modes and
366 effective action contrasts remain unchanged.

## Learning and exposure

Each complete population has eight antithetic directions, two signs and twelve
fresh shared serves: 192 full-rally rollouts, hold1, .35 opponent, 10 s limit. Fitness
is the mean terminal **win indicator**: 1 for positive raw score, 0 for a loss or
time-limit truncation. Raw signed score and end flags remain in every journal.
Hit counts and return opportunities are post hoc diagnostics only. Shared natural
serve wins cancel the mirrored contrasts but still consume compute.

Keep the established stable top-four selection, selected-return standard deviation
normalization, mean-head subtraction, zero-variance/tie handling and function-RMS
update clipping. Let R be the RMS of the three current action scores on the same
1,024 generic score-free calibration scenes used previously. Direction functions
have RMS1. Perturbation RMS is .50R; step coefficient and maximum update RMS are
.10R, recomputed before each population and pinned in its definition. R must be
positive and finite. The starting R is expected to be .30083477355236915; verify
from the retained theta/cache during preflight. This rule is algebraically
equivariant to positive score scaling; numerical checks use explicit tolerances,
not a claim of byte equality after floating-point rescaling. The smaller local
step may be slow in a finite run; it is fixed before reward observations.

Stop at the first of twelve complete populations, **200,000 new transition-camera
observations across fitness and anchors**, or 40 minutes training. Episode initial
images are reported separately and are additional to that transition cap.
Require at least three complete
populations, including legitimate zero updates, for a valid endpoint. Use four
single-threaded workers and the existing once-per-process imports-only GC freeze.
No intermediate DEV evaluation, best checkpoint, replacement seed, hidden bonus
training or budget extension is permitted. Incomplete populations never update.

Before fitness starts on every even local generation, pin the current pre-update
center and fixed starting policy on the same twelve serves. Execute these anchors
after the population commits. They never enter fitness, checkpoint selection or
stopping decisions. At generation zero the two identical policies reuse one exact
prospectively declared roster. Anchor camera and wall cost count in the training
budget. A budget-limited partial anchor preserves the last complete learning update
and is explicitly incomplete diagnostic work; it provides no replacement budget.
A logged anchor task failure stops training and makes the overall allocation
partial. A fitness task failure prevents a qualifying endpoint. All failures,
unused partial fitness rows and anchor rows remain distinct and durable.

The shared transaction loop reconciles a fully completed pinned population after
a result-to-state crash before checking remaining budget, without new cameras or
double application. An unfinished draw is skipped with its post-draw RNG/index
recorded at freeze. Resume preserves the original clocks and failure receipts.

Proposed independent roots are training **1374860291**, optimizer **1009827637**,
DEV **1826405731**, control actions **593170827** and bootstrap **1529704139**,
tag **29117**.
Before any expansion, collision-check the complete positive uint32 exclusion
snapshot including all actual prior training, action, calibration and DEV IDs,
current pendulum IDs, and reserved final roots. Never expand final streams. New
training serves exclude the entire frozen snapshot, new DEV/action IDs, all new
roots and earlier populations' training serves. Different domains do not share RNG.

## Endpoint and acceptance

Freeze the last complete population and evaluate once on 64 fresh shared DEV
serves, .35 opponent, 10 s, frozen vision and greedy hold1. Nine cohorts are:
new endpoint, unchanged acquired starting head, both original restricted initial
heads, constant down/no-op/up, random1 and random10. The acquired start is an eighth
comparator, preventing preserved return skill from being called new score learning.
Every cohort uses the same prespecified incoming geometry; random controls share
their pinned action stream. No final holdout is used.

Qualification requires **both** total-win and own-hit-followed-win point lifts
at least 10 percentage points over the strongest of all eight controls, with both
paired 95% confidence interval lower bounds above zero. Bootstrap 64 serve clusters
10,000 times, recomputing the respective strongest control within every draw.
Report endpoint-minus-start point estimates and intervals separately, losses,
timeouts and return metrics. Identical natural-win offsets mean the two win views
are not independent evidence. Replacing losses with timeouts cannot increase
the training win objective or satisfy a win lift.

Reserve 15 minutes for evaluation, annotation and scoring, starting at the durable
training freeze, within **55 minutes overall**. Allow 288,000 evaluation cameras
as an explicit upper bound; partial work remains partial. Evaluation annotation
replays only evaluation physics. Parent training/optimizer replay is separate audit
cost with zero new candidate observations. Retained completion binds all journals,
reports, optimizer/RNG/definition/anchor artifacts and original wall eligibility;
later zero-work verification cannot recompute eligibility using a later clock.

This is one predetermined warm-start adaptation, not replicated score acquisition
or complete M1B. A positive result motivates one prospectively fixed second-lineage
replication; a negative is retained and informs the next learning change. No
opponent sweep follows automatically.
