# Fixed radial-cap robustness and fresh .25 Pong learning

The paired 30k raw/sharing study failed numerically before either endpoint. Raw
committed 20,773 transitions before nonfinite weights; sharing was explicitly
stopped at durable transition23,400 after finite numerical divergence. There was
no trained endpoint evaluation. Exact replay and camera/entity histories show
visual public-ID hypothesis switches generating increasingly large finite
velocity differences. Current and relative-motion energies rose above2000 while
weights/Q were still small, then Dutch traces and weights expanded. Mean-energy
calibration did not bound these rare inputs. Preserve this failure and the
resource-stop protocol deviation; do not replace it with the new experiment.

Add one generic policy-boundary intervention: fixed radial L2 cap with **squared
norm bound256**, identity below the bound, positive global radial scaling above.
The independent 1024-transition score-free .25 calibration had norm² median77.54,
q99=219.76, maximum243.11; **every calibration input is unchanged**. No reward,
simulator state, entity role or game-specific threshold fits the cap. Apply it
after the optional base transform; the main new arm uses **raw base features**.
Preserve raw structured transport, sparse indices, relative feature values and
missing zeros. For any fixed readout, positive radial scaling preserves greedy
argmax/policy class. It changes the value approximation and TD trajectory; do not
claim unchanged Q functions. At alpha.001, capped alpha||x||²≤.256. This bounds the
Dutch trace update operator by its decay factor, not a full SARSA convergence proof.

Finite guards reject nonfinite traces, proposed weights and action values before
another journal sample can commit. Failed live sessions remain poisoned; restart
from the journal's last valid state, not the advanced in-memory environment/RNG.
There is no silent clipping of rewards, weights, traces or TD targets.

Before fresh learning, captured-input/fixed-action regression must reproduce all
uncapped weight prefixes exactly and show bounded capped updates on **both**
failure windows. The recorded raw86-step window reached max|w|5.60e70, capped.154;
sharing reached1.48e6, capped.0733. This is a mechanism/stability diagnostic, not
new learning data or competence. Preserve full replay/camera/feature artifacts.

Train **one from-zero raw+cap256 policy for exactly30,000 transitions**, original
development seed **172900127**, opponent speed **.25**, alpha **.001**, lambda
**.995**, gamma_per_second **.95**, epsilon **.2**, frozen prior/vision, score-only
terminal±1, legal actions, exact camera cadence, explicit resets and2400-tick
rally limit. Save6000/15000/30000 weights for descriptive diagnostics. The evaluated
endpoint is fixed at30000, irrespective of early results. No sharing, blockRMS,
tracker modification, reward shaping or extra exposure is automatically added.

Reuse the four completed24-rally .25 development controls after verified exact
cross-source action/physics replay. Freshly evaluate only the final trained
checkpoint on the same24serve stream **904271833**, same20s limit. Controls:
initial/down9wins(8natural+1own-hit), up10(8+2), random9(8+1), no-op10(8+2);
random-action seed447201113. The development promising-signal rule remains at
least **four additional total and own-hit-followed wins** beyond their respective
largest control counts: **≥14/24 total and≥6 own-hit wins**. Truncations count as
nonwins; report them, all losses/completion fractions, completed-rally score
differential, durations, return per physical second and contacts. This is a screen,
not final acquisition proof. Final holdout seeds remain unused. Any independent
seed replication/blank-camera evaluation receives a separate finite allocation.

Numerical resource-stop rule, frozen prospectively: stop on nonfinite update or
on any committed record with **max(|q|,|q_next|,q_spread)>1000**. Terminal score-only
returns are bounded by1, making this an extreme numerical-failure screen rather
than a policy-performance criterion. Preserve the last valid journal and failed
allocation; do not alter alpha or restart another seed under this protocol.

Maximum: **30,000 new learning +28,800 fresh evaluation transitions** (one24-rally
policy, conservative1200decisions/rally), **35 aggregate wall minutes**, <=200
transitions/durable chunk, up to five chunks/interpreter within160s work/180s kill
bounds. Expected learning cost about15–18minutes; evaluation depends on rally
durations. Source/runtime/cap calibration/config manifests, complete snapshots,
journals, hashes and process costs are mandatory. Use the spare core for a separately
declared diagnostic-only readout capacity audit; its oracle-derived coefficients
must never become candidate weights or actual learner inputs.
