# Single bounded exposure diagnostic after the failed 25-episode mechanism gate

The preceding raw/conditioned gate completed unchanged: all three paired policies
scored 10/20. Its failure is retained. Exact replay of all 1200 learning transitions
matched records and final weights and showed that terminal reward times the Dutch
trace accounts for essentially all learned first-cue discrimination. Conditioning
increased that discrimination 48–88 fold, but common bias still exceeded half the
left/right separation by factors 3.24, 1.35 and 1.66. Correct choices on both fixed
templates require this ratio below one. This supports testing exposure before
changing lambda or another feature setting.

Run fresh zero-weight raw and conditioned policies on the same three mechanism
training seeds (531028119, 1047199272, 1958971091), now for exactly **200 balanced
episodes = 1600 transitions each**, six jobs = **9600 learning transitions**.
Use the existing episode-schedule implementation: a fresh balanced 200-episode
permutation per seed, paired across arms. This is a new allocation, not continuation
of the earlier shuffled 25-episode schedule. Keep lambda=.8, alpha=.0001,
gamma_per_second=.95, epsilon=.2, frozen vision, actual .02-second cadence,
score-only terminal reward and the exact already fitted scaling artifact.

Evaluate the six final policies once, using the same fresh balanced 20-episode
mechanism schedules (62697349, 1197177281, 2007311671): **960 evaluation
transitions**. Reuse the completed initial/random baselines from the prior gate,
whose policy, environment, camera/reset, source/runtime and seed settings are
identical. Carry only frozen readout weights plus pinned scaling configuration.
No new calibration is performed. Charge the original 512 calibration transitions
separately as reused development cost; they are not new observations here.

The total cap is **10560 new transitions**, 9600 learning + 960 evaluation, with
20 minutes aggregate wall time and <=200 transitions/<=180 seconds per chunk.
Preserve partial allocations if capped. No extensions, best-checkpoint selection,
arm-specific budgets, replacement seeds or multiple new knobs. Report cumulative
training reward/correctness by episode from the full journal, all final side counts,
and Q discrimination/common-bias decomposition. Final evaluation uses exactly the
200-episode endpoint regardless of intermediate curves.

This diagnoses whether more exposure makes state-dependent credit overcome the
remaining greedy bias, and whether scaling helps at that exposure. It does not
require making scaling succeed. Afterward return to Pong with one justified,
finite staged intervention, retaining every failed control and leaving final Pong
seeds untouched. The repeated two-template evaluations do not measure visual
generalization, and deterministic correctness is restricted to 0, 10 or 20 of 20.
