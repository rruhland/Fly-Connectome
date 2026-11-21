# Learning when to prepare the next SO-101 stroke

The learner watched two unmarked fixed RGB cameras on the clamped table. It used the ordinary table edges to register the cameras, tracked the ball and blade, and kept the factory opening and catalogue of stroke commands. After seeing its outgoing ball on the far side, it estimated the rival reply's time to reach the paddle area from the recent optical track. The alternative timing rules change the window for starting a catalogue stroke. The earlier rule also checks visible blade settling against the blade's position at the start of the point. [Controller](../../scripts/experimental/tt21_recovery.py)

Practice began with the first 30 command frames of previous TT20 games. The live arm physically executed each prefix, then played the rest of the point from its own RGB against the stronger physical rival. The learner tried three timing rules over 15 chronological points. Each update used the ordinary point reward and a bonus for an optically observed forward arc that forecast a legal landing. The reward and optical data drove the learner. The physical scorer and event tapes measured outcomes. Practice cost 1,166 target control frames including the opening replays. [Practice observations and verified action tapes](recovery-practice.json.gz)

Practice produced two legal second returns and no point wins. One second return followed an early launch, and one followed the stock timing. The pooled global timing learner kept the factory rule. A contextual fit to the same practice stream, specified before held play, chose the earlier launch when the optical approach to the nominal strike region was at most ten frames away. The sparse fast group contained one factory trial and three early trials. The slow group contained five factory, two early, and four late trials. The slower group kept the factory timing. [Scored trial](../../scripts/experimental/tt21_recovery_trial.py)

Both held policies use this same practice budget. The frozen policy retains factory timing while the learned policy carries only the fitted timing selection. Held inputs begin from newly seeded points without replayed prefixes. Each point registers its two views from fresh RGB. The strong opponent plays through the same simulated physical arm and strict referee in every branch.

On held seed `638131348`, both policies issued identical commands through optical reacquisition. The factory policy landed the first return and never launched a second stroke. The learned policy launched at frame 48, hit after the near-side bounce at frame 55, and landed the second return at frame 68. The rival legally answered both shots. The point remained a loss and lasted 107 frames, up from 70. [Comparison video](recovery-paired-638131348.mp4) · [Held point and game tapes](recovery-held.json.gz)

The initial 12-point bank used six rival serves and six own serves. Both policies won two points, served legally on three of six, and landed a first return on five of six received serves. The learned policy landed six legal returns overall, up from five. The second return on `638131348` accounts for the difference. [Held point and game tapes](recovery-held.json.gz)

Both complete games ended 2–11, with four legal serves, five first returns and no subsequent legal return on seven points that the rival legally returned. In practice, the eight replies for which the learner never found a stroke were projected to cross the strike depth 33.2 cm from the centerline at the median. The median was 4.9 cm on the seven replies the learner attempted. The timing learner and factory reference used the same stroke catalogue, which struggled with those wide replies. [Practice tapes](recovery-practice.json.gz) · [Full games](recovery-held.json.gz)

## Independent receiving replication

The replication uses 32 distinct rival-service starts drawn in one fixed order. The practice checkpoint and recovery criteria stay fixed. Each branch has its own score on the same starts. A factory controller can consume the same native observations whenever both branches have taken exactly the same actions. After the first distinct command, both branches play physically and render their own images. Identical controls are also replayed in the physical arena for each branch. [Independent matched tapes](recovery-replication.json.gz)

The earlier preparation also produced a second legal return on replication seed `2065352219`. Both branches landed the first return. The learned policy began the next stroke at frame 49, hit at frame 56 and landed a legal return at frame 69. The rival answered both shots and won. The exchange lasted 110 frames compared with 72 for the factory policy. [Independent matched tapes](recovery-replication.json.gz)

The factory policy gained a second return and won seed `2146830181`. On that start the learned policy committed while the ball was still 12 cm onto the far half, three frames before the factory stroke. Its return landed on its own half. The 32-start replication ended with 16 first-return starts, 15 first legal rival replies, one second legal return and 17 total legal returns per policy. The learned policy won one point and factory won two. All 64 condition tapes replayed exactly. Across the initial six rival-service starts and the 32 independent starts, second returns were 2/19 learned and 1/19 factory per first legal rival reply. First-return starts were 21/38 in both conditions. [Independent matched tapes](recovery-replication.json.gz)

The optical detector classified two of its 14 fast approaches before the rival actually hit the ball. The recorded commands agreed between policies on both of these points. [Independent matched tapes](recovery-replication.json.gz)

## Narrower timing window

The two successful early strokes started at observed positions of −16.7 cm and −18.1 cm along the table length. The revised timing rule allows an extra attempt only on fast approaches between −24 cm and −15 cm. It uses the existing factory timing at other positions. The rule was fixed before scoring 24 new randomly generated rival services, with the same factory comparator, fitted timing checkpoint, opponent and native stereo views. [New matched tapes](recovery-near-side.json.gz)

Six of 24 starts reached the additional window with a fast classification. Neither policy produced a different motor command on any start. Both won two points and finished with 15 first-return starts, 13 first legal rival replies, two second returns and 17 legal returns altogether. All 48 condition tapes replayed exactly. [New matched tapes](recovery-near-side.json.gz)

## Reproduce

```sh
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBGL_ALWAYS_SOFTWARE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt21_recovery_trial --phase held
python -m scripts.experimental.tt21_recovery_trial --phase replicate
python -m scripts.experimental.tt21_recovery_trial --phase near-side
python -m scripts.experimental.tt21_recovery_trial --phase verify
```
