# Delayed credit for physically earned motor fragments

The two completed matched games scored **8 returns with delayed credit
versus 69 with ordinary recall**. On 8831 the score fell from 27 to 7,
and on 2741 it fell from 42 to 1. The first affected incoming offer was
still returned in both versions of both games. Their subsequent native
deliveries changed. These are fully scored real games from launch under
the same wide random opponent, not virtual search scores. The planned
roster was the first eight unfiltered official season practice cards.
The run was stopped after the two losses and the archive check below.
The result is an early rejection of this rule, not an eight-game estimate.
[Paired report](report.json) · [Issued joints and physical tape](paired.json.gz)

## What changed

The bounded 768-row memory, snapshots kept per legal return, nominal
joint proposals, modeled outgoing ranking and physical search budget
all stay unchanged. Each existing earned group tagged by game serial
and scored stroke gets a second label. Another scored return gives
favorable first-step credit, followed by more credit when a second
return actually scores. A natural loss after a real far-side legal
delivery lowers the predecessor's delayed score and reduces the
penultimate credit. The archive can be annotated from the public
referee's score, and live credit is not attached until the subsequent
reward or loss callback. When the opponent never returns the scored shot,
or a point is unfinished, the pending label is censored, not punished. Unresolved first successes remain
neutral and are never dropped.

During recall, a nearby scored fragment can move slightly in the stock
retrieval order only if the similar-state and similarly executed joint
pattern has support in at least two independent other games. Credit
contrasts that pattern with local state-only outcomes. Many nearly
duplicate frames from one hit do not create independent samples. The
learner still sees the usual stereo and encoders only, issues direct
bounded joint commands and does not use modeled outgoing trajectories
or the upcoming opponent target to assign delayed credit. The equal-data
sham shuffles resolved outcomes within each game and runs only as a
shadow diagnostic, never as an actuated arm.

This differs from TT47 retention, which removes redundant memory at
capacity, and TT49 sustained-value and tactical recall, which use an
outgoing-effect prediction or a flight envelope. It also differs from
TT46 failure credit, which directly grades a failed current action. The
unit tests exercise credit timing, residual independent-game support,
and the fixed within-game outcome distribution of the sham.

## Did the motor-history labels help?

The initial bank had 176 distinct resolved scored strokes from 22
completed source games. Another two returns followed 133 of them,
one return then a failed receive followed 21, and an immediate failed
receive followed 22. On a held-game observational check, a mean utility
from the other games had squared error .459. Nearby state alone gave
.502, nearby state plus joint motion .514, and the identical state-plus-
motor model with the same-data sham .504. The action grouping did not
show additional predictive information here.

Credit changed 25 of 41 retrieved shortlists across the two actual
games. The same-data sham would have changed 26 on exactly those
queries. Neither comparison is a causal estimate of whether the sham
would have improved or harmed future returns. All referee events
preceding the first differing issued control matched stock exactly,
at ticks 256 and 82, and all new score callbacks were verified against
the independent completed-event annotation. The eight returns produced
eight online reward updates and two observed terminal loss updates.
[Fixed roster and annotated parent checkpoint](protocol.json) · [Resumable joint checkpoint](paired.npz)

## Reproduce the completed prefix

The committed receipt covers the first two cards of the eight-card
protocol. These commands replay only that prefix into new paths.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt52_delayed_motor_credit plan --n 2 --plan /tmp/tt52-two.protocol.json
python -m scripts.experimental.tt52_delayed_motor_credit run --plan /tmp/tt52-two.protocol.json --out /tmp/tt52-two.json.gz
python -m scripts.experimental.tt52_delayed_motor_report
python -m pytest -q tests/test_tt52_delayed_motor_credit.py
```
