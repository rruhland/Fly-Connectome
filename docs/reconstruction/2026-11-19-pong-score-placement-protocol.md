# Score feedback for shot placement

Start from the acquired visual interception and motor policies and the
checkpoint saved by the timing stage after ordinary actuation returns. Keep
these inputs identical across score conditions. The timing estimator may still
track paddle motion within a rally. The selected delay and score estimates are
fixed during the paired evaluation. Further timing adaptation can be enabled
from the same checkpoint in a separate transfer test.

On an eligible incoming trajectory, choose one paddle center offset from a
fixed seven-action repertoire. The inherited timing controller steers to the
offset target and serves as the exact fallback when the offset is zero or the
visual arrival is unavailable. A candidate must fit the visible paddle range
and a conservative displacement estimate over the available time. The approach
holds its chosen offset, with mechanical limits enforced as the predicted
intercept changes. Selection pools nearby visual intercepts and mirrored halves.
Practice explores uncertain offsets and updates the last eligible choice using
the raw terminal score of that rally. Native contacts, opponent misses and
ground-truth ball state are evaluator fields. No angle table or contact labels
enter the score update.

Practice uses the native proportional tracker capped at .35. The first and
later return feasibility probes at .45 found zero own-hit wins under relaxed
legal-angle control. At .35 the first-return grid found winning angles on all
442 observed first-contact starts, with 2,917 winning angles out of 28,730
tested. At .65 the tracker still exceeds the ball's maximum vertical speed
of .43083. Its steady-state tracking error at that ball speed is approximately
.1077 against a collision reach of .1125. Use .65 as the interception-retention
condition. [Prior physics feasibility](2026-10-15-pong-scoring-feasibility.md)

Use disjoint practice, scoring and retention starts. On every scoring start
compare the same controller with frozen trained scores, frozen initial scores
with the inherited center, frozen initial scores with equal placement
exploration, and a centered forecast and braking control.
Policy exploration uses separately generated random streams keyed to trial
ordinal. Paired arms share the same exploration stream for each trial.
On retention starts compare frozen trained scores and the inherited center. Use
a 12-second cap. Count wins after an own return, natural-serve wins, losses,
timeouts, first returns and all return opportunities. Estimate paired own-return
win differences by resampling whole starts. Keep a fresh independent pairing
after the candidate checkpoint is fixed.
