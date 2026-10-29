# Direction review: learn visual consequences, then choose a command

Parent decision, 2026-10-29. The user has waived further architecture approvals
until reliable learned returns are achieved or work is stopped. No backpropagation,
privileged controller inputs, expert action targets or simulator planning is allowed.

## Why stop the previous route

Adaptive semi-Markov value learning improved from C0 to C1 on 32 matched TRAIN
games (44 to 50 returns), but remained below the preserved policy's 82. A second
experience/refit cycle regressed on its eight-game gate (16 to 13, first returns
unchanged). More data alone is not earning another cycle. No candidate was promoted.

## Competing explanations

- Sixteen-camera open-loop commands can prevent timely corrections. A controller
  that observes again after each primitive command tests a materially different
  control loop.
- Value estimates can generalize poorly and a greedy maximum can exploit those
  errors. Leaf-level action presence did not prevent poor native performance.
- The scene representation omits command history. A read-only inspection of
  `src/fly_connectome/pong.py` confirms actuator lag and inertia. Visual position
  and velocity therefore need not specify pending actuator response. This is a
  hypothesis about prior failures, not an established causal diagnosis. The new
  design included previous commands before this source inspection; no hidden
  activation, motor coefficients or equations enter its inputs or targets.
- Dense visual transitions may teach predictable action consequences more cheaply
  than sparse episode rewards teach a large policy. Prediction quality alone does
  not establish that those predictions are useful for control.

The parent proposed learned dynamics/planning as the next distinct route. Sol
independently recommended it over another critic/support amendment and specified
the primitive receding-horizon controller. The pilot changes prediction and action
timing together; it seeks useful learning, not isolation of one causal variable.

## Selected pilot

Reuse the existing legal scene stream, actual command histories and scalar events.
The inventory contains 452 unique traces, 114,229 transitions and 206 TRAIN starts;
114,097 transitions have successors. Event labels include 481 own hits and 328
natural ends. Missing successors do not invalidate observed event labels. Changing
presence/visibility/velocity-validity masks must remain explicit in the model.

Sol's fixed design uses a ridge model of visual changes plus a small residual tree
ensemble, and separate empirical event predictions. It includes only past actual
commands and public timing. No class-balanced sampling is misrepresented as an
event probability. It predicts discounted hit mass and surviving continuation
mass, so an anticipated hit is counted once and a natural end has no value tail.

An eight-step, width-eight beam searches primitive commands in the learned model.
At every depth it uses frozen C1 option3 as an approximate value tail, preserving
at least one candidate for each initial command. It executes only the first
command, then observes and replans. This tail was trained with future maxima; it
is not a proven value of following the preserved policy. Learned dynamics and
the tail can both be exploited. Baseline control remains active before the first
actual hit, and on unsupported/invalid visual inputs.

One fit/export/timing budget is 180 seconds. Pin the implementation and parameters
before fitting. Native evaluation starts with the same eight games, extends to
32 only on actual total/later-return gain with first-return parity, and uses fresh
games only after baseline qualification. Set the evaluation budget from a saved-
input timing probe without changing the horizon or width. One full chronological
replay plus fixed spot checks and frozen-model checks avoids paying twice for all
planning decisions. Record the actual audit coverage.

No prediction-error milestone, automatic promotion, or preauthorized sequence of
parameter variations follows this pilot. Interpret actual returns first.

## Outside evidence and independent fallback

[Black-DROPS](https://arxiv.org/html/1703.07261v2) motivates learning uncertain
dynamics and rewards before black-box control optimization. Its demonstrations
are robotics tasks, not this visual Pong problem. Our pilot is not a reproduction.
[Non-delusional Q-learning](https://research.google/pubs/non-delusional-q-learning-and-value-iteration/)
describes another way approximate greedy backups can become inconsistent; it
does not diagnose our implementation.

A separate Sol XHigh researcher investigated compact program evolution in
[the symbolic-policy note](2026-10-29-symbolic-policy-challenge.md). Structural
program search is distinct from weight-noise search, but published small deployed
programs can require thousands of native fitness evaluations. Keep it as a
low-confidence fallback; no new search allocation follows from its small size.

The milestone remains reliable fresh full-game returns, independent learning
replication and efficient inference/training. Neither C1 nor the planner currently
satisfies it.
