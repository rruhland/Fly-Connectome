# TT21 teaching a physical continuation stroke

The privileged teacher searched the real simulated arm state for legal replies to physically released balls. The student learned from the teacher’s executed normalized motor rates and the native contact timestamp. A separate factory model replayed those commands and inferred paddle pitch and forward speed at contact. The learned memory contains only the two inferred movement descriptors from each demonstration. This is demonstration learning with privileged teacher credit and a public CAD model. The student’s policy is frozen for held scoring. [Teacher data](coach-replies.json.gz) · [Physical oracle](../../scripts/experimental/tt21_coach_oracle.py)

The continuation controller forecasts the ball from its own native wrist, left and right RGB streams and predicts its own arm state from its command history. After the first outgoing ball, it considers taught pitch and speed at optically predicted contact points, adjusts the pose to the incoming trajectory, and simulates those options in an independent dry model. It commits only when the dry model predicts a legal shot. The stock receiver stays in charge until that launch. The generic control uses the same optics, timing search, model and scoring logic with three evenly spread factory styles. The stock control retains its usual catalogue. [Controller](../../scripts/experimental/tt21_apprentice.py)

The archive assay holds out entire original point seeds and retains at most one trajectory per opening, choosing a physically witnessed reply where available. Each condition reconstructs the same saved opening and released rival ball with the original command prefix. During that fixed prefix, original optical observations also restore the stock planner’s bookkeeping. The alternative planner already observes the fresh images and executed commands. After release, every branch makes its own continuation decisions from fresh RGB and acknowledged commands. This is a continuation assay on disjoint archive trajectories. The separate fresh assay, if triggered, starts the complete point from ordinary resets. The fresh schedule is fixed before archive evaluation. It runs only if the taught branch gains an actual contact or legal continuation over stock on the archive assay. [Assay](../../scripts/experimental/tt21_apprentice_trial.py)

| Assay | Condition | Starts | Legal rival replies | Continuation contacts | Legal continuations | Point wins |
|---|---|---:|---:|---:|---:|---:|
| Disjoint archived replies | Stock | 4 | 4 | 2 | 1 | 0 |
| Disjoint archived replies | Generic styles | 4 | 4 | 3 | 3 | 0 |
| Disjoint archived replies | Taught styles | 4 | 4 | 3 | 2 | 1 |
| Fresh ordinary resets | Stock | 6 | 2 | 1 | 0 | 2 |
| Fresh ordinary resets | Generic styles | 6 | 2 | 1 | 1 | 3 |
| Fresh ordinary resets | Taught styles | 6 | 2 | 1 | 0 | 2 |

On archived seed `1634221833`, both compact controllers made a legal continuation where stock missed contact. The taught shot landed at frame 82 and won on the second bounce. The generic shot landed at frame 78 and drew another legal reply. The generic controller also converted a stock fault on coached seed `746909458`, which became the only point in the pilot with two legal learner returns.

On fresh standard-rival seed `1394699776`, all policies shared the same opening serve and executed the same actions through frame 48. At frame 49 the generic controller launched a rate-limited stroke, landed a legal return at frame 72 and won when the rival hit long. It was the learner’s first legal return after serving. The taught and stock branches both contacted the reply and faulted. Across the full fresh set, only two starts produced legal rival replies.

The generic result demonstrates a usable RGB-and-model continuation option on one fresh point. The three privileged demonstrations yielded one archive win and fewer legal continuations than the equally sized generic set. Two of the three inferred pitches hit the preset lower bound, which limits how much of the demonstrated motion survived compression. The pilot does not establish a reliable benefit from learned stroke styles. These controllers apply a single continuation intervention and return toward the requested posture at launch. The experiment used zero online reward updates. All 30 complete strict action tapes reproduced their entire referee event lists and outcomes.

[Per-point observations, motor tapes and replay receipt](apprentice.json.gz)
