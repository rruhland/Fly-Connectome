# Keep the current motor action unless the model finds a safer replacement

Keeping an incumbent action cut search CPU by 70% per simulated second compared with the standard elastic search. It also returned six of eight openings instead of five. The tradeoff was shorter rallies and fewer later returns. The standard controller remains stronger if the goal is sustained play. [Paired results](joint-mpc-guarded-results.json)

The [elastic controller](joint-mpc.md) can begin a movement that the next recheck rejects. Reverting to the raw law immediately does not undo the motion the body has already made. The conservative branch first rechecks an active motor prefix. If that prefix still forecasts a playable return and at least one optical alternative agrees, it keeps the motion for another short lease. If the prefix is not viable, the model tries the actual raw feedback law. Two virtual sweeps apply that law to simulated ball and encoder states, rather than repeating one raw command and treating it as the whole controller. A legal raw forecast supported by at least one optical alternative leaves the actual raw controller in charge. [New branch](../../scripts/experimental/tt38_joint_mpc.py)

The expensive search runs only when neither incumbent is viable. It uses up to four diverse local fragments and a smaller 16-proposal sampling budget. A new motion must land in the nominal model and both optical alternatives. The branch will not issue a preparation motion without a viable candidate. An accepted motion expires within eight ticks unless rechecked. These changes test a combined switch and compute policy, so they do not isolate which individual threshold mattered.

Eight different measured launches were drawn from the other half of the novel TT37 roster before running this branch. None appeared in the primary controller comparison or the instruction archive. The raw, elastic and guarded runs use the same optics, card order, seeds and referee. Both searches share the same nominal body and start with empty reward memory. [Pinned independent roster](joint-mpc-guarded-protocol.json)

| Eight independent openings | Legal first | Legal later | Legal near total | Best near / alternating | Search CPU seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw feedback | 3 | 8 | 11 | 7 / 15 | 0 |
| Standard elastic | 5 | 31 | 36 | 15 / 31 | 69.58 |
| Guarded elastic | 6 | 14 | 20 | 5 / 11 | 13.31 |

All eight first trajectories matched across controllers. Guarded beat elastic on three points and lost on two. Those two losses were large. One standard rally reached 25 alternating strokes and the other 31. Guarded preserved raw feedback on 42 decision ticks, retained a previously issued motion on 19, and declined three possible replacements for weak optical agreement. Total CPU per simulated second fell from 1.97 to 1.31. Neither controller was shown to run in real time. [Point details and compute](joint-mpc-guarded-results.json)

An action-only independent replay of the guarded 11-stroke rally reproduced five legal negative-face near returns. Full arm CAD found no penetrations in 4,613 steps, with 16.15 mm minimum clearance. An additional matched diagnosis of three known failures found that more frequent fresh-frame checks did not recover those openings. [Guarded CAD replay](joint-mpc-guarded-audit.json) · [Cadence diagnosis](joint-cadence.md)

## Reproduce

Use the same physical simulation environment and measured launches as the [main comparison](joint-mpc.md).

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
optic=artifacts/tt37/rebound.npz
protocol=artifacts/tt38/joint-mpc-guarded-protocol.json
python -m scripts.experimental.tt38_joint_mpc guarded-protocol guarded_challenge "$protocol" --optic "$optic"
python -m scripts.experimental.tt38_joint_mpc raw guarded_challenge artifacts/tt38/joint-mpc-guarded-challenge-raw.json.gz --optic "$optic" --protocol "$protocol"
python -m scripts.experimental.tt38_joint_mpc live guarded_challenge artifacts/tt38/joint-mpc-guarded-challenge-elastic.json.gz --optic "$optic" --protocol "$protocol" --variant elastic
python -m scripts.experimental.tt38_joint_mpc live guarded_challenge artifacts/tt38/joint-mpc-guarded-challenge-guarded.json.gz --optic "$optic" --protocol "$protocol" --variant guarded --budget 16
python -m scripts.experimental.tt38_joint_mpc_report guarded --audit-best
```
