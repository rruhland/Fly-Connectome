# Complete matches and task transfer

Two ordinary first-to-eleven games alternated service against the same active physical opponent. The direct pupil drove all five arm joints from stereo RGB, its joint measurements and command history from the first frame of every point. The game assigned no service or receiving role to its policy. For the combined pupil we added 911 examples from five archived motor-labeled service demonstrations to its 48 receiving demonstrations. All six complete games reproduced the original referee events, final scores and service rotations when both physical arms' commands were replayed. These scores assess already-trained behavior.

| Pupil | First game | Second game | Legal arm returns, by game | Legal arm services | Longest alternating rally |
| --- | --- | --- | --- | ---: | ---: |
| Receiving training only | 0–11 | 0–11 | 5, 5 | 0/11 | 8 |
| Receiving and serving examples | 1–11 | 0–11 | 12, 4 | 6/12 | 10 |
| Same combined pupil, visual information delayed six frames | 0–11 | 0–11 | 0, 0 | 0/11 | 0 |

The five service examples enabled legal services and one point win. The arm still lost both games. The largest independent return-only rallies are in the [direct rally study](rally-improve.md). [Full games, traces, and referee records](full-games.json.gz) · [Match runner](../../scripts/experimental/tt24_full_game.py)

## Does a table-tennis player's joint coordination help it learn wrist bouncing?

A transfer test extracted a five-channel motor covariance from 2,067 joint-command frames the broad pupil issued in 24 table-tennis points. This shaped exploration on a different game. A shuffled control retained each joint's time series but broke its timing relative to other joints. Cold learning explored isotropically. All three wrist pupils received the same 24 scored practice drops per stream. On 48 fresh drops across four independent streams the genuine coordination prior earned 122 rebounds, shuffled 121, cold 125 and stationary 103. Preserving the table-tennis player's cross-joint timing offered no measurable advantage here. Native Pong's optical causal credits are a separate, more promising comparison. [Weights, exposures and scores](table-bounce-prior.json.gz) · [Native Pong transfer](recurrence.md)

## Raw optical imitation and a common motor memory

Three direct controls learned from wrist pixels and encoder features with motor labels. The random-feature controller's four training stages yielded mean longest runs of 0.58, 0.83, 0.25 and 0.67 versus 2.25 stationary on twelve held drops. A five-frame action predictor scored 2.08 against 2.67 stationary. Fitting selected pairs of visual features scored 1.58 on those same twelve. The underlying training examples and fitted policies are archived alongside the receipts. The separately calibrated perception and action pupil produced longer physical runs. [Random features](latent-stock.json.gz) · [Temporal fit](temporal-five.json.gz) · [Selected interactions](additive-stock.json.gz) · [Practice examples](latent-practice-examples.npz) · [Calibrated pupil](bounce-ident.md)

One direct memory shared across table tennis and four bounce demonstrations still made 17 legal table-tennis returns across 16 held starts, against 11 for table-only. It fell short of stationary on bouncing. Those results leave acquisition of both useful behaviors by a single shared readout unresolved. [Bounce study](shared-pupil.json.gz) · [Table-tennis retention](shared-retention.json.gz)

## Direct transfer of 3D contact experience

Thirteen exact table-tennis replays provided 52 legal near-arm returns and observable source action histories. A privileged diagnostic aligned incoming and outgoing movements in local paddle coordinates. The rich source feature distribution covered none of 88 fresh bounce contacts. A reduced source model predicted the normal exit velocity on those target contacts, yet its clean-state reactive controller and the shuffled version both averaged 1.75 consecutive target rebounds. The unchanged practiced anchor scored 3.00 and passive 2.50. This explains one limitation of bringing 3D game experience across directly. [Measured contacts and controller experiment](spatial-carry.md)

A second trial restricted motor practice on the real arm to 16 scored branches. Perturbing five-joint activity after its first reward reduced returns on all eight paired starts. The fitted hindsight controller earned 34 rebounds on sixteen fresh drops versus 42 with the original active policy. Only four intervened pairs yielded complete optical observations. [Hindsight tests](hindsight.md)
