# Learning a first return that leaves a reachable reply

A first-shot depth chooser kept the inherited stroke on every held and full-game context. A separate lateral sweep produced one additional two-return point on the first matched receiving cohort and lost two in the independent repeat. Across two complete games the sweep scored 1–11 and 8–11, compared with 1–11 and 7–11 for the inherited player. Its only extra multiple-return point came after own service, where it uses the inherited stroke and can differ through carried state from earlier receiving points. [Paired evidence](rotary-first-comparison.json.gz)

## Outcome learning on matched serves

All players started from the [TT15 blind-reset checkpoint after game three](../tt15/rotary-blind-game-3-player.json.gz). The depth policies changed only the first shot after an optically recognized rival serve. The choices were the inherited stroke, a modeled far-half depth of 0.27 m and a modeled depth of 1.16 m. The latter two changed requested pitch within 0.14 rad of the inherited stroke. Own serving, subsequent strokes and between-stroke recentering used the inherited controller. The measured contact position and incoming velocity supplied the context. [Player](../../scripts/experimental/tt16_rotary_player.py)

The player scored each practice shot using a public reward and evidence from its stereo ball track. It looked for a visible far-half landing, a coherent incoming trajectory and a visible second landing. On a detected reply it weighted the lateral and vertical motion required by the receiving plan and the frames available. The learner received no referee labels. A post-play check on 64 archived points associated the first-return cue with all 19 actual first returns and the reply cue with 11 of 12 actual replies. It detected one of three actual second returns, so the learning signal relied mostly on first-return and reply reach evidence. [Optical measurement audit](rotary-first-comparison.json.gz)

Each depth played the same 24 practice starts, balanced across ordinary, soft-racket, delayed and mixed conditions. Players reset between starts. The learned selector weighted optically similar contacts and shrank their scores toward the average for each target. The context-free choice was the inherited stroke. On the same development data, pairing each shot's optical payoff with the inherited payoff on that incoming ball also yielded no positive improvement under leave-one-out evaluation across 12 neighbourhood settings. [Frozen fit](rotary-first-fit.json.gz)

| Practice stroke | First returns on 24 starts | Second returns and opportunities | Points won | Mean outcome score, with prior |
| --- | ---: | ---: | ---: | ---: |
| Inherited | 9 | 3/6 | 5 | 0.209 |
| Shallow target | 9 | 2/5 | 5 | 0.199 |
| Deep target | 8 | 0/6 | 3 | 0.160 |
| Lateral sweep | 9 | 3/6 | 6 | — |

The shallow depth sustained one point on which the inherited stroke made one return, and lost two other sustained points. The frozen context selector picked the inherited stroke on all 46 later receiving contexts across the held cohort and full games, reproducing all 62 inherited point action tapes including own service. [Matched practice tapes](rotary-first-shallow-practice.json.gz) · [Frozen held tapes](rotary-first-context-held.json.gz)

## Lateral sweep

The sweep kept the inherited angle and contact plane. When the predicted contact lay more than 0.10 m laterally from the ready pose, it requested a tangential blade velocity toward a flight that would move the ball nearer that pose on the rival side. This objective was bounded at 0.055 m per frame and applied around the first contact. The alternative used the same incoming cues and never acted on own service. Its optical development score improved by 0.0125 over the inherited stroke on the 16 eligible matched starts, below the +0.025 adoption threshold. The fixed sweep was tested as a physical comparison. [Sweep practice](rotary-first-center-practice.json.gz)

| Policy | First returns on 32 held serves | Second returns and opportunities | Held points won | Complete games | Second returns on rival service in games | Second returns after own service in games |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Inherited | 9 | 1/5 | 5 | 1–11, 7–11 | 0/1 | 0/1 |
| Frozen depth choice | 9 | 1/5 | 5 | 1–11, 7–11 | 0/1 | 0/1 |
| Fixed sweep | 11 | 2/6 | 7 | 1–11, 8–11 | 0/0 | 1/1 |

The sweep acted on 19 of the 32 held balls. On that subset the inherited stroke made two first returns and no second returns. Sweeping made four first returns and one second return. The added second came on seed 1372491515 and won the point. The first landing moved from (−0.113, 0.676) m to (−0.109, 0.648) m in the table plane. The rival's legal reply bounced at the same frame with a different measured trajectory. The other sustained point contained four returns under both branches and used the inherited stroke. [Matched held sweep](rotary-first-center-held.json.gz)

In the first full game, own-service seed 953902430 went from one to three returns with the sweep branch and lost under both players. The initial own service commands and first legal opponent reply were matched. The sweep was inactive on that point, which followed earlier changed rival-served points while ordinary motor and reward updates carried through the game. [Inherited game](rotary-first-base-one.json.gz) · [Sweep game](rotary-first-center-one.json.gz)

## Independent receiving repeat

On 28 fresh matched opponent serves, the sweep triggered on 20 starts. It landed seven first returns, two second returns on seven opportunities and won one point. The inherited stroke landed nine first returns, four second returns on six opportunities and won four points. The two points sustained by the sweep were among the four also sustained by the inherited stroke. Across both held cohorts, the sweep tied first returns at 18/60 and landed second returns on four distinct points versus five for the inherited stroke. It won eight points versus nine. [Inherited repeat](rotary-first-base-repeat.json.gz) · [Sweep repeat](rotary-first-center-repeat.json.gz)

## Verification and checkpoints

The optical fit used 72 scored depth practice plays, and the sweep comparison added 24 scored practice plays. Together these contributed 6,043 stereo frame pairs. Practice, held tests and qualification games accounted for 339 verified physical point replays. All 192 shared rival openings agreed through the legal service bounce. All 32 shared own-service command tapes agreed through the strike.

The frozen depth checkpoint and the sweep checkpoint reproduced all 18 and 19 RGB point tapes from their respective second games. Each then matched uninterrupted play on four new continuation points. [Depth checkpoint](rotary-first-context-two-player.json.gz) · [Depth continuation](rotary-first-context-two-audit.json.gz) · [Sweep checkpoint](rotary-first-center-two-player.json.gz) · [Sweep continuation](rotary-first-center-two-audit.json.gz)

## Reproduction

Use the [hall environment](../tt6/README.md) from the repository root.

```sh
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TT3_COMPONENT_LIBRARY=/tmp/tt3-components-final.so
python=/workspace/Fly-Connectome/.venv/bin/python
output=/tmp/tt16-rotary
for mode in base shallow deep; do
  "$python" -m scripts.experimental.tt16_rotary_trial "$output" play practice "$mode"
done
"$python" -m scripts.experimental.tt16_rotary_trial "$output" fit
"$python" -m scripts.experimental.tt16_rotary_trial "$output" play practice center
for mode in base context center; do
  for fixture in held one two; do
    "$python" -m scripts.experimental.tt16_rotary_trial "$output" play "$fixture" "$mode"
  done
done
for mode in base center; do
  "$python" -m scripts.experimental.tt16_rotary_trial "$output" play repeat "$mode"
done
"$python" -m scripts.experimental.tt16_rotary_report "$output"
"$python" -m scripts.experimental.tt16_rotary_trial "$output" audit context two
"$python" -m scripts.experimental.tt16_rotary_trial "$output" audit center two
```
