# Choosing receiving timing on a gantry and a six-joint arm

## Fast blade motion

Both new installations started from the TT8 integrated player and commissioned a motor map from eight physical RGB probe episodes. The alternative changed only actuator settling curves and their lag summaries. It was fitted on 16 scored cursor targets on the same physical arm. The selector then compared these arrays on eight disjoint dry swings with the same commands, using the two-frame stereo displacement of the blade. Each candidate had the same 48 usable pairs per installation. The selection required at least 0.8 m/s median forward speed. [Integrated player](../tt8/span-after-four.json.gz)

| Mechanism | Ordinary timing error | Cursor-fit timing error | Median peak forward speed | Choice |
| --- | ---: | ---: | ---: | --- |
| Four-axis gantry | 12.68 mm | 13.34 mm | 1.59 m/s | Ordinary |
| Six-joint rotary | 44.36 mm | 40.83 mm | 1.85 m/s | Cursor fit |

Each installation used 272 commissioning commands, 384 cursor commands and 168 dry-swing commands. Each fixed comparison and the selected branch shared these observations. The choice held at each of the two swing paces and after omitting any one swing. [Gantry probe tape](body-gantry-probes.json.gz) · [Rotary probe tape](body-rotary-probes.json.gz)

## Separating the serve from the return

Each serving profile played the same eight development tosses at its installation. Gantry forward drive made three legal serves versus two for the classic stroke. The rotary classic stroke at a target height of 0.39 m made six legal serves and drew five legal replies. Its closest alternatives made five legal serves each. The development comparison used referee legality to choose one shared service profile per installation. [Gantry service practice](body-gantry-practice.json.gz) · [Rotary service practice](body-rotary-practice.json.gz)

The player received stereo RGB, its preceding command and scalar point reward. The service controller used the commissioned timing and the same initial latency belief on each toss. The receiving servo also observed the commands and blade from the opening frame. A rival-side ball sighting gave it control on opponent service. An optical own toss kept the service controller active until two observed ball sightings across the net. The timing comparison covers receiving and following up on a serve. All three branches used the same timing to strike the toss.

Two disjoint fixture schedules per body alternated service to 11 points with a two-point lead. Independent seed streams supplied the nth own toss and nth rival opening to each branch regardless of the score. Games restarted from the same commissioned checkpoint. Racket response varied across ordinary, soft, delayed and mixed conditions. Delay and mixed points applied hidden command noise and changed transport lag during play. The ordinary and cursor-fit branches kept their respective settling curves fixed. The latency observer and ordinary reward updates kept running.

## Full games

The selector won one of four fixtures, as did each fixed policy. It produced 19 points, 29 legal returns and eight sustained rallies. Fixed ordinary timing produced 17, 25 and eight. Fixed cursor timing produced 25, 34 and eleven. The optical choice improved first returns on the rotary arm. On the gantry it selected ordinary timing, where the cursor fit produced six more points over the two games.

| Mechanism | Timing | Fixture scores | Legal returns | Sustained points |
| --- | --- | --- | ---: | ---: |
| Gantry | **Ordinary, selected** | 11–6, 3–11 | 22 | 7 |
| Gantry | Cursor fit | 11–8, 9–11 | 27 | 10 |
| Rotary | Ordinary | 1–11, 2–11 | 3 | 1 |
| Rotary | **Cursor fit, selected** | 1–11, 4–11 | 7 | 1 |

| Mechanism | Timing | First returns on rival serve | Later returns on rival serve | Points won on rival serve | Legal own serves | First returns on own serve | Later returns on own serve | Points won on own serve |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gantry | **Ordinary, selected** | 11/15 | 8/13 | 10/15 | 5/16 | 2/4 | 1/1 | 4/16 |
| Gantry | Cursor fit | 15/20 | 7/12 | 13/20 | 7/19 | 5/6 | — | 7/19 |
| Rotary | Ordinary | 2/12 | 1/3 | 0/12 | 8/13 | 0/5 | — | 3/13 |
| Rotary | **Cursor fit, selected** | 6/13 | 0/5 | 1/13 | 9/14 | 1/5 | 0/1 | 4/14 |

First-return denominators count legal rival serves or legal rival replies after an own serve. Later denominators count the remaining legal rival replies. A sustained point has two legal returns on rival service or one following own service. The gantry won three of the four points with a legal reply to its serve with selected ordinary timing, and six of six with fixed cursor timing. The rotary won zero of five with either model. Both gantry branches had one legal own-serve reply on which the receiving servo activated after the first reply bounced. The other own-serve replies switched by the bounce. The independent comparison receipts include all starts and timing-route counts.

The four independently played selected games matched the corresponding fixed branch point for point. All 29 matched own tosses had identical requested and applied commands through the service strike or miss, and identical service events. The audit matched all RGB dry-swing tapes and physically replayed 64 service-practice points and 180 fixture points, reproducing applied commands, events and final scores. [Gantry fixtures and audit](body-gantry-comparison.json.gz) · [Rotary fixtures and audit](body-rotary-comparison.json.gz)

## Carrying the player back to the gantry

The three-game sequence carried reward, optical flight, racket and delay state through a gantry game, a rotary game, and a return to the original gantry wiring and appearance. The receiving curves and service profiles came from the archived calibration for each installation. On the rotary entry and the gantry revisit, two additional RGB probes per entry tested the assigned local map. Each map predicted motion better than simply holding at the ready position, and all four probes fell outside the existing strict recall criterion. The carry comparison measures continuation with known installations.

| Visit | Score | Legal returns | Sustained points | Legal own serves |
| --- | ---: | ---: | ---: | ---: |
| Gantry | 9–11 | 15 | 5 | 5/10 |
| Rotary | 3–11 | 1 | 0 | 5/8 |
| Return to the original gantry | 9–11 | 11 | 4 | 7/10 |
| Fresh player on the revisit schedule | 11–7 | 13 | 4 | 6/8 |

The fresh player won the matched revisit and landed eight of ten first returns on rival service. The carried player landed seven of ten. Both used the same timing and service profiles, and the complete 72-point course and control replayed through the physics. [Carry comparison](body-carry-comparison.json.gz) · [Carried gantry game](body-gantry-carry-1.json.gz) · [Carried rotary game](body-rotary-carry-2.json.gz) · [Gantry revisit](body-gantry-carry-3.json.gz) · [Post-revisit checkpoint](body-gantry-carry-3-player.json.gz) · [Fresh revisit](body-gantry-revisit-fresh.json.gz)

The post-revisit checkpoint reproduced all 180 reference native Pong action tapes. It earned 41 points and made 129 of 160 first returns and 195 of 206 later returns. [Native Pong check](body-carry-pong.json.gz)

## Reproduction

Use the [hall environment](../tt6/README.md) from the repository root. The carry command plays gantry, then rotary, then the original gantry, followed by a fresh player on the same final schedule.

```sh
output=/tmp/tt12-cross-body
for body in gantry rotary; do
  python -m scripts.experimental.tt12_body_trial prepare "$body" "$output"
  python -m scripts.experimental.tt12_body_trial practice "$body" "$output"
  for fixture in 1 2; do
    for branch in inherited cursor selected; do
      python -m scripts.experimental.tt12_body_trial play "$body" "$output" --fixture "$fixture" --branch "$branch"
    done
  done
  python -m scripts.experimental.tt12_body_trial audit "$body" "$output"
done
python -m scripts.experimental.tt12_body_trial carry gantry "$output"
python -m scripts.experimental.tt8_pong_retention "$output/body-gantry-carry-3-player.json.gz" "$output/body-carry-pong.json.gz"
```
