# A dedicated low-ball wrist pulse

Among the 126 original full-random practice offers, the evaluator-only projected receive height below .99 m was associated with 14 near contacts but just two legal returns in 15 offers. The 1.04–1.09 m band returned all 50. Those bins use realized offline crossing telemetry, not the actor's camera. This experiment gates only on its own RGB/encoder-based observation at the incumbent's −1.56 m receive depth.

The prototype first searched two specialized lift families on four physically replayed practice prefixes. The first family tried a shallower rising brush. The second looked for a reachable undercut and controlled forward and upward blade speed. The cold geometry bank found no nominal legal replacement in 724 virtual trial evaluations including the four incumbents. Six individual physically legal low-return command segments per trial point, transported and phase-warped, added one nominal legal proposal out of 768 extra templates. Its optical uncertainty score did not beat the incumbent and it was never physically issued. These results are negative for the tested private planner, not proof that an SO-101 has no possible low stroke. [Cold geometry](low-lift-practice.json.gz) · [Transported legal motions](low-lift-earned-practice.json.gz)

## Learn by actually playing the low receives

The second experiment begins later, around the last camera decision actually reached by the stock motion. Before the perturbation, the measured serve, live opponent seed, action tape, native referee event history, and optical/encoder state all match the original practice point. The policy adds a short smooth wrist-flex, shoulder-lift, elbow and optional roll pulse around the unchanged incumbent motor shape. It physically plays 18 versions (including zero control) of each of the four low practice opportunities, with the same original full-random opponent protocol. The private model neither chooses nor labels the winner. Score comes only from an actual legal landing.

| Practice release | RGB-estimated receive height | Stock | Same twice-rewarded pulse |
| --- | ---: | --- | --- |
| 14951 | .952 m | Own-half bounce | Floor fault |
| 11846 | .967 m | Out | Out, no touch |
| 4239 | .965 m | Own-half bounce | Legal |
| 13718 | .988 m | Floor fault | Legal |

This grid issued 72 physical receiving attempts with 55 contacts with a recorded racket separation and four legal returns. Three pulse choices recovered 4239 and only one recovered 13718. The pulse shared by both uses −.25 rad wrist flex, +.10 rad shoulder lift and +.04 rad elbow flex with no roll variable. On 4239 it changed outgoing vertical speed from 2.12 to 3.13 m/s while keeping 3.38 m/s of forward speed. It landed at x = .024 and y = .358 m. On 13718, outgoing vertical speed went from −2.68 to +2.60 m/s, with 3.59 m/s forward and a legal landing at x = .329 and y = .247 m. The earlier two cases remain unsolved. Both repaired shots were then exactly reissued, and the original opponent sent a real, legal next return on each. Native CAD found zero penetration samples at minimum separations of 16.05 and 17.17 mm. [Physical practice](low-pulse-practice.json.gz) · [Longer independent replays](low-pulse-witness.json.gz)

## Deployment rule

Only a motion rewarded on at least two distinct practice opportunities is eligible. In assessment the rule is frozen. The stock policy chooses the underlying action, and keeps it unchanged on higher trajectories, when the candidate is already well centered, or when the current camera flight and ETA are outside observed successful practice support. The added pulse neither reads live simulator state nor performs a private physics query to choose an action. Six preassigned release-disjoint full-random assessments retain the original opponent targets and governor. These are exploratory held games for transfer, not additional training data.

## Held transfer result

The frozen specialist added **no legal returns** on six release-disjoint, preassigned full-random pairs. Each arm made 26 near-side legal returns and 57 alternating legal strokes in total. Both arms' longest point was 11 near-side returns and 23 alternating strokes.

| Held release | Stock own legal | Specialist own legal | First different issued tick |
| --- | ---: | ---: | ---: |
| 9550 | 1 | 1 | 171 |
| 15665 | 3 | 3 | None |
| 11884 | 11 | 11 | 1126 |
| 3401 | 1 | 1 | None |
| 14760 | 0 | 0 | None |
| 7901 | 10 | 10 | None |

The other four games were exactly unchanged in actions and all referee events. The two affected games had common physical offers and the same pre-intervention actions and events. The pulse was selected three times, including one earlier proposal in 11884 that did not yet change the issued command. Both changed receiving attempts still failed to return. In 9550 it increased upward speed from 3.16 to 3.60 m/s but reduced forward speed from 2.56 to .96 m/s. In 11884 it caused a real contact on a ball that stock missed, but the departure was still toward the near player at −3.37 m/s. Previously earned physical returns around the visible low band were enough to identify a plausible motor primitive, not enough to decide when a new low ball needs more lift versus more forward drive. A broader learner should condition its trial selection on the already planned stroke and reachable arm state as well as incoming optical height and velocity. [Paired physical tapes](low-pulse-held.json.gz) · [Matched outcome audit](low-pulse-held-summary.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt40_low_lift practice artifacts/tt40/low-lift-practice.json.gz
python -m scripts.experimental.tt40_low_lift practice artifacts/tt40/low-lift-earned-practice.json.gz --earned
python -m scripts.experimental.tt40_low_pulse practice artifacts/tt40/low-pulse-practice.json.gz
python -m scripts.experimental.tt40_low_witness artifacts/tt40/low-pulse-witness.json.gz --cad
python -m scripts.experimental.tt40_low_pulse held artifacts/tt40/low-pulse-held.json.gz --limit 6
```
