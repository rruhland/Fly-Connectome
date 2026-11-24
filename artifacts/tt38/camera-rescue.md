# A camera-only fallback after no legal candidate

This test adds a motion search to the value-economy optical teacher. It only runs after the existing reuse and baseline search declare no legal candidate. It changes neither the court nor the opponent. The search receives the same camera and encoder estimate already used by `OpticalClinic`, not the live simulator, ball state, placement card, or game seed. It checks that the private court has never launched a point. A separately sampled, prospectively fixed random stream drives all proposal exploration. [Rescue implementation](../../scripts/experimental/tt38_camera_rescue.py)

The previous search compared a weak collision and a pure miss in one ranking. A backward departure could score worse than a small gap, throwing away evidence that an actual pose was capable of touching the ball. The rescue retains separate pools for no contact, weak or backward contact, and forward contact. It examines multiple contact depths from the visual flight estimate and varies normal, phase lead, joint velocity, and wrist curvature. A local signed outgoing-velocity fit generates small action changes around existing contacts. The simulated contact and velocity labels come from the private optical twin. They do not open the physical referee ledger. Once a legal option exists, the historical value readout and measured post-bounce geometry rank it, with two small sensitivity reruns around its visual estimate. The real arm is still driven by the existing bounded servo command. Recovery actions are excluded from the incompatible 10-parameter motion memory.

The paired groups are the 12 value-economy school cases and 24 separately seeded memory-course cases. The two groups do not share a release ID or a point seed triple. Both conditions start from the same stored clinic memory and use identical first and reply draws on every paired card. They continue to learn within their own physical run. Later realized offers may therefore diverge after a different arm stroke even though the opponent's independent requests match. The requested cap is 200 legal alternating returns in all conditions. Private seeds are 282206268004005062 and 210435408683012463, respectively. Each rescue gets at most three rounds of 56 normal candidates and up to 12 sensitivity trials on the successful round.

A separate intervention replies to the archived 6785 optical estimate during its two failed last decisions. This reconstructs the observation from the real cameras and the command/encoder history, verifies the physical action prefix independently, and exposes only the reconstructed optical twin to the rescue. The intervention is a targeted diagnosis and is not counted as a held full-point result. Its separate private seed is 153795806199820457.

The two reconstructed observations matched the original RGB/encoder contexts exactly. At tick 1513 the rescue found only four weak contacts and no legal schedule. At tick 1518 its successive rounds found 20, 31 and 31 contacts. The number moving forward increased from zero to one to five as the weak-contact pool generated 9 and 12 signed inverse proposals. None produced a legal private trajectory, so nothing was sent to the real arm in this diagnostic. The intervention improved the quality of evidence retained in search, but did not rescue that rally ending. [Matched-context diagnostic](camera-rescue-6785.json.gz)

## Physical games

The fallback did not lengthen the rallies. Across both complete paired groups, it found six nominally legal motions in 7,684 additional rollouts and issued five of them. All five made native physical contact within 2.1 ms of the predicted time. Only one was an actual legal return, and the unmodified controller had also returned that ball legally. The 15-return rally on 6785 stayed at 15 with the fallback in the full run. No point approached the 200-exchange cap.

The physical first offers were identical on all 35 paired points in which the opponent made an offer. The remaining point, 8126, ended identically before either opponent could make a first contact. The two controls and interventions also shared 169 identical opponent requests before any differing reply count. Neither evaluation roster contains a release from the fixed motor-value fitting set.

| Points | Arm legal returns, stock → rescue | Alternating legal returns | Longest point, arm / both | Point wins | Process CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12 matched school | 66 → 66 | 144 → 144 | 15 / 31 → 15 / 31 | 0 → 0 | 165.9 s → 184.9 s |
| 24 independent releases | 46 → 45 | 113 → 109 | 5 / 11 → 5 / 11 | 2 → 4 | 151.4 s → 175.9 s |

The measured increase in point wins is not an increase in the ability to keep a rally going. On independent release 14233 both actors played identical commands up to tick 457 and made five legal returns. The fallback's fifth stroke clipped the net and landed wide and short. The live opponent could not get to it, so the arm won on the second bounce. The stock fifth stroke was answered and the arm lost. An exact replay with the CAD links checked every solver state in both points, 4,653 stock and 4,359 fallback, with zero penetration. On independent release 14843 no fallback motion was issued, but carried memory already differed. Its second successful stroke ended the point before the opponent could answer, while the stock arm made a further return and then lost. [CAD receipt](camera-rescue-14233-cad.json)

Contact timing was not the main failure mechanism in the five selected motions. The other four failed after a touch: three landed on the near half and one hit the side of the table. Two of those strokes left the blade moving distinctly downward, at −2.36 and −2.81 m/s, despite legal flight in the private court. None of the five selected candidates was legal under all three small uncertainty perturbation settings. Three were legal only under the nominal estimate. Better scoring for weak contacts helped the search preserve useful examples, but the court copy was too confident about the outgoing direction. The next useful change to test is calibrating uncertainty at impact, particularly signed vertical velocity and blade incidence, rather than increasing the number of candidate schedules against the same estimate.

[School physical tape](camera-rescue-matched.json.gz) · [School paired summary](camera-rescue-matched-summary.json) · [Independent physical tape](camera-rescue-independent.json.gz) · [Independent paired summary](camera-rescue-independent-summary.json)

## Reproduction

Run `python -m scripts.experimental.tt37_motor_clinic optical-learn OUTPUT` with the flags below. Use `--plan artifacts/tt37/school-plan.json --start 78 --n 12 --rescue-seed 282206268004005062` for the matched cases and `--plan artifacts/tt37/memory-course.json --n 24 --rescue-seed 210435408683012463` for the independent cases. For the control remove all four rescue flags.

```text
--samples 35 --rounds 3 --depth -1.56 --cut .18 --guided --refine --economy --body
--visual artifacts/tt37/rebound.npz --cap 200 --memory artifacts/tt37/scored-memory.json.gz
--motor-value artifacts/tt37/motor-value.json.gz
--camera-rescue --rescue-batch 56 --rescue-rounds 3
```

Use `python -m scripts.experimental.tt38_camera_rescue summarize BASELINE SUMMARY --tested OUTPUT` to enforce matching cards, controller settings, independent opponent requests, and report the first physical offers, actual alternating returns and projection-to-contact checks. The full CAD audit is `python -m scripts.experimental.tt38_camera_rescue clearance BASELINE AUDIT --tested OUTPUT --point-id 14233` on the independent cases.
