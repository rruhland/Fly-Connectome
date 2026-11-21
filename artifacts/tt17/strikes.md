# Learning the interception plane on a familiar arm

After practice, choosing a stroke from point reward changed a gantry loss into a 14–12 win. The next matched comparison on the same gantry reversed the advantage. Across all six comparisons, the inherited and point-reward players each won four. An optical rally cue changed the stroke choices but reduced the number of points with multiple legal returns on the four primary fixtures. Keep the inherited receiving stroke as the default. [Decision and checkpoint audit](strike-summary.json.gz)

Each cell contains the final score, legal returns and distinct points with at least two returns. The two repeat games used fresh seeds. The optical method was evaluated on the four primary fixtures.

| Familiar installation | Inherited plane | Learned from point reward | Learned from visual crossings and point reward |
| --- | --- | --- | --- |
| Gantry A | 11–5 / 9 / 0 | 11–7 / 14 / 2 | 11–4 / 15 / 4 |
| Gantry B | 8–11 / 18 / 5 | 14–12 / 19 / 2 | 11–9 / 17 / 3 |
| Five-axis A | 10–12 / 12 / 2 | 8–11 / 9 / 1 | 11–13 / 8 / 2 |
| Five-axis B | 11–4 / 15 / 5 | 11–4 / 15 / 5 | 11–7 / 16 / 2 |
| Gantry A repeat | 11–3 / 19 / 3 | 11–3 / 19 / 3 | — |
| Gantry B repeat | 12–10 / 21 / 6 | 9–11 / 20 / 6 | — |

On each arm, 36 rival-served practice points used balanced randomized choices of three interception locations along the incoming flight. Each location shifts the established player's preferred plane by 0, +12 or −9 cm. Context consists of incoming position and velocity features extracted from stereo. Each arm spent 2,681 (five-axis) or 3,474 (gantry) control frames on practice. Only the resulting stroke examples transferred from practice into each matched game. The learner continued updating its empirical action values between scored points. The ordinary scorer used the sign of the terminal point reward. The shaped version also counted visible flights crossing from the player's end toward the opponent. The offsets, perception and optical reward rule were supplied by the experimenter. [Policy](../../scripts/experimental/tt17_strike.py) · [Practice and evaluation](../../scripts/experimental/tt17_strike_trial.py)

On the gantry, the rearmost option landed 19 returns but won only 3/12 practice points. The inherited plane landed 11 and won 8/12. Different options saw different practice trajectories. During actual evaluation the point-reward policy never chose the rearmost plane. The shaped reward favored the forward plane on three primary fixtures. Across the 72 practice tapes an independent replay counted 98 optical crossings versus 76 legal returns. Counts matched exactly on 51/72 points. A visually successful crossing can still land illegally. [Optical replay](../../scripts/experimental/tt17_strike_optics.py) · [Choice reconstruction](../../scripts/experimental/tt17_strike_report.py)
