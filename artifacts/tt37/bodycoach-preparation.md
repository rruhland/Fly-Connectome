# Where the articulated arm can strike, and why preparation alone did not fix later balls

Moving the strike location along the incoming flight produced collision-clear pose/face hypotheses for all 100 incoming balls from the long random witness. That is very different from the 24/100 pose fits obtained by copying the carriage's exact strike point. The problem then split into finding a useful outgoing flight and having enough time for the correct joint branch. A preparation posture synthesized from the witness improved offline reach coverage but did not produce a later legal return in four paired fresh-seed points. [Full time-window census](bodycoach-windows-native.json.gz) · [Live paired preparation test](bodycoach-preparation-summary.json)

## Flight windows and arm constraints

Each witness forecourt flight started from the measured native ball state after its physical near-table bounce. A separate native solver recovered the carriage's actual, later arrival positions with a median 4.45 mm and a maximum 4.88 mm difference. We sampled 1,164 eligible post-bounce locations, about 24 ms apart, spanning candidate strike depths from 0.88 to 1.75 m from the net. No actor received these records during deployed play.

At each location we tested several pitch angles, face fits, and bounded projected blade velocities using the five-joint arm Jacobian. We evaluated full CAD geometry at the fit and along sampled linear joint-space routes from two readiness poses. Arrival tests used a 3.45 rad/s joint speed bound and a further 135 ms reserve. A finite-mass isolated paddle supplied a projected departure. Two separate tests then graded that departure. One asked only for a native post-bounce track on the far half. The stricter test asked for a playable outgoing arc into the far receiver window. These are sampled endpoint-and-drive checks, not physical articulated stroke executions or a mathematical ceiling on all possible techniques. [Census implementation](../../scripts/experimental/tt37_bodycoach_windows.py) · [Original literal-endpoint experiment](bodycoach.md)

| Lane at carriage contact | Collision-clear pose/face | Native landing proxy | Playable far-receiver arc | Playable with a route from either initial pose |
| --- | ---: | ---: | ---: | ---: |
| Left | 39/39 | 34/39 | 33/39 | 32/39 |
| Middle | 27/27 | 13/27 | 4/27 | 4/27 |
| Right | 34/34 | 25/34 | 23/34 | 23/34 |
| All | 100/100 | 72/100 | 60/100 | 59/100 |

The nominal raw home supported routes for 41 of the 60 playable arcs and the alternate centered home for 38. They overlapped on only 20. The middle balls made the distinction particularly clear. A pose and face could be fitted for every such incoming flight, but only 13 met the native landing test and only four yielded a continuation into the receiver window. Across all 72 native-landing flights the sampled strike depths ranged from 0.976 to 1.607 m. The median widest time window was 25 ms, and 23 of the balls had only single sampled times. Reaching the ball is not the same as imparting the necessary upward and forward departure, and even the better fit can leave a small timing margin.

## Physical later balls and a matched intervention

We replayed the actual articulated teacher's four earlier later-ball misses. Its observed joint state at each opponent hit was the starting point. All four had geometric endpoint fits. Two, 14832 at x=+0.150 m and 8239 at x=+0.027 m, had no native landing in the sampled departure family. The other two had landing windows but no route within the arrival budget. For 5530, center x=−0.054 m, the closest candidate needed 286 ms of joint travel plus the 135 ms reserve, versus 408 ms available. For 3025 on the right at x=+0.188 m, even the optimistic angle travel was 578 ms versus 383 ms available. That latter shortfall persists without the reserve. [Actual later-ball diagnoses](bodycoach-windows-later.json.gz)

We built a bank of 20 intermediate positions, sampled from the synthetic witness and screened with articulated CAD. Its best single pose had routes into 57 of the 60 playable arcs, with the blade near x=−0.005 m, y=−1.139 m, z=1.011 m. The real-arm teacher could move after its first stroke toward a reachable choice from this bank without knowing the next randomized opponent goal. This was ordinary bounded actuator movement, with the same geometry and no point or body reset. The reference teacher held its previous follow-through. Both ran on four independently seeded fresh replies. [Preparation endpoints](bodycoach-hubs.json) · [Preparation implementation](../../scripts/experimental/tt37_bodycoach_preparation.py)

The first and second physical offers matched exactly for all four pairs. Both teachers returned all four first shots and none of the four later ones. Preparation moved on all four, and only differed from the reference after the first hit. The reference touched one second ball without landing it. In the matched fresh set three of four later balls still had no native landing in the same sampled departure family. Only the fourth, a middle ball on 5530, had one, and the sampled arrival deficit was larger after preparation, 202 vs 152 ms. This experiment gives no evidence that a new neutral ready position alone improves later play. [Raw paired tapes](bodycoach-preparation-baseline.json.gz) · [Prepared tapes](bodycoach-preparation-prepared.json.gz) · [Matched later-flight diagnoses](bodycoach-preparation-windows.json.gz)

All eight paired action tapes reproduced exact physical contacts, placements and scores. A separate full-CAD replay covered 12,074 solver states with zero penetrations and 18.6 mm minimum separation among the sampled pairs. [Independent replay and clearance audit](bodycoach-preparation-audit.json)

The controller remains a privileged motion teacher. None of these geometry or simulator inputs were added to a deployed direct learner. The earlier 266 camera/encoder-only demonstration samples remain separate and use only practice points. [Practice observations and physical coach](bodycoach.md)

## Reproduce

Run `python -m scripts.experimental.tt37_bodycoach_windows --output artifacts/tt37/bodycoach-windows-native.json.gz` for the full scan. Add `--diagnose` and a list of saved coach tapes to use actual encoder states from their later receptions. `python -m scripts.experimental.tt37_bodycoach_preparation derive` builds the preparation bank. Run `prepared --feeds /tmp/tt25-rallies.json` and `baseline --feeds /tmp/tt25-rallies.json` in the same module, then `summary` for exact pair matching. Use `python -m scripts.experimental.tt37_bodycoach_audit --output /tmp/tt37-body-prep-audit --verify` with the paired action tapes for physical replay.
