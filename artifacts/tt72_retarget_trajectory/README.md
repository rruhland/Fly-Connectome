# TT72 · Can the centered clamp replay an earned +12 cm stroke?

The original centered body made a real broad-face contact on the exact central-low 9254 delivery, but did not reproduce the teacher's forward and upward impulse. It returned **0 legal landings and 0 ordinary replies in 40 physical retargets** across three known +12 cm wins. There is no new stock lesson or held result. This is a bounded movement-transfer result, not a proof that the stock body has no legal solution.

| Exact developmental release | +12 cm earned contact radius | Stock retargets | Stock CAD-clean contacts within 58 mm with  an absolute face alignment of at least 0.75 | Stock legal landings / answered |
| --- | ---: | ---: | ---: | ---: |
| 9254, central low | 16.5 mm | 13 | 2 | 0 / 0 |
| 8831, left-adjacent low | 39.0 mm | 14 | 0 | 0 / 0 |
| 15456, left low | 43.8 mm | 13 | 6 | 0 / 0 |

The original stock qualification had no clean legal return on 9254 or 8831. Its separate mechanics coach did return 15456 three times at 70.9–75.9 mm from blade center. Raw shifted commands on stock had no legal landing on any of the three. These three were selected because the lateral teacher had already earned its returns. They are not an estimate of season performance. [Teacher qualification](../tt63_fixtures/lateral/README.md)

## What the actual contact measurements show

- **9254** produced centered-stock contacts at 55.6 and 56.3 mm, both on the blade face with zero sampled CAD penetration. The closest of those to the teacher's outgoing velocity came off at `(1.68, 1.17, 1.48) m/s`, then bounced on the learner's own half. The teacher left at `(0.41, 2.83, 2.27) m/s`. At the actual stock contact, the face normal was tilted 17.7° from the teacher's and the center was 47.0 mm away. Using rigid-body center velocity plus angular velocity crossed with the contact offset, the local blade patch was moving **down at 0.18 m/s**, versus **up at 0.86 m/s** on the scored teacher tape. This separates access to the broad face from executing the stroke.
- **8831** reached only the oblique rim on stock. At the donor clock, the centered joint-box fits missed the blade center by 26–28 mm and the signed normal by 17–18°, both with wrist flex at its limit.
- **15456** had a much closer stock fit at the donor clock, with roughly 3 mm of center error and 4.4° of signed face error in static inverse kinematics. But its actual wide-face contacts on the examined trajectories sent the ball **down**, despite 3.85–4.34 m/s forward separation. The teacher sent it upward at 2.15 m/s. A static fit did not imply the needed dynamic stroke.
- Retiming did not solve 9254 by itself. The permissive stock fit two ticks later matched the center within 1.0 mm and the signed normal within 0.63° only with an 85.5° in-plane blade rotation and an optimistic residual angular twist error of 3.96 rad/s. The two actual attempts intended for the later clock families incurred 29 and 88 sampled CAD intersections and did not touch the ball. The nonpenetrating candidates still failed on their own merits.

## Design and reproducibility

1. Replay three already-paid donor strokes at the original +12 cm commissioned geometry. Each entire event sequence reproduced its previously verified digest. Capture ordinary stereo estimates, encoders, encoder speed and actuator references along with evaluator-only world paddle center, full signed rotation and rigid-body twist. The 457 control samples included 413 valid RGB ball estimates.
2. Hold the exact real incoming far contact and near bounce fixed. Retarget the complete stock movement from neutral through preparation and follow-through under joint and per-tick limits. Search five contact clocks over ±4 control ticks with preparation warping and servo lead. The phase-equivalence comparison gives earlier preparation additional freedom, checks four constant rotations around the disc normal, and binds local joint motion to the donor's full twist. A constant in-plane phase preserves the aimed world disc plane and world twist, but changes the articulated handle route. Smooth trajectory-level changes are graded in the native motor model before physical replay.
3. The two bounded stages ran 1,065 model trajectories including repeated elites and 40 physical retargets. All physical tests had bit-identical far impact and near bounce, and the receive line matched when reached before contact. They kept the original centered clamp, stock actuator/rate bounds, ordinary opponent and referee. Of the physical tests, 38 had zero sampled CAD penetration. Three representative stock contacts were then re-observed in the native stereo and encoder apparatus with unchanged full event hashes.
4. The mechanism fit can use the archived true time and blade state. Neither a legal stock skill nor an RGB controller is claimed. Independent stock releases were not opened because no target produced a new repeatable legal interior.

If revisiting the centered arm, use the clean 9254 contact as a practice state and ask whether a CAD-safe preparatory route can change **local blade patch vertical and forward speed at the same depth** while keeping the normal close enough to the scored one. Merely repeating the contact-position search would miss the measured deficit. Keep an earned legal landing with an ordinary answer as the threshold before any camera policy is trained or neighboring releases are opened.

Run `python -m scripts.experimental.tt72_retarget_trajectory harvest`, then `stock`, `equiv`, and `measure` with the project's EGL and one-thread native settings. The ledgers checkpoint by physical release.

[Protocol](protocol.json) · [Donor trajectories](donors.json.gz) · [Stock retargets](stock.json.gz) · [Phase-equivalent retargets](equivalence.json.gz) · [Measured stock witnesses](stock-witnesses.json.gz)
