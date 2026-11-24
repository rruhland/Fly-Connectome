# A longer preparation opens a new low-ball stroke

The same clamp and SO-101 rescued the 0.934 m projected-height miss on development release 14843, reply index 1. The change began just after the previous near racket had separated from the ball, at control tick 92. Moving through an intermediate joint waypoint during the following 53 ticks placed the arm in a substantially untucked pose before the next low ball arrived. The second encoder was approximately −0.23 rad at tick 145, versus −1.75 rad at the nominal ready posture.

The incoming far collision and near-side bounce were unchanged, as was the opponent's random placement draw. The branch took the same ball at **y = −1.122 m and z = 0.912 m** on its rising approach. The original failed glancing hit was at y = −1.656 m. The new shot landed legally at x = +0.296 m, y = +0.816 m, and the same live opponent answered with a legal near-side bounce. All scored events reproduced bit-for-bit in the independent physical replay. Its full CAD audit saw zero penetrations over 1,792 actual solver steps.

The kinematic screen tests positive and negative signed blade faces and depth targets out to y ≈ −1.92 m, including below typical tabletop-level contact. The short exact-state stroke optimizer is compiled with a private y cutoff of −1.97 m, solely to avoid treating the default private −1.82 m bound as a physical rule. The real referee, ball, opponent, collision model and mounted arm are never edited. The successful hit was in front of this extended region. The proof is of **preparation before the far shot**, not yet of a defensive lob behind the endline.

This is an existence proof from a matched development prefix, using privileged planning and direct bounded joint commands. It is not performance of a learner. The proof tape is `long-low.json.gz`. The archived trainable input stream `long-low-optical.json.gz` contains only camera-derived measurements, own encoders and issued six-axis commands from ticks 78–223. Native contact geometry and CAD results are in the separate `long-low-audit.json`.

## Complete physical comparison

All four tapes reproduced their full native event histories on separate replays. The target far impact and inbound near-table bounce matched the original event records exactly in all four. All new legal landings were answered by a native far-blade collision and a legal return.

| Case, projected low height | New stroke y/z | Useful preparation | Full CAD outcome |
| --- | --- | --- | --- |
| 14843, 0.934 m | −1.122 / 0.912 m | 0.883 s, positive blade face | 0 penetrations in 1,792 steps |
| 14843, opposite face | −1.247 / 0.927 m | 0.883 s, faster wrist-flip route | 0 penetrations in 1,832 steps |
| 4322, 0.912 m | −1.128 / 0.898 m | 1.333 s, positive blade face | 5 blade/table overlap steps, up to 2.80 mm |
| 8302, 0.890 m | −1.136 / 0.905 m | 1.167 s, positive blade face | 4 blade/table overlap steps, up to 1.55 mm |

The two lower tapes are accepted by the default simulator guard and referee, but **not** by the stricter full-CAD requirement. A local follow-up kept each long preparation fixed, lifted the blade and varied timing and drives while passing the private optimizer a +1.5 or +3.0 mm minimum gap. It found no legal candidates in 1,920 trials. Do not promote the two lower tapes as clean demonstrations. The separate `long-low-cad-clean-optical.json.gz` contains only the two physically verified 14843 orientations, 297 observable control ticks in total. Native contact normals confirm opposite signed face alignments, about +1 and −1. The complete four-tape solver audit has 12,152 checked states and is kept separate from the observation export.

## Behind-endline check

The default fast forecast suppresses sub-tabletop samples even when the real referee would let the ball continue. A separate isolated physical flight collector retained low samples to z = 0.71 m. It provided 17 candidate ball points across the three actual arcs in the requested y ≈ −1.72 to −1.92 m band. The deepest sampled ball was at y = −1.918 m, z = 0.729 m. Across both signed faces and the tested defensive-lob orientations, the mounted-arm pose/route screen found no statically cleared approach there. This is a finite route and pose search, not a proof of mechanical impossibility. All observed useful strokes came in early on the rising ball instead of extending the private horizon.

## Strict follow-up on the two lower shots

Changing the preparation to the opposite blade face succeeded where trimming the original positive-face stroke did not. A single fast flip preparation per case and a +0.6 mm guard in the private search yielded two additional physical proofs. The +0.6 mm check was used only to rank isolated search simulations. Physical execution used the same stock court and referee from the exact earlier command prefix.

| Lower case | Actual clean stroke y/z | Full-CAD replay | Nearest audited gap |
| --- | --- | --- | --- |
| 4322 | −1.106 / 0.892 m | 0 overlap states out of 4,816 | 5.17 mm |
| 8302 | −1.105 / 0.896 m | 0 overlap states out of 3,528 | 16.03 mm |

Both used the verified negative blade face, landed legally and were answered. Both independent physical action replays reproduced the complete event histories. **All three formerly unresolved low balls now have a legal, answered, full-CAD-clean rescue from a long preparation.** This conclusion supersedes the unqualified positive-face trials on the lower two. It does not show a behind-endline lob or deployment transfer. Use `long-low-alternate-strict-optical.json.gz` for the two lower clean demonstrations, together with `long-low-cad-clean-optical.json.gz` for the two 14843 variants. The two clean files give four physical action demonstrations from three development cases. No held release was used.
