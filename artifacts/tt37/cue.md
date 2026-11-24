# TT37 visual anticipation of the far paddle

The unmarked gray paddle is visible in the existing 384×288 stereo RGB. Its two far-end silhouettes are often only 9×10 pixels each. Stereo yields a blade center, recent center motion and two sets of contour moments. The latter are face-shape proxies, not an exact recovered face normal. The blade can disappear below the table during a backswing, so all comparisons report the paired visible subset and its coverage.

## What was learned from the pixels

The dataset used the privileged finite-mass capacity receiver only to generate varied physical exchanges under the full continuous TT37 random opponent. The receiver's plans, opponent requests, simulator ball state and far carriage state were not fed to the models. Learning used regularized least squares with splits by measured opening release. The label is the later optically measured incoming position and velocity at `y = -0.40 m`, obtained from stereo frames after a real far collision and before the learner's next collision. The independent physical contact/landing audit places each capture in time. It never supplies a position, velocity or aim label.

A causal gate used the first fresh outward ball fit after `y = +1.00 m`. The ball model and paddle models were evaluated on the same samples, including the identical decision time. The uniform/no-aim baseline is the unconditional mean of practice outcomes. It does not know the true random placement. The training sets supplied 84 paired examples across 18 measured releases at this gate.

| Held situation | Visible paired / labelled | Median lead | Unconditional p / v error | Ball only | Ball + motion | Ball + motion + silhouette |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh continuous random openings | 23 / 24 | 142 ms | 14.95 cm / .780 m/s | 12.73 cm / .567 m/s | 12.50 cm / .553 m/s | 12.37 cm / .565 m/s |
| Independent later continuous random replies | 16 / 16 | 132 ms | 9.94 cm / .684 m/s | 8.18 cm / .501 m/s | 8.32 cm / .505 m/s | 8.20 cm / .516 m/s |

Position is Euclidean error in x/z, and velocity is Euclidean error in vx/vy/vz. For causal fresh openings the release-bootstrap interval for the full paddle cue's position change versus ball only was −1.16 to +0.46 cm. For later random replies it was −0.35 to +0.25 cm, and velocity error was slightly higher with full silhouette. These data do not establish a useful incremental prediction at the +1.00 m gate.

The retrospective test also held out entire releases and reply seeds, but aligned each decision to the actual physical contact exclusively for diagnosis. At ~250 ms lead the full paddle cue reduced held opening velocity error from .586 to .507 m/s (release bootstrap −.127 to −.036 m/s) and held later reply error from .588 to .504 m/s (−.125 to −.039 m/s). It did not materially improve position. This real early signal is not a deployable timing method. A follow-up froze the model from the 234 ms practice bin and used the causal `y = +0.70 m` gate, selected from the outgoing position in that **practice** bin. Its confirmation openings use 18 new measured releases and its later shots use 16 new independent continuation seeds with the four established anchors.

| Fresh confirmation at +0.70 m | Visible paired / labelled | Median lead | Unconditional p / v error | Ball only | Ball + motion | Ball + motion + silhouette |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh continuous random openings | 15 / 16 | 244 ms | 12.50 cm / .610 m/s | 9.62 cm / .500 m/s | 9.95 cm / .436 m/s | 10.29 cm / .443 m/s |
| New independent later random replies | 16 / 16 | 239 ms | 14.55 cm / .601 m/s | 12.53 cm / .486 m/s | 11.77 cm / .450 m/s | 10.92 cm / .425 m/s |

Early paddle motion improved opening velocity prediction versus ball-only input (95% release bootstrap −.109 to −.028 m/s). The full silhouette model was chosen on practice only and its opening velocity interval crossed zero (−.125 to +.003 m/s). On the independent later draws, the full cue improved both position (−2.21 to −1.06 cm) and velocity (−.073 to −.050 m/s). The later intervals are grouped across four measured releases, so they describe consistency on those releases, not all possible shots. The randomized later draws were unseen at training or model selection.

## Physical motor test

The actor is the unchanged TT33 RGB/encoder joint policy with the TT36 visual reach correction at handoff. The preparation layer retrieves encoder postures from earlier *physically scored* TT36 approaches, using only a frozen visual forecast to choose its neighbor. It changes only the formerly blind interval, up to 480 ms and .42 rad per axis from the ready posture, within the already enforced slew and motor limits. At the first trustworthy inbound fit it hands off immediately to the usual direct RGB controller. No target search, inverse kinematics, simulator/partner input, or future label is available to the deployed actor. There is no online update in this test.

The paired conditions are unchanged ready posture, preparation from ball-only prediction, and preparation from ball plus paddle cue. The coarsely aligned position/velocity model is the same across these conditions except for its visible inputs. All arms use the same first-release and opponent draws, sighting gate and sensor frames up to any real movement effect. Continuing rallies respond physically to the arm's actual returns and need not have identical future balls.

For the original +1.00 m gate, all three arms scored the same 7 legal near-side returns across the 24 fully random opening cards. On 16 anchored continuation cards with independent random reply placements, unchanged/ball/motion scored 14/13/13 near-side legal returns, of which 2/1/1 were later returns. Motion beat ball on one point and lost on one. The motion branch issued 544 preparatory control ticks across the two decks. Of 45 preparations, 42 began before a subsequent independently audited far collision, two began after it and one did not see a subsequent far collision. Optical reversal can lag a physical collision; the forecast itself is scored only for genuinely pre-contact decisions. At handoff the motion branch's median blind interval was 200 ms in each deck.

The result is a real control null for this motion signal and joint preparation layer, not just a failure of a forecast score.
