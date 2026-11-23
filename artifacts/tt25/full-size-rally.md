# From a recorded incoming ball to a live two-arm exchange

[Watch a practiced point followed by an unseen point](recorded-two-arm-points.mp4) · [Shot distribution, teaching and the 96-shot assessment](recorded-receiving-lessons.md)

The camera pupil's seven independently recorded first returns survived when the far arm played live. The opponent contacted and legally returned four of them. The original block pupil held its pose and missed all four next balls. The incoming human–robot recording supplies the opening ball state once. Both arms and the free ball then evolve under contact physics. The far arm uses the existing privileged opponent planner. The SO-101 has access to its fixed camera pair, encoders and its own issued commands. The first incoming feed is excluded from the opposing reply count. These are mid-rally receiving points with a regulation-size table. [All seven two-arm action tapes](recorded-live-opponent.json.gz)

## Is a moving second return possible?

We used the six original successful training feeds to acquire a continuation. Five elicited a legal reply from the live opponent. On four of these the simulated practice teacher could propose a second receptive posture. Across 96 posture trials it reached a second racket contact 60 times and made zero second legal returns. The fifth reply had no candidate under this bounded posture search. This gap between contacts and returns explains why rewarding touches alone can produce misleading progress. [Practice starts](recorded-live-development.json.gz) · [Static receiving proposals](recorded-second-ball.json.gz)

A teaching variant explored moving the five joints through contact. It used simulator kinematics, prior contact timing and the recorded rival reply to propose motion during development. Physical landings selected the examples. It found a second legal SO-101 return on two training feeds. This is privileged simulated teaching. The resulting pupil reads the visual and proprioceptive history, recognizes observations resembling its two demonstrations, and retrieves direct motor rates from those examples. It has no prescribed stroke solver, incoming state vector or contact-trigger signal at deployment. Its orange-object segmentation and demonstration-based action memory are explicit scaffolds. [Teacher proposals](recorded-through-contact.json.gz) · [Camera-to-rate memory](recorded-second-pupil.npz)

The pupil reproduced two own returns and two opposing replies on **both practiced feeds** with an active opponent. On the original seven unseen first-return feeds it retained all seven legal blocks and all four legal opposing replies. It made **zero second returns**. Its learned switch to the new motion activated in only one of the seven cases. The video shows one practiced point and one previously unseen point in that order. Every complete point passed motor-tape and referee replay. [Deployment results](recorded-second-pupil.json.gz) · [Video verification](recorded-two-arm-points.json)

Even small physical perturbations broke the second return. We replayed eight variants of the two practice feeds with independent per-axis standard deviations of 5 mm in launch position, 0.02 m/s in velocity and 0.5 rad/s in spin. Seven retained the first block and none retained the second. At 3 cm, 0.15 m/s and 5 rad/s, three of eight retained the first block and none retained the second. Both controllers produced the same legal return counts on all 16 paired points. These are simulated variants of two recorded feeds. [Paired scores and replay checks](recorded-rally-tolerance.json.gz)

## Does either game teach the other faster?

* **Pong to the full-size receiving pupil.** We transferred the frozen optical delay compensation gain previously selected by reward in 720 native Pong practice points. The pupil received camera information one or two frames late. Across 96 focused real launches, with one, three or six demonstrated arm postures, the Pong gain gave 2, 4 and 6 legal returns. Omitting compensation or using shuffled Pong credit gave 2, 4 and 7. The same number of demonstrations, action repertoire and body knowledge were available in every comparison. This test gives no evidence of faster table-tennis acquisition from that temporal prior. [Physical outcomes and source receipt](recorded-pong-clock.json.gz)
* **Table tennis to wrist-only bouncing.** We fitted an optical time gain to the six full-size training matches using ordinary camera measurements from 2,595 overlapping short prediction examples. We transferred it into an existing wrist-only bounce policy without target practice. On 12 fresh simulated wrist releases with variable camera delay, the sum of longest consecutive streaks was 23 with tennis experience, 17 with temporally shuffled experience, 18 with unit extrapolation and 30 with zero correction. Beating the shuffled comparison alone would overstate transfer. This change made the strongest matched baseline worse overall. [Source optic fit](recorded-delay-visual-fit.json) · [Wrist-only drops and paired action tapes](recorded-delayed-wrist.json.gz)

These comparisons concern a particular delay mechanism. A score shared across separate game heads is insufficient to establish general cross-game strategy. Source and target task experience, body configuration, controls and camera latency should stay explicit. The table-tennis checkpoint and the wrist head remain individually loadable.

## Reproduction

Use the DeepMind rallies JSON identified in the [physical benchmark](recorded-benchmarks.md).

```bash
python -m scripts.experimental.tt25_live_opponent /path/to/rallies.json
python -m scripts.experimental.tt25_live_opponent /path/to/rallies.json --development
python -m scripts.experimental.tt25_second_ball /path/to/rallies.json
python -m scripts.experimental.tt25_through_contact /path/to/rallies.json
python -m scripts.experimental.tt25_second_pupil /path/to/rallies.json
python -m scripts.experimental.tt25_rally_tolerance /path/to/rallies.json
python -m scripts.experimental.tt25_receiving_movie /path/to/rallies.json --full-points
python -m scripts.experimental.tt25_pong_clock /path/to/rallies.json
python -m scripts.experimental.tt25_match_to_delayed_wrist /path/to/rallies.json
```
