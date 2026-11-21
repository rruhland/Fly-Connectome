# Learning when to raise the six-joint racket

The [mixed-condition follow-up](contact-mixed.md) plays complete games against aligned and earlier fixed lifts.

After 24 all-soft practice points, the player returned 14 of 20 openings across two new six-joint installations. The inherited receiver returned two. The learner that began with an empty timing memory returned ten. Both timing learners kept updating on every held point. The carried player won 12 points and the empty-memory player won ten, compared with two for the inherited receiver. [Timing learners](contact-soft-online.json.gz) · [Inherited and coasting controls](contact-soft-controls.json.gz) · [Practice](contact-lift-practice.json.gz)

| Player | Legal openings, site A | Legal openings, site B | Legal returns | Points with a second return | Wins–losses |
|---|---:|---:|---:|---:|---:|
| Inherited | 1/10 | 1/10 | 2 | 0 | 2–18 |
| Inherited stroke with longer visual coasting | 1/10 | 1/10 | 2 | 0 | 2–18 |
| Fixed aligned lift | 3/10 | 9/10 | 13 | 1 | 10–10 |
| Starts timing learning on the held course | 4/10 | 6/10 | 10 | 0 | 10–10 |
| Carries 24 points of timing practice | 5/10 | 9/10 | 15 | 1 | 12–8 |

The fixed aligned phase tied the carried learner on the second site. On the first site, the carried learner added two openings and two wins while selecting earlier strokes for five opportunities. [Fixed timing](contact-soft-fixed.json.gz)

The first all-soft game finished 7–11 for the carried learner and 5–11 for the inherited receiver. Across the 16 common starts the learner landed 14 legal returns against six. Its final point included five legal returns from each side and lasted 299 control frames. The inherited receiver had zero points with a second legal return. The complete tapes recorded 19 legal returns and three sustained points for the learner. [Live timing game](contact-game-carried.json.gz) · [Inherited game](contact-game-inherited.json.gz) · [First-person ten-return rally](contact-soft-five-returns.mp4)

On a second disjoint installation, the carried learner won **11–9** where the inherited player lost **3–11**. The learner landed 11 legal returns across its 20 points, compared with four in 14 points for the inherited receiver. Every legal return in this second game opened a point. [Independent matched game](contact-game-repeat.json.gz)

The player uses its stereo ball and racket estimates to plan a collision, as before. It can sample a small upward stroke one frame early, aligned, or one frame late. Each phase uses the same rise rate and bounded height excursion. It also has a choice to keep the inherited stroke. Regularized score estimates use the incoming ball's lateral position, approach speed, strike height and visually inferred actuator delay. At the end of a point, four coherent stereo sightings support a local outgoing flight fit when available. The player's preference update uses that flight's estimated landing and clearance plus the public point reward. Every timing choice and its ensuing evidence appears in the point ledger.

Every group received eight public motor-probe episodes at each new installation. They used the same pre-practice shared hall donor, controller random state, commissioning data, ball starts, and softened racket. The carried group transferred only its contact timing memory from the two practice installations. The six-joint games on new installations used the same first-to-11, win-by-two rules and a 420-frame point cap. Every scored command tape in the held test and games was replayed against the applied commands, physics events and full point result. Camera coasting after occlusion changed isolated actions and tied the inherited receiver on all aggregate outcomes in the short courses.

## Screening shorter changes

On 16 old development starts, bounded one- to three-frame changes to the rate commands could help or harm. One live variant produced a second legal return on each installation and reduced legal openings from six to four overall. An earlier attempt moving the strike plane forward by nine centimetres landed zero of eight openings. [Development tapes](contact-development.json.gz)

A separate tape-only screening evaluated 380 edits per development start, using the recorded camera forecast as the timing anchor. The best one-frame count was eight of 16 openings. A two-frame up-and-forward change recovered four failures and preserved all six previous openings. The screening used the referee to score landings after replaying each candidate. All anchor times came from camera forecasts. [Optical-clock tape audit](contact-rate-audit.json.gz)

The live fixed two-frame rate change landed ten of 20 openings on the same held courses. The separately trained rate policy selected the three-frame intervention on 19 of 20 held opportunities and landed four openings. The inherited receiver landed two. All three groups finished with zero sustained points. [Rate curriculum and live comparison](contact-soft-rate.json.gz) · [Rate memory](contact-player-rate.json.gz)

## Reproduce or continue

Use the MuJoCo and EGL environment in the [TT6 instructions](../tt6/README.md). The following runs take the shared 42-point hall checkpoint as the donor.

```sh
python -m scripts.experimental.tt7_contact_trial /tmp/contact-practice.json.gz \
  --test-seeds --checkpoint /tmp/contact-memory.json.gz
python -m scripts.experimental.tt7_contact_trial /tmp/contact-held.json.gz \
  --practice-seeds --resume /tmp/contact-memory.json.gz \
  --conditions baseline coast cold carried fixed
python -m scripts.experimental.tt7_contact_trial /tmp/contact-game.json.gz \
  --practice-seeds --resume /tmp/contact-memory.json.gz \
  --test-seeds 68059317 44593197 --conditions baseline carried --race-to 11 --max-steps 420
python -m scripts.experimental.tt7_contact_trial /tmp/contact-rate.json.gz \
  --family rate --practice-seeds 39187163 --conditions carried fixed
python -m scripts.experimental.tt7_contact_audit /tmp/contact-tape-screen.json.gz
```

Continue contact practice at a new installation using `--resume` with a `--practice-seeds` value and a new `--checkpoint` path. The saved [lift memory](contact-player-lift.json.gz) preserves the sufficient statistics from all 24 practice points. Loading it reproduced the first held point at site B, including the command hash and optical timing evidence. `--family rate` exercises the learned two- or three-frame rate changes and a single-frame comparison. Each family carries its own timing preference state.
