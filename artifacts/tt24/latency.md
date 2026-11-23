# Learning an unknown wrist-image delay

An optical clock recovered the unknown age of **99.46%** of wrist images under the changing delay schedule during 16 fresh physical games. With image alignment and forecasting, the same frozen motor scored **80 rebounds versus 25** with uncorrected delayed delivery. The clock estimates image age from the agreement between static scene edges and recent joint positions. The controller decodes the image at the inferred capture pose and forecasts to the current control tick. [Clock and forecaster](../../scripts/experimental/tt24_latency.py) · [Held real-physics trial](../../scripts/experimental/tt24_latency_trial.py)

The clock compares three visible natural edges against line projections for the current and previous five encoder packets. A persistent six-state filter combines absolute alignment and changes between consecutive delivered images. Repeated frames and sudden restoration are observable through the delivered image stream. The controller receives current encoders but no delay value or acquisition timestamp. Test injection switches the image alone from prompt delivery to an unknown 2–5-tick delay (33–83 ms), restores it, then changes it again on games that survive. [Frozen scene registration](wrist-egomotion.md)

## Reused stock play

Six archived stock action tapes were replayed to reconstruct 966 wrist frames and encoder packets, representing 16.1 seconds of stock experience. Recorded commands produced those replays. Only the reconstructed observations entered the fit. An additional 360 frames from two old stock reels selected ridge penalties. A line residual model learns small stock geometry corrections. Five horizon-specific displacement and velocity predictions learn from later optical observations on those tapes. The fit used no target play, action labels, rewards, ball metrology or camera timestamps. The stock motor and its earlier four-demonstration learning history are shared by every gameplay condition. [Fit receipt and exposure](latency-fit.json.gz) · [Frozen adaptation](latency-model.npz) · [Frozen motor provenance](bounce-ident.md)

On the old optical validation clips, forecast error was 0.30 m/s at one tick and 0.30 m/s at five ticks. Carrying the estimated velocity forward yielded 0.35 and 0.64 m/s. The score averages a displacement-rate and velocity residual. These errors measure agreement with optical estimates. The fresh gameplay trials score physical rebounds and execute actual joint dynamics.

## Fresh physical wrist gameplay

The stock arm ran the same 16 new resets in each condition. Intact delivery supplies the ordinary input stream. Naive delivery uses current joints with stale images. The learned clock condition estimates image age and forecasts from it. The oracle uses the same forecaster and receives the true age for comparison. The learned-clock retention condition receives intact delivery. Every run replays its actuator commands to reproduce the full physical scorer record. [Complete matched trials](latency-held.json.gz)

| Delivery and controller | Total rebounds | Sum of longest streaks | Best streak | At least five consecutive |
| --- | ---: | ---: | ---: | ---: |
| Intact, stock | 91 | 87 | 16 | 7/16 |
| Delayed, naive stock | 25 | 24 | 3 | 0/16 |
| Delayed, estimated age | 80 | 73 | 12 | 7/16 |
| Delayed, oracle age | 76 | 65 | 8 | 6/16 |
| Intact, learned clock enabled | 91 | 87 | 16 | 7/16 |

Estimated delivery age and forecasting recovered 55 of the 66 rebounds lost to naive delay, or 83%. The four-drop groups with first delays of 2, 3, 4 and 5 ticks scored 27, 23, 13 and 17 rebounds with the learned clock, versus 7, 6, 8 and 4 with naive delayed input. Games had a 330-tick cap. One intact game reached the cap.

The learned clock yielded more scored rebounds on 13 starts and tied three. Its mean improvement was +3.44 rebounds per drop, with a paired 95% percentile bootstrap interval of +2.19 to +4.81. The best adapted run was 12 consecutive rebounds. On that matched start, seed `369105`, the fixed controller with naive delivery scored two. The longest naive run across the roster was three. [Matched witness replay](latency-held.mp4)

The clock was exact on 99.42% of delayed images and 99.56% of prompt or restored images. A three-frame sequence of correct estimates began within two ticks for 20 of 21 delay arrivals and within one tick for all 11 restorations. The remaining game ended after two matching frames at the switch. [Per-frame ages and change audit](latency-held.json.gz)

On intact input the enabled clock selected zero delay on all 2,081 frames. The actions and complete judge records were identical on all 16 intact pairs. All 80 gameplay tapes replayed exactly. The learned correction was tested on the stock wrist camera and the carried scene geometry. [Retention and exact replay](latency-held.json.gz)
