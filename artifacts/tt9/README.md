# From a legal five-axis serve to the next return

After 12 practice tosses per installation, the player landed 23 of 24 new serves, compared with 15 for its fixed stroke. It learned from where the descending toss crossed the moving blade face in stereo, including tosses the blade missed entirely. The learned controller lost both full alternating-service games 5–11. The fixed controller scored 4–11 and 6–11. [Same throws side by side, first-person replay](five-axis-same-toss.mp4) · [Service experiment and checkpoints](precontact.md)

| Research question | Result |
|---|---|
| Does measuring the approach help when there is no departing ball to observe? | The camera estimate recovered the blade-face crossing on all 24 complete misses in the timing experiment. Adding the sideways correction subsequently turned eight held faults into legal serves while preserving all the fixed stroke's legal serves. [Approach experiment](precontact.md) |
| Does the correction carry to a third five-axis installation? | Carried and reset learners both lost 3–11. Returning to the practiced station after that game cost points compared with going there directly. All 180 reference Pong tapes survived. [Installation switch](installation-switch.md) |
| Does extra time and lift help return the opponent's reply? | On seven matched legal replies, it added one first follow-up but reduced total legal returns from ten to eight. The original controller won more points in both full games. [Matched histories and full games](recovery.md) |
| Does receiving-only practice improve alternating-service play with six joints? | The practiced lift scored 5–11 and 1–11. A learner starting with empty lift memory scored 5–11 and 2–11. [Six-joint games](rotary.md) |
| Does learning from earlier transitions improve a new cart game? | The source-conditioned map caught 364/512 versus 317 for direct transfer, 356 for pooled practice, and 361 for a fixed 50/50 blend. The evidence for an advantage over the simple blend is uncertain. [Transfer and controls](catch-transfer.md) |

To reproduce the 11-second comparison, run the following in the project Python environment. Each panel shows the left camera on the same held throw at half speed. The first learned serve is answered and the arm loses the point. The second wins.

```bash
python -m scripts.experimental.tt9_serve_replay artifacts/tt9/precontact-registration-b.json.gz /tmp/same-toss.mp4 226729254 2094545355
```
