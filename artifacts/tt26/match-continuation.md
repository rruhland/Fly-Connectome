# Learning the next return after a live reply

On 14 held opening variants the frozen camera pupil landed one second return. The original continuation landed zero on the same variants. The successful held point used mount +0.22 radians and variant 17 from recorded donor 11953. The camera pupil selected development variant 9 from the same donor. The live opponent also landed its following shot, giving four successive legal returns after the initial feed. Both arms' commands reproduced the exact referee events. [Held points and action tapes](match-continuation.json.gz)

The development search found nine new physical second returns in 629 probes on 14 opening variants. The recorded first-return controller had produced one near return and one legal opposing reply on each opening. Each successful stroke reproduced the referee events when rerun from its saved physical branch point. [Development probes and held points](match-continuation.json.gz)

The course used the 28 points with a legal live reply in the existing two-mount experiment. Opening variants were divided at index 12, leaving seven per mount for development and seven per mount held. Both mounts used the same ball launch at a given variant index and stayed in the same split. Three original recorded donor IDs supplied variants in both splits, 4104, 11953 and 13009. Donor 2999 appeared only in development, and 10542 only in the held split. [Opening variants and original action tapes](live-points.json.gz)

Both courses replayed the existing first-return commands. A stereo RGB detector tracked the ball going toward the far side and then returning. In development, the teacher used the true post-reply ball state to propose rate-limited blocking postures. It searched through-contact swings around proposals that made a paddle contact and retained one action per opening that landed legally. During held play, the pupil chose the closest successful example for the same mount using stereo position and velocity. It used its own issued joint targets and the ticks since optical detection. The opposing arm continued to act live, and both arms’ issued commands were replayed against the referee. [Implementation](../../scripts/experimental/tt26_match_continuation.py)

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt26_match_continuation
```
