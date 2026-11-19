# Carrying visual pursuit evidence across Pong points

The point-local guard estimates opposing pursuit from public positions. The carry wrapper starts the next point with 88% of the highest retained estimate. Earlier evidence can protect the first attack of a new point, even before the new ball has caused the opponent to pursue visibly. The estimate fades across points that supply no stronger evidence. A newly restored ordinary state can earn placement scores with the original learned table.

The pursuit-only controller accepts the same goal, score, and timing settings as the point-local guard. Its extra input is a serializable `PursuitMemory`. The wrapper copies the visual estimate when a point ends. [Pursuit-only controller](../../scripts/experimental/pong_carried_pursuit.py)

## Paired development

The reused course replayed the original faster-opponent phase with its carried remap and 12-camera delay, then removed all changes on the original ordinary-again starts. The fresh development course used four distinct starts per phase. Both arms began with the same motor, timing, and score checkpoints, carried them between points, and updated placement from scores. The point-local guard and the carried guard used the same 16-second limit.

| Course segment | Inference | First returns | Later returns | Wins after own return |
| --- | --- | ---: | ---: | ---: |
| Reused harder state | Carry pursuit | 7/12 | 26/26 | 0 |
| Reused harder state | Point-local pursuit | 7/12 | 22/23 | 0 |
| Reused ordinary restored | Carry pursuit | 11/11 | 17/17 | 8/12 |
| Reused ordinary restored | Point-local pursuit | 11/11 | 18/18 | 6/12 |
| Fresh harder state | Carry pursuit | 3/4 | 11/11 | 0 |
| Fresh harder state | Point-local pursuit | 3/4 | 8/9 | 0 |
| Fresh ordinary restored | Carry pursuit | 3/3 | 0/0 | 3/4 |
| Fresh ordinary restored | Point-local pursuit | 3/3 | 0/0 | 3/4 |

The earlier development wrapper also tracked a penalty after a missed reply, which used a supplied contact event. That penalty never activated in these 32 games. The pursuit-only replay used public visual motion and matched every original request hash, contact result, score-update count, cancellation, and carried estimate over 14,727 camera transitions. The combined controller adds a separate estimate based on terminal scores of delayed shot attempts. [Portable replay evidence](../../artifacts/m1b/2026-11-19-pong-mixed-courses.zip)
