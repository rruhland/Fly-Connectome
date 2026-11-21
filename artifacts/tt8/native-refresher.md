# Native Pong refresher after three hall games

The checkpoint after three alternating table-tennis games ranked numeric relation 1 first. It had ranked relation 0 first before the hall games. The roles and geometry coefficients were unchanged. On the original 180-point native Pong schedule, the three-game checkpoint changed 156 command tapes. First returns went from 129 to 126, later returns from 195/206 to 188/215, and earned points from 41 to 36.

The refresher tried all four carried relations on 16 disjoint native Pong starts with two points per start. Each policy saw the camera frames, its preceding command, and the public point score. The practice policies played the same starts and reidentified the signed action mapping from the remapped camera scenes. The archived 180 points were reserved for frozen evaluation. A nominee had to outperform the incumbent on another 16 starts with three points each. Promotion required a positive one-sided 90% bootstrap lower bound on paired mean score, with resampling by match. Both sets of practice starts, including the continuation serves, were disjoint from the archived evaluation.

Discovery scores were −1, −4, −4, and −9 in bank order. The discovery-only policy therefore selected relation 0. Independent revalidation scored relation 0 four points below relation 1 over 48 matched rallies. The average difference was −.083 points per rally with a 90% lower bound of −.208. The gate kept relation 1. Adding all practice evidence to the long-running shared averages also kept relation 1. Those outcomes depend on play alone, with scored contact histories used only for retrospective metrics.

| Read-only policy on the archived Pong schedule | Changed action tapes | First returns | Later returns | Earned points |
| --- | ---: | ---: | ---: | ---: |
| Before three hall games | 0/180 | 129 | 195/206 | 41 |
| After three hall games | 156/180 | 126 | 188/215 | 36 |
| After adding the refresher scores to the carried totals | 156/180 | 126 | 188/215 | 36 |
| Confidence-gated | 156/180 | 126 | 188/215 | 36 |
| Discovery-only | 0/180 | 129 | 195/206 | 41 |

The discovery-only control recreated the entire archived point history. Its checkpoint differed from the carried checkpoint only in the numeric evidence gathered from the fresh starts. Independent public scores failed to confirm its ordering, so this practice budget yielded an exact recovery on the archived schedule and insufficient evidence for promotion.

## Return to the hall

The inherited and discovery-only checkpoints each played one fresh mixed-condition gantry game with the direct renderer. Both used the same installation, calibration, serve-seed sequence, and initial service memory. They kept relations 1 and 0 respectively throughout their games. The inherited player lost 10–12 with 23 legal returns over 23 played points. The discovery-only player lost 6–11 with 21 legal returns over 18 played points. On the 14 shared starts with the same server and physical condition, the discovery-only player made 14 legal returns versus 10 and won five points versus eight. This fixture preserved some receiving skill after the change but gave no evidence that replacing the shared ranking protects table-tennis scoring. Both games passed the command-tape replay checks.

## Reproduce

```bash
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt8_native_refresher /tmp/native-refresh.json.gz \
  /tmp/native-refreshed-player.json.gz --practice-checkpoint /tmp/native-practice-player.json.gz
for checkpoint in artifacts/tt8/after-three-games.json.gz /tmp/native-practice-player.json.gz; do
  python -m scripts.experimental.tt8_service_trial "/tmp/native-table-$(basename "$checkpoint")" \
    --resume "$checkpoint" --seed 327046119 --installation-seed 128754010 --limit 35 \
    --race-to 11 --no-compiled-geometry --first-server -1 --phases ordinary mixed soft delay
done
```

[Refresher and frozen evaluation](native-refresh.json.gz) · [Confidence-gated checkpoint](native-refreshed-player.json.gz) · [Discovery-only checkpoint](native-practice-player.json.gz) · [Inherited table game](native-table-frozen.json.gz) · [Discovery-only table game](native-table-practiced.json.gz)
