# Removing prior SO-101 opening practice

Removing the practiced opening bank eliminated legal learner returns in both matched games. The generic bank served legally on every own-service attempt in the center game and still lost. [Matched game results](opening-practice-removal.json.gz)

This ablation compares the frozen center and full-course players to a fresh opening bank on the same native RGB game seed, 36325598. It retains their detector, calibrated views, public serving turn, bank lookup, carried ball law and racket response, factory motion catalogue, late-rally planner and strict physical scoring. The only change is the learned bank that supplies the opening motion for each role. Both conditions use zero held practice. The existing trained game tapes provide the paired trained result. [Sampled banks and complete tapes](opening-practice-removal.json.gz)

The fresh bank has 16 independent unpracticed proposals per role, drawn from the same zero-history proposal distribution used by the contextual learner. Each program gets the same neutral cue and prior success so the unchanged lookup selects the first random draw. These entries carry zero simulator interactions. This is a single generic draw per role, shared across the two feeders. The comparison measures the value of the practiced opening bank on these starts while retaining experience and engineering in the rest of the player.

| Feeder | Opening bank | Score | Legal serves | Legal returns | Points with two or more returns |
| --- | --- | ---: | ---: | ---: | ---: |
| Center | Practiced | 11–3 | 5/6 | 7 | 0 |
| Center | Fresh proposals | 7–11 | 8/8 | 0 | 0 |
| Full | Practiced | 11–9 | 8/10 | 6 | 0 |
| Full | Fresh proposals | 0–11 | 0/5 | 0 | 0 |

The center games shared their first 14 point seeds and serving turns. The full games shared their first 11. On the first two starts, both on rival serve, the center trained player won both and the full trained player won the first. The fresh branches lost both starts. The next two starts used learner serve. Both center branches won them. On the full feeder, the practiced branch won both and the fresh branch lost both. These are paired comparisons on one seed with previously completed trained games. All 63 trained and fresh tapes replayed under the same strict scorer. [Point-by-point receipt](opening-practice-removal.json.gz)

The stored center and full banks used 5,168 and 5,424 prior SO-101 practice and qualification points respectively. Their recorded control frames were 228,177 and 231,353. The held ablation charges zero new practice to both conditions and holds the shared factory catalogue and other policy priors fixed. [Audit and reproduction code](../../scripts/experimental/tt19_match_practice_ablation.py)

```bash
PYTHONPATH=src:scripts/experimental LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m scripts.experimental.tt19_match_practice_ablation
```
