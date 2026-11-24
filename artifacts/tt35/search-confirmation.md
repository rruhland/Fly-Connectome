# Frozen joint adjustment on 24 confirmation releases

The frozen learned adjustment landed **9 near-arm returns versus 7** for the exact nominal raw policy. It made four later replies versus one for nominal, while returning five opening replies versus six. The strongest point went from three to nine alternating legal returns. The same checkpoint from the earlier search was carried into this assessment. [Paired receipt](search-confirmation-results.json) · [Frozen checkpoint](search-checkpoint.json) · [Original learning study](search.md)

| Controller | First near replies landed | Later near replies landed | All near returns | Alternating legal returns |
| --- | ---: | ---: | ---: | ---: |
| Nominal raw policy | 6 | 1 | 7 | 38 |
| Frozen learned adjustment | 5 | 4 | 9 | 41 |

The 24 release IDs were disjoint from the policy demonstrations, search practice and both earlier held rosters. The roster included nine left, six middle and nine right first deliveries. Far contact and near bounce matched exactly on every point. The ball crossing the receiving line also matched exactly on 23 points. On 10809 the two line samples differed by at most 3.3 × 10⁻¹⁶ m in position and 2.0 × 10⁻¹⁵ m/s in velocity, and both arms missed. [Confirmation roster](confirmation-plan.json.gz) · [Nominal commands](confirm-nominal.json.gz) · [Frozen-search commands](search-held-confirmation.json.gz)

The gain concentrated on 5530. Both controllers landed its first left reply. The learned controller then landed three more left replies, taking the point from one to four near returns. It lost the opening on 12166. The score tied on the other 22 releases, including 14832, where both landed two near replies. The three held cohorts together produced 19 near returns with the learned adjustment versus 16 with nominal, including six later replies versus two. These results show longer rallies on a few releases and a small drop in the number of openings returned. [Per-release differences](search-confirmation-results.json) · [Earlier held results](search-results.json)

The learned confirmation run used 43.95 process CPU seconds. Action-only replay matched the referee events of all five scoring games. Full-CAD replay found zero penetrations across 9,911 solver steps, with 18.6 mm minimum clearance. [Command-only audit](search-held-confirmation-audit.json)

## Reproduce

Use the pinned recorded launches and the saved confirmation plan and nominal control. The search checkpoint can be loaded directly.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt35_search assess /tmp/tt25-rallies.json confirmation
python -m scripts.experimental.tt35_search summary --confirmation
python -m scripts.experimental.tt35_search audit artifacts/tt35/search-held-confirmation.json.gz artifacts/tt35/search-held-confirmation-audit.json
```
