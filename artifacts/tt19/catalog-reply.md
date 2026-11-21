# Earlier planning for a second incoming ball

The two conditions used the same practiced opener, native wrist-plus-fixed RGB, strict score rule and three held game-start seeds. The intervention only changed the catalog policy that takes over after the opener observes an outgoing ball and recovers its posture. The existing policy used 2,048 dry motion tapes and began checking once the incoming ball passed y = −0.24 m. The intervention used 3,072 wider tapes and began checking at y = +0.56 m. The middle 99% of lateral blade coordinates in the dry tapes expanded from −262…+256 mm to −367…+363 mm. These changes were tested together. [Wider dry catalog](catalog-reply-wide.npz) · [Native baseline](match-full-from-wide.json.gz)

| Held game | Score in both conditions | Legal learner returns in both | Legal rival replies in both | Late and early reply plans |
|---:|---:|---:|---:|---:|
| 4531227 | 11–9 | 4 | 2 | 0 and 0 |
| 29175201 | 11–5 | 4 | 0 | 0 and 0 |
| 17634507 | 10–12 | 3 | 2 | 1 and 1 |

Both controllers finished with zero points containing two legal learner returns. Three of the four legal rival replies bounced near the lateral edge of the learner's half, from x = −0.41 m to −0.38 m, and drew no plan in either condition. On the remaining reply, the wider planner committed at tick 44, six frames earlier than the baseline. Both made contact at tick 55. The late shot bounced on its own half, and the earlier shot went long. The motion catalog offered few options for wide replies in this sample, and the reachable reply still ended in a fault despite the earlier start. [Paired native receipt](catalog-reply-trial.json.gz)

Opening actions matched exactly up to the first reply plan or point end on 55 of 58 shared points. Two more prefixes differed by at most 1.7 × 10⁻⁷ command units. The remaining opening differed after the previous point's divergent reply. Camera registration state persists between points. The baseline and candidate each replayed all 58 physical action tapes, including their original scores and contact events. The assay uses the full mini-table opening distribution and detailed arm geometry in the native renderer.

```bash
python -m scripts.experimental.tt19_catalog_reply_trial artifacts/tt19
```
