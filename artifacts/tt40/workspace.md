# A longer recovery window for low-ball workspace

The stock SO-101 chart starts close to its shoulder and elbow limits. The latest low-ball test asks whether moving out of that fold during the previous outward flight can do more than choosing a better swing after the next ball appears. The preparation is a joint command to the real simulated arm, not a replacement or reset of its posture.

The calibration sampled 75 ball sites across both wings and the center, five depths, and five heights. Each site allowed three practical face angles. A pose had to keep its blade within 47 mm of the ball-centered face target, within 23 mm in the face-normal direction, and pass full-CAD geometry clearance. The comparison contrasts one IK solve seeded from the nominal stance with an alternate, multi-seed solve that also keeps the shoulder at or above −1.58 rad. This is a finite geometric sample, not a proof over all continuous poses or a score forecast.

Among 45 low sites, 13 were accessible to the nominal-seeded solve and 10 to the untucked solve. Eight overlapped, so the untucked solution gained two and lost five. The only two new low sites were the outer wings near the endline at z = 0.99 m. Both called for shoulder pan of roughly ±1.8 rad and over 2.1 rad of elbow movement from the common hub, so the added static poses are not automatically useful late. Neither solve reached any of the 15 sites at 0.89 m ball height. At 0.94 m the sampled center sites also remained out of reach for the untucked branch. Among 30 high sites the respective counts were 22 and 23. The chosen connected hub shortened the median maximum single-joint travel to a feasible pose from 0.947 to 0.564 rad for low sites and from 0.997 to 0.711 rad for high sites. The expected benefit is more time to reach existing useful poses, not new access to the full low band. [Workspace atlas](workspace-atlas.json)

The live diagnostic pairs replay the exact archived motor and physical prefix after a previously legal near stroke. Once RGB confirms an outbound ball, the preparation arm drives to the hub, ending that request when the camera sees a new incoming ball. The stock control retains its chart preparation. Both then use the same local pose-and-velocity search on their own independent RGB/encoder-driven model, and execute any shortlisted action on the physical simulator. The diagnoses preserve the opponent's random state and the requested placement, and stop after the tested stroke and any legal reply. The failures and high control were selected offline, so these are not held whole-policy results.

## Physical result

On the center-low release 15294, the new controller actually moved for 35 control ticks before seeing the incoming ball. The measured shoulder, elbow and wrist flex changed from approximately (−1.746, −1.690, 1.587) to (−1.554, −1.344, 0.600) rad at the first incoming observation. The physical incoming events were identical in both variants. At three subsequent optical decision times, the search found no acceptable contact pose from either starting position. Stock grazed the ball with a backward departure and the untucked fallback missed. Both naturally ended at 13 arm returns. [Low physical pair](workspace-low.json.gz)

The 3920 high control preserved the ability to return a higher ball. Its tested actual contact was at roughly 1.11 m height. After 46 outbound preparation ticks the alternate really attained the new shoulder, elbow and wrist configuration. From the same physical incoming ball, the planner found 22 nominally legal schedules in stock and 47 from the prepared configuration, out of 112 candidates each. Both made the tested legal return and received a legal opponent reply, reaching five arm returns and eleven alternating exchanges. The selected high stroke was allowed to move back toward the stock joint branch, which protects high reach. An exactly replayed suffix resolved each airborne ball and its reply, rather than treating an in-flight ball as a failed stroke. A full-CAD native replay reproduced both physical histories and checked 4,328 stock and 4,304 alternate solver states, with zero penetration in both. [Bounded high receipt](workspace-high.json.gz) · [Resolved high physical pair](workspace-high-resolved.json.gz) · [High CAD replay](workspace-high-cad.json)

This establishes that the body can attain a less folded joint configuration using the outbound interval and still hit a high ball. It does not establish a useful broad low-ball controller. The geometry sample and physical low case point to the blade-face and height envelope at center-low, not just the time necessary to move out of the stock fold. A future test should select incidence and contact-depth families within the actual reachable envelope, and preserve higher motions independently. The unmodified plant and finite solver scenarios do not justify a universal reachability limit.

## Reproduction

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt40_workspace atlas artifacts/tt40/workspace-atlas.json
python -m scripts.experimental.tt40_workspace low artifacts/tt40/workspace-low.json.gz --atlas artifacts/tt40/workspace-atlas.json
python -m scripts.experimental.tt40_workspace high artifacts/tt40/workspace-high.json.gz --atlas artifacts/tt40/workspace-atlas.json
python -m scripts.experimental.tt40_workspace finish RESOLVED --saved BOUNDED
python -m scripts.experimental.tt40_workspace clearance CAD_RECEIPT --saved RESOLVED
```
