# Discovering the controlled paddle in rotated Pong

The controller starts with the sealed state from the first continuous Pong course. It retains the visual association model, motor and timing belief, interception relation, and score-placement state. A camera adapter isolates the moving silhouettes and supplies explicit public track IDs to the existing relation. This replaces the assumption that exactly three observed objects can be ranked by position to bind the paddle and ball. [Post-course checkpoint](../../artifacts/m1b/2026-11-19-pong-after-mixed-course.json) · [Camera adapter](../../scripts/experimental/pong_role_transfer.py)

The adapter segments the binary camera image and considers elongated silhouettes near image edges as paddle candidates. A compact silhouette that visibly moves supplies the ball candidate. The adapter changes its requested buttons and measures movement along each candidate rail. With two rails in view it looks for motion reversing after the request reverses. A sole rail can bind after the initial pulse moves it at least half a pixel. The choice of quarter-turn uses the direction and side of the detected rail. The same frozen vision model tracks the isolated pixels in their canonical orientation. The role decision uses binary frames and requested buttons exclusively.

Frozen role inference keeps the ordinary left-paddle geometry fixed. Motor and timing updates run in both active arms. In the online arm, a carried rail prior is discarded when the first three frames of a new scene lack a matching silhouette. This check can detect a rail that moves from the left to the bottom. A similarly shaped object at the same rail can still make a carried prior ambiguous. The adapter supports quarter-turns and uses shape and motion priors tailored to a visible paddle and ball.

## Matched evaluation

Each of 24 independently prepared starts is played through four camera orientations with four static distractors. The camera rotates while the underlying game, paddle forces, and button map stay fixed. The same starts are played by a passive paddle and by the controller with its role geometry fixed to ordinary Pong. The historical controller also plays the unrotated starts without distractors. Each trial has the 0.65-speed opponent and an eight-second limit. [Evaluation runner](../../scripts/experimental/run_pong_role_transfer.py)

The evaluator counts crossings of the real paddle collision plane and checks that the first incoming trajectory is identical across each matched triplet. It checks both published positions against the simulated objects at the first policy decision and every tenth camera frame, with a two-pixel tolerance. Unavailable and occluded samples are counted separately from errors. Role-bound play forwards only the sign of cumulative visible reward at the end of a life to the inherited score policy. Physics hit counts and true positions are reserved for evaluation.

| Camera rotation | Correct rails | Correct first pairs | Opening returns, online | Opening returns, passive | Opening returns, frozen | Later returns, online |
|---|---:|---:|---:|---:|---:|---:|
| 90° | 24/24 | 24/24 | 16/24 | 8/24 | 8/24 | 15/16 |
| 180° | 24/24 | 22/24 | 16/24 | 8/24 | 1/24 | 15/16 |
| 270° | 24/24 | 19/24 | 15/24 | 8/24 | 8/24 | 14/15 |
| All rotations above | 72/72 | 65/72 | 47/72 | 24/72 | 17/72 | 44/47 |

Median time to identify the controlled rail was 0.333 seconds in all three rotated conditions. Of 1,378 sampled frames with both roles visible, 1,352 identified both within the two-pixel tolerance. An additional 650 sampled frames lacked one or both visible tracks, including the time spent probing. The matched 95% bootstrap intervals for the online arm's opening-return gain over passive were 12.5 to 54.2 percentage points at 90°, 8.3 to 58.3 at 180°, and 4.2 to 54.2 at 270°.

On the original orientation, the inherited controller without clutter and the adapter with a fixed correct rail both saved 24/24 openings. They saved 47/48 and 48/48 total incoming balls, respectively. Learning the original rail from scratch took the same median 0.333 seconds and saved 17/24 openings and 33/41 total balls. The first probe costs time even in the original orientation. No arm scored after an own return against the strong opponent on these starts.

The candidate was fixed at `64cd0f0` before the independent schedule was disclosed. The archive contains the summary, per-start results with matched indices, selected role IDs and errors, physical contact clocks, action hashes, and file checksums. [Evaluation receipts](../../artifacts/m2/2026-11-20-rotated-pong-role-transfer.zip)
