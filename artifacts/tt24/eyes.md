# Learning a correspondence between wrist and outside views

A single-camera policy uses recent RGB positions, measured joint angles and its own issued commands to choose joint rates from the first frame. The camera providing deployment images is the only live camera the near learner sees. The full fit uses twelve earlier motor demonstration clips from five releases, totaling 1,999 frame-level examples. A separate transferred fit uses 1,684 other-camera motor labels and 174 simultaneous wrist/outside image pairs from one additional clip to learn a visual correspondence. The correspondence fit uses the pair of observed tracks and measured joint state. The near arm receives zero teacher commands during assessment.

| Legal opening returns on nine held archive releases | Wrist | Outside |
| --- | ---: | ---: |
| One local motor demonstration | 1/9 | 1/9 |
| All twelve local motor demonstrations | 0/9 | 4/9 |
| Other camera's experience + one paired clip | 4/9 | 3/9 |
| Full policy with six-frame visual delay | 0/9 | 0/9 |

The outside-only full pupil earned a second return on 2/9. The transferred wrist pupil earned the opening on 1/9 with scrambled training correspondence and 0/9 with delayed live images. On **24 fresh, unselected serves** it earned the opening on **1/24**, versus 0/24 one local clip, 0/24 with scrambled correspondence and 0/24 with delayed vision. It earned zero second returns on this fresh set. The original wrist gain was real on those releases but generalized poorly. [Single-camera fit](../../scripts/experimental/tt24_rally_eye.py) · [Wrist](rally-eye-wrist.json.gz) · [Outside](rally-eye-left.json.gz) · [Fresh](rally-eye-fresh.json.gz)

## Adding angular size to the view estimate

A separate regression takes the moving silhouette's angular center and area-derived size plus the five joint angles. It fits the other camera's pixel position from simultaneous images. No camera pose, ball diameter or world position is supplied. The direct motor readout was learned from 1,684 earlier outside-camera motor samples on other release seeds. In leave-one-release-out optical prediction the median error on paired, visible examples ranged from 2.1 to 3.3 pixels at the common 480-pixel reference width. Predicting a constant gave 18.8 to 26.2 pixels. Only 665 of 1,999 frames had valid paired detections.

On sixteen new starts, estimating the outside view from a single 174-frame paired clip (60 usable examples) earned 6/16 first and 3/16 second returns with wrist-only live control. The full optical mapping earned 5/16 and 2/16, delayed full mapping 2/16 and 1/16, and shuffled target correspondence 1/16 and 0/16. This shows some correspondence learning, with substantial failures in turning a precise visible-frame estimate into a sustained rally. [Physical scores and optical fold errors](rally-eye-depth.json.gz) · [Mapping and deployment](../../scripts/experimental/tt24_eye_depth.py)
