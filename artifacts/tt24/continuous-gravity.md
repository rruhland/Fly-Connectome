# Wrist-only bouncing through changes of gravity

On twelve new stock SO-101 releases, seven stayed in play through five gravity changes and all 30 seconds. The other five ended before the first change at four seconds. The first registered release reached 93 consecutive rebounds with the camera-based controller. [Watch the full first drop](continuous-gravity-first.mp4)

Gravity took the values 1.00, 0.76, 1.25, 0.86, 1.12 and 1.00 times Earth gravity, changing every four seconds and staying at the last setting for the final ten seconds. Each change applied to both the ball and the arm. There were no resets, announcements, dropped balls or edits to the policy at the switches. The actuator weights were learned on other hinged arms from bounce rewards. Deployment used the existing wrist RGB and encoders. [Controller and perturbation](../../scripts/experimental/tt24_continuous_gravity.py) · [Motor training](tt24_donor.md)

| Condition on the same 12 releases | Mean longest consecutive | Maximum consecutive | Total qualifying rebounds | Full 30 seconds |
| --- | ---: | ---: | ---: | ---: |
| Constant gravity, original motor controller | 57.08 | 101 | 735 | 7/12 |
| Switching gravity, original controller | 48.08 | 99 | 717 | 7/12 |
| Switching gravity, acceleration estimated from wrist images | 49.25 | 99 | 722 | 7/12 |
| Switching gravity, actual gravity supplied as a diagnostic | 51.58 | 118 | 727 | 7/12 |
| Switching gravity, stationary paddle | 2.42 | 3 | 29 | 0/12 |

The adaptive treatment fits the curvature of short, visually supported ball trajectories, compares it with a previously measured nominal reference, and rescales the observed ball velocity before choosing joint commands. The underlying square-root relationship between acceleration and velocity is supplied. The median wrist estimates after the first second of each segment were 0.80, 1.30, 0.90 and 1.17 for actual changed gravities of 0.76, 1.25, 0.86 and 1.12. Most of the robustness came from the existing feedback controller. The normalization improved the summed longest streak from 577 to 591 without rescuing any of the five early losses. Every command tape reproduced its complete physical referee record. [Frozen mixed-gravity runs](continuous-gravity-frozen.json.gz) · [Online optical estimates and actions](continuous-gravity-optical.json.gz) · [Supplied-gravity comparison](continuous-gravity-oracle.json.gz) · [Constant-gravity comparator](continuous-gravity-constant.json.gz) · [Stationary control](continuous-gravity-stationary.json.gz)
