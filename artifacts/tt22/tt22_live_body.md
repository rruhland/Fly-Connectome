# Learning the motor wiring during a scored point

The clamped SO-101 started a scored receiving point with a hidden remap of its five arm command channels. During its first 10 control frames, it issued two `+0.45` rate commands to each public motor port in turn. The serve and referee kept advancing through these taps. Both branches used that prefix. The frozen branch issued the rally controller's raw commands throughout. The online branch could correct the port commands as soon as all five assignments passed its optical confidence test. The first point counted toward the score and the taps took place in its regular reset. [Scored experiment](../../scripts/experimental/tt22_live_body.py)

The correction assumed a signed permutation, an unchanged gripper, the known stock CAD model and standard reset pose, and the nominal wrist optics. During each tap, the fitter compared motion in unmarked wrist RGB with a separate nominal model seeded independently of the target. It computed motion from tiled pixel differences with orange ball regions masked. The model replayed earlier taps using assignments inferred so far, then tried both signs for each remaining canonical axis. The fitter accepted a complete map when every winning cost was at most 0.35 and every runner-up cost was at least 0.03 higher. Its observations were wrist images and requested commands. Fixed stereo supplied ball observations to the rally controller, whose command-driven simulation incorporated the inferred earlier movements before the first corrected command.

The 12 matched launches follow the original predefined wider receiving course, with the learned five-axis partner serving every point. The robot resets at ordinary point boundaries and its wiring persists across them. [Original course](paired-course.json.gz)

## Acquisition

The wrist learner applied the exact correction at tick 10, after 0.33 seconds of simulated interaction. The fit used six wrist images and compared 30 signed canonical branches spanning 140 nominal model frames. Matching took 2.17 seconds of wall time and the lowest runner-up margin was 0.147. The fitted public-to-canonical order was `[0, 2, 3, 4, 1]` with signs `[−, −, +, −, +]`. The corrected arm landed two legal returns in that same point, at frames 39 and 75. Both branches lost the point and the frozen arm made zero legal returns.

## Results for all 12 pairs

| Outcome | Frozen public commands | Online wrist correction |
| --- | ---: | ---: |
| SO-101 legal returns | 0 | 33 |
| Far partner legal returns | 0 | 33 |
| Longest alternating exchange | 0 | 14 |
| Launches reaching 10 alternating returns | 0/12 | 2/12 |
| Launches reaching 20 alternating returns | 0/12 | 0/12 |
| Final score, SO-101 to partner | 0–12 | 0–12 |
| Frames where realized rates differed from controller requests | 456/493 | 10/1,615 |

Corrected lengths in launch order were `4, 2, 10, 14, 8, 0, 2, 8, 6, 6, 2, 4`. The frozen lengths were all zero. The final 11 points of each branch matched the corresponding earlier dedicated-calibration comparison in controller requests, both robots' physical commands, scores, referee events and stereo image digests. The first point establishes acquisition on a single simulated signed remap with a matched nominal CAD model and known wrist optics. [Prior scored comparison](tt22_learned_body_held.json.gz)

The completed receipt preserves the first seven scored traces from the interrupted run. At the next scheduled point, the continuation restored both scores and control clocks along with the online branch's fitted map. All 24 tapes reproduced their public-to-physical commands, complete stereo digests, point durations, scores and referee events. The six wrist-image hashes matched in both first points, and the refit reproduced all candidate costs and the accepted map. [Complete score and replay receipt](tt22_live_body.json.gz)

## Reproduce

```sh
tt22_native=$(mktemp -d)
c++ -O3 -std=c++17 -shared -fPIC scripts/experimental/tt3_components.cpp -o "$tt22_native/components.so"
export TT3_COMPONENT_LIBRARY="$tt22_native/components.so"
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBGL_ALWAYS_SOFTWARE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt22_live_body
python -m scripts.experimental.tt22_live_body --phase verify
```

Add `--resume` to the course command to continue from its last saved point. Use `--output` to supply a custom receipt path for either command.
