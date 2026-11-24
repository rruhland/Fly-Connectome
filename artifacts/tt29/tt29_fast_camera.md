# Native RGB timing with the faster physical partner

All six paired live runs completed four alternating regulation returns. On each measured opening, the original loop, native detector with native flight forecast, and packed stereo produced identical near-motor command and referee event digests. Each run recomputed near commands from 384 × 288 external stereo frames at 60 Hz and used the same fast physical far carriage. The opening launch was excluded. [Matched live timings and digests](tt29_fast_camera.json.gz)

| CPU milliseconds per simulated second | Original | Native detector and flight forecast | Packed stereo plus native kernels |
| --- | ---: | ---: | ---: |
| RGB rendering | 1,987 | 2,143 | 1,981 |
| Orange-component detection | 102 | 28 | 26 |
| Other optical estimation | 72 | 89 | 70 |
| Near command generation and simulated stroke search | 707 | 575 | 624 |
| Physical solver and fast far partner | 252 | 286 | 258 |
| Total | **3,125** | **3,128** | **2,968** |

The values weight the two measured starts by their simulated durations. The detector cut CPU cost by about three quarters, and running the existing isolated forecast through its native kernel lowered near-command cost. Rendering was the largest cost. At this resolution and capture rate the packed branch consumed about three CPU seconds per second of play on the software renderer. The shared environment took about 11 and 16 seconds of wall time per simulated second on the two packed runs. Compilation and held-image validation are outside the timing window. [Profiling code](../../scripts/experimental/tt29_fast_camera.py)

The native RGB detector reproduced all 556 held component measurements from 278 native images across six measured launch trajectories. Full packed stereo also reproduced all 556 measured components. Its background shading differs from the separately rendered images by a mean 1.92 channel levels on the 0–255 scale. The policy still sees ordinary RGB in both paths. [Detector and renderer validation](tt29_fast_camera.json.gz) · [Native RGB component kernel](../../scripts/experimental/tt29_fast_camera.cpp)

## Static color and depth cache

The fixed-camera renderer cached the static color and depth buffers once, then composited native dynamic RGB against them on every exposure. Depth stayed inside the rendering pipeline. The pupil received the resulting stereo RGB frames. The cache is defined for fixed camera poses and a fixed background. Transparent surfaces can shade differently in a separate pass. All 268 held ball measurements on 134 frames matched the original detector exactly. [Static cache implementation](../../scripts/experimental/tt29_fast_camera.py) · [Held image validation](tt29_fast_camera_cache.json.gz)

| Launch 7271, CPU milliseconds per simulated second | Original | Cached stereo plus native kernels |
| --- | ---: | ---: |
| RGB rendering | 1,952 | 1,031 |
| Orange-component detection | 101 | 29 |
| Other optical estimation | 62 | 67 |
| Near command generation and simulated stroke search | 196 | 96 |
| Physical solver and fast far partner | 254 | 242 |
| Total | **2,569** | **1,471** |

Both loops produced four alternating legal returns on launch 7271 with exactly the same commands and native events over 3.48 simulated seconds. The cache reduced total CPU by 43%. It used 1.47 CPU seconds per simulated second at this resolution and 60 Hz. Elapsed wall time in the shared environment fell from 11.64 to 7.04 seconds per simulated second. The live cache comparison covers one opening. [Actual closed-loop cache timing](tt29_fast_camera_cache.json.gz)
