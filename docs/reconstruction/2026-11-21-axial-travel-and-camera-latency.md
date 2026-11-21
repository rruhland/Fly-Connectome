# Pong through traveling switches at 30 Hz

The practiced Pong player won **12 of 16 matches** through simulated traveling switches with 50 ms camera latency. It won 15 with zero transport delay. At 50 ms, the fresh Pong learner won three matches with the same arm practice and timing compensation. The practiced player also won three when it steered without timing compensation.

## Simulated arm and switch travel

This follow-up uses the trained visual Pong checkpoint and the native first-to-three game from the earlier arm study. Both the game and the servos run continuously at 120 Hz. The player issues three bounded relative joint commands. The proximal joints move the hand across the panel, and the third now pitches the 20 cm wrist link to lift or lower the fingertip. The three original angular speed limits remain 1.48, 1.63 and 1.93 radians per second.

The fingertip starts 9 mm above the panel. Each stationary button well has a circular cap with a 39 mm radius. Each cap starts flush and travels a seeded 3.6–5.2 mm along its vertical axis. To close the switch, the cap must move through 60–72% of its total stroke. It reopens when the spring returns it to 24–42% of that stroke. The fingertip can hover inside the cap footprint with zero input. A panel lip constrains lateral movement when the fingertip is below the surface. The switch can reopen as the arm retracts while its fingertip is still inside the footprint. The spring return uses a prescribed speed and the model contains hard position stops. Contact forces, switch bounce, real camera transport, and physical hardware are outside this simulation.

The controller sees rendered RGB marks in an overhead view and a magnified orthographic side view of the fingertip and caps. It estimates height and compression from vertical pixel displacements in the calibrated side view. It estimates horizontal image motion from issued commands, starting from 20 observations in a separate switch-free arm scene. It has approximate wrist timing and can associate cap pixels with button identities. It observes paddle movement after compressing each cap to learn the two signed effects independently. The harness alone sees the electrical threshold, native game state and actual contact. The old study describes what the reused Pong checkpoint learned from direct play. [Earlier arm experiment and Pong training](2026-11-21-pong-through-an-arm.md)

## Camera timing and matched games

The two arm views and the native Pong raster are captured synchronously every four native ticks. In the 50 ms conditions, they arrive six native ticks after capture. The latest joint command stays active between arrivals. New points keep the arm pose, switches, motor model and wiring belief. Game images captured during a finished point are discarded when they arrive. The compensated controller estimates current fingertip position from capture age and the joint commands it issued. It advances the visible paddle position and shortens the estimated time to interception by the frame age. The uncompensated controller tracks motion from capture-time observations and steers as if the latest pixels were current. Every match uses the same axial mechanism and no direct game-input access.

The confirmation uses 16 matched starts with balanced wiring, the native 0.35 opponent and a 55-second cap. Every condition receives the same start, physical panel and switch properties. The trained player starts from the archived visual crossing and shot-choice evidence. The fresh player gets the same scaffold and arm practice and starts with empty Pong evidence. Both continue learning. Win fractions count each scheduled match, including capped games. Earned goals are native points after at least one own return. A stationary paddle can return an in-range ball, so the disconnected matches provide that reference.

| Pong player and camera condition | Wins / losses / caps | Points for–against | Earned goals | Returns / incoming | Median game seconds |
|---|---:|---:|---:|---:|---:|
| Practiced, zero transport delay | 15 / 0 / 1 | 47–10 | 39 | 92/102 | 24.4 |
| Fresh, zero transport delay | 2 / 1 / 13 | 18–9 | 13 | 230/239 | 55.0 |
| Practiced, 50 ms with compensation | 12 / 2 / 2 | 40–24 | 30 | 114/138 | 30.4 |
| Fresh, 50 ms with compensation | 3 / 2 / 11 | 23–24 | 17 | 195/219 | 55.0 |
| Practiced, 50 ms without compensation | 3 / 13 / 0 | 20–42 | 11 | 32/74 | 10.4 |
| Practiced, 50 ms with disconnected output | 0 / 16 / 0 | 10–48 | 2 | 9/57 | 7.5 |

At 50 ms with compensation, prior Pong practice added 0.81 earned goals per match (paired 95% bootstrap interval 0.25 to 1.38) and 56.3 percentage points to the scheduled win rate (31.3 to 81.3). The fresh player returned a larger share of incoming balls and reached the time cap more often. With the practiced player, timing compensation added 1.19 earned goals per match at 50 ms (0.56 to 1.81). Relative to zero transport delay, the practiced player with compensation still earned 0.56 fewer goals per match (0.06 to 1.13 fewer). Each interval resamples the full matched starts.

Twelve opening balls were outside the stationary paddle’s reach. With 50 ms latency, the compensated practiced arm saved five, the uncompensated practiced arm saved two, and the disconnected arm saved zero. The fresh compensated player also saved five.

The compensated practiced arm correctly identified both switch effects by the end of all 16 matches. It closed a switch 1,016 times and reopened one 1,011 times. At every reopening, the fingertip was still inside the switch footprint. The disconnected arm physically compressed a switch on all 16 starts and produced zero game input and zero wiring assignments.

## Reproduction and audit

The compressed receipt includes all 96 matches, game and arm seeds, switch depths and thresholds, contact counts, score histories, camera delivery counts, and source hashes. The first scheduled match in each condition also has a joint-command tape and individual switch transitions. All six tapes reproduced the stored native scores, returns, fingertip motion, cap depth and switch transitions when replayed without a controller. [Match receipt and action tapes](../../artifacts/arm-pong/2026-11-21-axial-travel-latency.json.gz)

Run from the repository root with the project dependencies installed.

```sh
export PYTHONPATH=src:scripts/experimental
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/experimental/travel_button_matches.py artifacts/arm-pong/direct-pong-trained.json /tmp/axial-travel-latency.json.gz --workers 2
python -m pytest -q tests/test_travel_button.py tests/test_arm_plays_pong.py tests/test_arm_pong_arrival.py tests/test_arm_pong_camera_rate.py tests/test_arm_pong_calibration.py tests/test_arm_pong_independent_effects.py tests/test_arm_pong_timing.py tests/test_button_pong_skill.py tests/test_pong_button_panel.py
```

All 30 targeted tests passed with Python 3.12.14, NumPy 2.5.4 and PyTorch 2.14.1 CPU. They cover painted depth at the full wrist range, a nonconducting hover, cap depression, reopening while the tip stays within the cap footprint, the panel lip, 30 Hz capture with exact six-tick delivery lag, and disconnected action replay.
