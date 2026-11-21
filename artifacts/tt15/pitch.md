# One RGB pitch probe at the return to table tennis

The invisible-source search recovered the correspondence of all translational motor ports on four installations. Three of those maps had the wrong sign on the weakest translation channel. One short first-person stereo probe corrected all three and kept the already-correct map. The corrected players won three of four complete games and landed 61 legal returns, compared with one win and 20 returns for search alone. A player given the identical probe without the search-derived correspondence found the probed axis and sign on all four fixtures, then lost all four games with zero legal returns. [Full paired audit](pitch-comparison.json.gz)

## What the probe observed

The carried source map's Jacobian identified the original channel with the smallest positional response. Its norm was about 5.1 mm per unit on the gantry and 3.1 mm per unit on the five-axis arm. The same channel had a fitted pitch response of 0.18 and 0.16 rad per unit, respectively. The search-derived correspondence mapped that source axis to one public port per fixture. One positive motion on that port used three commands of 0.70. Two preceding zero commands and seven following zero commands measured the onset and settling.

Each player used 12 commands, 13 stereo pairs and 0.40 simulated seconds for the transition. The resident tracker returned pitch on all 13 pairs in each fixture. The acquisition executed 48 commands and collected 52 pairs over the four fixtures. A receipt serialization error on the first execution required repeating the same deterministic tapes, for 96 commands and 104 pairs in total. Each fitted player consumed observations from one tape. [Probe implementation](../../scripts/experimental/tt15_pitch_probe.py)

The sign decision compared the centered pitch error under the existing search key and the key with just this sign reversed. It flipped only with at least five pitch observations, at least 0.06 rad of observed motion and an error reduction greater than 0.025 rad. The post-play audit consulted the evaluator wiring to verify the result.

| Fixture | Probed public port, zero-based | Pitch error with old sign | Pitch error with opposite sign | Decision |
| --- | ---: | ---: | ---: | --- |
| Gantry 0 | 3 | 0.012 rad | 0.304 rad | Keep |
| Gantry 1 | 0 | 0.304 rad | 0.012 rad | Flip |
| Five-axis 0 | 4 | 0.269 rad | 0.031 rad | Flip |
| Five-axis 1 | 1 | 0.263 rad | 0.026 rad | Flip |

Centered position error also favored the correct sign in three tests. On five-axis 1 it favored the wrong sign, with 12.7 mm error for that sign and 16.1 mm for the correct sign. [Gantry 0 probe](pitch-gantry-0-fit.json.gz) · [Gantry 1 probe](pitch-gantry-1-fit.json.gz) · [Five-axis 0 probe](pitch-five-0-fit.json.gz) · [Five-axis 1 probe](pitch-five-1-fit.json.gz)

## The matched-budget control

The probe-only control received exactly the same public command history, observation times and tracked RGB positions and pitch. Its prior was the native table motor map. It compared every candidate source axis and sign for the activated port using centered spatial error and pitch error weighted at 0.12 m/rad. It retained the original assignments of every unobserved port except the one displaced by the newly identified axis, and retained their original signs. Among losses within 1 mm of the minimum it preferred the original axis, then the original sign, then the lowest source-axis index. All four choices were unique within that tolerance.

The probe-only control selected the correct source axis and sign for the one activated port in every case. The rest of the resulting correspondence still differed from the physical wiring on all four. None of its full keys coincided with an archived comparator, so all four control games were played. [Gantry 0 control fit](pitch-gantry-0-control-fit.json.gz) · [Gantry 1 control fit](pitch-gantry-1-control-fit.json.gz) · [Five-axis 0 control fit](pitch-five-0-control-fit.json.gz) · [Five-axis 1 control fit](pitch-five-1-control-fit.json.gz)

## Full-game qualification

The three branches used the same initial portable player, installation and seeded point and disturbance schedules. The game runner froze each selected signed motor map and kept ordinary stroke and point-reward updates active. All 106 shared starts across the paired comparisons matched the original phase and service role. Keeping the gantry 0 map reproduced all 21 original point command tapes.

| Fixture | Search alone | Search and pitch | Probe alone | Legal returns, search / combined / probe |
| --- | --- | --- | --- | ---: |
| Gantry 0 | 11–9 | 11–9 | 0–11 | 17 / 17 / 0 |
| Gantry 1 | 0–11 | 9–11 | 0–11 | 0 / 21 / 0 |
| Five-axis 0 | 2–11 | 11–9 | 0–11 | 1 / 14 / 0 |
| Five-axis 1 | 5–11 | 11–9 | 1–11 | 2 / 9 / 0 |
| **Total** | **1 win** | **3 wins** | **0 wins** | **20 / 61 / 0** |

The combined player landed 34 legal own serves and had 18 sustained points, compared with 15 serves and six sustained points for search alone. The probe-only control landed one legal own serve. The audit reconstructed both selections from the saved optical values, checked the frozen map in each endpoint and replayed all 188 game point tapes. [Machine-readable audit](pitch-comparison.json.gz) · [Gantry corrected endpoint](pitch-gantry-0-player.json.gz) · [Five-axis corrected endpoint](pitch-five-1-player.json.gz)

## Reproduction

Run from the repository root with the [hall environment](../tt6/README.md).

```sh
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TT3_COMPONENT_LIBRARY=/tmp/tt3-components-final.so
python=/workspace/Fly-Connectome/.venv/bin/python
for site in gantry five; do
  for index in 0 1; do
    "$python" -m scripts.experimental.tt15_pitch_probe artifacts/tt15 "$site" "$index" probe
    "$python" -m scripts.experimental.tt15_pitch_probe artifacts/tt15 "$site" "$index" qualify
    "$python" -m scripts.experimental.tt15_pitch_probe artifacts/tt15 "$site" "$index" control
  done
done
"$python" -m scripts.experimental.tt15_pitch_report artifacts/tt15 --replay
```
