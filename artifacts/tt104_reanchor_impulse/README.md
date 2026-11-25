# Reanchor the paid canted-arm stroke at its physical hit plane

The preceding whole-waveform experiment transferred 11 known ordinary stock demonstrations to the canted SO-101 but restored none of four new ordinary misses. Its compact readout often retained the wrong starting reference and usually failed to attain the new ball. This follow-up keeps the same body, camera, arm, learned carry, table and standard continuous native opponent in simulation. It changes how the paid motor path is retimed and positioned.

The observer advances the visible ball in a separate private twin to the donor's actual learned contact plane. The controller retimes the path at that encounter, then converges from the current requested encoder position to the paid trajectory's **absolute** references. It relocates the observed face, changes the target normal for the difference in incoming impulse, and fits the new world face rate and translation jointly. The old command remains authoritative when its search finds a robust plan. A high post-bounce projection permits bounded alternate starts and speeds. These candidates reach the physical plant only if the independent camera-based twin accepts them. The gate uses *predicted* height, not the native realized height shown in this assessment.

## Release-disjoint matched control

| Actually received height group | Original | Absolute alignment |
|---|---:|---:|
| Five previously missed high routine balls | 0/5 | 0/5 |
| Two previously scored routine balls | 2/2 | 2/2 |
| Three low controls below 1.01 m | 1/3 | 1/3 |
| One 1.017 m transition ball | 0/1 | 1/1 |

On release 14540, leg 19, the learned camera estimated 1.055 m and issued a different motor stroke for the same past opponent strike and legal near bounce. The actual 1.017 m transition ball took a 52.14 mm broad face hit, crossed without touching the net, bounced legally on the far half and drew a real legal answer. The actual own count at the paired fork grew from 19 to 20. All 664 checked states of the changed suffix were CAD clear.

On 12116 an issued different 23.04 mm interior movement also paid, retaining the original result. The original successful low 4317 was left intact, with exactly the same joint commands, native events and score. Of the five higher misses, two never met the causal height or bounce gate. For the remaining three, 202 optical trajectories found no eligible return. One of nine separate unissued native diagnostics, on 3527, reached a 46.56 mm broad-face hit and landed an answered net cord under the standard referee. Its optical twin rejected it. This is a candidate, not a deployed gain. All 22 exact physical control, intervention and diagnostic suffixes were clear across 9,150 solver states. [Preassigned distinct cases](protocol.json) · [Native fork, visibility decision and motor tapes](held.json.gz)

## Whole points

The regular eight-point comparison uses the regime already fixed across the preceding experiment, with the same measured releases, opponent seeds and carried preparation. Its release IDs are disjoint from these matched cases and all TT98 paid donors. Unmodified baseline play is shared with TT98. Each variant independently earns online credit from its own actual returns. [Frozen paired schedule](../tt98_world_waveform/games.protocol.json)

The reanchored controller scored **32** own returns versus **44** without it. Seven points tied. The remaining point, release 10313, ended at 14 instead of 26. Its only issued change came at control tick 688, after six credited own returns. Both arms scored the same physical 1.088 m incoming ball and got a new legal reply. The changed stroke landed at x=.113 m and y=.203 m, versus x=.010 m and y=.414 m for the ordinary stroke. The actual later deliveries then differed. The reanchored game eventually failed on a new, low ball without touching it. A different credited carry also changed a motor command on the next point but did not change its score. All 73,624 solver samples across both arms were CAD clear, and every opponent draw used the original random law.

This limited result does not support promoting the correction. Landing an alternative on the current ball does not validate the future course it creates. [Whole-point scores, paired prefixes and CAD](whole-point-comparison.json) · [Exact local and next-reply effects](issued-effects.json) · [Reanchored full command and event tapes](transport.json.gz)

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt104_reanchor_impulse held
python -m scripts.experimental.tt104_reanchor_impulse games
python -m scripts.experimental.tt104_reanchor_impulse paired
python -m pytest -q tests/test_tt104_reanchor_impulse.py
```
