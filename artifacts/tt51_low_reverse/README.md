# RGB-only low teacher reverse course

The two physically rewarded low teachers are converted to motor templates using
only recorded RGB ball estimates, own joint encoders, issued references and
installed paddle geometry. Scalar physical legal-return rewards authorize them.
Neither native contact coordinates nor native receiving-line coordinates enter
template extraction or retrieval. Three focused tests cover this boundary.

| Declared physical practice | Cases | Cold | Frozen two teachers | Reward-growing |
| --- | ---: | ---: | ---: | ---: |
| Exact teacher reproduction | 2 | 0 | 2 | 2 |
| New goal neighbors | 8 | 0 | 7 | 7 |
| Six control ticks earlier, new goals | 4 | 0 | 2 | 2 |
| Untouched broad reply streams, late handoff | 4 | 0 | 0 | 0 |

Every counted return was also physically answered by the opponent. Paired incoming
far-contact and near-bounce records matched in all 18 triples. The six-tick earlier
handoff changes the observed arm by only .147/.221 maximum joint radians; this
does not establish a normal-start skill. Nine newly earned templates were retained,
but aggregation made no causal difference relative to the original two templates.
The independent broad draws failed. No official full-game promotion is claimed.

Practice scaffolding physically replays an actual teacher command prefix to a
declared handoff, then uses ordinary RGB/encoder control. Neighbor goals change
the actual far paddle's goal; no ball or arm teleport is used. Prefix 3999 is paid
once; each subsequent practice prefix and each of three actual suffixes is paid.
Total: 9062 actual control ticks, 632.016 seconds wall and 164.352 seconds CPU
(last completed-row lower bounds). The fixed bound was 900 seconds/12000 ticks.
Suffix CAD clearance is recorded per branch. The altered prep prefix was not
independently CAD-audited, so this artifact does not certify an entire new course.

`course.json.gz` retains all actions, RGB/body observations, actual offers,
positive and negative outcomes. `executed_source.py` is the exact frozen source
that ran. Its final hash-report expression failed after all 18 rows were saved
because it passed a string instead of a Path. `tt51_low_reverse_report.py`
reconstructed only final metadata and the already earned template list, with
zero physical steps; this repair and its source hash are recorded in the artifact.
The current runnable source fixes that type error and accepts JSON template rows.

Reproduce the native course with the documented single-worker EGL environment:

```sh
PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1 /workspace/Fly-Connectome/.venv/bin/python -m scripts.experimental.tt51_low_reverse_curriculum --out artifacts/tt51_low_reverse/reproduction.json.gz
```

`RGBTaught(rows).proposals(forecast,path,x)` is the shared candidate API. Rows
contain `p=[x,z,vx,vy,vz]`, RGB-derived `depth`, relative phase and actual next
reference arrays. Candidate acceptance still requires the normal optical/body
forward checks; template identity is not a live input.
