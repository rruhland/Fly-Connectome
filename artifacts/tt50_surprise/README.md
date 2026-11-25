# Scored joint prefixes when the native legal screen is empty

The native joint search can report no legal trial at a control tick even though the real arm later scores on that same incoming ball. Most examples are recovery opportunities, not cases where the arm struck without any later viable physics hypothesis. In the 59 compatible direct-joint archives, after excluding every release from the official season, 499 actually won returns across 109 releases had a suitable earlier empty search. A selected note had to precede actual contact by at least five and at most 21 control ticks so the first five previously issued incremental rates could be real preparation. Four separately enumerated literal cases had an empty search within the last five ticks of contact in the defined y window. They are not pooled as stronger proof of a long plan. The discovery also found 598 qualifying unreturned offers across 153 releases. Identical native incoming evidence and issued motion prefixes count once across mirrored tapes. [Frozen discovery and roster](plan.json)

The capture replays the original bounded rates on the original articulated arm and continuously reacting opponent. It recomputes RGB and encoder values and keeps a row only when the archived physical events replay exactly through the actual score or point fault. The capture also checks the current optical position against the saved no-legal note. It records missing camera support or replay failures. Data are selected by release diversity before looking at any new physical results.

All 146 captured trajectories passed the physical replay and causal observation checks. They cover 113 releases, with 64 physically scored and 82 unreturned. Every one of the 64 scored preparations had a later legal native search. Their median selected empty search was 16 control ticks before contact. The fitted action association therefore predicts recovery after preparation, not the correctness of a never-validated strike. Leaving entire releases out gave Brier .2488 for prevalence, .0614 for current and recent sensory state alone, and .0565 after conditioning on the actual five issued rates. This modest action signal is observational. [Reconstructed inputs and commands](captured.json.gz) · [Grouped validation](validation.json)

The actor looks at current RGB flight, four ticks of causal history, current encoder position and issued reference. At a normal replan where the original search is empty, it may recall five original bounded rates from a local scored movement. Both an ordinary control and nearest physically positive state-only recall are compared to action-conditioned recall. The latter needs independent successful supporting movements and a higher local physical reward association for that action than for the state alone. Neither condition borrows based on the measured launch ID or the actual future. The original search still updates while the rates are in force, but does not veto the short supported movement it did not propose.

The held assay uses exact whole openings on the ordinary frozen history/risk controller, with the original season checkpoint and unchanged broad draws. The same recorded release ID can have a weak opening on one independent reply stream and a scored opening on another. The separate evaluation roster freezes the specific original failed assignment for each four weak releases and four ordinary scoring controls, before trying any recalled action. Its selection does not change which release IDs are withheld from training. The controller follows the original whole native game until the first legal arm return or actual terminal fault. This tests a receiving intervention on the exact physical offer. It does not establish rally-length effects. [Exact held assignments](held-roster.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt50_surprise_data plan
python -m scripts.experimental.tt50_surprise_data capture
python -m scripts.experimental.tt50_surprise_actor plan-held
python -m scripts.experimental.tt50_surprise_actor validate
python -m scripts.experimental.tt50_surprise_actor run
python -m scripts.experimental.tt50_surprise_report --audit
python -m pytest -q tests/test_tt50_surprise.py
```
