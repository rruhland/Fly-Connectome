# Learning direct movement on the canted stock SO-101

TT84 verifies a different supported installation of the same table-tennis
arm. TT90 keeps that exact registered geometry, ready encoders, RGB rig,
normal command envelope, continuous reactive opponent and referee. It
does not use the TT84 demonstration motions, preliminary points, design
archive or any TT80 release as training. This is not a qualification
claim for the unchanged centered original.

The release shuffle has no outcome input. Forty distinct recorded
release states supply normal uninterrupted practice with live reward
credit. Twenty other releases are held for complete physical points.
Forty neutral scouts from a third, separate pool fixed eight low arrivals
and four ordinary counterparts. Seven of those low scout arrivals are
below 1 m. The scout determines the test stratum only. It supplies no
future or truth signal to the acting pupil.

## Acquisition and controls

The installed registered controller directly earns its ordinary movement
fragments across practice. TT90 logs the same causal stereo estimate,
encoder poses, starting references and exact issued targets that were
available in the original loop. Load reconstruction checks them against
the actual bounded commands. An offline practice coach separately
resamples at most 16 lost low, wing or other native deliveries from the
body already attained at far separation. It saves an RGB/encoder lesson
only for an independently replayed, nonpenetrating legally returned
same-ball movement with a real far answer. Exact coach state never
enters a deployment feature.

The inherited registered policy has a FIFO movement bank. A second
body-specific archive retains rewarded movements across practice and
does not replace regions just because there were more recent scores
elsewhere. A proposal needs at least two distinct supportive releases
in the local optical/encoder neighborhood. It uses a bounded short phase
alignment and blends only mutually compatible joint paths. At most two
slots replace the least preferred existing movement seeds. No trial
budget is added and the original search still authorizes commands.

The frozen comparison separates the original registered foundation,
the ordinary FIFO after mounted practice, and the extra body archive
with the same FIFO. The hard-offer control additionally shuffles action
labels while preserving the search inputs, capacity and path inventory.
Every assigned full point counts. Hard first-offer attribution also
requires the same real far strike and learner-side bounce, with a CAD
replay of the complete issued tape.

[Frozen full roster](protocol.json) · [Neutral hard roster](hard.protocol.json)

## First exposure checkpoint

The first 16 live practice points produced 39 native returns. The
largest uninterrupted point scored 14 and passed exact event and CAD
replay. Two recorded openings yielded optical- and encoder-controlled
returns below 1 m before any exact-state coaching. On release 6737 the
0.9834 m ball hit the negative face at 55.72 mm radius. On 8129 the
0.9853 m ball hit at 74.91 mm. Neither tape had CAD penetration. These
are practice results, not a matched gain from the new library.

[16-point issued tape](practice-dose16.json.gz) · [Earned motor checkpoint](practice-dose16.npz) · [Exact physical replay](practice-dose16-audit.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt90_installation_learning plan
python -m scripts.experimental.tt90_installation_learning hard-plan
python -m scripts.experimental.tt90_installation_learning practice --n 40
python -m scripts.experimental.tt90_installation_learning coach
python -m scripts.experimental.tt90_installation_learning fit
python -m scripts.experimental.tt90_installation_learning hard
python -m scripts.experimental.tt90_installation_learning assess --n 20
python -m scripts.experimental.tt90_learning_report
python -m pytest -q tests/test_tt90_joint_library.py
```
