# Designing the whole end-clamp installation

This study asks whether the whole stock six-joint arm is poorly placed for the actual spread of receiving balls. It jointly varies lateral and depth placement, pitch, roll and yaw. The original table and clamp shoes, arm and active joints, racket, camera hardware, motor limits, ball, continuous opponent law and scoring are retained. It is explicitly a different rigid installation, not the original benchmark or a manufactured load certification.

## Workspace and impulse screen

The source is 884 measured ordinary deliveries across 128 uninterrupted stock training points. A fixed random sample retained 72 each from low, transition and routine heights. The screen used mostly continuous measured lines, not a fixed center or a selected successful donor. It projected each recorded receiving-line state to four depths and asked for two useful outgoing pace bands. A precomputed stock joint atlas transforms rigidly with the proposed mounting pose. It checks blade-face plane, contact offset, approximate travel time and local blade-patch velocity in both the normal and upward tangent directions. Mixed directions prevent the same joint from being counted twice at full rate. The geometry predicts support, not a legal return. It ignores loaded motion, complete route clearance and later rebound errors.

Of 192 five-dimensional proposals, 27 fit the original CAD base footprint on the supported deck with the declared height bound. Ranking required ordinary and transition coverage broadly comparable to stock, then traded low and regular support across the complete sample. A no-wedge +12 cm translation is included as a simpler reference.

| Installation | Low / 72 | Transition / 72 | Routine / 72 |
|---|---:|---:|---:|
| Stock | 36 | 45 | 69 |
| +12 cm translation only | 47 | 45 | 66 |
| Combined installation | 49 | 43 | 70 |

The combined design is +119.72 mm lateral and 5.61 mm farther inboard, with +17.96° pitch, −7.60° roll and −4.00° yaw in the declared extrinsic XYZ convention. The taper has at most 61.94 mm thickness. Two relieved lower pads bear on the original end plate. CAD measured a base-to-deck gap below 8 nm and 1.00 mm and 4.00 mm clearance from the upper jaws and playing table, respectively. The rig was separately commissioned to the original ready blade center within 2.2 mm. It retains 24 of 25 ordinary routed atlas sites. The screening advantage over lateral-only remains small. [Protocol and source checksums](protocol.json) · [Joint workspace screen](screen.json) · [Compiled support and calibration](build.json)

## Powered tests on independent releases

The same native coach and movement budgets compare stock, lateral-only and the canted work cell on independent measured openings. They keep the actual opening far impact and legal near bounce. On all four reserved low balls, the combined installation scored and drew a legal ordinary answer without sampled CAD penetration. The centered controller returned none and lateral-only returned two. The combined body made three broad-face hits within 58 mm. On 15541 and 7398 it returned previously unreturned low balls on the +12 cm station. These are physical corrections selected by the native mechanism coach, not by the RGB learner.

| Low release | Centered | +12 cm | Combined minimum qualifying radius |
|---|---|---|---:|
| 14429 | Miss | Interior return | 22.68 mm |
| 15541 | Miss | Miss | 62.38 mm |
| 7398 | Miss | Miss | 51.11 mm |
| 10762 | Miss | Rim return | 57.75 mm |

[Original and canted actual actions](physical.json.gz) · [Same-offer lateral-only control](lateral-control.json.gz)

All four reserved ordinary controls also landed and drew an ordinary answer on the combined installation. Each had a clean interior hit from the same native-coach budget. The unchanged archived joint-rate tape alone transferred just two of the four routine returns. The other two needed a different movement even though the incoming physical ball did not change. Independent command replay reproduced all 12 actual scoring tapes and checked every reactive draw and 15,664 CAD solver states with no penetration. This is a reason to measure actual online learning on the new body, not to interpret the four native rescues as an ordinary policy result. [Independent replay, referee and CAD receipt](replay.json)

The eight-game preliminary stream and subsequent unfiltered 128-game season use the original archived carry, actual stereo and own encoders. Installation registration changes only action-memory coordinates and the private body's known geometry. The extensive season is the same release and reply-seed roster as the centered TT80 comparator, and starts from the original carry again. Uninterrupted points end at the standard physical loss or existing cap. Any 100 own returns would be saved and audited on the *actual canted body* before being counted. [Preliminary stream](games.protocol.json) · [Full mounted season's frozen roster](season.protocol.json)

## Optical pilot versus the ordinary centered actor

| Release | Centered online | Canted online |
|---|---:|---:|
| 4888 | 0 | 1 |
| 6299 | 4 | 5 |
| 5222 | 2 | 2 |
| 8883 | 0 | 5 |
| 5335 | 11 | 8 |
| 12866 | 1 | 25 |
| 12144 | 6 | 2 |
| 3482 | 0 | 0 |
| **Total** | **24** | **48** |

All eight points in each arm were played once and all 68,287 CAD samples had no penetration. The entire +24 net gain came from the one distinct 12866 rally. The remaining seven points tie at 23 own returns. This confirms a working real RGB/encoder controller and one longer witness on the canted body. It does not show uniform improvement or a 100-return qualifier. [Centered camera-controlled pilot](stock-online.json.gz) · [Mounted camera-controlled pilot](installed-online.json.gz)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt84_installation_system plan
python -m scripts.experimental.tt84_installation_system screen
python -m scripts.experimental.tt84_installation_system build
python -m scripts.experimental.tt84_installation_system physical
python -m scripts.experimental.tt84_installation_compare
python -m scripts.experimental.tt84_installation_report
python -m scripts.experimental.tt84_installation_play plan
python -m scripts.experimental.tt84_installation_play stock
python -m scripts.experimental.tt84_installation_play installed
python -m scripts.experimental.tt84_installation_play pilot-audit
python -m scripts.experimental.tt84_installation_play pilot-frozen
python -m scripts.experimental.tt84_installation_play season-plan
python -m scripts.experimental.tt84_installation_play season-live
python -m scripts.experimental.tt84_installation_play summary
python -m pytest -q tests/test_tt84_installation_system.py
```
