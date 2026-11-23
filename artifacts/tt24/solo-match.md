# Carrying a solo-bounce motor response into live table tennis

The earlier bounce-to-match comparison transferred an encoder-based wiring decoder. The earlier match residual study learned its corrections from table-tennis episodes. The earlier donor-volley study used a static receiving region. This experiment tests a motor feedback component from the reward-trained repeated-bounce head inside the existing from-release match controller, against the live learned far arm. [Wiring transfer](bounce-to-match.json.gz) · [Match feedback and opponent](rally-improve.md) · [Solo source](tt24_donor.md)

The current match arm uses the frozen optical interaction policy learned from its existing demonstration bank. It reads two actual camera views, own joint angles and its own issued commands. It produces all five active near-arm joint rates from the first frame. The experiment uses no new demonstrations or supplied opening. Legal exchanges require legal underlying returns by both physical paddles. The far arm responds to the actual evolving point. [Match controller](../../scripts/experimental/tt24_rally_improve_optical.py) · [Experiment code](../../scripts/experimental/tt24_solo_match.py)

## Source extraction and target practice

The extraction replays four archived solo drops and reads 42 optical and encoder snapshots. It then queries the frozen solo motor head about small local displacement and normal perturbations, retaining ball-to-paddle translation in each synthetic sensitivity query. A public encoder geometry model maps this response to a common paddle frame built around the face normal and shaft. The symmetric part has two locally restoring eigenvalues, −0.1049 and −0.0264. Only that part is carried. The shuffled control rotates the feedback operator in dimensionless paddle coordinates and preserves both eigenvalues and norm. The generic isotropic controller has the same Frobenius norm.

In the match the residual acts on the discrepancy between the measured and issued paddle poses. It uses the measured joint angles, the match head's own accumulated target and static vendor geometry. It sees no ball variables, anticipated impact, target placement, or preset trajectory. The frozen match head determines the nominal movement. A damped joint solve turns the learned tool feedback into a bounded correction.

The two gains were assigned to the same four fresh calibration releases for each residual. Each residual used eight scored practice episodes plus the same four frozen episodes as a shared comparison, for 28 physical practice trajectories across all conditions. Calibration selected the gain with the highest mean number of legal two-arm exchanges, capped at eight, and kept the frozen control on ties. Held testing also includes a pre-specified low-gain donor and a low-gain shuffled diagnostic even if calibration selects zero. [Plan](solo-match-plan.json) · [Frozen source stabilizer](solo-match-stabilizer.npz)

| Practice component | Frozen average | Gain .32 | Gain .64 | Selected gain |
|---|---:|---:|---:|---:|
| Solo-derived | 2.00 | 0.50 | 0.25 | 0 |
| Shuffled orientation | 2.00 | 0.50 | 0.25 | 0 |
| Generic isotropic | 2.00 | 0.75 | 0.75 | 0 |

Each practice branch lost to the frozen match head, so calibration preserved the original match commands.

## Held two-arm match

| Controller on 24 held releases | Own legal returns | Far legal replies | Complete exchanges | Longest alternating run |
|---|---:|---:|---:|---:|
| Frozen match and all three calibrated selectors | 18 | 18 | 18 | 12 |
| Fixed low-gain solo-derived feedback | 20 | 19 | 19 | 8 |
| Fixed low-gain shuffled feedback | 29 | 28 | 28 | 16 |

The two fixed corrections had a mean rate-change norm of 0.01064 and a peak of 0.096 normalized rate. The solo-derived correction gained 0.042 complete exchanges per release over frozen, with a paired 95% bootstrap interval of −0.54 to 0.58. The shuffled correction scored nine more exchanges than the solo-derived correction on this cohort. These results show no clear advantage from the solo-derived component. The schedules were checked for overlap against 713 prior receipts. The study independently replayed 28 practice and 72 distinct held action tapes, including both paddles and the full point referee. [Full paired results](solo-match-trial.json.gz)

The initial service shutdown preserved the source matrix and registration. The restarted runner reused that matrix, reran practice tapes that had never been saved, and checkpointed complete scored points. The original solo policy remained byte-for-byte identical. On four fresh ordinary wrist drops after the match comparison, it produced streaks of 18, 2, 2 and 3, with exact replay of every score. [Source fit](solo-match-source.json.gz) · [Source retention](solo-match-trial.json.gz)

## Reproduce

```sh
export PYTHONPATH=src LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
python -m scripts.experimental.tt24_solo_match --held-workers 2
```
