# Transferring the sign of a motor step

All five independently trained native Pong fits chose the same actions on 64 fresh simulated arm openings. Those actions earned **24 legal returns**, versus **13** from center and **one** with the sign reversed. The prespecified pooled shuffled fit made the same choices and also earned 24. The fixed promotion rule required more legal returns than both center and shuffle, so the arm experiment ended at the opening stage. [Frozen design](direction-design.md) · [Arm choices and scores](direction-arm.json.gz) · [Physically played option bank](smooth-jog-arm.json.gz)

The training sources contained 264 distinct native Pong starts, each with seven physically played fixed actions. Within a start, the learner compared actual point duration for opposite directions at each of three equal magnitudes. It averaged the signs of those three differences. A reflected kernel estimated the signed preference from the existing anonymous motion descriptors in the calibrated motor frame. The model selected the direction of one arm step. The step length came from the pre-existing arm action grid and was **7.632 cm**. The same body stroke executed every choice. This separation was devised after the earlier seven-action experiment showed that individual source fits chose different magnitudes. [Source data and frozen models](direction-source.json.gz) · [Earlier magnitude experiment](anonymous-action.md)

| Frozen policy | Mean held native Pong seconds | Held native first returns | Arm contact starts | Arm legal return starts |
|---|---:|---:|---:|---:|
| Learned ordinal direction | 3.348 | 15/48 | 63/64 | 24/64 |
| Center | 2.695 | 8/48 | 25/64 | 13/64 |
| Within-start shuffled durations | 2.920 | 10/48 | 63/64 | 24/64 |
| Reversed learned direction | 3.015 | 8/48 | 2/64 | 1/64 |

Four independent fits chose the pooled direction on all 48 held native scenes. The fifth agreed on 37. Each held Pong scene physically played the three available options, and all four directly executed policy choices on the first held scene reproduced the corresponding action digest and outcome. The 64 arm starts physically played all seven options, including the three available to this selector. Three arm scenes lacked a qualifying motion observation and used center in every branch. The learned arm gained 23 legal returns and lost 12 compared with center. Its results were 11 versus 8 legal returns on moderate openings and 13 versus 5 on wide openings. [Held native trials and direct check](direction-native.json.gz)

After seeing the pooled tie, a follow-up permuted duration labels within each native start and kept fits from the five source sets independent. For each of 4,096 joint draws, it scored each fit on the same 64 arm scenes and averaged their legal-return counts without pooling the fits. The null mean was 12.53, with a central 90% interval of 5.6 to 19.4. A mean of at least the observed 24 occurred in 117 draws, giving a one-sided Monte Carlo permutation p of **.0288**. All five shuffled fits beat center together in 130 draws, giving **.0320**. These diagnostics condition on the one arm bank and quantify variation across independently shuffled native training outcomes. [Independent-fit permutation receipt](direction-permutation.json.gz)

```sh
python -m scripts.experimental.tt23_direction fit
python -m scripts.experimental.tt23_direction native
python -m scripts.experimental.tt23_direction arm
python -m scripts.experimental.tt23_direction_permutation
```
