# SO-101 follower simulation assets

TheRobotStudio and contributors published `so101_new_calib_camera.xml` and the files in `assets/` under Apache License 2.0. This directory carries those files byte for byte from commit `141aea7bcac2744d92998e30268c413202a56f96` of `TheRobotStudio/SO-ARM100`, under `Simulation/SO101`. [Source tree](https://github.com/TheRobotStudio/SO-ARM100/tree/141aea7bcac2744d92998e30268c413202a56f96/Simulation/SO101) · [Apache License 2.0](LICENSE)

The files in `assets_preview/` are modified visual derivatives of those STLs. We imported each STL into Blender 4.3.2 at unit scale with Y forward and Z up, applied its collapse decimator with target `max(420, 0.096 × input triangle count)`, and exported binary STL with the same axes and scale. The preview keeps the source mesh coordinate frames. MuJoCo uses the pinned source model's link inertias and separate paddle, table and ball physics in both modes.
