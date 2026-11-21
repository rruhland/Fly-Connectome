# Continual learning across bodies, views and physics

A useful transfer is a faster improvement in play after the world changes. Keep the general task, reward, controller limits and stream of observations fixed across the comparison. Give the player its sensor frames, its previous public commands and the ordinary point reward. Reserve true arm state, camera poses, installed masses, bounce settings, contact labels and the change schedule for simulation and evaluation.

## The unit of experience

At deployment, one checkpoint plays one uninterrupted chronological stream. Carry memory through points, games and unannounced changes. Count dry movements, resets used for learning and repeated views as interaction costs. The primary self-directed comparison uses zero target-side calibration movements before scoring. A separate calibration study may use visible landmarks or known camera layout if the experiment explicitly identifies that assistance. Track seconds of simulated interaction independently of wall-clock throughput.

Evaluate an experienced learner, a learner without the transferred component and a frozen control with matched starts and budgets. Identify which components are shared: a cold motor model in a practiced stroke controller measures transfer of motor knowledge specifically. Treat a fully cold perceptual/motor/task learner as a separate longer-term comparison. Score the first legal serve, first legal return, first point with two returns, complete game and recovery after a hidden change. Track retention on return to earlier conditions.

## What to vary

| Factor | Comparisons |
| --- | --- |
| Arm and attached objects | Change joint inertia, accessible travel and link geometry. Mount a physical camera payload at different links. Separate mass-only from image-only changes before testing both together. |
| Ball and table | Vary ball mass, uniform or location-dependent bounce, and table dimensions. Verify that identical inputs cause measurably different trajectories or outcomes, and that the simulator's dimensions agree with rendering and scoring. |
| Observation | Move cameras, swap their order, remove a view, restore it, or add a camera on a moving arm. Measure 3D error, uncertainty/coverage, and actual game play. Preserve uncertainty along an unobserved depth direction. |
| Timing | Continue the existing changing transport latency, action noise and soft-racket courses. Count real chronological experience needed for recovery. |
| Unknown combinations | Include unseen combinations, each factor alone and at least one revisit. Retain strong simple controls; winning on one changed machine calls for a new paired fixture. |

## Parallel experience

Use concurrent simulators for independent sources, matched controls, disjoint seeds and evaluator replay. Save each stream and its seed. When combining source experience, identify exactly what crosses to deployment and charge the total source frames. Keep parallel target rollouts, model selection on test seeds and oracle-assisted competitors separate from single-stream online evidence. A repeat from a chosen winner is evidence about that winner, not the entire population.

## Staged research

1. Establish that the physical perturbations work and that the arm responds to changes during scored play with no target-side dry routine. Test optical pose registration separately, then put changed views in the scoring loop.
2. Learn what to explore from related environments. Compare the same update on cold and experienced priors after surprise, and matched passive play against directed movement. Retire a method if it improves motion prediction but consistently hurts games.
3. Combine unannounced body, view and physics changes within one game series, fork only for evaluation controls and revisit the starting world. Require improvements on independent arms and ordinary Pong retention before treating a checkpoint as general.
4. Move to a real arm only after a single-camera or partial-view fallback, physical bounds, latency budgets and simulator-to-simulator transfer have been qualified. All TT18 comparisons use simulation.

Peng et al. demonstrated that randomized simulated dynamics can transfer to a physical pushing arm. Kumar et al. separated a base locomotion policy from rapid adaptation to changing physical conditions. The questions for this project are how much experience a non-backprop learner needs on a new environment, and whether it can identify what changed from the camera and the consequences of its own actions. A policy that works across several environments without updating provides a useful baseline. [Dynamics randomization](https://arxiv.org/abs/1710.06537) · [Rapid motor adaptation](https://arxiv.org/abs/2107.04034)
