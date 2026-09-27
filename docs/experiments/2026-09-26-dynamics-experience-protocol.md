# Dynamics transfer: experience diversity and causal online adaptation

The frame-aware state reduces missing forecast samples from 68/1044 to 3/1044.
Its learned dynamics beats fixed controls on aggregate, especially curved motion,
but eight-frame straight-motion forecasts regress substantially. Do not hide
that failure in the mean or tune a motion-specific threshold.

Test two large questions with the unchanged model and training budget:

1. Does varied **observed camera experience** solve quantized direction/scale
   transfer? Keep 24 training streams; sample their direction uniformly and
   their speed scale in [.8,1.2] with fixed seed 810. Held-out directions, phases,
   shapes and scale 1.25 remain unchanged.
2. Does **causal online adaptation** solve novel dynamics promptly? In a separate
   evaluation, reset each scene to the trained weights, update only when the
   current actual observation arrives, then issue forecasts. No future positions
   or simulator truth may enter updates. The shuffled-credit control draws only
   from previously observed displacements in the same scene.

Compare diverse/frozen, restricted/online and diverse/online with the preserved
restricted/frozen run. Preserve all 1/4/8 horizons and fixed controls. Require the
same aggregate advantage and report every family's regression. One run of each;
no exposure, memory-size, learning-rate or horizon sweep. Unit-test prefix
causality of the prequential evaluator before interpreting its results.
