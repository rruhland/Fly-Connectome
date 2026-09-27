# Paired measured-motion feature ablation

Use the recovered measured T4/T5 graph as a frozen optional input to the learned
context state. Preserve all anatomical edges, signs, delays and learned weights;
retain its already-investigated rest/graded experimental settings. Feed each
observed event frame for eight neural ticks. Map stage-specific spike fractions
back through measured retinal columns, and pool the same local 21x21 patch used
by the visual key. Append eight unnamed stage activities to the 51 sensory
components. The ablation supplies zeros in those eight channels, with equal
memory capacity and learning rule. No oracle object crop.

One bounded comparison uses the 16 speed-two development scenes for each of the
two learned context relations at horizontal and vertical orientations. Resize
rotated arrays to the graph's fixed 32x64 retinal input; both arms see identical
arrays. Evaluate the existing held-out relation scenes at both orientations.
Report baseline and augmented scores, activity and wall time. This is a matched
contribution audit, not a claim of real-camera transfer or proof the graph is
necessary. Do not tune stage gains or force a positive contribution.
