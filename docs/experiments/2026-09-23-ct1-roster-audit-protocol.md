# CT1 cross-side roster audit

The pinned MaleCNS M1 graph selects visual cells by right **soma** side. Primary T5 physiology identifies CT1 GABAergic input as a null-direction suppressive route. Before adding a T5 local timing mechanism, check whether the measured CT1 route into the already selected T5 cells is present in the raw connectome and whether the pinned roster retains its source cell.

Read the frozen checkpoint, raw body annotations, transmitter evidence, and raw minimum-confidence contact table. For each traced CT1 body, count measured edges and contacts to selected T5 targets and from selected Tm1/Tm9 sources. Report its soma side, pinned membership, transmitter ground truth, and the contact threshold-3 counts. Do not change graph extraction, checkpoints, signs, or dynamics. A missing route is an anatomical-selection finding, not a functional rescue claim; test its effect separately in an opt-in shadow graph before considering production changes.
