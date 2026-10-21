# Equivalent bulk fingerprint validation

The retained-memory pendulum prefix profile attributed7.20/9.07 profiled seconds
to fingerprinting; per-tensor bulk-path guards alone cost1.58 s. Encoder deepcopy
was only0.10 s, so it was not the next optimization target.

The narrow change lets `torch.stack` validate homogeneous shape/device/layout
before any child seen-ID/retained-reference mutation, keeping exact type/dtype and
size guards plus scalar fallback. It resolves negative/conjugate/noncontiguous
logical values as the scalar NumPy-force path does. Primitive dtype/shape metadata
is constructed once per homogeneous sequence. Full bytes, canonical nodes, alias
IDs, mutation detection, graph format and recovery semantics are unchanged.

An isolated process injected old/new functions into OTHERWISE UNCHANGED2055109
source. On one uninstrumented ABBA,128-camera retained-memory frozen B loops took
6.047/5.827 s old versus4.381/4.666 s new: means5.937→4.524 s (23.80% less wall).
This is one warmed prefix, excluding construction; it is not a universal speed
claim. All512 existing record/state/chain rows and4 initial images matched exactly.
There was no source-metadata normalization or new candidate observation.

287 synthetic old/new digest/exception comparisons covered mixed shapes/layouts,
repeated/external aliases, conjugate/negative/noncontiguous/expanded views, signed
zeros/NaN payloads, RNGs and NumPy-alias mutation. After promotion,28 focused hash
and mutable-session/recovery tests passed2.43 s (36 diagnostic cameras/41 images,
2 replayed cameras included, no candidate data). Original candidate source/archives
remain intact. Newly committed codec bytes naturally change future session source
signatures; source bridges must preserve that provenance rather than claim complete
old/new session-hash identity after the source signature itself changes.

Artifacts: `/workspace/m1b-experiments/pendulum-hash-guard-prototype-20261021/`.
The prior profile separately reexecuted128 cameras+1 initial in9.06 profiled seconds;
profiling and uninstrumented benchmarking are counted separately in the ledger.
