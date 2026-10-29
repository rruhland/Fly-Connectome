# Exact visual transport copying for newly staged fitness workers

The new `scripts/experimental/exact_visual_transport_copy.py` specializes only the
two entity/forecast copies inside `VisualStateEncoder.encode`. It reuses the
original encoder function's identical Python code object with a private `copy`
global. Every encoder arithmetic operation and its order remain unchanged. It
preserves owned tensors and repeated object aliases; unsupported metadata, views,
shared storage, manual tensor gradients and other unsupported cases retain normal
`deepcopy`. Existing pinned runners and source files remain unchanged.

Call `install_exact_transport_copy(session)` immediately after evaluator
construction and before its initial camera. Installation rejects an already
started or modified encoder and preserves the initialized learned-contents hash.
New workers must pin this helper alongside their own source. Adoption is limited
to the newly staged stateful reward-fitness worker for the unlaunched servo pilot.

## Evidence and scope

Artifacts: `/workspace/m1b-experiments/pong-runtime-profile-20261029-r1`.

The single granted profile game used zero symbolic policy, seed `3351729059`, the
unchanged reward-only evaluator, one physics action per camera and full normal
vision. It completed with one own hit in 284 transitions plus one initial camera.
The 4.025s cProfile wall attributed 78% of profiled time to observation; encoder
transport deepcopies accounted for 0.492s, or 12.3%. Diagnostic norms and global
RNG bookkeeping were approximately 1.4% and 1.15%, respectively.

Offline tests compared complete encoded outputs and encoder fields on four
reconstructed transports from existing archived snapshots. Seven focused alias
and fallback cases included distinct shared-storage views and a manually assigned
`.grad` on a tensor whose `requires_grad` is false. Full output byte/object/storage
parity and source isolation passed. No new camera acquisition was used for these
tests. The helper also matched the independently native-tested prototype on all
four captured inputs.

Two separately authorized, uncProfiled zero-policy games then compared the
original encoder with the prototype on the same seed. Both completed 284
transitions plus one initial camera. Actual records and outcomes, every initial
and post-transition feature/scene/encoded/runtime-state hash, rolling digest,
decision digest and initial/final frozen hash were identical. Runtime comparison
represents encoder fields directly so the intentional encoder class name is the
only normalized type difference.

Summed native begin/step time was 3.36385s versus 2.98819s; instrumented episode
wall including hash capture was 6.04551s versus 5.66033s. This is one fixed-order,
same-seed pair. The exact full-vision and runtime-output parity supports adoption;
the timing is evidence of a modest improvement, not a general 11.17% speed claim.

Native accounting is explicit: original profile **284 transition + 1 initial**;
paired proof **568 transition + 2 initial**; combined **855 cameras**. No repeats,
additional native tests, policy changes, learning, camera skips or cadence changes
were performed. `native-pair/accounting.json` and `native-pair/parity.json` retain
the grants, charges and comparison; `new-helper.json` pins the helper SHA-256.
