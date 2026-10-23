# Confirm an actually executed search child

## Decision after the fresh-confirmation result

The weighted-proposal confirmation trial completed six cycles, accepted zero updates and retained RIGHTtheta6 exactly. Both independent raw checks passed. Fresh confirmation phase nets (LEFT,RIGHT) were (+2,-1), (0,0), (+2,-9), (-1,+2), (+2,-2), (+1,-2), each out of 24 paired starts per phase. Total gains were +1,0,-7,+1,0,-1: none reaches the fixed +2 threshold even if phase protection is removed. The result therefore cannot be explained solely by an overly strict phase-preservation constraint. Proposed function-RMS changes were substantial (.07945–.10); applied change was zero. This is a failed learning procedure under its budget, not a heldout null for the rejected heads or a representation-capacity result.

Actual costs: 159,824 TRAIN transitions =157,443 complete-cycle used +2,381 incomplete-cycle unused; 1,755 initial images; zero DEV. Runtime981.600 seconds (979.847 training,1.753 reporting). The prospective unchanged-policy rule avoided redundant evaluation. Parent reconstruction2.731 seconds; Sol reconstruction13.029 seconds; both zero new observations/replays. Descriptive pooled confirmation differences are LEFT+6/144, RIGHT-12/144 across six different proposals, not a single-policy performance estimate.

Before seeing outcomes, Sol favored a DIRECT-start path test if several accepted proposals still failed, or actual-child selection if acceptance starved. Astra independently considered path dependence and constrained phase-specific search directions. After observing zero accepted changes, Astra selected Sol's actual-child alternative as the one final follow-up. DIRECT initialization remains an untested alternative, not the next automatic run. This respects the research-batch limit: one primary trial and one justified follow-up, then synthesis.

## Fixed follow-up

Keep the same preserved RIGHTtheta6, frozen generic vision/T, existing linear greedy head, strong native opponent and first-hit scalar reward. Run the same eight mirrored .25-RMS search directions and shared six LEFT plus six RIGHT discovery starts: 192 episodes. Select the highest mean-fitness actually executed child among the sixteen candidates, with stable direction-major/sign-major ties (plus before minus). Do not build a weighted top-four center, clip, interpolate or select using DEV.

Confirm that exact child against its incumbent on 24 fresh starts per phase, another96 TRAIN episodes. Accept only D_LEFT>=0, D_RIGHT>=0 and total>=2; otherwise preserve exact incumbent weights and advance RNG only after a complete cycle. Distinct fresh schedules, fixed original launch clock, min5/max12 cycles,160k TRAIN/1200s plus900s DEV/2100s total. Keep zero-acceptance futility, last-complete endpoint selection, original576 heldout jobs, eight screens, separate80% competence and replication requirement. Partial work remains charged and cannot commit. No architecture alteration or backpropagation.

This deliberately replaces a <=.10-RMS weighted update with an exact .25-RMS executed child. It changes selection AND deployment step size; any improvement would not isolate candidate-to-center mismatch. The existing perturbation budget already executes such .25 children, but retaining one can change behavior substantially. Fresh empirical confirmation is not a statistical retention guarantee. The prior native-win validated-child experiment remains a material negative precedent; dense first-hit reward and two-phase confirmation differ here.

## Decision boundaries

A changed endpoint with genuine heldout gain and retention supports independent recipe replication. Accepted TRAIN gains that fail heldout screens expose the limitation of finite confirmation. Another zero-acceptance result is another unsuccessful bounded search around this carried policy, not proof that no joint policy exists. Do not loosen thresholds, change noise, launch a DIRECT restart or expand architecture within this session afterward. Preserve the candidate and return a synthesis with the next research question. No new full allocation after18:04:30 UTC; session hardstop18:49:30 UTC with ten-minute closure reserve.

## Storage closure before follow-up

After coordinator EXIT0 and both raw audits, the scoped completed-futility archive eligibility was extended only to the exact checked confirmation root; status was not altered and interrupted allocations remain excluded. Empty sidecars archived/verified first (EXIT0,1.221s), then1,755 journals compressed from5,161,914,368 to685,392,174 bytes in50.211s. Every original stream was reconstructed and hash-verified before unlinking;0new observations/replays. Readouts, protocols, source, metadata and checkpoints remain plain.

## Prelaunch review

Eleven focused fixtures passed3.09s; source unchanged68d15a7. Parent independent schedules/directions/candidates review passed0.456s; all42,777 prior IDs excluded,912 distinct accepted canonical starts,881 rejected candidates. Frozen driver2a28f52f47d8f08a3a9e80fab377bddd44067db6edc16a9786df033c2285ccd0; protocol5c2453ed64088802de15111c195ede1e2b780ee8815d54207c20d9643af9c011. Final allocation authorized only after both reviews,0prelaunch observations.
