# Frozen visual-motif feasibility preflight

Continue the approved controlled experiment by checking representability and
actual precursor response before any additional training. No model change.

The direct L2->L1 anatomical inventory found 136/137 edges within the same mapped
column and no edge spanning the tested rightward 2-4-pixel separation. This is
an anatomical observation, not a functional search or proof against indirect paths.

Keep previous L1 target38366 and the same dot trajectory (row16,x28..36), polarity,
speed and eight-tick forecast lead. Include all three measured predictive sources
into this target and ALL direct parents of these sources, plus the trajectory's
sensory cells. Keep the induced measured graph unchanged:63neurons/274edges.
It has Mi1/Tm3 excitatory prediction inputs and an inhibitory C2 input. Other target
types are not scored; this tests one explicitly selected mixed-sign L1 motif.

Run only ten frozen traversals, blank intervals12-36frames from seed9011. Each
trial and its controls start from an identical copy of the normally warmed-up
initial network; these are paired probes, not training resets. For each ON/OFF
target event separately, blank all earlier images in the control while retaining
images from that event onward. Compare forecasts eight ticks before the event;
the controls therefore cannot use the target event as input to that forecast.

Require both measured output signs before executing. For each polarity require
8/10 trials with: correctly signed actual forecast, correctly signed forecast
increase over the blank control>0.001, and an individual compatible incoming
signed physical trace increasing in the correct direction>0.001. This prevents
tonic firing alone from counting as a precursor response. This tests current
causal signal availability, not successful learning or M1A acceptance. Correctly
signed initial net current is not mathematically necessary for eventual learning:
reweighting could reveal a signal currently canceled by competing inputs.

Reconstruct signed traces from recorded actual spikes and fixed delays. Warmup
history is not included in that reconstruction; require reconstructed frozen
current to match actual forecasts within1e-6 at every tested issuing tick, after
at least12blank frames. Reject measurements that fail this check. Save edge
identities, source spike timing, actual event timing, paired predictions and all
small probe traces. Verify source checkpoint and weights remain unchanged.

Bound: one fixed motif, ten paired traversals with two controls each,120seconds
of neural execution. No training or candidate/speed/horizon search in this pass.
Stop on failure with the trace-level cause. If it passes, the next bounded
training comparison can use this preflight-verified motif; architecture changes
still require the user's approval.
