# Causal local reference replay

Diagnostic only. Preserve the saved split event-balanced amplitudes and all
neural weights. No fitting or parameter sweep. Use the existing sensory decay.

At each observed local event, replace a scalar reference with the absolute
sensory state stored one camera frame earlier. A forecast issued now uses only
the reference already available now; it cannot use its future target. Keep its
amplitude when abs(current sensory state) is within half a camera frame of decay
on either side of the reference (multiplicative exp(+/-4*dt/tau_sensory)).
Before a nonzero reference exists, suppress output. Do not reset at trial
boundaries, inspect phase, or supply tempo. This added reference is an offline
diagnostic mechanism, not an implemented biological learning rule.

Reconstruct complete sensory histories from recorded injected events and check
against saved issue states. Include first-cycle observations in causal updates,
while scoring the unchanged held-out capacity rows. Report ON/OFF retention,
quiet alarms and MSE together, separately by tempo including unseen 3.

Separately stress the same selector using sensory-only streams: 24 alternating
events per interval block 2,3,6,4 frames, with an unannounced omitted event pair
in the middle of the tempo-6 block. Report event hits and quiet activations by
event ordinal after switches and around omissions. These streams establish
selector adaptation only, not neural amplitudes or successful learning. Never
stitch independent neural histories into a claimed continuous simulation.
