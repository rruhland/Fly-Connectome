# Learned relational feedback: qualified improvement, milestone incomplete

The frozen relational feedback endpoint improved actual TRAIN32 full-game returns83→130, first returns unchanged29, and later returns54/76→101/110. Fresh63313 improved90→131, first returns unchanged30, and later returns60/84→101/113. Fresh overall return fraction was131/145=90.34%; later fraction89.38%. This is useful held-out progress below the95% reliability milestone, not an automatic deployment or completed replication.

Search used two fixed rounds of32 challengers with the same nested4→8→16 TRAIN starts. Only complete16-game scores could replace the incumbent. Round one selected64 returns versus baseline44; round-two best61 did not replace64. Selection chose observed cardinality3, horizontal ranks0 and1, coordinate0(public[y,x] order), polarity−1. All five numeric coefficients(ci,cj,offset,deadzone,hysteresis) remained0. Binding/axis/polarity were selected by actual own hits; no semantic object roles, teacher actions or simulator geometry entered the controller.

Fresh mean hits gained1.28125/game, paired bootstrap95% interval[0.65625,1.90625]. At least two returns increased21→26 games; at least three17→25. Seventeen games gained hits and four lost hits. Later-fraction difference+0.17952 had paired interval[0.09245,0.28742], conditional on actual exposures. Maximum streak6 occurred within the fixed2400-physics-tick window. Candidate timeouts18/32 versus baseline6/32 mean counts are censored by that window and do not establish indefinite rally reliability.

TRAIN32 executed24,747 cameras, all raw-Q/stateful decisions replayed exactly:2,820 baseline-before-hit decisions,5,086 cardinality fallbacks,16,841 relational decisions and9,703 changed commands. The16 paid optimized-fitness tapes matched their original-runtime final executions exactly on12,020 actions/hits/durations/ticks/end flags, without extra cameras. Fresh candidate25,366 stateful decisions also replayed exactly. Source and asset hashes stayed unchanged.

Search cost161,545 transition cameras/352 initials in428.306 seconds; complete experiment231,307 transitions/448 initials in632.380 seconds, within900 seconds/320,000 transitions. Challenger tapes stayed compact; full raw inputs were captured only for the endpoint. The separately proved exact-copy helper was pinned; old runtime sources stayed unchanged.

- Endpoint `/workspace/m1b-experiments/pong-relational-search-20261029-r1/models/final-policy.npz`, SHA256 `026ff8da73457238209708a03060bcaf10a5f94bbb83805d9f7a60c0644beedf`.
- Final spec SHA256 `aa9b357ca0c2f354f1f078bcf41d25614024bd7cc25942d257dcd9da73c557c7`; original preserved as adjacent `spec.pre-exact-copy.json`.
- Outcomes: adjacent `summary.json`, `final32/summary.json`, `final32/live-aggregate.json`, `fresh/summary.json`.
- Core `1100d44`; runner/adapter `7238313`, `e144648`; helper `3e9081d`. Twelve focused tests passed before launch.
