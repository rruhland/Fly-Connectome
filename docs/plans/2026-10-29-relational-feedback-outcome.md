# Learned relational feedback: qualified improvement, milestone incomplete

The frozen relational endpoint improved actual TRAIN32 returns from 83 to 130, with first returns unchanged at 29 and later returns improving from 54/76 to 101/110. Fresh tag 63313 improved from 90 to 131 returns, with first returns unchanged at 30 and later returns improving from 60/84 to 101/113. Fresh overall return fraction was 131/145 (90.34%); later fraction was 89.38%. This remains below the 95% milestone and requires further validation.

Search used two fixed rounds of 32 challengers on the same nested 4→8→16 TRAIN starts. Only complete 16-game scores could replace the incumbent. Round one selected 64 returns versus baseline 44; round two's best 61 did not replace 64. Selection chose observed cardinality 3, horizontal ranks 0 and 1, coordinate 0 (public [y,x] order), and polarity −1. All five numeric coefficients (ci, cj, offset, deadzone, hysteresis) remained 0. Actual own hits selected the binding, axis and polarity; no semantic roles, teacher actions or simulator geometry entered the controller.

Fresh mean hits gained 1.28125/game, paired bootstrap 95% interval [0.65625, 1.90625]. Games reaching at least two returns increased from 21 to 26; at least three increased from 17 to 25. Seventeen games gained hits and four lost hits. Later-fraction difference was +0.17952 with paired interval [0.09245, 0.28742], conditional on actual exposures. Maximum streak was 6 within the fixed 2,400-physics-tick window. Candidate timeouts were 18/32 versus baseline 6/32; this window does not establish indefinite rally reliability.

TRAIN32 executed 24,747 cameras, all raw-Q and stateful decisions replayed exactly: 2,820 baseline decisions before the first hit, 5,086 cardinality fallbacks, 16,841 relational decisions and 9,703 changed commands. The 16 paid optimized-fitness tapes matched their original-runtime executions exactly on 12,020 actions, hits, durations, ticks and end flags, without extra cameras. All 25,366 fresh candidate decisions also replayed exactly. Source and asset hashes stayed unchanged.

Search cost was 161,545 transitions and 352 initials in 428.306 seconds; the complete experiment cost 231,307 transitions and 448 initials in 632.380 seconds, within 900 seconds / 320,000 transitions. Challenger tapes stayed compact; full raw inputs were captured only for the endpoint. The separately proved exact-copy helper was pinned; old runtime sources stayed unchanged.

- Endpoint `/workspace/m1b-experiments/pong-relational-search-20261029-r1/models/final-policy.npz`, SHA256 `026ff8da73457238209708a03060bcaf10a5f94bbb83805d9f7a60c0644beedf`.
- Final spec SHA256 `aa9b357ca0c2f354f1f078bcf41d25614024bd7cc25942d257dcd9da73c557c7`; original preserved as adjacent `spec.pre-exact-copy.json`.
- Outcomes: adjacent `summary.json`, `final32/summary.json`, `final32/live-aggregate.json`, `fresh/summary.json`.
- Core `1100d44`; runner/adapter `7238313`, `e144648`; helper `3e9081d`. Twelve focused tests passed before launch.
