# Fixed short-history velocity pilot

Current goal: reliable learned ball returns throughout matches, without backpropagation.
Selected model remains3ac7f87f. The approved64-RBF test produced no verified candidate;
no further RBF sweep is authorized.

Parent and Sol independently select a smaller motion-estimation experiment. Keep
frozen vision,4389-dimensional transport layout, scaler/cap, actuator and3ac linear
head. Replace only existing observed-velocity estimates with fixed least-squares
slopes over the last four observations of each generic tracked entity, using observed
positions/IDs and declared nominal sample timestamps. No game-state association,
ball labels, privileged timing/state, expert actions, or new policy input channels.
Use an experimental adapter, preserving existing default code/checkpoints.

An earlier selected miss had improved velocity error under this estimator, but
that privileged passive diagnostic was not learning evidence. Smoothing may lag
rebounds and shift the readout's input distribution. Test generic constant-motion,
reversal and reset behavior before16 predeclared phase-balanced TRAIN games:
eight later misses and eight stable/high-return successes where available. Compare
actual total/first/later returns with existing3ac receipts and capture new raw inputs.
Old raw trajectories cannot certify reuse under the changed estimator.

Fixed filter; no window/reset/bounce-rule sweep. Experimental budget10minutes,
implementation separately recorded. Clear actual improvement with no opening loss
permits planning one own-reward linear learning check using filtered inputs; a frozen
head improvement alone is not new learning. Neutral/harmful result ends this branch.
Parent reviews before further learning/full-bank expansion.

Agent: reused `/root/rally_integration`, Sol6.1XHigh. Parent owns selection, independent
review and interpretation. Bridge work remains assigned to the other thread.

## Fixed four-observation velocity pilot — rejected

Root `pong-rally-velocity-ls-20261028-r1`, agentdc5d29a→main0d8f6b6. Experimental generic LS estimator used only observed tracked positions and nominal sample times, changing existing velocity/relative-velocity slots. Frozen3ac head, vision/scaler unchanged; no new inputs/learning. Thirteen focused tests passed; synthetic constant motion exact, scene/new-track reset correct, sharp reversal one-sample direction lag. Implementation192.4s separately.

All16 predeclared phase-balanced TRAIN cases actually executed:66→52total,16→16first,50/58→36/46later (86.21→78.26%). Eight selected later-miss games improved18→26; eight previously strong games fell48→26. Four gains/eight losses/fourunchanged. Cost10,819transition cameras+16initials,44.37s, source/base frozen. Reject estimator under the unchanged head and stop this pilot; no window/reset/bounce-rule sweep or filtered-input learning. This is not proof that no learner could adapt to filtered inputs. No checkpoint promotion.

Parent next hypothesis: keep original transport and relax incidental post-burst imitation, while retaining the target's approach and own-reward burst choices plus known successful openings. Await independent Sol critique before implementation.
