# Local credit separation rescues fit, but recurrent state adds little

The final bounded variant kept the frozen learned dictionary, locally learned latent transition, 128 unlabeled training episodes, and combined event forecast. It changed only the delayed local emission updates: observed-source and imagined-source synapses each received their own branch's next-event residual instead of one residual from the combined output. The test suite and all controls were unchanged. The run took 207 seconds.

| Event F1 | Shared credit recurrent | Separate credit recurrent | No-recurrence emission | Archived learner | Fixed sensory control |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 0.452 | **0.583** | 0.563 | 0.621 | 0.909 |
| Unseen single patterns | 0.528 | **0.644** | 0.640 | 0.669 | 0.971 |
| Independent movers | 0.539 | **0.678** | 0.668 | 0.681 | 0.977 |
| Crossings | 0.497 | **0.580** | 0.564 | 0.583 | 0.817 |
| Speed changes | 0.484 | **0.584** | 0.583 | 0.634 | 0.893 |
| Brief disappearance | 0.366 | **0.451** | 0.446 | 0.430 | 0.716 |
| Sensor-bit noise | 0.399 | **0.502** | 0.489 | 0.487 | 0.798 |

This confirms that a shared scalar event error was causing substantial credit interference: separate local residuals improved the recurrent model's training and held-out fit without labels or backpropagation. The new model retained imagined latent activity on **216/488** blank-input frames. Its **102** quiet-target false-alarm pixels stayed within the registered 20% tolerance relative to the archived learner's 97.

The more important recurrence gate failed. Disappearance improved only **0.005** F1 over the matched no-recurrence emission model, and noise only **0.013**; both are well below the required 0.05. The separate-credit recurrent model stayed just below the archived clean learner and far below the fixed sensory control. Thus local credit separation is a useful principle, but the present event-site latent transition is not earning its extra state machinery. Do not promote this two-bank circuit to production.

The next architectural question is **what latent transition target can teach continuity when no new event-site code exists**. The current recurrent bank learns to predict the next observed event-site winner and receives no informative target during a blank interval; carrying a prediction forward is not the same as learning where an unseen pattern should be. The next opt-in investigation should compare a locally learned state transition trained on generic partial-observation sequences against a nonrecurrent state with the same emission rule, using an explicit evaluation of hidden-state continuity and reappearance. It should not continue gain, threshold, or emission-bank sweeps on this design. Production M1A remains unchanged.
