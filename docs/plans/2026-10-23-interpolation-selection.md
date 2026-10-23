# Combine learned interception changes in one existing head

The checked balanced trial learned direct returns (+18.75pp, paired95%CI[9.375,28.125]) while damaging opponent-return interception (−21.875pp, CI[−34.375,−10.9375]). Balanced lift was −1.5625pp. Nine substantial updates rule out the earlier tiny-update explanation for this allocation. This is evidence of a tradeoff under that training recipe, not proof that the linear policy class cannot hold both skills.

Sol independently proposed a fresh balanced restart from the direct-trained ancestor to test path dependence. Astra challenged the cost of another160k allocation and proposed first testing the parameter segment between retained RIGHTtheta6 and the newly direct-improved balancedtheta9. Sol endorsed this order after considering that greedy actions and closed-loop returns are nonlinear in weights: an interior policy could combine behaviors even when endpoints trade off. Astra added prospective unchanged-policy futility. This is a deliberate direction change, not another noise/rate sweep.

## Fixed experiment

Five coefficients0,.25,.5,.75,1 define literal theta=(1−alpha)START+alphaDONOR, with exact endpoint copies. The result is one unchanged linear head with existing frozen generic vision; no runtime ensemble, backpropagation, teacher or privileged policy input. Explicit assistance comprises the research-selected segment, two reward-trained ancestors and balanced native-start curriculum. DONOR failed the joint gate and is not promoted.

Fresh32LEFT+32RIGHT starts shared across all five heads give320 ordinary binary first-hit training episodes. Select highest mean training fitness, lowest coefficient on ties, only after all320 complete. No adaptive grid and no heldout selection. If alpha0 wins, stop with unchanged policy and no fresh heldout claims. Otherwise evaluate the selected head on fresh64LEFT,64RIGHT and64ordinary-game starts against the same controls and all eight prior gates. Report detectable gains below10pp honestly while retaining the preregistered minimum-worthwhile screen. Absolute80% competence is separate. No result establishes an autonomous continuous learner or complete rally learning.

Immutable caps:50k training transitions/420s,900s evaluation,1320s overall; no extension. All prior accepted, rejected and unused schedules excluded. Inherited donor ancestry718686 training transitions; interrupted14879 remain separately paid unused exposure. Reuse no partial fitness. Parent independent stage PASS0.252s, nine focused tests PASS4.32s and Sol stage/check PASS, all0 observations/replay. Source68d15a7 unchanged. Freeze through both independent result checks.

Full protocol and driver: `/workspace/m1b-experiments/pong-interpolation-interception-20261023-r1/`. Parent review artifacts: `/workspace/m1b-experiments/pong-mechanism-review-20261022/interpolation-parent-stage.json` and corresponding checker. User session ends08:09UTC; this allocation fits with closure reserve.

## Outside evidence and limits

Astra read the abstract/introduction of [Wortsman et al., WiSE-FT, CVPR2022](https://arxiv.org/html/2109.01903v3). Their weight interpolation motivates testing adaptation/retention combinations; with a fixed linear readout, logits interpolate algebraically without added inference cost. Their supervised gradient-trained image classifiers neither establish a Pong guarantee nor supply our learning rule. Here fresh scalar game reward chooses the coefficient without gradients. The experiment was selected from the observed tradeoff and cost before this source was consulted.

If a changed head passes, independently replicate the viable recipe before promotion. If the segment fails, reject this bounded grid/selection, not the whole class; no finer-grid rescue. Reconsider the direct-ancestor balanced restart or mechanisms with the actual outcome and remaining session budget.

## Blind follow-up challenge (06:56UTC, before current outcomes)

Astra asked Sol to compare direct-ancestor restart against changing the scalar objective to protect the weaker phase. Merely balancing exposure does not prevent mean fitness from buying one phase with losses in the other. A max-min phase fitness would target that mechanism without changing policy inputs or architecture, but six starts per phase give coarse and noisy worst-phase estimates; this can select noise or sacrifice useful progress. A different start tests path dependence rather than this objective mismatch. Neither is authorized as the automatic next allocation.

Astra read abstract/introduction and setup§2 of [Sagawa et al., ICLR2020 group DRO](https://arxiv.org/html/1911.08731v2). Their worst-group objective and distinction between training group information and inference inputs help formulate this alternative. They also show why optimizing worst-group training performance alone need not improve its generalization. Our Pong phases, black-box episodic rewards and finite populations differ from their supervised experiments; their gradient algorithm and guarantees are not adopted. Any chosen worst-phase test would retain fresh heldout controls and declare training-only phase information as scaffold assistance.

Sol independently favors the worst-phase objective over the direct restart if the current segment fails: it directly changes permission to sacrifice a phase. Preserve6+6 starts, perturbation/update scales and all other search mechanics to avoid bundling coverage/dose changes. A fresh unpaired training run would remain a mechanism-informed attempt, not a clean causal comparison against the prior mean run. Stage only after outcome review and only if the session closure reserve remains.
