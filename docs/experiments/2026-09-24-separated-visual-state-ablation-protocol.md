# Imagined-source emission ablation

The first separated-state run sustained latent activity on blank input and reduced quiet-target false alarms, but recurrent prediction lost recall and F1 in every scene family. Freeze that trained recurrent model and zero **only** its predicted-only-to-event emission weights at evaluation. Its observed-event emission weights, latent transition weights, dictionary, and state dynamics remain unchanged. Compare this ablation with the trained recurrent model and the matched no-recurrence emission arm on the same training and unseen-scene suites.

If F1 returns within 0.03 of the no-recurrence arm, the imagined emission channel is the principal failure. If it remains low, the recurrent arm's observed emission learning or its state dynamics is underfit; do not continue gain/threshold sweeps on this architecture. This evaluation adds no training or production change.
