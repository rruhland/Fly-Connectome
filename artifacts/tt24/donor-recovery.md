# Live correction by the reward-trained wrist policy

The five-joint policy learned from rebound rewards on three donor arms. It then played 16 new stock-arm drops from wrist RGB and encoders. Ten started with three consecutive verified rebounds and qualified for an unexpected sideways impulse during the following airborne phase. The other six are included in the launch denominator. The motor readout received neither the impulse nor a contact flag. Its stock sensing was calibrated in eight earlier unlabelled drops. [Source policy](tt24_donor.md) · [Physical comparison and action tapes](donor-recovery.json.gz)

| Same ten eligible starts from the instant of the impulse | Subsequent uninterrupted rebounds, summed | Drops lost inside the comparison window |
| --- | ---: | ---: |
| Ordinary undisturbed play | 49 | 2 |
| Disturbed ball, live wrist feedback | 41 | 3 |
| Disturbed ball, replay previously successful joint commands | 10 | 10 |
| Disturbed ball, hold joints after the kick | 11 | 10 |

Interventions added 0.16 or 0.24 m/s on a randomly allocated horizontal axis and sign. The four branches originated from the same hashed simulated state. For each eligible start, the shared comparison window covered up to two seconds and stopped at the end of the original command tape if earlier. The window therefore measures early recovery. The reference point for consecutive returns was the last verified impact before the impulse. All 46 full or perturbed physical tapes reverified. [Method](../../scripts/experimental/tt24_donor_recovery.py)
