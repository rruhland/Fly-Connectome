import unittest
from run_rally_raw_fresh_screen import paired_metrics


def episodes(hits):
    return [dict(seed=i+1,report=dict(hits=h),annotation=dict(first_incoming=dict(success=h>0,tick=10),own_opportunities=h+1)) for i,h in enumerate(hits)]


class RawFreshTests(unittest.TestCase):
    def test_distinct_carried_references_keep_their_paired_differences(self):
        original=episodes([1]*32);scaffold=episodes([3]*32);candidate=episodes([2]*32)
        versus_original=paired_metrics(original,candidate);versus_scaffold=paired_metrics(scaffold,candidate)
        self.assertEqual(versus_original['mean_hits']['difference'],1.)
        self.assertEqual(versus_original['mean_hits']['paired_bootstrap95'],[1.,1.])
        self.assertEqual(versus_scaffold['mean_hits']['difference'],-1.)
        self.assertEqual(versus_scaffold['mean_hits']['paired_bootstrap95'],[-1.,-1.])
        self.assertEqual(versus_original['at_least2']['gained_games'],32)
        self.assertEqual(versus_scaffold['at_least3']['lost_games'],32)
        self.assertAlmostEqual(versus_original['later_return_fraction']['difference'],.5)
        self.assertAlmostEqual(versus_scaffold['later_return_fraction']['difference'],-1/6)

    def test_nonpaired_or_missing_incoming_context_is_rejected(self):
        original=episodes([1]*32);candidate=episodes([2]*32);candidate[0]['seed']=100
        with self.assertRaises(AssertionError):paired_metrics(original,candidate)
        candidate=episodes([2]*32);candidate[0]['annotation']['first_incoming']=None
        with self.assertRaises(AssertionError):paired_metrics(original,candidate)


if __name__=='__main__':unittest.main()
