import unittest
from run_rally_repertoire_collection import select_starts
import run_practical_pong_v2 as base


class RepertoireScheduleTests(unittest.TestCase):
    def test_balanced_reproducible_outcome_independent_schedule(self):
        a=select_starts(1943906127,63025,set());b=select_starts(1943906127,63025,set());self.assertEqual(a,b);self.assertEqual(len(a['TRAIN']),64)
        self.assertTrue(all(base.leftward(s) for s in a['TRAIN'][::2]));self.assertTrue(all(not base.leftward(s) for s in a['TRAIN'][1::2]));self.assertEqual(len({s%base.MODULUS for s in a['TRAIN']}),64)

    def test_prior_seed_aliases_are_excluded_without_mutating_reservations(self):
        a=select_starts(1943906127,63025,set());excluded={s%base.MODULUS for s in a['TRAIN'][:12]};before=excluded.copy();b=select_starts(1943906127,63025,excluded)
        self.assertEqual(excluded,before);self.assertFalse({s%base.MODULUS for s in b['TRAIN']}&excluded);self.assertEqual(len(b['TRAIN']),64);self.assertTrue(any(r['reason']=='prior_or_reserved_canonical_seed' for r in b['rejected']))


if __name__=='__main__':unittest.main()
