import unittest
from pong_rally_credit import outcome_from_record,query_positions,burst_plan
class NextReturnTests(unittest.TestCase):
 def test_past_hits_do_not_turn_future_miss_into_success(self):
  self.assertEqual(outcome_from_record(dict(hits=0,reward=-1,terminated=True,truncated=False)),(0,'own_miss'))
  self.assertEqual(outcome_from_record(dict(hits=1,reward=0,terminated=False,truncated=False)),(1,'next_own_hit'))
 def test_censoring_not_failed_return(self):
  self.assertIsNone(outcome_from_record(dict(hits=0,reward=1,terminated=True,truncated=False))[0])
  self.assertIsNone(outcome_from_record(dict(hits=0,reward=0,terminated=False,truncated=True))[0])
 def test_segment_boundary_and_effectively_identical_bursts(self):
  positions=query_positions(200,100);self.assertEqual(positions,[101,120,160])
  p=burst_plan(200,positions);self.assertEqual(len(p),33)
  self.assertEqual(len({(v['transition'],v['effective_hold'],v['action']) for v in p}),len(p))
  self.assertEqual({v['action'] for v in p},{0,1,2})
if __name__=='__main__':unittest.main()
