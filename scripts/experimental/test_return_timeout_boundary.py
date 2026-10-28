import unittest
import torch
from fly_connectome.pong import Pong
from return_metrics import ReturnCounter,native_count_status

class TimeoutBoundaryTest(unittest.TestCase):
    def test_missed_plane_can_precede_timeout_without_native_goal(self):
        game=Pong([123]);game.body.position[0]=.1;game.ball[0]=torch.tensor([game.config.paddle_x+game.config.ball_radius+.001,.9]);game.ball_velocity[0]=torch.tensor([-.55,0.])
        counter=ReturnCounter();out=counter.step(game,game.step,torch.zeros(1),auto_reset=False)
        self.assertEqual(float(out.scores[0]),0.);self.assertEqual(counter.opportunities,1);self.assertEqual(counter.returns,0)
        annotation=counter.summary(game,truncated=True,terminated=False,reward=0.,seconds=game.config.dt)
        report=dict(complete=True,hits=0,losses=0,wins=0,truncations=1)
        status=native_count_status(annotation,report);self.assertTrue(status['valid_terminal_counts']);self.assertTrue(status['pending_native_loss_at_timeout'])
        self.assertFalse(native_count_status(annotation,dict(report,truncations=0))['valid_terminal_counts'])
        self.assertFalse(native_count_status(annotation,dict(report,hits=1))['valid_terminal_counts'])

if __name__=='__main__':unittest.main()
