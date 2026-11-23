import unittest

import numpy as np

from scripts.experimental.tt24_motor import MotorEffects, MotorPolicy, optical_sightings


class MotorTest(unittest.TestCase):
    def test_orange_ball_is_separate_from_yellow_housing(self):
        image = np.full((48, 64, 3), 30, dtype=np.uint8)
        image[18:23, 8:13] = (233, 106, 18)
        image[34:48, 17:32] = (202, 29, 32)
        image[24:47, 35:64] = (255, 255, 40)

        mover, paddle = optical_sightings(image)

        np.testing.assert_allclose(mover.center, np.array((10. - 31.5, 20. - 23.5)) / 48)
        np.testing.assert_allclose(paddle.center, np.array((24. - 31.5, 40.5 - 23.5)) / 48)

    def test_blind_and_unsampled_frames_skip_optical_reading(self):
        class Detector:
            def __init__(self):
                self.calls = []

            def observe(self, rgb, previous_action):
                self.calls.append(previous_action)
                return None, None

        detector = Detector()
        effects = MotorEffects(5)
        policy = MotorPolicy(np.zeros((5, 11)), effects, detector=detector)
        observation = dict(joints=np.zeros(5), joint_velocity=np.zeros(5), rgb=np.zeros((4, 4, 3), dtype=np.uint8))
        policy.command(observation)
        policy.command(observation, sampled=False)
        policy.command(observation, blind=True)
        policy.command(observation)
        self.assertEqual(len(detector.calls), 2)
        self.assertIsNone(detector.calls[0])
        np.testing.assert_array_equal(detector.calls[1], np.zeros(5))

    def test_drive_tracking_adapts_without_discontinuity(self):
        effects = MotorEffects(5)
        effects.readout[0, 0] = 1.
        initial = effects.gain().copy()
        effects.track_change()
        np.testing.assert_allclose(effects.gain(), initial)
        effects.readout[0, 0] *= .8
        np.testing.assert_allclose(effects.gain()[0], initial[0] / .8)


if __name__ == '__main__':
    unittest.main()
