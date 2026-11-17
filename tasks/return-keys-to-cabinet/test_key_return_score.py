"""Regressions for hanging rings whose blades never enter their guide."""
from copy import deepcopy
import unittest

from scipy.spatial.transform import Rotation

from audit_key_return_trial import audit
from key_return_score import HOOK_Y, settled_hangs


class KeyReturnScoreTest(unittest.TestCase):
    cabinet_x, cabinet_y = .55, -.06

    def setup_rows(self):
        orientation = Rotation.from_euler('z', 90, degrees=True).as_quat().tolist()
        keys = {
            name: {
                'pose_xyzw': [self.cabinet_x + .112, y + self.cabinet_y - .0015, .982, *orientation],
                'gripper_force_n': [0.],
            }
            for name, y in HOOK_Y.items()
        }
        frame = {'keys': keys, 'hinge_deg': 0., 'contacts_n': {'door': [0.]}}
        return [deepcopy(frame) for _ in range(20)]

    def test_both_blades_seated_and_closed_door(self):
        self.assertEqual(audit(self.setup_rows(), self.cabinet_x, self.cabinet_y)['score'], 1.)

    def test_both_rings_threaded_with_blades_outside_guides(self):
        rows = self.setup_rows()
        for row in rows:
            row['keys']['blue_key']['pose_xyzw'][0] += .020
            row['keys']['orange_key']['pose_xyzw'][2] += .006
        self.assertTrue(all(settled_hangs(rows, self.cabinet_x, cabinet_y=self.cabinet_y).values()))
        self.assertEqual(audit(rows, self.cabinet_x, self.cabinet_y)['score'], 0.)

    def test_one_frame_out_of_guide_removes_credit_for_that_key(self):
        rows = self.setup_rows()
        rows[0]['keys']['orange_key']['pose_xyzw'][0] += .002
        self.assertTrue(all(settled_hangs(rows, self.cabinet_x, cabinet_y=self.cabinet_y).values()))
        result = audit(rows, self.cabinet_x, self.cabinet_y)
        self.assertEqual(result['keys']['orange_key']['blade_seated_frames'], 19)
        self.assertEqual(result['score'], .40)


if __name__ == '__main__':
    unittest.main()
