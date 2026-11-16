"""Read-only physical packing measurements shared by model-run recording.

Observe stability after release so objects can be supported by other packed items.
No planning or action generation.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from layout import ACCESSORIES, BAG_POS, BAG_R, ITEMS, MIN_END_GAP, PACKING_POINTS, SHOES, WORLD_ROT, corners, end_gap, task_pos
from geometry import vertices


def array(value):
    if hasattr(value, 'torch'):
        value = value.torch
    return value.detach().cpu().numpy()


class PackingObserver:
    def __init__(self, base):
        self.base = base
        self.held = dict.fromkeys(ITEMS, 0)
        self.separated_steps = 0
        self.points = {name: vertices(name) for name in ITEMS}

    def capture(self, action, step):
        from yam_arena.runtime import tcp_world
        base = self.base
        robots = {side: {'joint_positions': array(base.scene[side + '_robot'].data.joint_pos)[0].tolist(),
                         'tcp': tcp_world(base.scene[side + '_robot']).tolist()}
                  for side in ('left', 'right')}
        row = {'step': step, 'time_s': (step + 1) / 20, 'actions': np.asarray(action).tolist(),
               'robots': robots, 'objects': {}}
        points_earned = 0
        bounds = {}
        for name, cfg in ITEMS.items():
            data = base.scene.rigid_objects[name].data
            pose = array(data.root_pose_w)[0]
            pos = pose[:3]
            rotation = Rotation.from_quat(pose[3:]).as_matrix()
            velocity = array(data.root_lin_vel_w)[0]
            angular = array(data.root_ang_vel_w)[0]
            points = (self.points[name] @ rotation.T + pos - task_pos(BAG_POS)) @ (WORLD_ROT @ BAG_R)
            sensor = base.scene.sensors['bag_contact_' + name].data
            normal = array(sensor.normal_force_matrix_w).reshape(-1, 7, 3).sum(0)
            friction = array(sensor.friction_force_matrix_w).reshape(-1, 7, 3).sum(0)
            force = normal[0] + friction[0]
            gripper_force = float(np.linalg.norm(normal[1:], axis=1).sum())
            low, high = points.min(0), points.max(0)
            bounds[name] = (low[0], high[0])
            contained = bool(np.all(low >= [-.30, -.18645, .0435]) and
                             np.all(high <= [.30, .18645, .245]))
            supported = bool(force[2] > cfg['mass'] * 9.81 * .5)
            center = (corners(name) @ rotation.T + pos).mean(0)
            released = gripper_force < .02 and all(
                np.linalg.norm(np.asarray(robots[side]['tcp']) - center) > .16 for side in robots)
            qualified = contained and released and np.linalg.norm(velocity) < .01 and np.linalg.norm(angular) < .1
            self.held[name] = self.held[name] + 1 if qualified else 0
            if self.held[name] >= 40:
                points_earned += PACKING_POINTS[name]
            row['objects'][name] = {
                'pos': pos.tolist(), 'quat_xyzw': pose[3:].tolist(),
                'velocity': velocity.tolist(), 'angular_velocity': angular.tolist(),
                'bag_min': low.tolist(), 'bag_max': high.tolist(),
                'bag_force': force.tolist(), 'contained': contained, 'supported': supported,
                'grippers_clear': bool(released), 'gripper_normal_force_N': gripper_force,
                'held_steps': self.held[name],
            }
        gap = end_gap(bounds)
        group_qualified = all(self.held[name] > 0 for name in (*SHOES, *ACCESSORIES))
        self.separated_steps = self.separated_steps + 1 if group_qualified and gap >= MIN_END_GAP else 0
        row['end_gap_m'] = gap
        row['separated_steps'] = self.separated_steps
        row['score'] = (points_earned + 20 * (self.separated_steps >= 40)) / 100
        return row
