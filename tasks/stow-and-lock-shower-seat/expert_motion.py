"""Shared observed-state motion primitives; all outputs are robot actions.

The caller supplies robot-state readers and a joint-action sink. This module has
no simulator object setters. Asset-specific grasp/contact geometry and scoring
remain outside these primitives.
"""
import math
import numpy as np
from yam_arena import ik


def cartesian_move(get_q, send, base_pos, orientation_xyzw, point, seconds, label, hz=20,
                   interpolate_orientation=False):
    """Interpolate from the observed TCP, solve IK, then send joint targets."""
    q = np.asarray(get_q()).copy()
    start, start_rotation = ik.tcp_pose(q, base_pos)
    if interpolate_orientation:
        from scipy.spatial.transform import Rotation
        end_rotation = Rotation.from_quat(orientation_xyzw).as_matrix()
    count = round(seconds*hz)
    if count < 1:
        raise ValueError('Motion duration must contain at least one control step')
    for i in range(count):
        fraction = ik.smoothstep((i+1)/count)
        target = start + fraction*(np.asarray(point)-start)
        rotation = (ik.matrix_to_quat_xyzw(ik.slerp_matrix(start_rotation,end_rotation,fraction))
                    if interpolate_orientation else orientation_xyzw)
        q, pe, re = ik.solve(target, rotation, q, base_pos=base_pos, iters=100)
        if pe > .001 or re > math.radians(.3):
            raise RuntimeError(f'{label} IK failed: {pe} m, {re} rad')
        send(q, label)


def feedback_press(move, measure, initial_target, inward_axis, target_travel,
                   increment=.00025, max_corrections=48, force_limit=8., neighbor_limit=.1):
    """Press by contact; compensate compliance using observed joint travel.

``measure`` returns (travel_m, target_contact_N, max_neighbor_contact_N).
``move`` sends actual robot controls and must enforce its per-step guards.
The returned target is a command, never the achieved button displacement.
"""
    axis = np.asarray(inward_axis, dtype=float)
    if not np.isclose(np.linalg.norm(axis), 1.):
        raise ValueError('Press axis must be a unit vector')
    point = np.asarray(initial_target, dtype=float).copy()
    move(point, 2., 'press')
    corrections = 0
    while True:
        travel, force, neighbor = measure()
        if not np.isfinite([travel, force, neighbor]).all():
            raise RuntimeError('Non-finite press feedback')
        if force > force_limit or neighbor > neighbor_limit:
            raise RuntimeError('Press contact guard exceeded')
        if travel >= target_travel:
            return {'corrections': corrections, 'command_target_m': point.tolist(),
                    'observed_travel_m': float(travel)}
        if corrections >= max_corrections:
            raise RuntimeError('Press feedback reached displacement guard')
        point += axis*increment
        move(point, .25, 'feedback press')
        corrections += 1
