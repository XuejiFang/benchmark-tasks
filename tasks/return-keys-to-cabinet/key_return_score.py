"""Geometric identification of a key ring hanging from its assigned hook."""
import math

import numpy as np
from scipy.spatial.transform import Rotation

HOOK_Y = {'blue_key': .09, 'orange_key': -.09}


def threaded_on_shaft(pose_xyzw, cabinet_x, hook_y):
    """Require the shaft to cross inside the ring's aperture, clear of its tip."""
    pose = np.asarray(pose_xyzw)
    to_ring = Rotation.from_quat(pose[3:]).inv()
    shaft = np.array([[cabinet_x + .082, hook_y, .997],
                      [cabinet_x + .145, hook_y, .997]])
    first, second = to_ring.apply(shaft - pose[:3])
    span = second - first
    # A nearly parallel shaft or a crossing outside the shaft is not a hang.
    if abs(span[1]) < .025:
        return False
    fraction = -first[1] / span[1]
    if not 0. <= fraction <= 1.:
        return False
    crossing = first + fraction * span
    # Slightly looser than the centred pass-through needed when threading.
    # A released hanging ring settles with its inner wall touching the shaft.
    return bool(np.linalg.norm(crossing[[0, 2]]) <= .020 and pose[2] < .997)


def settled_hangs(rows, cabinet_x, *, steps=20, cabinet_y=0.):
    window = rows[-steps:]
    if len(window) < steps:
        return dict.fromkeys(HOOK_Y, False)
    result = {}
    for name, y in HOOK_Y.items():
        poses = np.array([row['keys'][name]['pose_xyzw'] for row in window])
        if not np.isfinite(poses).all():
            result[name] = False
            continue
        delta = poses[:, :3] - poses[-1, :3]
        rotations = Rotation.from_quat(poses[:, 3:])
        angular_drift = (rotations[-1].inv() * rotations).magnitude()
        released = all(sum(row['keys'][name].get('gripper_force_n',())) <= .1 for row in window)
        stable = (released and np.max(np.linalg.norm(delta, axis=1)) <= .003
                  and np.max(angular_drift) <= math.radians(1.))
        result[name] = bool(stable and all(threaded_on_shaft(pose, cabinet_x, y+cabinet_y) for pose in poses))
    return result
