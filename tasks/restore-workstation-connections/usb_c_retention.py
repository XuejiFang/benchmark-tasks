"""Depth-dependent axial snap force for the stock Type-C mating pair."""

import numpy as np

PORT_X = .390
PORT_Z = .803
MALE_PORTS_Y = (-.220, -.193, .220)
TARGET_SEATED_DEPTH = .0062
MIN_SEATED_DEPTH = .0058
MAX_SEATED_DEPTH = .00665
PHYSICS_HZ = 960

# Positive depth is into the receptacle. The negative lobe compresses the
# latches during entry, and the positive lobe pulls them into the detent.
# These are nominal equivalent-joint targets, not a measured Molex curve.
DETENT_DEPTH_M = (0., .001, .0032, .00475, .00575, .0062, .00665)
DETENT_FORCE_N = (0., 0., -9.7, 0., 12., 0., 0.)
VISCOUS_NS_M = 14.


def axial_force(depth, axial_speed=0.):
    """One-axis bench model with no held/attached state and no position writes."""
    spring = np.interp(depth, DETENT_DEPTH_M, DETENT_FORCE_N, left=0., right=0.)
    gate = np.clip(np.minimum((depth-.001)/.0005, (.00665-depth)/.0003), 0., 1.)
    return spring - VISCOUS_NS_M*gate*axial_speed


def install_retention(observer, paths, device):
    """Apply the same passive detent in demonstrations and model runs."""
    import omni.physx
    import torch

    views = [observer.bodies[path] for path in paths]
    if any(view.count != 1 for view in views):
        raise RuntimeError('Both live USB-C plug bodies are required for retention.')
    depth_knots = torch.tensor(DETENT_DEPTH_M, dtype=torch.float32, device=device)
    force_knots = torch.tensor(DETENT_FORCE_N, dtype=torch.float32, device=device)
    ports_y = torch.tensor(MALE_PORTS_Y, dtype=torch.float32, device=device)
    index = torch.zeros(1, dtype=torch.int32, device=device)
    zero = torch.zeros(1, dtype=torch.float32, device=device)
    last = torch.zeros(len(paths), dtype=torch.float32, device=device)

    def update_forces(_dt):
        for i, view in enumerate(views):
            poses = view.get_transforms()
            velocities = view.get_velocities()
            x, y, z, qx, qy, qz, _qw = getattr(poses, 'torch', poses)[0]
            speed = getattr(velocities, 'torch', velocities)[0, 0]
            depth = x - PORT_X
            section = torch.bucketize(depth, depth_knots).clamp(1, len(DETENT_DEPTH_M)-1).reshape(1)
            start, end = depth_knots[section-1], depth_knots[section]
            fraction = ((depth-start)/(end-start)).clamp(0., 1.)
            spring = torch.lerp(force_knots[section-1], force_knots[section], fraction)
            envelope = torch.minimum((depth-.001)/.0005, (.00665-depth)/.0003).clamp(0., 1.)
            # A misaligned or backwards plug cannot be attracted through a
            # socket wall. The third, unmarked, receptacle has the same force.
            aligned = (
                ((y-ports_y).abs().min() <= .00030) &
                ((z-PORT_Z).abs() <= .00030) &
                (1.-2.*(qy*qy+qz*qz) >= .99863) &
                ((1.-2.*(qx*qx+qy*qy)).abs() >= .99863)
            )
            force_x = (spring - VISCOUS_NS_M*envelope*speed) * aligned
            last[i].copy_(force_x[0])
            forces = torch.stack((force_x, zero, zero), dim=1)
            view.apply_forces_and_torques_at_position(forces, None, None, index, True)

    subscription = omni.physx.get_physx_interface().subscribe_physics_on_step_events(
        update_forces, True, 0)
    return subscription, last
