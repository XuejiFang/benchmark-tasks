"""Dimension the trim ballast and print the unloaded seat's force targets.

The seat's side view uses u towards the front, z upwards, and the hinge
at the origin. Low-speed readings exclude the rotary damper by extrapolating
the mean of upward and downward pull readings to zero speed.
"""

import argparse
import math


GRAVITY = 9.81
TARGET_MASS = 5.9505827655
TARGET_U = .005834417
TARGET_Z = -.000025345531
TARGET_HINGE_INERTIA = .1024673405
GAUGE_RADIUS = math.hypot(.230, .051)
FRONT_RAIL = .330
REAR_RAIL = -.130


def ballast(bare_mass, bare_u, bare_inertia):
    """First-pass masses assuming small ballast blocks centred at z=0."""
    moment = TARGET_MASS * TARGET_U - bare_mass * bare_u
    inertia = TARGET_HINGE_INERTIA - bare_inertia
    determinant = FRONT_RAIL * REAR_RAIL * (REAR_RAIL - FRONT_RAIL)
    front = (moment * REAR_RAIL**2 - inertia * REAR_RAIL) / determinant
    rear = (inertia * FRONT_RAIL - moment * FRONT_RAIL**2) / determinant
    shaft = TARGET_MASS - bare_mass - front - rear
    if min(front, rear, shaft) < 0:
        raise ValueError('This bare assembly needs different ballast positions or a lighter deck.')
    return front, rear, shaft


def lift_force(degrees):
    """Tangential force at the rail due to gravity, with the keeper lifted."""
    angle = math.radians(degrees)
    offset = TARGET_U * math.cos(angle) - TARGET_Z * math.sin(angle)
    return TARGET_MASS * GRAVITY * offset / GAUGE_RADIUS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bare-mass', type=float, help='Unballasted rotating assembly mass in kg')
    parser.add_argument('--bare-u', type=float, help='Unballasted horizontal centre of mass in m')
    parser.add_argument('--bare-inertia', type=float, help='Unballasted hinge inertia in kg m²')
    args = parser.parse_args()
    values = (args.bare_mass, args.bare_u, args.bare_inertia)
    if any(value is not None for value in values):
        if any(value is None or not math.isfinite(value) for value in values):
            parser.error('Provide all three finite bare assembly measurements.')
        if args.bare_mass <= 0 or args.bare_inertia <= 0:
            parser.error('Mass and inertia must be positive.')
        front, rear, shaft = ballast(*values)
        print(f'First-pass ballast in kg: front {front:.3f}, rear {rear:.3f}, at hinge {shaft:.3f}')
    print('seat_angle_deg,tangential_gravity_force_n')
    for angle in (0, 15, 30, 45, 60, 75, 85, 89):
        print(f'{angle},{lift_force(angle):.3f}')
