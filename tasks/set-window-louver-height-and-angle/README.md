# Set a window blind's height and louver angle

Raise the blue lower rail handle to the stripe on the window frame. Press the orange lever until all the slats angle 28 degrees down toward you. Release both handles and leave the blind in this position.

The lower rail height and louver angle control different parts of the view through the window. The operation must end with both settings holding without hand support.

## Run and score

Use the Linux environment in the [RoboCurve YAM walkthrough](https://tasks.robocurve.org/walkthrough). The reference stack uses Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, YAM adapter `0059d02a1a91bc6834c1d0846022891df0c78730`, Inspect Robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14. It also uses NumPy, SciPy, Pillow, PyTorch and imageio-ffmpeg.

From this directory, use a fresh output directory per trial.

```bash
isaac_python=/data/IsaacLab-Arena/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=2

"$isaac_python" run_blind_robot_probe.py --asset assets/blind.usda --output runs/expert
"$isaac_python" audit_blind_trial.py runs/expert

# Configure Dream Labs credentials according to the walkthrough.
"$isaac_python" run_blind_robot_probe.py --asset assets/blind.usda --dream --output runs/dream
"$isaac_python" audit_blind_trial.py runs/dream

"$isaac_python" run_blind_robot_probe.py --asset assets/blind.usda --setup-only --output runs/setup
```

Seed 0 fixes the published starting layout. Each video frame in the action recordings represents one control. The run limit is 2,400 controls at 20 Hz, or 120 simulated seconds. The trace records the observed rail displacement, each slat angle, and gripper contact in every control. The expert changes only the arm and finger commands after reset.

The independent expert scores 1.0 in 676 controls or 33.8 simulated seconds. Released rail lift is 82.68 mm and released slat angles range from −29.84° to −29.85°. Dream-1 scores 0.0 in 2,400 controls and 50 requests. The Dream-1 run takes about 11 wall-clock minutes on an L40S under a concurrent workload.

## Fixed setup

Positions in the table are metres unless labelled otherwise. Positive X extends forward from the robot bases, positive Y points left, and positive Z points up. The origin of the blind fixture is `(0.360, 0.035, 0.630)` with its horizontal slats spanning the Y direction.

| Object | Size and represented material | Reset |
| --- | --- | --- |
| Top rail and side guides | Painted aluminium, 95 × 376 × 26 mm for the top rail and 35 × 20 × 303 mm per guide | Top rail centered 361 mm above the fixture origin. Guides sit 173 mm on each side of its centerline. |
| Sliding bottom rail and handle | 44 × 308 × 18 mm painted aluminium rail with two 23 × 11 × 201 mm guide tongues and a 38 × 35 × 22 mm blue moulded grip | Rail center starts 64 mm above the fixture origin. Grip center is offset −131 mm in X and +169 mm in Y from the origin at the same height. |
| Linked slats and tilt lever | Three aluminium slats, each 75 × 315 × 3 mm, joined by a rigid 132 mm linkage. The orange lever tab is 37 × 41 × 16 mm. | Slats start horizontal and centered 115, 169 and 223 mm above the fixture origin. The lever projects from the middle slat toward the user. |
| Height stripe and slide brake | 8 × 17 × 8 mm blue painted mark proxy, fixed aluminium track and spring-loaded rubber shoe | Stripe is 149 mm above the fixture origin. The shoe preloads the moving carriage against the track, and the mechanism permits 108 mm of travel. |
| Window surround, pane, and floor | 32 × 590 × 700 mm painted surround, 21 × 342 × 350 mm blue glass proxy, 3 × 3 × 0.04 m matte floor | Centers are `(0.535, 0.058, 0.760)`, `(0.504, 0.035, 0.825)` and `(0.450, 0, −0.020)` respectively. |
| YAM pair and adapter marker | Two six-axis arms and one 10 mm bookkeeping marker outside the work area | Bases at `(0, ±0.305, 0.750)`. Marker center at `(−0.5, 0, −0.05)`. |

## Outcome

Score the final 20 controls and add the following independently.

- **0.45** if the lower rail is raised by 85 ± 8 mm throughout, with at most 0.1 N of summed gripper contact on the rail.
- **0.55** if all three slats are tilted 28 ± 4° downward toward the user throughout, with at most 0.1 N of summed gripper contact on the tilt body.

A ruler and angle gauge can measure the corresponding state after both handles are released in a physical trial. Reset the rail to the lower stop, the slats to horizontal and the arms to their starting poses before each run.

The short three-slat mock-up models a single guided unit. The counterweighted slats are linked by physical hinges. Lift and tilt both persist after the hands leave because of spring-preloaded friction and balance. The window is an opaque stand-in, and the assembly omits flexible cords, fabric tape and lighting effects.
