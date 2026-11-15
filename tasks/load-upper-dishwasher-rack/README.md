# Load two washable cups into the upper dishwasher rack

Open the upper rack, load the two washable cups openings down and slide the loaded rack back in. Keep the hand-wash wooden cup upright at its worktop.

## Run and score a trial

Use the Linux image in the [RoboCurve YAM walkthrough](https://tasks.robocurve.org/walkthrough). The reference environment uses Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, YAM adapter `0059d02a1a91bc6834c1d0846022891df0c78730`, Inspect Robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14. The runner also uses NumPy, SciPy, Pillow, PyTorch and imageio-ffmpeg.

Run from the task folder with fresh output directories.

```bash
isaac_python=/data/IsaacLab-Arena/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1

"$isaac_python" run_dishwasher_rail_probe.py --usd assets/dishwasher/dishwasher_ldfn4542b.usd \
  --load assets/cups --place --output runs/expert
"$isaac_python" audit_dishwasher_trial.py runs/expert

# Configure the Dream Labs credential using the walkthrough.
"$isaac_python" run_dishwasher_rail_probe.py --usd assets/dishwasher/dishwasher_ldfn4542b.usd \
  --load assets/cups --dream --output runs/dream
"$isaac_python" audit_dishwasher_trial.py runs/dream

"$isaac_python" run_dishwasher_rail_probe.py --usd assets/dishwasher/dishwasher_ldfn4542b.usd \
  --load assets/cups --setup-only --output runs/setup
```

The fixed scene uses seed 0 and disables object jitter. The controller receives 20 actions per simulated second for up to 2,400 actions. Each H.264 video frame corresponds to one action. The trace includes robot commands, rack position, cup poses and gripper contact. The setup screenshot uses a higher front camera to show all three cups at reset.

The reference expert scored 1.0 in 1,084 actions (54.2 simulated seconds). The Dream-1 run scored 0.0 in 2,400 actions (120 simulated seconds) and used 50 API requests. Recording took roughly 8 minutes for the expert and 14 minutes for Dream-1 on the reference L40S, which was also running other trials.

## Fixed setup

Coordinates are metres in the dual-arm frame. Positive X points into the appliance and positive Y is to the robots' left.

| Object | Dimensions and represented material | Reset |
| --- | --- | --- |
| Dishwasher | Approximately 665 mm deep, 612 mm wide and 858 mm tall. Steel chassis, polymer door trim, metal wire racks | Base at `(0.68, 0, 0)` with 180° yaw. Door starts at the 90° horizontal stop. The upper rack settles at its retracted position. |
| Upper rack grip | Orange polymer clip, 32 × 17 × 21 mm | Fixed to the right front edge of the upper rack |
| Tall blue cup | Glazed ceramic, 70 mm outside diameter, 102 mm tall | Inverted, native base at `(0.219, 0.391, 0.692)` |
| Short light-tan cup | Glazed ceramic, 70 mm outside diameter, 96 mm tall | Inverted, native base at `(0.312, 0.479, 0.686)` |
| Dark wooden cup | Sealed wood, 70 mm outside diameter, 64 mm tall | Upright on the opposite worktop at `(0.294, -0.442, 0.590)` |
| Worktops | Wood, 201 × 205 × 40 mm and 186 × 180 × 40 mm | Centred at `(0.257, 0.44, 0.57)` and `(0.268, -0.446, 0.57)` |
| Two YAM arms | Six axes and parallel grippers on 150 × 150 mm pedestals, 590 mm high | Bases `(0, 0.39, 0.59)` and `(0, -0.39, 0.59)` |
| Floor and adapter marker | Concrete floor 4 × 4 × 0.04 m and 10 mm adapter marker outside the working area | Floor centre `(0.45, 0, -0.02)`, marker at `(-0.5, 0, -0.05)` |

## Scoring

- **0.20 for opening the rack.** It travels outward at least 225 mm from its starting rail position and remains out by that amount for five consecutive observations.
- **0.30 per cup.** In the final 20 observations, the cup stays within 32 mm of its designated rack-local site, with its opening direction within 20° of straight down and the combined fingertip contact at most 0.1 N. Sites are `(0.154, -0.099, 0.0773)` m and `(0.167, -0.190, 0.078)` m.
- **0.20 for completion.** After opening and loading both cups, the released rack sits within 10 mm of its starting position and the wooden cup stays within 18 mm and 12° of its upright start. Rack contact force is at most 0.1 N throughout the final observations.

The appliance starts anchored with the door held horizontal. The rack retains its source rail and roller, and the cups move through gravity and physical contact. The setup packages all source composition dependencies and identifies the licensed dishwasher asset in [ATTRIBUTION.md](ATTRIBUTION.md).
