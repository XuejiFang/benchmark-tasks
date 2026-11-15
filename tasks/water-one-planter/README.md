# Water one planter and save half the pitcher

Pour about half the water from the teal pitcher into the terracotta planter. Save the rest in the pitcher for later, keep the green planter dry, and return the pitcher upright to its coaster. Release the pitcher.

## Run and score

Use the Linux environment in the [RoboCurve YAM walkthrough](https://tasks.robocurve.org/walkthrough). The reference stack runs Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, YAM adapter `0059d02a1a91bc6834c1d0846022891df0c78730`, Inspect Robots 0.60.0, inspect-robots-dream 0.1.0, and dreamlabs-client 1.0.14. NumPy, SciPy, Pillow, PyTorch, and imageio-ffmpeg are also required.

Run from this folder. Use a fresh output path for each trial.

```bash
isaac_python=/data/IsaacLab-Arena/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=2

"$isaac_python" run_watering_robot_probe.py --asset assets/watering.usda --output runs/expert
"$isaac_python" audit_watering_trial.py runs/expert

# Configure the Dream Labs credential according to the walkthrough.
"$isaac_python" run_watering_robot_probe.py --asset assets/watering.usda --dream --output runs/dream
"$isaac_python" audit_watering_trial.py runs/dream

"$isaac_python" run_watering_robot_probe.py --asset assets/watering.usda --setup-only --output runs/setup
```

Seed 0 fixes the published starting layout. The limit is 2,400 controls at 20 Hz or 120 simulated seconds. Each video frame corresponds to a control. The trace records the observed water counts, pitcher pose, and gripper contact for each control. The expert changes only arm and finger commands after reset. PhysX advances the pitcher and all individual water proxies.

The reference expert scores 1.0 in 848 controls or 42.4 simulated seconds. It delivers 108 drops and saves 88, with no water in the green planter or on the worktop. Dream-1 scores 0.0 through 2,400 controls and 50 requests. The run took about 19 minutes on an L40S with other simulation jobs running.

## Fixed setup

Positions are in metres in the dual-arm frame. Positive X is forward from the robot bases and positive Y is to their left.

| Object | Size and represented material | Reset |
| --- | --- | --- |
| Teal pitcher | 69 × 69 × 64 mm polypropylene body, about 270 g empty. Two rubberized grip tabs, a pouring lip and a 36 × 40 mm opening. | Upright with base at approximately `(0.24, 0, 0.64)` on the coaster. The opening faces positive X. |
| Water | 196 equally weighted, 5.2 mm round blue proxies, 0.074 g each | All begin in the pitcher in a 7 × 7 × 4 grid. |
| Terracotta planter | 111 × 109 × 71 mm unglazed clay tray with a small seedling | Fixed at `(0.415, 0, 0.64)`. Seedling sits in the rear corner. |
| Green planter | 111 × 109 × 71 mm glazed ceramic tray with a small seedling | Fixed at `(0.415, 0.153, 0.64)`. Seedling sits in the rear corner. |
| Coaster | 112 mm diameter, 1 mm cork outline rendered on the worktop | Center at `(0.24, 0, 0.64)`. |
| Worktop and floor | Wooden 650 × 720 × 40 mm surface and 3 × 3 × 0.04 m matte floor | Worktop center `(0.415, 0.03, 0.62)`. Floor center `(0.45, 0, -0.02)`. |
| YAM pair and adapter marker | Two six-axis arms. 10 mm bookkeeping marker outside the working area. | Bases at `(0, ±0.285, 0.60)`. Marker center at `(-0.5, 0, -0.05)`. |

## Outcome

The green planter must receive zero water throughout. The following conditions hold for the final 20 controls. At most eight drops may remain outside either planter and the pitcher.

- **0.5** when the terracotta planter holds 45–134 drops and at least 50 remain in the pitcher.
- **0.3 more** when the terracotta planter holds 80–116 drops, roughly half the initial dose, and at least 76 remain in the pitcher.
- **0.2 more** when the pitcher is upright within 8°, its base is within 18 mm horizontally and 9 mm vertically of the coaster datum, and summed gripper contact is at most 0.1 N.

The rigid proxies model delivered volume and spillage. Their flow differs from water at small scales, and plant stems and leaves are visual. The planters collect the dose without modelling absorption. A physical equivalent can use a scale to weigh the two planters and the pitcher separately before and after. Isolating catches beneath the planters allows spilled water to be weighed separately from the amount that reached the planters.
