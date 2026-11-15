# Restore a monitor's height and tilt

Support the monitor and press the orange paddle to unlock the height. Lower the monitor to the middle slot, let the paddle return, and tilt the screen back 10 degrees from vertical. Release the monitor.

The height lock and the tilt hinge hold different ergonomic settings on a shared desk. The final settings must hold after the hands leave.

## Run and score

Use the Linux environment in the [RoboCurve YAM walkthrough](https://tasks.robocurve.org/walkthrough). The reference stack uses Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, YAM adapter `0059d02a1a91bc6834c1d0846022891df0c78730`, Inspect Robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14. It also uses NumPy, SciPy, Pillow, PyTorch and imageio-ffmpeg.

Run from this directory and use a fresh output directory for each trial.

```bash
isaac_python=/data/IsaacLab-Arena/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=2

"$isaac_python" run_monitor_robot_probe.py --asset assets/stand.usda --output runs/expert
"$isaac_python" audit_monitor_trial.py runs/expert

# Configure Dream Labs credentials according to the walkthrough.
"$isaac_python" run_monitor_robot_probe.py --asset assets/stand.usda --dream --output runs/dream
"$isaac_python" audit_monitor_trial.py runs/dream

"$isaac_python" run_monitor_robot_probe.py --asset assets/stand.usda --setup-only --output runs/setup
```

Seed 0 fixes the starting positions. Each video frame in the action recordings represents one control. The limit is 2,400 controls at 20 Hz, or 120 simulated seconds. Each control records slide height, hinge angle, pin extension and gripper contact. The expert commands only robot joints and fingers.

The independent expert scores 1.0 in 493 controls or 24.65 simulated seconds. The released screen holds −9.21° with a maximum middle-slot error of 0.014 mm. Dream-1 scores 0.0 in 2,400 controls and 50 requests. The Dream-1 run takes about 15 minutes 26 seconds on an L40S under a concurrent workload.

## Fixed setup

Positions are in metres in the dual-arm frame. Positive X is forward from the arms, positive Y points left, and positive Z points up.

| Object | Size and represented material | Reset |
| --- | --- | --- |
| Monitor and support grip | 27-inch monitor with ABS housing and a simulated glass display. The screen assembly, including grip, is approximately 170 × 605 × 362 mm in monitor coordinates and 7.27 kg. The support grip is 36 × 24 × 28 mm. | Mounted to the moving bracket, screen vertical, at the highest slot. |
| Stand and hinge bracket | Approximately 249 × 308 × 447 mm metal stand and 116 × 115 × 63 mm metal bracket | Foot at `(0.280, 0.090, 0.490)`, yaw 90° from the asset frame. Slide travel is 112.88 mm. Tilt travel is −15° to +5°. |
| Lock rail and orange paddle | A roughly 30 × 22 × 151 mm fixed steel rail with three slots. The 36 × 14 × 48 mm moulded paddle moves a spring-loaded steel pin. | Rail is fixed to the stand frame. Pin starts engaged in the top slot; the middle-slot datum is 56.44 mm below the top stop. |
| Worktop and floor | 430 × 510 × 35 mm wood worktop and 3 × 3 × 0.04 m matte floor | Centers are `(0.385, 0.100, 0.4725)` and `(0.450, 0, −0.020)`. |
| YAM pair and adapter marker | Two six-axis arms and one 10 mm bookkeeping marker outside the work area | Bases at `(0, ±0.305, 0.750)`. Marker center at `(−0.5, 0, −0.05)`. |

## Outcome

The final 20 controls must have at most 0.1 N summed gripper contact on the pin and display. Add the following independently when that release condition holds.

- **0.60** for the slide within 4 mm of the middle slot and the pin within 2 mm of its engaged position throughout that window.
- **0.40** for the tilt hinge within 2° of 10° backward from vertical throughout that window.

A ruler, an angle gauge and visual confirmation of the released pin measure the equivalent real stand state. Reset to the top slot, vertical screen and released orange paddle. Reinitialize the arm poses before a new trial.

The original monitor mass and orientation limits are retained. The model adds a passive counterbalance, a capped hinge with balanced mass, a support extension and a physical three-slot spring pin. The monitor foot is fixed to the worktop. Electronics and cable routing are omitted. [Source asset and attribution](THIRD_PARTY_NOTICES.md)
