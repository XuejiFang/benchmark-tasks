# Set and arm a bedside alarm clock

Set the clock's alarm to 07:15 and enable it. Press and release the keys to make each adjustment.

## Run and score

Use the Linux environment in the [RoboCurve YAM walkthrough](https://tasks.robocurve.org/walkthrough). The reference environment runs Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, YAM adapter `0059d02a1a91bc6834c1d0846022891df0c78730`, Inspect Robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14. NumPy, SciPy, Pillow, PyTorch and imageio-ffmpeg are also required.

Run from this folder with fresh output directories.

```bash
isaac_python=/data/IsaacLab-Arena/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=2

"$isaac_python" run_alarm_clock_probe.py --fixture assets/clock.usda --output runs/expert
"$isaac_python" audit_alarm_clock_trial.py runs/expert

# Configure a Dream Labs credential according to the walkthrough.
"$isaac_python" run_alarm_clock_probe.py --fixture assets/clock.usda --dream --output runs/dream
"$isaac_python" audit_alarm_clock_trial.py runs/dream

"$isaac_python" run_alarm_clock_probe.py --fixture assets/clock.usda --setup-only --output runs/setup
```

Trials use seed 0 and disable object jitter. The action limit is 2,400 controls at 20 Hz or 120 simulated seconds. Each H.264 frame corresponds to one control. The trace contains the action, each key's travel and fingertip contact, and the display's time and enable settings.

The reference expert reached 1.0 in 960 controls (48 simulated seconds). The Dream-1 run took about seven minutes on an L40S and scored 0.0 after 2,400 controls and 50 API requests.

## Fixed setup

Positions are in metres in the dual-arm frame. Positive X is forward from the robot bases and positive Y is to their left.

| Object | Size and represented material | Reset |
| --- | --- | --- |
| Clock housing | 150 × 158 × 30 mm molded polymer | Base at `(0.302, 0.025, 0.750)` with zero yaw. Fixed to the nightstand. |
| Display | 55 × 137 × 3 mm glass panel with green digits and an orange status lamp | Clock-local center `(-0.029, 0, 0.031)`. Shows 06:25, with the lamp off. |
| Three keys | 26 × 26 × 8 mm polymer caps, steel spring proxies and 3.2 mm mechanical travel | Button centers at X 0.348 m, Z 0.785 m and Y 0.077, 0.025, and −0.027 m. Blue + advances one hour, gray − moves five minutes back, orange toggles arming. |
| Nightstand and floor | Wooden 480 × 500 × 40 mm worktop and 3 × 3 × 0.04 m floor | Worktop centered at `(0.36, 0.05, 0.73)`, floor centered at `(0.45, 0, -0.02)`. |
| YAM pair and adapter marker | Two six-axis arms. 10 mm bookkeeping marker outside the working area | Bases at `(0, ±0.305, 0.75)`. Marker at `(-0.5, 0, -0.05)`. |

## Outcome

The scorer reconstructs the clock from the recorded key travel. A press starts at 2.15 mm and generates one key event after returning within 0.50 mm. Each key returns through its own spring. Holding a key down does not repeat a command.

- **0.5** for 07:15 with all keys released during the final 20 controls. Each key travels at most 0.50 mm and has at most 0.1 N of gripper contact in that window.
- **0.5 more** when the alarm is also armed.

The alarm time stays fixed between button events. The active light shows the enable state. The task does not simulate audible playback.
