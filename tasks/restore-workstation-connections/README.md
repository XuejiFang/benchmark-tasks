# Connect and route a workstation cable

Connect two marked ports and route the cable through the open guide while both connectors stay seated.

## Run and score

Use the Linux environment in the [RoboCurve YAM walkthrough](https://tasks.robocurve.org/walkthrough). The reference image uses Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, YAM adapter `0059d02a1a91bc6834c1d0846022891df0c78730`, Inspect Robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14. Runtime dependencies also include NumPy, SciPy, Pillow, PyTorch and imageio-ffmpeg.

Run these commands from this task directory. Each trial creates its own output directory.

```bash
isaac_python=/data/IsaacLab-Arena/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=2

"$isaac_python" run_workstation_socket_probe.py --fixture assets/workstation.usda --output runs/expert
"$isaac_python" audit_workstation_trial.py runs/expert

# Configure a Dream Labs credential according to the walkthrough.
"$isaac_python" run_workstation_socket_probe.py --fixture assets/workstation.usda --dream --output runs/dream
"$isaac_python" audit_workstation_trial.py runs/dream

"$isaac_python" run_workstation_socket_probe.py --fixture assets/workstation.usda --setup-only --output runs/setup
```

All runs use seed 0 and disable object jitter. The limit is 2,400 controls at 20 Hz, or 120 simulated seconds. Each H.264 frame corresponds to one recorded control.

The reference expert scored 1.0 after 1,733 controls (86.65 simulated seconds). Dream-1 scored 0.0 after 2,400 controls and 50 API requests. The Dream run took about 20 minutes on the reference L40S while other trials were running.

## Fixed setup

Positions use metres in the dual-arm frame. Positive X points toward the devices and positive Y points to the robots' left.

| Object | Dimensions and represented material | Reset |
| --- | --- | --- |
| Two device enclosures | 83 × 105 × 44 mm polymer bases, metal sockets with a polymer tongue | Target port mouths at `(0.390, -0.220, 0.803)` and `(0.390, 0.220, 0.803)`. Each marked port is beneath a green dot. The negative-Y device has an additional unmarked socket. |
| Two plugs | 21 × 12 × 8 mm molded blue grips, 8.6 mm long hollow metal shells | Tips at `(0.230, -0.290, 0.803)` and `(0.230, 0.290, 0.803)`. Both connectors point along +X. |
| Plug supports | 53 × 42 × 49 mm polymer stands, 26 × 22 × 14 mm strain-relief shelves | Stands at `(0.210, ±0.290, 0.7745)`. Shelves sit in front of each green-marked port. |
| Cable and collar | Cable diameter 5.6 mm, flexible polymer represented by articulated capsules. The gold collar measures 16 × 16 × 11 mm with a 24 × 15 × 18 mm raised grip. | The fixed USD contains the S-shaped layout and collar at the cable midpoint. |
| Cable guide | Polymer channel, 55 × 135 mm bed with two 16 mm high side lips | Centered at `(0.232, 0, 0.752)`. |
| Worktop and floor | 600 × 720 × 40 mm wood worktop and 3 × 3 × 0.04 m floor | Worktop top at 0.75 m and center at `(0.42, 0.045, 0.73)`. Floor top at 0 m. |
| YAM arms and adapter marker | Two six-axis arms and 10 mm bookkeeping marker outside the working area | Arm bases at `(0, ±0.305, 0.75)`. Marker at `(-0.5, 0, -0.05)`. |

## Scoring

- **0.35 per plug.** Establish a released connection for 20 consecutive controls and keep the plug seated through the end. A seated plug is 7.5–10.5 mm deep and within 1.0 mm horizontally and vertically of the socket center. Its long axis and shell profile each align within 6° of the socket. Either USB-C roll orientation is allowed. Summed gripper contact must be at most 0.1 N when establishing the connection and for the final 20 controls.
- **0.30 for routing.** Both plugs qualify, and for the final 20 controls the collar stays within 18 mm across and 45 mm along the guide center, 2–16 mm above the worktop, with at most 0.1 N gripper contact.

The packaged USD is self-contained. Both shells contain open metal walls and slide around physical tongues. The cable segments have spherical constraints and transmit tensile loads. The scorer measures these mechanical states directly.
