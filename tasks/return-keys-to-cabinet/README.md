# Return two keys to an entryway cabinet

Open the hinged door, hang the blue and orange key rings on their assigned hooks, release both keys and close the door. Shallow tapered guides receive the key blades and reduce sway while the door moves.

## Run and score a trial

Use the Linux image in the [RoboCurve YAM walkthrough](https://tasks.robocurve.org/walkthrough). The reference environment uses Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf` and the YAM adapter `0059d02a1a91bc6834c1d0846022891df0c78730`. The Dream run uses Inspect Robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14. The Python runner uses NumPy, SciPy, Pillow, PyTorch and imageio-ffmpeg.

Run these commands from this task's folder. Choose a fresh output directory for each trial.

```bash
isaac_python=/data/IsaacLab-Arena/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1

"$isaac_python" run_key_return_fixture.py --assets assets \
  --phase return-both --video --output runs/expert
"$isaac_python" audit_key_return_trial.py runs/expert

# Configure the Dream Labs credential using the walkthrough.
"$isaac_python" run_key_return_fixture.py --assets assets \
  --phase return-both --dream --output runs/dream
"$isaac_python" audit_key_return_trial.py runs/dream
```

The fixed setup uses seed 0 and disables object jitter. PhysX advances at 240 Hz and the arms receive controls at 20 Hz, with up to 2,400 controls (120 simulated seconds) per trial. The expert earns 1.0 in 2,002 controls, or 100.1 simulated seconds. Dream-1 earns 0.0 after 2,400 controls and 50 API requests. Each video frame corresponds to one control, and `trace.jsonl` contains the observed states and the commanded joints.

## Fixed setup

Coordinates are metres in the robot frame. Positive X points toward the cabinet. Positive Y is to the left when facing the closed door. The tray holds both keys flat on their fobs, as shown in the setup image.

| Object | Dimensions and represented material | Reset |
| --- | --- | --- |
| Wall | 90 mm thick, 1.2 m wide, 1.65 m high, painted plaster | Centre `(0.764, 0, 0.825)` |
| Cabinet | Painted steel, 360 mm wide and 340 mm tall, 160 mm nominal front-to-back dimension | Case origin `(0.55, -0.06, 0.76)`, fixed to the wall |
| Door and handle | Oak panel, 336 × 324 × 18 mm, with an 80 mm grip bar | Closed at 0°, vertical hinge at `(0.55, 0.11, 0.93)` |
| Two hooks | Coated metal with an 8 mm diameter shaft and a raised tip | Shaft Z is 0.997 m. Left Y is 0.03 m, right Y is −0.15 m. |
| Two blade guides | Rubber-lined seats, 28 × 36 × 27 mm overall with a taper at the opening | One directly below each hook, spanning Z = 0.874–0.901 m |
| Console and tray | 320 × 560 × 50 mm wood console. Tray floor 260 × 480 × 12 mm, moulded plastic. | Console centre `(0.31, 0, 0.686)`. Tray surface at Z = 0.723 m, with 18 mm raised rims. |
| Blue key and orange key | Each 68 g, with a steel blade, 36 × 14 × 51 mm plastic fob, and metal wire ring with approximately 38 mm clear diameter | Blue at `(0.365, -0.05, 0.732)`, orange at `(0.305, -0.117, 0.732)` before settling. Each imported key rotates 90° around local X, then 180° around world Z. |
| Two YAM arms | Six axes, each with a parallel gripper and 180 × 180 × 750 mm pedestal | Bases `(0, 0.305, 0.75)` and `(0, -0.305, 0.75)` |
| Floor | 3 × 3 × 0.04 m concrete slab | Centre `(0.45, 0, -0.02)` |

## Scoring

The audit reads the last 20 controls. Each correctly returned key earns 0.40. In every scoring frame, the assigned hook must cross the plane of the ring within the shaft segment and within 20 mm of the aperture centre, above the ring centre. The measured key displacement from the last frame must stay within 3 mm and its rotation within 1°. Total finger and gripper contact on the key must stay at or below 0.1 N. Returning both keys and closing the released door adds 0.20 when the door is within 3° of closed and its gripper contact stays at or below 0.1 N in every scoring frame.

The rings use 24 convex wire segments each. The hooks, tips and blade guides have collision surfaces. The key, fob and ring each move together as a rigid body, and the hinged door is not kinematically driven. The expert moves them through finger and panel contact. Each starting pose is restored by resetting the scene.
