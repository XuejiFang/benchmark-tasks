# Dose dry tea and seat the filled infuser

Remove the tea tin lid, scoop a measured dose into the loose infuser, then place the loaded infuser in its cup. The tin and infuser retain a fixed inventory of 160 separate dry pieces.

## Run and score a trial

Provision the Linux image described in the [RoboCurve YAM walkthrough](https://tasks.robocurve.org/walkthrough). Tested versions are Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, YAM adapter `0059d02a1a91bc6834c1d0846022891df0c78730`, Inspect Robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14. The Python runner also uses NumPy, SciPy, Pillow, PyTorch and imageio-ffmpeg. The scene uses CPU PhysX at 480 Hz with continuous collision detection and one control every 24 physics updates.

From the task folder, choose a fresh output folder for each run.

```bash
isaac_python=/data/IsaacLab-Arena/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1

"$isaac_python" run_tea_robot_probe.py --tea assets/tea.usda --scoop assets/scoop.usda \
  --accessories assets --scoop-end-x .0135 --top-up --trim-dose --output runs/expert
"$isaac_python" audit_tea_robot_trial.py runs/expert

# Configure the Dream Labs credential through the walkthrough before this run.
"$isaac_python" run_tea_robot_probe.py --tea assets/tea.usda --scoop assets/scoop.usda \
  --accessories assets --dream --output runs/dream
"$isaac_python" audit_tea_robot_trial.py runs/dream
```

Both runs use scene seed 0 and the original procedural leaf seed 4. Object jitter is disabled. The action budget is 2,400 at 20 Hz. The expert reaches full score in 2,314 actions, or 115.7 simulated seconds. The recorded Dream-1 run uses 2,400 actions and 50 API requests. Each video frame corresponds to one control step. `trace.jsonl` retains observed states and commanded joints, and the audit script independently writes `audit.json`.

## Fixed setup

Coordinates are metres in the shared robot frame. Positive X points across the worktop and positive Y is the robots’ left. The station-local tea coordinates used for the particles have their origin at the tin base and rotate 90° about world Z.

| Object | Dimensions and represented material | Initial configuration |
| --- | --- | --- |
| Worktop | 600 × 750 × 40 mm, wood | Centre `(0.38, 0.13, 0.73)` with work surface at Z = 0.75 |
| Green tea tin | 76 mm inner diameter, 81 mm outer diameter, 55 mm tall, coated metal | Base centred at `(0.27, 0.20, 0.75)` |
| Lid | 83 mm diameter, 29 mm overall height, coated metal with a 27 g model | Centred on the green tin at `(0.27, 0.20, 0.8058)` before settling |
| Tea | 160 approximately 7–9 mm rolled pieces, 0.042 g each | Reset into the tin in the positions authored in `tea.usda`, then settled under gravity |
| Scoop | 31 × 27 × 74 mm, 34 g, metal with resin stem | Base at `(0.22, 0.26, 0.751)` before settling, yaw 90° |
| Loose infuser | 58 mm inner diameter, rim at 33 mm above its base, 37 g, metal with a grasp tab | Base at `(0.287, 0.345, 0.751)` before settling, yaw 90° |
| Brown cup | 70 mm inner diameter, 76 mm outer diameter, 48 mm high, ceramic | Base at `(0.18, 0.32, 0.75)` |
| Robots | Two fixed-base six-axis YAMs | Base coordinates `(0, 0.305, 0.83)` and `(0, −0.305, 0.83)` |
| Floor and adapter marker | Concrete floor 3 × 3 × 0.04 m, 10 mm rigid adapter marker outside the work area | Floor centre `(0.45, 0, −0.02)`, marker at `(−0.5, 0, −0.05)` |

## Scoring

The scorer examines the last 20 controls.

- **0.20 for the parked lid.** Its centre stays within 22 mm horizontally of `(0.34, 0.12)` and 6 mm vertically of Z = 0.753 m, with at most 2 mm drift.
- **0.35 for the dose.** The infuser contains 24–33 pieces in every scoring frame. Piece centres count inside it at radius below 28 mm and height from 4 to 33 mm in its measured local frame.
- **0.35 for seating the infuser.** Its centre stays within 4.5 mm of the cup centre and 4 mm of target Z = 0.7665 m, with at most 1.5 mm drift. Its up-axis dot product with the vertical is at least 0.998.
- **0.10 for preserving the inventory.** A qualifying dose leaves every piece in the tin or infuser. The tin’s inclusion region uses radius below 37 mm and height from 4 to 55 mm in the station frame.

The authored materials and collision bodies are under `assets/`, with exact input hashes in `asset_manifest.json`. The dry infuser uses compound convex walls and a solid collision floor. Every tea piece maintains its own rigid-body state. During the expert run the robot moves the scoop, lid and infuser through finger contact.
