# Separate nested bowls and lay out two breakfast settings

Separate the nested bowls, then establish two upright place settings. Each bowl goes on its corresponding mat, with the assigned spoon on the inner side of the mat. This tests unstacking fragile tableware from a shared storage arrangement, then preparing two different positions within the same table area.

## Run and score a trial

Provision the Linux simulation environment in the [RoboCurve YAM walkthrough](https://tasks.robocurve.org/walkthrough). The tested versions were Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena revision `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, YAM adapter revision `0059d02a1a91bc6834c1d0846022891df0c78730`, Inspect Robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14. The runner also uses NumPy, SciPy, Pillow and imageio-ffmpeg.

From this folder, use a fresh output directory for each run.

```bash
isaac_python=/data/IsaacLab-Arena/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1

"$isaac_python" run_breakfast_fixture.py --assets assets --output runs/expert --full --video
"$isaac_python" audit_breakfast_fixture.py runs/expert --two-settings

# Configure the Dream Labs credential as described in the walkthrough first.
"$isaac_python" run_breakfast_fixture.py --assets assets --output runs/dream --full --dream --video
"$isaac_python" audit_breakfast_fixture.py runs/dream --two-settings
```

The fixed layout uses seed 0, with object jitter disabled. The time budget is 2,400 actions at 20 Hz, or 120 simulated seconds. The successful expert demonstration takes 1,410 steps and 70.5 simulated seconds. Model-run wall time also includes API latency. Each recorded MP4 has one frame per executed control step. Check the `audit.json` and `result.json` artifacts for the score and run completion.

## Fixed starting setup

Positions are in metres in the two robots' shared world frame. Positive X points from their bases across the table.

| Object | Dimensions and represented material | Initial configuration |
| --- | --- | --- |
| Dining tabletop | 68 × 100 × 4.5 cm, wood | Centre `(0.46, 0, 0.7275)`, flat with its top at `z = 0.75` |
| Teal and tan mats | Each 23 × 23 × 0.4 cm, rigid model of a flat woven placemat | Centres `(0.46, +0.37, 0.752)` and `(0.46, −0.37, 0.752)` |
| Cream lower bowl and teal upper bowl | Each Ø15.8 × 4.6 cm, 310 g, glazed ceramic | Nested at `(0.41, 0.12)` after gravity settles the reset heights of `0.752` and `0.768` |
| Dark- and light-handled spoons | Each approximately 17.1 × 4.2 × 0.8 cm, coated handle and metal head | `(0.26, 0.23)` and `(0.28, −0.24)`, starting rotations 75° and 105° around world Z before settling |
| Arm bases and supports | Two six-axis YAM arms on 15 × 15 × 83 cm rigid stands | Left base `(0, 0.305, 0.83)`, right base `(0, −0.21, 0.83)` |
| Floor | 300 × 300 × 4 cm, concrete | Centre `(0.45, 0, −0.02)` |

The released bowls target their mat centres. The dark and light spoons target `(0.46, 0.235)` and `(0.46, −0.235)`, with the heads pointing toward the robot bases along negative X. The scorer gives 0.25 for each object that remains within 26 mm of its target in the plane and 8 mm of its resting height for the final 20 controls, with at most 3 mm of centre drift, 2° of angular drift and 0.1 N total contact with the two grippers. Each object must be right side up within 5.13°. Each spoon's head must align within 15.21° of the specified direction. The two bowls use compound convex wall and base colliders to preserve their open interiors, and all placements use physical contact. Spoons and mats are rigid. Fracture and woven surface deformation are outside the simulation model.

The USD files under `assets/` are authored for this task. Their exact input hashes are in `asset_manifest.json`.
