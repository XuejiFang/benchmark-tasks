# Refill and reseal a pantry storage jar

Open the green storage jar, pour in the dry contents of the carton, then return the carton upright and reseal the green jar. The brown jar provides a second, visually distinct destination.

## Run and score a trial

Use the Linux image in the [RoboCurve YAM walkthrough](https://tasks.robocurve.org/walkthrough). The reference environment uses Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf` and the YAM adapter `0059d02a1a91bc6834c1d0846022891df0c78730`. The official model run uses Inspect Robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14. The runner also uses PyTorch, NumPy, SciPy, Pillow and imageio-ffmpeg.

Run these commands in this task folder. Choose a fresh output folder for each run.

```bash
isaac_python=/data/IsaacLab-Arena/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1

"$isaac_python" run_refill_robot_probe.py --assets assets --with-lid \
  --pour-trace assets/pour_path.jsonl --output runs/expert
"$isaac_python" audit_refill_trial.py runs/expert

# Configure the Dream Labs credential using the walkthrough.
"$isaac_python" run_refill_robot_probe.py --assets assets --dream --output runs/dream
"$isaac_python" audit_refill_trial.py runs/dream
```

Both runs use a fixed layout with seed 0 and disabled object jitter. Each trial has up to 2,400 robot controls at 20 Hz, or 120 simulated seconds. The expert earns 1.0 in 1,546 controls (77.3 simulated seconds). Dream-1 earns 0.0 over 2,400 controls and 50 API requests. CPU PhysX runs at 480 Hz with continuous collision detection. The video has one frame per control and `trace.jsonl` records robot controls, free-object poses, grain positions and finger forces.

## Fixed setup

Coordinates are metres in the dual-arm frame. Positive X points across the worktop and positive Y is to the robots' left. Reset restores both containers and the grain inventory.

| Object | Dimensions and represented material | Initial configuration |
| --- | --- | --- |
| Worktop | 620 × 620 × 40 mm wood | Centre `(0.40, 0, 0.62)` and working surface Z = 0.64 |
| Cardboard carton | 90 × 60 × 126 mm, 32 g, coated interior. Two rubber-faced 18 × 34 × 12 mm grip tabs | Base at `(0.24, 0, 0.64)`, standing upright, opening facing up |
| Dry grain | 80 separate convex particles, 7.4 mm diameter, 0.16 g each | In the carton, authored in layers and then settled under gravity |
| Green jar | Moulded polymer, 104 mm interior diameter, 110 mm outside diameter, 88 mm high | Base at `(0.445, 0, 0.64)` |
| Lid | 112 mm diameter polymer cap and 96 mm locator ring, 45 g including the overhanging grasp tab | Initial centre `(0.445, 0, 0.7225)`, seated on the green jar |
| Brown jar | Moulded polymer, 92 mm interior diameter, 98 mm outside diameter, 82 mm high | Base at `(0.445, 0.15, 0.64)`; stays empty |
| Two fixed-base YAM arms | Six axes and parallel grippers | Bases at `(0, 0.285, 0.60)` and `(0, -0.285, 0.60)` |
| Floor and adapter marker | 3 × 3 × 0.04 m concrete slab and 10 mm adapter marker outside the work area | Floor centre `(0.45, 0, -0.02)`; marker at `(-0.5, 0, -0.05)` |

## Scoring

The scorer examines every frame in the final 20 controls.

- **Up to 0.55 for loading the green jar.** Multiply 0.55 by the smallest observed fraction of grains in the green jar. The centre of a grain counts at radius below 50 mm and height between 3 and 88 mm above the worktop.
- **0.20 for complete, clean transfer.** All 80 grains must be in the green jar, with zero grains in the brown jar or outside the vessels.
- **0.15 for reseating the lid after a qualifying transfer.** With at least one grain in the green jar, its lid centre must be within 4 mm of the seat, its up axis within 2° of vertical, and finger contact below 0.1 N per finger.
- **0.10 for returning the carton after a qualifying transfer.** With at least one grain in the green jar, the carton floor must be within 18 mm of its starting point, its up axis within 5° of vertical, and finger contact below 0.1 N per finger.

The physical model approximates dry grains as stiff round particles and the two jars as compound collision walls fixed to the worktop. The removable lid rests on a shallow locating ring. The saved desired pouring path supplies targets for the robot's wrist solver. The carton, lid and grains evolve under gravity and contact throughout the trial.
