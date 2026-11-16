# Pack a gym bag and keep used shoes away from clean accessories

Put the five objects named in [task.json](task.json) in the open gym bag. Keep the used shoes together at one end and leave at least 12 cm between them and the wallet and headphones at the other end. Measure the gap edge to edge along the long axis of the bag. Both ends are valid for the shoes.

## Run and score

Use the Robocurve [YAM simulation walkthrough](https://tasks.robocurve.org/walkthrough) to provision the NVIDIA Linux simulation environment and configure Dream Labs credentials. The verified environment used Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena revision `c7edf3a9d5d8276565208f72422f8f8a0806d6cf` and YAM adapter revision `0059d02a1a91bc6834c1d0846022891df0c78730`. Python packages include Inspect Robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14. Use the corresponding interpreter and GPU driver runtime. NumPy, SciPy, OpenCV, Pillow and imageio-ffmpeg are also used.

From this task directory, use a fresh output path for each trial:

```bash
isaac_python=/data/IsaacLab-Arena/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1

"$isaac_python" run_dream.py --output runs/dream
"$isaac_python" audit_dream_run.py runs/dream

"$isaac_python" run_expert.py --output runs/expert
"$isaac_python" verify_run.py runs/expert
```

The model receives three RGB cameras at 320 × 176 and measured dual-arm joint/gripper state. A trial allows up to 3,000 actions at 20 Hz, including the adapter's initial hold while gathering observations, for 150 simulated seconds. The expert uses measured simulator geometry for grasp planning and commands the same robot actuators. Both scores come from the final state after the same continuous 40-step qualification window. The scorer permits an object to rest on another object in the bag.

The output contains `result.json`, `trace.jsonl`, mesh vertices for offline scoring and synchronized camera streams in their own timelines. `audit_dream_run.py` checks the model's final score, source snapshots and actual API requests. `verify_run.py` checks the expert's final score by recomputing mesh containment, separation, release and settling from its trace. Check the result files and the audits in addition to the process exit code.

The reference expert scores 1.0 in 2,590 controls (129.5 simulated seconds). The final 40 controls maintain at least 15.72 cm of separation. Dream-1 scores 0.0 after 3,000 controls (150 simulated seconds).

## Reproduce the starting configuration

The standalone scene layers and textures are under `assets/`. `asset_manifest.json` records SHA-256 hashes of all dependencies. Placement uses the fixed entity order in `task.py` and the pose constants in `layout.py`. The arm bases are at world `(0, +0.355, 0.55)` and `(0, -0.355, 0.55)` metres. Positions in `layout.py` are expressed in the room frame and transformed to the arm frame by `(x, y, z) → (0.90 − x, 0.705 − y, z)`. The room and target reset use simulator defaults and embodiment seed 0 with object jitter disabled. [Scene inventory](SETUP.md) lists the visible objects and dimensions.

## Asset sources

The [Innerscene Free Parts Library](https://www.innerscene.com/tools/library/terms) publishes the [sofa](https://www.innerscene.com/tools/library/3d-parts/link-armless-modular-sofa-c0827105), [duffel](https://www.innerscene.com/tools/library/3d-parts/duffel-bag-644a9785), [bottle](https://www.innerscene.com/tools/library/3d-parts/water-bottle-d7c3525e), [shoes](https://www.innerscene.com/tools/library/3d-parts/running-sneaker-c7a5a7eb), [wallet](https://www.innerscene.com/tools/library/3d-parts/wallet-b3861aab), headphones and shopping tote under CC0. Poly Haven publishes the coffee table, plant, throw pillows, notebook, lamp, basket, bowl, wood floor and daylight HDRI under [CC0](https://polyhaven.com/license). The other room, furniture and decor geometry is procedural. The asset layers preserve the input meshes and textures with prepared collision and material data. The bag opening, lining and handle positions are authored for this task, the bag is scaled as documented and shoes are split into separate bodies.
