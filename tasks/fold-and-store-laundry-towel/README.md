# Fold a towel on a board and store it

Use the folding board to fold the tan towel. Reopen the board, place the folded towel fully inside the wooden compartment, and release it.

The compartment is narrower than the unfolded towel. The folding board determines the fold, after which the cloth must retain that fold during transfer.

## Run and score

Use the Linux environment in the [RoboCurve YAM walkthrough](https://tasks.robocurve.org/walkthrough). The reference stack uses Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, YAM adapter `0059d02a1a91bc6834c1d0846022891df0c78730`, Inspect Robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14. It also uses a PhysX-capable GPU, NumPy, SciPy, Pillow, PyTorch and imageio-ffmpeg.

From this directory, use a fresh output path per trial.

```bash
isaac_python=/data/IsaacLab-Arena/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=2

"$isaac_python" run_towel_robot_probe.py --asset assets/folding_board.usda --output runs/expert
"$isaac_python" audit_towel_trial.py runs/expert

# Configure Dream Labs credentials according to the walkthrough.
"$isaac_python" run_towel_robot_probe.py --asset assets/folding_board.usda --dream --output runs/dream
"$isaac_python" audit_towel_trial.py runs/dream

"$isaac_python" run_towel_robot_probe.py --asset assets/folding_board.usda --setup-only --output runs/setup
```

Seed 0 fixes the published starting layout. The run limit is 2,400 controls at 20 Hz or 120 simulated seconds. Each video frame corresponds to one control. The trace records the flap angle, live textile dimensions, opposing end positions and both wrist positions. The expert moves only the two arms and fingers after reset.

The independent expert scores 1.0 in 1,202 controls or 60.1 simulated seconds. It leaves the towel about 180 mm wide in the compartment with both arms withdrawn. Dream-1 scores 0.0 in 2,400 controls and 50 requests. The Dream-1 run takes about 26 minutes 39 seconds on an L40S under a concurrent workload.

## Fixed setup

Positions are in metres in the dual-arm frame. Positive X is forward, positive Y points left, and positive Z points up.

| Object | Size and represented material | Reset |
| --- | --- | --- |
| Tan towel | 360 × 240 mm cotton proxy modelled as a 2 mm thick, 289-node PhysX surface | Unfolded, centered at `(0.310, 0, 0.747)`, long direction along X, resting across both board halves after settling. |
| Folding board | Fixed 191 × 180 × 10 mm and mobile 190 × 291 × 10 mm moulded-plastic panels. Two 26 mm grip knobs, one accessible on each side of the moving panel. | Open, with hinge at `(0.310, 0, 0.735)` along Y. Fixed panel is on the side nearer the robot bases. |
| Compartment | Wood floor 208 × 268 × 14 mm, low back and side rims, and a wood support | Empty. Floor centered at `(0.025, 0, 0.692)`, top at 0.699 m. The low front side is open. |
| Worktop and floor | 510 × 286 × 40 mm wood worktop and 3 × 3 × 0.04 m matte floor | Centers at `(0.390, 0, 0.690)` and `(0.450, 0, −0.020)`. |
| YAM pair and adapter marker | Two six-axis arms and one 10 mm bookkeeping marker outside the work area | Bases at `(0, ±0.305, 0.750)`. Marker center at `(−0.5, 0, −0.05)`. |

## Outcome

Award **0.55** when the board reached at least 140° with the towel at most 225 mm wide during the run, and the final 20 controls meet these conditions. The towel stays at most 210 mm wide in X, the opposing end regions are at most 92 mm apart along X, the board stays within 4° of open and both grippers are clear. Clearance means each wrist center is at least 22 mm from the cloth's bounding box and contact on the flap is at most 0.1 N.

Award **another 0.45** when the folded towel stays on the compartment floor with all nodes within ±102 mm in X and ±127 mm in Y of its center. The lowest tenth percentile of nodes must remain within 6 mm of the floor surface, and the entire towel must lie between 3 mm below and 38 mm above that surface.

Use a ruler, overhead view and a side view or touch check to evaluate a corresponding real trial. Reset the board to open and the towel to an unfolded rectangle before each run.

The textile is a single compressed layer with calibrated stiffness and friction. The board hinges freely except for damping and travel stops. The two rigid knobs provide repeatable purchase through the folding cycle.
