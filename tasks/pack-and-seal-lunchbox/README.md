# Pack and seal a divided lunchbox

Stand the divider in its two guides, put one wrapped item in each specified compartment, then seat the lid and close all four latches.

## Run and score a trial

Use the Linux environment in the [RoboCurve YAM walkthrough](https://tasks.robocurve.org/walkthrough). The reference stack uses Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, YAM adapter `0059d02a1a91bc6834c1d0846022891df0c78730`, Inspect Robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14. Run from this folder with fresh output paths.

```bash
isaac_python=/data/IsaacLab-Arena/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1
"$isaac_python" run_lunchbox_pack_probe.py --usd assets/box/packing.usd \
  --contents assets/contents --centered --latch --align-lid --aim-mm 1.05 -0.93 \
  --staged-brace --max-controls 3000 --output runs/expert
"$isaac_python" audit_lunchbox_trial.py runs/expert

# Configure the Dream Labs credential using the walkthrough.
"$isaac_python" run_lunchbox_pack_probe.py --usd assets/box/packing.usd \
  --contents assets/contents --dream --max-controls 3000 --output runs/dream
"$isaac_python" audit_lunchbox_trial.py runs/dream

"$isaac_python" run_lunchbox_pack_probe.py --usd assets/box/packing.usd \
  --contents assets/contents --setup-only --max-controls 3000 --output runs/setup
```

Seed `0` resets the same layout each time. Trials allow up to 3,000 controls at 20 Hz, or 150 simulated seconds. The reference expert completes all parts in 2,959 controls, or 147.95 simulated seconds.

The Dream-1 baseline scored 0.0 in 3,000 controls and 62 model requests. It took approximately 23 minutes on an L40S with other simulation jobs running.

## Starting layout

Robot-frame coordinates are metres. Positive X points forward into the worktop and positive Y points to the left arm. The boxes and packets start with a yaw of −90°.

| Object | Material and dimensions | Starting position |
| --- | --- | --- |
| Lunchbox tub with four molded guide pairs | Polypropylene, approximately 200 × 150 × 80 mm including the sealed lid | `(0.30, 0.08, 0.75)` |
| Lid and four pivoting retaining tabs | Frosted polypropylene lid, plastic tabs and top pull handle | Lid authored at `(0.39, 0.32, 0.78)` and settles beside the tub. Tabs start with the revolute joints open. |
| Pull-tab divider | Rigid polypropylene, 146 × 18 × 45 mm including feet | `(0.257, 0.316, 0.75)` |
| Orange and green wrapped packets | Rigid packet proxies, each 56 × 31 × 27 mm plus a folded paper grab tab | Orange `(0.181, 0.252, 0.75)`, green `(0.279, 0.225, 0.75)` |
| Worktop | Wood, 650 × 720 × 40 mm | Centre `(0.42, 0.065, 0.73)` |
| Floor | Concrete, 3 × 3 × 0.04 m | Centre `(0.45, 0, -0.02)` |
| Two YAM arms | Metal links and parallel grippers | Bases `(0, 0.305, 0.75)` and `(0, -0.305, 0.75)` |

The tub is secured to the worktop to resist latch loads. The divider, packets, lid and latches move through contact and gravity. The lid's concealed locator has 2.7 mm extra clearance radially so the rigid collision surfaces can seat under ordinary gripper force. ForgeCAD supplied the source polypropylene container and material. [Pinned container asset, CC BY 4.0](https://huggingface.co/datasets/ForgeCAD/simready-assets/tree/b94ea770666b6c0f82e09a78e9fcb375b223e3b1/latching_food_container/simready_usd).

## Scoring

Every condition must hold for the final 20 observations. Objects must be released with at most 0.1 N of summed fingertip contact.

- **0.15** for the divider within 10 mm longitudinally and 9 mm across the centerline, no more than 12 mm above the tub floor datum, at most 5° from vertical.
- **0.15 per packet** for resting in its designated compartment. The packet reference point is 23–57 mm to the correct side of the centerline, within 31 mm along the compartment, and 0–16 mm above the tub datum, with at most 22° tilt.
- **0.15** for the lid within 10 mm of the tub centre horizontally and 9 mm of the seated height, with at most 5° tilt.
- **0.10 per latch** for a released latch no more than 4° from closed, with the lid seated.
