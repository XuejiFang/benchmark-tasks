# Put a book on its stand and turn the marked page

Pick the green book up with both arms and set it on the wooden reading stand. Open its cover, turn the blue-marked page, and release the book.

## Run and score

Use the Linux environment in the [RoboCurve YAM walkthrough](https://tasks.robocurve.org/walkthrough). The reference environment uses Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, YAM adapter `0059d02a1a91bc6834c1d0846022891df0c78730`, Inspect Robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14. It also uses NumPy, SciPy, Pillow, PyTorch and imageio-ffmpeg.

Run these commands from this task folder with fresh output directories.

```bash
isaac_python=/data/IsaacLab-Arena/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=2

"$isaac_python" run_book_robot_probe.py --fixture assets/book.usda --output runs/expert
"$isaac_python" audit_reading_trial.py runs/expert

# Configure the Dream Labs credential according to the walkthrough.
"$isaac_python" run_book_robot_probe.py --fixture assets/book.usda --dream --output runs/dream
"$isaac_python" audit_reading_trial.py runs/dream

"$isaac_python" run_book_robot_probe.py --fixture assets/book.usda --setup-only --output runs/setup
```

The fixed trial uses seed 0 with object jitter disabled. The limit is 2,400 controls at 20 Hz or 120 simulated seconds. The encoder writes one H.264 MP4 frame per control. The trace records both robots' actions, the book pose and joint angles, and fingertip contacts. The reference expert scores 1.0 in 1,064 controls or 53.2 simulated seconds.

The reference Dream-1 run scored 0.0 after all 2,400 controls and 50 model requests. It took approximately 16 minutes on an L40S with other simulation jobs running.

## Fixed setup

World coordinates use metres. Positive X is forward from the YAM robot bases and positive Y is to their left.

| Object | Size and represented material | Reset |
| --- | --- | --- |
| Green notebook | 167 × 225 × 34 mm closed body, 182 × 267 mm footprint including tabs. Cloth-backed cardboard covers and paper. Moving assembly mass approximately 0.50 kg. | The spine origin is `(0.435, 0.055, 0.682)`. The book lies flat and closed, with its fore edge toward positive X. |
| Blue page marker | 34 × 18 × 3 mm reinforced paper tab on one independently movable page | At the fore edge of the marked page, offset by `(0.143, 0.056, 0)` from its pivot frame. The page starts flat and closed. |
| Wooden reading stand | 326 × 256 × 28 mm wooden plinth, 132 × 232 × 26 mm folded-cover support, and four 14 mm wide, 20 mm high locating stops | Stand origin at `(0.25, 0.055, 0.710)`. The support lies on the negative X side of the book. The stops flank the stand at Y offsets of ±126 mm. The stand starts empty. |
| Reading table and floor | Wooden 580 × 870 × 40 mm worktop and 3 × 3 × 0.04 m matte floor | Worktop center `(0.38, 0.08, 0.662)`. Floor center `(0.45, 0, -0.02)`. |
| YAM pair and adapter marker | Two six-axis arms. 10 mm rigid bookkeeping marker outside the work area. | Mount centers `(0, ±0.305, 0.75)`. Marker center `(-0.5, 0, -0.05)`. |

The simulation uses two contact-driven hinges on a free book. The marked leaf represents a paper page with a reinforced tab. The remaining bound pages are a single block. The stand has guides for repeatable placement and supports the folded panels.

## Outcome

The final 20 controls must satisfy each credited condition throughout the window. The scores are cumulative.

- **0.35** when the released book's spine is within 16 mm horizontally and 10 mm vertically of the stand datum and the book tilts no more than 9°. Gripper contact on the book spine must remain at or below 0.1 N.
- **0.30 more** when the cover has opened by at least 150° and contact on the cover is at or below 0.1 N.
- **0.35 more** when the blue-marked page has turned by at least 150° and contact on the page is at or below 0.1 N.
