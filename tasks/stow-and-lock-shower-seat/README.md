# Fold and secure a wall-mounted shower seat

Hold the orange retaining lever clear, fold the seat up, release the lever and withdraw both grippers.

Use the Linux environment in the RoboCurve YAM walkthrough. The reference environment uses Isaac Sim 6.1.0.0, IsaacLab-Arena revision c7edf3a9d5d8276565208f72422f8f8a0806d6cf, YAM adapter revision 0059d02a1a91bc6834c1d0846022891df0c78730, Inspect Robots 0.60.0 and inspect-robots-dream 0.1.0.

From this directory, download the pinned source and use a fresh output path for every trial.

    python3 fetch_assets.py
    isaac_python=/data/IsaacLab-Arena/.venv/bin/python
    export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=2
    opts="--source assets/shower_seat.usda --round-grip --side-grip .275 --counterbalance .97 --radial-grip .23 --left-offset .015 --left-y .49 --left-height .80 --entry-pitch=-75 --wrist-follow .50 --lead-limit 9"
    "$isaac_python" run_shower_seat_robot.py $opts --output runs/expert
    "$isaac_python" audit_shower_seat_trial.py runs/expert
    "$isaac_python" run_shower_seat_robot.py $opts --dream --output runs/dream
    "$isaac_python" audit_shower_seat_trial.py runs/dream
    "$isaac_python" run_shower_seat_robot.py $opts --setup-only --output runs/setup

The fixed layout uses seed 0. The limit is 2,400 controls at 20 Hz. The expert reads simulator state and moves the robot through joint commands. The independent audit checks the lever and seat angles and measured gripper contact.

The independent package run scores 1.0 in 1,214 controls, with the released seat at least 85.46° and the keeper at or below −49.82° throughout the final 20 controls. Dream-1 scores 0.0 in 2,400 controls and 50 requests. In a separate diagnostic starting from the same robot-operated stow, the robot reopens the keeper and the seat falls from 85.46° to 64.09°.

The robot demonstration uses a left arm at (0.015, 0.49, 0.80) metres and a right arm at (0, −0.305, 0.795) metres. The source seat is 450 × 430 × 52 mm, weighs 5.951 kg and is mounted on a wall. A physical counterbalance approximation moves the effective centre of mass toward the hinge. A rigid cylindrical grab rail and a spring-return orange keeper are added. The fixture is at (0.51, 0, 0.035) metres and yawed −90°. The floor is 3 × 3 metres.

The source shower-seat asset is by [ForgeCAD](https://huggingface.co/datasets/ForgeCAD/simready-assets/tree/b94ea770666b6c0f82e09a78e9fcb375b223e3b1/shower_seat/simready_usd), distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The fetch script downloads files at that revision and verifies SHA-256 hashes.
