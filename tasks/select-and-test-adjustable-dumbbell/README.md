# Select and test an adjustable dumbbell

Move the orange U-shaped selector to the lower slot, lift both weight pairs together, return the assembly to its cradle and release.

Use the Linux environment in the RoboCurve YAM walkthrough. The reference stack uses Isaac Sim 6.1.0.0, IsaacLab-Arena revision c7edf3a9d5d8276565208f72422f8f8a0806d6cf and YAM adapter revision 0059d02a1a91bc6834c1d0846022891df0c78730.

Run from this folder with a fresh output directory for each trial.

    isaac_python=/data/IsaacLab-Arena/.venv/bin/python
    export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=2
    "$isaac_python" run_dumbbell_robot_probe.py --asset assets/dumbbell.usda --clearance-feedback --mass-scale .72 --output runs/expert
    "$isaac_python" audit_dumbbell_trial.py runs/expert
    "$isaac_python" run_dumbbell_robot_probe.py --asset assets/dumbbell.usda --dream --output runs/dream
    "$isaac_python" audit_dumbbell_trial.py runs/dream
    "$isaac_python" run_dumbbell_robot_probe.py --asset assets/dumbbell.usda --setup-only --output runs/setup

Seed 0 fixes the layout. The limit is 2,400 controls at 20 Hz. The trace records body poses, arm commands and gripper contact forces. The expert observes simulator state and sends joint commands. The reference expert run scored 1.0 in 1,187 controls, with maximum final return error 14.987 mm. Dream-1 scored 0.0 in 2,400 controls and 50 requests.

The selector assembly starts at (0.405, -0.065, 0.757) metres on a 47 mm pedestal. The worktop is 700 × 760 × 40 mm. The left and right arm bases are (0, 0.305, 0.797) and (0, -0.305, 0.8435) metres. The steel U-selector tines are 113 mm long, and the two slots are 26 mm apart. The handle and weight bodies are collision-enabled rigid assemblies.

The source USD is procedurally authored. The assembly uses rigid contacts and enlarged guides, and the weights are lighter than typical gym weights.
