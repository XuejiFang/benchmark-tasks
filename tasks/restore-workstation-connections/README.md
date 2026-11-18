# Connect and route a workstation cable

Connect two marked ports and route the cable through the open guide while both connectors stay seated.

## Run and score

Use the Linux environment in the [RoboCurve YAM walkthrough](https://tasks.robocurve.org/walkthrough). The reference environment uses Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, YAM adapter `0059d02a1a91bc6834c1d0846022891df0c78730`, Inspect Robots 1.0.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.1.7. Runtime dependencies also include NumPy, SciPy, Pillow, PyTorch and imageio-ffmpeg.

Run these commands from this task directory. Each trial creates its own output directory.

```bash
isaac_python="${ISAAC_PYTHON:-/data/IsaacLab-Arena/.venv/bin/python}"
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=2

"$isaac_python" run_workstation_socket_probe.py --fixture assets/workstation-usbc.usda --usb-c --output runs/expert
"$isaac_python" audit_workstation_trial.py runs/expert

# Configure a Dream Labs credential according to the walkthrough.
"$isaac_python" run_workstation_socket_probe.py --fixture assets/workstation-usbc.usda --usb-c --dream --output runs/dream
"$isaac_python" audit_workstation_trial.py runs/dream

"$isaac_python" run_workstation_socket_probe.py --fixture assets/workstation-usbc.usda --usb-c --setup-only --output runs/setup
```

All runs use seed 0 and disable object jitter. The Type-C detent runs at 960 physics steps per second with 20 Hz robot controls. The limit is 2,400 controls at 20 Hz, or 120 simulated seconds. Each H.264 frame corresponds to one recorded control.

The scripted demonstration uses simulator pose feedback to align each held plug and sends joint position commands. During insertion its controller scales the arm position gains by 5 and the damping gains by 2.2, within the YAM actuator effort and velocity limits. Dream-1 uses its standard controller and the normal camera and joint observations.

The reference scripted run scores 1.0 in 1,509 controls (75.45 simulated seconds). The two released plugs end at 6.319 and 6.321 mm of insertion and remain seated during routing. The Dream-1 reference run scores 0.0 after 2,400 controls.

## Fixed setup

Positions use metres in the dual-arm frame. Positive X points toward the devices and positive Y points to the robots' left.

| Object | Dimensions and represented material | Reset |
| --- | --- | --- |
| Two device enclosures | 83 × 105 × 44 mm polymer bases, Molex 1054500101 sockets with a polymer tongue | Target port mouths at `(0.390, -0.220, 0.803)` and `(0.390, 0.220, 0.803)`. Each marked port is beneath a green dot. The negative-Y device has an additional unmarked socket. |
| Two plugs | 21 × 12 × 8 mm blue grip housings around Molex 1054440001 plugs. Simulated shells are 6.65 × 8.25 × 2.40 mm | Tips at `(0.230, -0.290, 0.803)` and `(0.230, 0.290, 0.803)`. Both connectors point along +X. |
| Plug supports | 40 × 42 × 49 mm polymer stands | Stands at `(0.225, ±0.290, 0.7745)`, with the cord exits behind the rear edge. |
| Cable and collar | Cable diameter 5.6 mm, flexible polymer represented by articulated capsules. The gold collar measures 16 × 16 × 11 mm with a 24 × 15 × 18 mm raised grip. | The fixed USD contains the S-shaped layout and collar at the cable midpoint. |
| Cable guide | Polymer channel, 55 × 135 mm bed with two 16 mm high side lips | Centered at `(0.232, 0, 0.752)`. |
| Worktop and floor | 600 × 720 × 40 mm wood worktop and 3 × 3 × 0.04 m floor | Worktop top at 0.75 m and center at `(0.42, 0.045, 0.73)`. Floor top at 0 m. |
| YAM arms and adapter marker | Two six-axis arms and 10 mm bookkeeping marker outside the working area | Arm bases at `(0, ±0.305, 0.75)`. Marker at `(-0.5, 0, -0.05)`. |

## Assemble the fixture with stock USB-C parts

Use two Molex **1054440001** plugs and three Molex **1054500101** mating receptacles. Mark two ports and leave the third unmarked. Example single-unit prices are $2.90 per plug and $2.74 per receptacle, or about $14 for the five components before shipping. [Plug listing](https://www.newark.com/molex/105444-0001/usb-conn-3-1-usb-type-c-plug-surface/dp/01AH8106), [receptacle listing](https://www.digikey.com/en/products/detail/molex/1054500101/5843891).

1. Mount each receptacle by its shell tabs on a backed 0.80 mm PCB coupon with the manufacturer's footprint. Bolt the backing into a plastic device base so the shell carries insertion load without rocking. Leave the mouth face flush and accessible at the listed coordinates. Set the unmarked receptacle 27 mm along +Y from the negative-Y target. [Molex receptacle drawing](https://www.molex.com/content/dam/molex/molex-dot-com/products/automated/en-us/salesdrawingpdf/105/105450/1054500101_sd.pdf).
2. Attach each stock plug to a 0.80 mm carrier coupon using the manufacturer's footprint. Encapsulate the backs in two 21 × 12 × 8 mm blue printed grips. Leave the complete nose exposed and use the stock metal shells, contacts and retention springs. Anchor 908 mm of flexible 5.6 mm jacketed cord between the recessed exits. Keep the cord and circuit cards unpowered. Verify that the metal ends and carriers stay fixed in their grips under a 20 N axial proof pull. [Molex plug drawing](https://www.molex.com/content/dam/molex/molex-dot-com/products/automated/en-us/salesdrawingpdf/105/105444/1054440001_sd.pdf).
3. Print or machine the [stands, channel and collar](fixture.scad); these parts only guide the cable and support the initial plug poses. Fit the collar at mid-span. Match the worktop and fixed layout coordinates above.
4. Mark the actual fully seated shoulder position on the device using a ruler and a known-good insertion in each orientation. Use the stock connector's clean, level engagement when marking. Use a 0.40 mm witness band toward the robot from that seated reference. Confirm the gauge readings separately for both ports.

## Calibrate insertion and retention

Molex's [qualification summary](https://www.molex.com/content/dam/molex/molex-dot-com/products/automated/en-us/testsummarypdf/105/105444/1054440001-TS.pdf) gives 5–20 N mating and 8–20 N initial extraction limits. It calls for slow axial travel and extraction after five conditioning cycles. Use a calibrated 0–50 N axial force gauge and a guided screw slide: align the plug axis with the mouth, travel at up to 12.5 mm/min, and record force and depth through mating and unmating. Test both plug/port pairs on the sixth cycle, in both orientations. Record the insertion and extraction peaks and where the latch settles. Use the measured curve to update the simulation if the peak load or engagement position differs materially. For a separate seated load check, apply axial pull below the measured extraction peak and observe no disconnect, then exceed the peak and confirm normal release.

The simulator uses independently authored colliding shells and a tongue. The stock receptacle opening is 8.34 × 2.56 mm, and the plug shell is 8.25 × 2.40 mm with a 6.65 mm exposed nose. Convex strips approximate the rounded profile and chamfers. The simulator applies a passive axial force curve at every physics step near an aligned port. The same curve applies to the unmarked socket. Nominal peak forces are 9.7 N for insertion and 12.0 N for extraction, based on the published limits. In a guided simulation bench trial the released plug sat at 6.20 mm, held 5.90 mm of insertion under an 8 N axial pull, and detached as the load passed about 12 N. The physical fixture can be calibrated against the stock pair with the procedure above.

## Physical scoring with a camera

Use synchronised fixed views of both socket mouths and the guide at 20 fps or higher. For each port, put two witness lines on a small stationary ruler beside the connector, one at the calibrated fully seated grip position and one 0.40 mm towards the robot. Use a close-up looking across the mouth so the grip front and the two lines can be compared and the front can be seen entering the marked port. The visible gap between the grip front and the fully seated mark must be at most 0.40 mm. Use a second view if an arm covers a mouth; an uncertain interval must be resolved from the footage before awarding credit.

- **0.35 per plug.** Its tip is inside the correct marked opening, its grip front is in the witness band, and no part of a gripper touches it for a continuous second. From the start of that second through the end of the run the grip may not leave the witness band, including while the collar is being placed. The plug must also remain free of gripper contact in the last second.
- **0.30 for routing.** Both plugs qualify. The centre mark on the collar lies inside the outlined rectangle, the collar rests in the channel, and both hands are clear of it throughout the final second. A collar suspended above the bed or held by a finger earns no routing credit.

The rest of the outcomes score the sum of earned awards, including zero if neither plug qualifies. Use the video to identify release and visible seating throughout the trial. The fixture does not require an electrical test.

## Simulation scoring

- **0.35 per plug.** Establish a released connection for 20 consecutive controls and keep the plug seated through the end. A seated plug is 5.8–6.65 mm deep and within 1.0 mm horizontally and vertically of the socket center. Its long axis and shell profile each align within 6° of the socket. Either USB-C roll orientation is allowed. Summed gripper contact must be at most 0.1 N when establishing the connection and for the final 20 controls.
- **0.30 for routing.** Both plugs qualify, and for the final 20 controls the collar stays within 18 mm across and 45 mm along the guide center, 2–16 mm above the worktop, with at most 0.1 N gripper contact.

The packaged USD is self-contained. Both shells contain open metal walls and slide around physical tongues. The cable segments have spherical constraints and transmit tensile loads. The scorer reads depth, pose and contact directly as proxies for the witness-band and release observations. A solid backstop at 6.35 mm guards against overtravel. The window allows numerical contact penetration and a small clearance from the 6.20 mm model detent centre.
