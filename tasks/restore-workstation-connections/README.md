# Connect and route a workstation cable

Connect two marked ports and route the cable through the open guide while both connectors stay seated.

## Run and score

Use the Linux environment in the [RoboCurve YAM walkthrough](https://tasks.robocurve.org/walkthrough). The reference image uses Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, IsaacLab-Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, YAM adapter `0059d02a1a91bc6834c1d0846022891df0c78730`, Inspect Robots 0.60.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.0.14. Runtime dependencies also include NumPy, SciPy, Pillow, PyTorch and imageio-ffmpeg.

Run these commands from this task directory. Each trial creates its own output directory.

```bash
isaac_python=/data/IsaacLab-Arena/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=2

"$isaac_python" run_workstation_socket_probe.py --fixture assets/workstation.usda --output runs/expert
"$isaac_python" audit_workstation_trial.py runs/expert

# Configure a Dream Labs credential according to the walkthrough.
"$isaac_python" run_workstation_socket_probe.py --fixture assets/workstation.usda --dream --output runs/dream
"$isaac_python" audit_workstation_trial.py runs/dream

"$isaac_python" run_workstation_socket_probe.py --fixture assets/workstation.usda --setup-only --output runs/setup
```

All runs use seed 0 and disable object jitter. The limit is 2,400 controls at 20 Hz, or 120 simulated seconds. Each H.264 frame corresponds to one recorded control.

The reference expert scored 1.0 after 1,733 controls (86.65 simulated seconds). Dream-1 scored 0.0 after 2,400 controls and 50 API requests. The Dream run took about 20 minutes on the reference L40S while other trials were running.

## Fixed setup

Positions use metres in the dual-arm frame. Positive X points toward the devices and positive Y points to the robots' left.

| Object | Dimensions and represented material | Reset |
| --- | --- | --- |
| Two device enclosures | 83 × 105 × 44 mm polymer bases, metal sockets with a polymer tongue | Target port mouths at `(0.390, -0.220, 0.803)` and `(0.390, 0.220, 0.803)`. Each marked port is beneath a green dot. The negative-Y device has an additional unmarked socket. |
| Two plugs | 21 × 12 × 8 mm molded blue grips, 8.6 mm long hollow metal shells | Tips at `(0.230, -0.290, 0.803)` and `(0.230, 0.290, 0.803)`. Both connectors point along +X. |
| Plug supports | 53 × 42 × 49 mm polymer stands, 26 × 22 × 14 mm strain-relief shelves | Stands at `(0.210, ±0.290, 0.7745)`. Shelves sit in front of each green-marked port. |
| Cable and collar | Cable diameter 5.6 mm, flexible polymer represented by articulated capsules. The gold collar measures 16 × 16 × 11 mm with a 24 × 15 × 18 mm raised grip. | The fixed USD contains the S-shaped layout and collar at the cable midpoint. |
| Cable guide | Polymer channel, 55 × 135 mm bed with two 16 mm high side lips | Centered at `(0.232, 0, 0.752)`. |
| Worktop and floor | 600 × 720 × 40 mm wood worktop and 3 × 3 × 0.04 m floor | Worktop top at 0.75 m and center at `(0.42, 0.045, 0.73)`. Floor top at 0 m. |
| YAM arms and adapter marker | Two six-axis arms and 10 mm bookkeeping marker outside the working area | Arm bases at `(0, ±0.305, 0.75)`. Marker at `(-0.5, 0, -0.05)`. |

## Build the physical fixture

Use the passive machining replica in [fixture.scad](fixture.scad). The custom connector is the geometry tested in simulation. Commercial USB-C parts have different shape and retention and are not drop-in replacements. The replica has no electrical contacts. It uses visible seating marks for the physical trial.

1. Make two blue grips from nylon, PETG or machined acetal. Build their shells from 0.50 mm stainless sheet (top/bottom) and 0.70 mm stainless strip (side walls), or machine the equivalent open tube. The finished nose is 8.60 mm long, 8.80 mm wide and 2.80 mm high. The through opening is 7.40 × 1.80 mm. Remove burrs and keep seam filler out of the opening. Attach the back of each shell to the grip with an internal backing tab or adhesive fillet confined behind the mating length, leaving the outside grip and nose profile as drawn. Add internal ballast in the grip as needed for a 46 g finished plug.
2. Make three identical socket blocks. Machine or assemble the rectangular opening from 1.50 mm side plates and 1.20 mm lip plates. The opening is 10.20 × 4.30 mm, with the stop face 9.00 mm behind the mouth. Fit a centred 6.80 × 0.68 mm acetal or nylon tongue whose front is 4.85 mm behind the mouth. The tongue is supported by a centred slit in the rear stop; glue from the rear and remove excess. Hold the shell, cavity and tongue mating dimensions to ±0.10 mm. Verify clearance with a plug before attaching the blocks. No added snap lock or spring pad is used.
3. Screw or clamp the bases to the bench. Use the `preview` coordinates to fixture the two green-marked mouths, the extra unmarked mouth and the two shelves. A non-contact rear riser between the base and back of each socket supports the block; keep it behind the stop face so it does not interfere with insertion. Use double-sided mounting tape for the plug stands and guide. Mark the guide bed with a 36 × 90 mm rectangle centred on it.
4. Use 908 mm of 5.6 mm flexible PVC-jacketed cable between the two grip exits. The cable can be offcut stock without connected conductors. Fasten it to both backs with recessed cord anchors. Fit the two-piece gold collar at the midpoint and secure it with a thin strap or adhesive outside the wire groove. The drawing includes the groove and the raised tab. Lay the cable in the S path shown by the setup view, with the plug tips on their start coordinates, the collar centred at approximately X = 168 mm, Y = 0 mm, and the cable on the table between stands. Print a top-view template from the packaged USD if mounting several fixtures.

Set `part` in OpenSCAD to preview the assembly or export individual parts. The thin shell and socket features are dimensioned for machining or sheet assembly rather than an ordinary filament print. Before a trial, insert a loose plug by hand to the rear stop and calibrate its fully inserted position on each target receptacle using a steel ruler or feeler gauge. Repeat this with the reverse orientation. Both fits should bottom out without crushing the shell. If loose mounting hardware moves, or the shell binds before the stop, correct the assembly and redo calibration before testing the robot. This assembly specification is derived from the submitted CAD geometry; the reference videos are simulation runs.

## Physical scoring with a camera

Use synchronised fixed views of both socket mouths and the guide at 20 fps or higher. For each port, put two witness lines on a small stationary ruler beside the connector, one at the position of the grip front when the tip touches the 9.00 mm stop and one 1.50 mm towards the robot. Use a close-up looking across the mouth so the grip front and the two lines can be compared and the front can be seen entering the marked port. With the nose inside the opening and the grip front between the lines, insertion is 7.50–9.00 mm. Use a second view if an arm covers a mouth; an uncertain interval must be resolved from the footage before awarding credit.

- **0.35 per plug.** Its tip is inside the correct marked opening, its grip front is in the witness band, and no part of a gripper touches it for a continuous second. From the start of that second through the end of the run the grip may not leave the witness band, including while the collar is being placed. The plug must also remain free of gripper contact in the last second.
- **0.30 for routing.** Both plugs qualify. The centre mark on the collar lies inside the outlined rectangle, the collar rests in the channel, and both hands are clear of it throughout the final second. A collar suspended above the bed or held by a finger earns no routing credit.

The rest of the outcomes score the sum of earned awards, including zero if neither plug qualifies. Use the video to identify release and visible seating throughout the trial. The fixture does not require an electrical test.

## Simulation scoring

- **0.35 per plug.** Establish a released connection for 20 consecutive controls and keep the plug seated through the end. A seated plug is 7.5–10.5 mm deep and within 1.0 mm horizontally and vertically of the socket center. Its long axis and shell profile each align within 6° of the socket. Either USB-C roll orientation is allowed. Summed gripper contact must be at most 0.1 N when establishing the connection and for the final 20 controls.
- **0.30 for routing.** Both plugs qualify, and for the final 20 controls the collar stays within 18 mm across and 45 mm along the guide center, 2–16 mm above the worktop, with at most 0.1 N gripper contact.

The packaged USD is self-contained. Both shells contain open metal walls and slide around physical tongues. The cable segments have spherical constraints and transmit tensile loads. The scorer reads depth, pose and contact directly as proxies for the witness-band and release observations. The 10.5 mm numerical upper bound guards against impossible pass-through; the solid stop sets the nominal physical upper bound of 9 mm.
