# Starting setup

The five target objects are free rigid bodies. The bag is kinematic and the room objects are static. The arm mounts, camera rig and lights are loaded automatically.

| Target or bag | Material appearance | Size (cm) | Mass (g) | Room-frame XY and bottom Z (m) |
| --- | --- | --- | --- | --- |
| Open duffel | Canvas, leather | 67.2 × 46.2 × 29.5 body | Fixed | Origin (0.55, 0.65, 0.182), yaw 90° |
| Teal capped bottle | Steel shell, plastic cap | Ø7.3 × 25.6 | 250 | (0.49, 0.08), 0.453; sideways |
| Blue-white left shoe | Fabric and rubber | 29.7 × 11.0 × 10.5 | 300 | (0.90, −0.08), 0.019; yaw −90° |
| Blue-white right shoe | Fabric and rubber | 29.7 × 11.0 × 10.5 | 300 | (1.03, −0.08), 0.019; yaw −90° |
| Black wallet | Leather | 11.5 × 9.0 × 1.5 | 120 | (0.62, 1.35), 0.558; flat |
| Over-ear headphones | Plastic, metal, padding | 20.2 × 8.35 × 17.75 | 250 | (0.83, 1.40), 0.558; laid sideways |

The fixed background is the USD stage at reset. Dimensions below are room-frame axis-aligned envelopes of its components in centimetres. Groups with several separated parts use their combined envelope; source and mesh names identify the exact fixed geometry. Lighting and cameras are scene infrastructure.

| Room object | Material appearance | Envelope (cm) |
| --- | --- | --- |
| Coffee table | stone, timber | 132.0 × 65.9 × 42.8 |
| Sofa | linen upholstery | 252.0 × 91.6 × 98.0 |
| Throw pillows | fabric | 60.1 × 36.9 × 27.3 |
| Plant and pot | foliage, earthenware | 122.4 × 116.4 × 122.0 |
| Basket | wicker | 66.0 × 55.7 × 17.8 |
| Side table | wood, metal | 55.0 × 55.0 × 52.7 |
| Lamp | painted metal | 38.1 × 41.7 × 64.0 |
| Charging cable and plug | polymer jacket, plastic, metal | 37.1 × 28.9 × 1.6 |
| Notebook | paper and card | 30.5 × 16.6 × 1.3 |
| Mug with tea | ceramic and fixed colored surface | 12.9 × 8.6 × 9.0 |
| Coaster | cork | 12.0 × 12.0 × 0.4 |
| Fruit bowl | wood | 22.7 × 22.5 × 6.3 |
| Tangerines | peel appearance | 13.8 × 8.8 × 7.3 |
| Keys and ring | metal | 6.0 × 7.2 × 0.5 |
| Remote | plastic and rubber | 6.4 × 16.4 × 2.2 |
| Spare bottle | painted steel, cap plastic | 8.7 × 8.7 × 25.6 |
| Shopping tote | canvas | 35.8 × 25.4 × 47.0 |
| Blanket | woven cloth appearance | 62.1 × 90.0 × 34.8 |
| Sweatshirt | knit fabric appearance | 49.2 × 38.7 × 2.7 |
| Folded towel | cotton appearance | 32.2 × 38.2 × 1.7 |
| Slippers | fabric and rubber | 31.7 × 32.2 × 6.7 |
| Rug | wool appearance | 365.0 × 285.0 × 1.8 |
| Books on floor | paper and card | 29.7 × 35.2 × 9.6 |
| Wall shelf | oak | 105.0 × 36.0 × 3.5 |
| Books on wall shelf | paper and card | 35.2 × 17.0 × 27.1 |
| Framed abstract artwork | timber, canvas and pigment | 132.0 × 5.4 × 83.0 |
| Curtains | linen | 11.0 × 357.0 × 257.4 |
| Window frames | painted timber | 15.0 × 335.0 × 200.5 |
| Room flooring | oak | 1000.0 × 1000.0 × 10.0 |
| Walls and skirting | plaster, painted timber | 1400.0 × 640.0 × 500.0 |
| Bottle tray | silicone appearance | 29.0 × 16.0 × 1.9 |
| Robot pedestal, base and feet | metal | 40.0 × 102.0 × 53.5 |
| Bag footstool, cushion and legs | wood and fabric | 52.0 × 78.0 × 16.3 |

The dual six-axis arms use the packaged YAM robot description with nominal 26 cm upper arms, 25 cm forearms and approximately 25 cm from the wrist to the grasp point. Each parallel jaw spans up to 9.4 cm.
