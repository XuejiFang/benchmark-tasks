// Physical stand mounting layout, mm in height_lock.scad's foot frame.
// Set height to top, middle, or bottom. Transparent parts are clearance
// envelopes. Fastener and clevis dimensions come from the purchased parts.
height = "top";
drop = height == "middle" ? 56.44 : height == "bottom" ? 112.88 : 0;
damper_retracted = 225; // replace with measured clevis-to-clevis value
damper_fixed_z = 395 - damper_retracted - 131.44;
trim_eye_z = 427; // adjust after measuring the installed constant-force strips
$fn = 32;

module block(c, size) {
    translate(c) cube(size, center=true);
}

module pin_y(c, size, diameter) {
    translate(c) rotate([90, 0, 0]) cylinder(h=size, d=diameter, center=true);
}

module vertical_span(x, y, a, b, diameter) {
    translate([x, y, (a+b)/2]) cylinder(h=abs(a-b), d=diameter, center=true);
}

// Frame base and rear crossmembers; make the under-load joints rigid.
color("dimgray") block([-100, 0, -3], [300, 250, 6]);
for (y = [-84, 84]) block([-129, y, 225], [40, 20, 450]);
block([-129, 0, 440], [40, 185, 20]);
block([-87, 0, 311], [12, 12, 200]);
for (z = [345, 390]) block([-87, 0, z-drop], [16, 27, 35]);

// Captive strip tails and rear carriage clevis ledges.
color("slategray") block([-156, 0, 340-drop], [8, 184, 6]);
for (side = [-1, 1]) {
    color("silver", .35) block([-156, side*60, 475], [52, 36, 56]);
    for (ds = [-11, 11]) {
        y = side*60 + ds;
        w = ds*side < 0 ? 19.05 : 9.53;
        color("silver") block([-152, y, (465 + 340-drop)/2], [.4, w, 465-(340-drop)]);
    }
}
color("orange") vertical_span(-152, 25, 340-drop, trim_eye_z, 10);
color("gray", .35) block([-152, 25, (390+469)/2], [8, 20, 79]);

// Damper flex clevises and mounting slot are dimensioned to their axes.
color("steelblue") vertical_span(-202, 0, damper_fixed_z, damper_fixed_z+180, 26);
color("silver") vertical_span(-202, 0, damper_fixed_z+180, 395-drop, 8);
color("gray", .35) block([-218, 0, 50], [6, 26, 90]);
pin_y([-202, 0, (395-drop)], 16, 6);
pin_y([-202, 0, damper_fixed_z], 16, 6);

// VESA hinge axis, pivot washer stacks and horizontal pin return pocket.
pin_y([-21.56, 0, 406-drop], 132, 8);
for (y = [-55, 55]) pin_y([-21.56, y, 406-drop], 6, 25);
color("orange", .4) pin_y([-86, 33, 406-drop], 56, 12);
pin_y([-86, 5, 406-drop], 2, 12);
pin_y([-86, 61, 406-drop], 2, 12);
