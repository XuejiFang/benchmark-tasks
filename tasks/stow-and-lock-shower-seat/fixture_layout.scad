// Side/front assembly drawing in millimetres. Origin is the seat shaft.
// X is across the seat, U (OpenSCAD Y) is towards the robot, Z is up.
// Drawn at the horizontal reset position, with the keeper open.
$fn = 48;

module block(pos, size) {
    translate(pos) cube(size, center=true);
}

module rod_x(pos, length, diameter) {
    translate(pos) rotate([0, 90, 0]) cylinder(h=length, d=diameter, center=true);
}

module rod_u(pos, length, diameter) {
    translate(pos) rotate([90, 0, 0]) cylinder(h=length, d=diameter, center=true);
}

// Unloaded seat skin envelope, trim positions, and contact rail.
color("lightgrey") block([0, 215, 0], [450, 430, 52]);
rod_x([0, 0, 0], 610, 16);
color("steelblue") rod_x([-275, 230, 51], 52, 32);
for (x = [-304, -246]) translate([x, 230, 31]) cylinder(h=40, d=9, center=true);
rod_x([-251, 230, 10], 135, 14);

// Blocks are nominal envelopes. Trim the actual masses using fixture_balance.py.
for (s = [-1, 1]) {
    color("darkslategrey") block([s*165, 330, 0], [32, 25, 8]);
    color("darkslategrey") block([s*260, -130, 0], [35, 55, 55]);
    block([s*260, -65, 0], [14, 130, 8]);
}
color("darkslategrey") rod_x([0, 0, 0], 265, 45);
// Fixed bearing supports and frame mounting plane.
for (s = [-1, 1]) block([s*297, -32, -16], [16, 70, 30]);
color("grey", .25) block([0, -70, 65], [450, 8, 680]);

// Keeper axle, the finger, and the orange release pad.
translate([246, 36, 365]) {
    rod_u([0, -6, 0], 36, 10);
    color("orange") block([0, 0, 36], [15, 10, 75]);
    rod_u([0, 57.5, 0], 115, 12);
    color("orange") translate([0, 115, -26]) cylinder(h=52, d=18, center=true);
}
block([258, -20, 364], [24, 86, 26]);
