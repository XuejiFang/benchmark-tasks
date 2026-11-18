// Mounting accessories for the stock Molex USB-C pair, in millimetres.
// The plug and receptacle are purchased components and are not printed.
part = "preview"; // preview, stand, guide, collar

module block(pos, size) {
    translate(pos) cube(size, center=true);
}

module stand() {
    block([0, 0, 24.5], [40, 42, 49]);
}

module guide() {
    block([0, 0, 2], [55, 135, 4]);
    for (s = [-1, 1]) block([s*29, 0, 8], [6, 135, 16]);
}

module collar() {
    difference() {
        union() {
            block([0, 0, 0], [16, 16, 11]);
            block([0, 0, 12], [24, 15, 18]);
        }
        rotate([90, 0, 0]) cylinder(d=5.6, h=18, center=true, $fn=40);
    }
}

if (part == "preview") {
    translate([232, 0, 0]) guide();
    for (s = [-1, 1]) translate([225, s*290, 0]) stand();
    translate([168, 0, 5.5]) color("goldenrod") collar();
} else if (part == "stand") stand();
else if (part == "guide") guide();
else if (part == "collar") collar();
