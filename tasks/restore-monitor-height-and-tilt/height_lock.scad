// Three-position monitor height lock, top slot at reset. Units are mm,
// relative to the stand foot. Set part to assembly, rail, or pin.
part = "assembly";
$fn = 48;

module block(centre, size) {
    translate(centre) cube(size, center=true);
}

module rail() {
    block([-101, 104, 347.56], [10, 22, 150.88]);
    for (shelf = [[279.62, 15], [321.34, 44.44], [377.78, 44.44], [417.50, 11]])
        block([-86, 104, shelf[0]], [20, 22, shelf[1]]);
}

module pin() {
    translate([-86, 87, 406]) {
        rotate([90, 0, 0]) cylinder(d=8, h=32, center=true);
        block([0, -16, 0], [17, 7, 15]);
        block([0, -16, -20], [11, 8, 40]);
        block([-47, -23, -39], [100, 9, 10]);
        block([-95, -26, -39], [24, 11, 23]);
        block([-106, -37, -39], [46, 38, 16]);
        color("orange") block([-120, -51, -39], [36, 14, 48]);
    }
}

if (part == "assembly") {
    rail();
    pin();
    // Envelope of the moving pin guide. Drill and bush the through bore
    // on the pin centreline before bolting the block to the carriage.
    color("silver", .45) block([-86, 61, 406], [17, 37, 13]);
} else if (part == "rail") rail();
else if (part == "pin") pin();
