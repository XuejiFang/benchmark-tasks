// Workstation connector fabrication geometry, all dimensions in millimetres.
// Set part to preview, plug, grip, shell, socket, tongue, stand, guide, or collar.
// The socket and shell drawings describe finished mating surfaces.
part = "preview";

module block(pos, size) {
    translate(pos) cube(size, center=true);
}

module grip() {
    block([-19.2, 0, 0], [21, 12, 8]);
}

module shell() {
    for (s = [-1, 1]) {
        block([-4.3, s*4.05, 0], [8.6, .7, 2.8]);
        block([-4.3, 0, s*1.15], [8.6, 8.1, .5]);
    }
}

module tongue() {
    block([7.1, 0, 0], [4.5, 6.8, .68]);
}

module socket() {
    for (s = [-1, 1]) {
        block([5.4, s*5.85, 0], [10.8, 1.5, 7]);
        block([5.4, 0, s*2.75], [10.8, 13.2, 1.2]);
    }
    block([10, 0, 0], [2, 14, 8]);
}

module guide() {
    block([0, 0, 2], [55, 135, 4]);
    for (s = [-1, 1]) block([s*29, 0, 8], [6, 135, 16]);
}

module collar() {
    // Two polymer halves with a longitudinal cable groove. Add a thin strap
    // around the body; do not obstruct the raised tab or the guide.
    difference() {
        union() {
            block([0, 0, 0], [16, 16, 11]);
            block([0, 0, 12], [24, 15, 18]);
        }
        rotate([90, 0, 0]) cylinder(d=5.6, h=18, center=true, $fn=40);
    }
}

module enclosure(target_y, extra_port=false) {
    block([433, target_y, 22], [83, 105, 44]);
    for (offset = extra_port ? [0, 27] : [0])
        translate([390, target_y+offset, 53]) {
            socket();
            tongue();
        }
    block([392.5, target_y+13.4, 53], [5, 13.4, 9]);
    block([392.5, target_y-14.5, 53], [5, 16, 9]);
    block([368, target_y, 41.7], [26, 22, 14]);
}

if (part == "preview") {
    enclosure(-220, true);
    enclosure(220);
    translate([232, 0, 0]) guide();
    for (s = [-1, 1]) {
        block([210, s*290, 24.5], [53, 42, 49]);
        translate([230, s*290, 53]) {
            color("steelblue") grip();
            shell();
        }
    }
    translate([168, 0, 5.5]) color("goldenrod") collar();
} else if (part == "plug") { grip(); shell(); }
else if (part == "grip") grip();
else if (part == "shell") shell();
else if (part == "socket") socket();
else if (part == "tongue") tongue();
else if (part == "stand") block([0, 0, 24.5], [53, 42, 49]);
else if (part == "guide") guide();
else if (part == "collar") collar();
