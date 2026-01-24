$fn=200;

// Copyright 2025 Jack Kingsman <jack.kingsman@gmail.com>
// Licensed under Creative Commons Attribution 4.0
// CC-BY 4.0 https://creativecommons.org/licenses/by/4.0/

//////
// customizable inputs
//////

// how rounded the corner is
corner_radius_config = 0.1;

// how big the thing is, as a factor of size from the phi-unit dodecahedron
scale_factor_config = 18;

// radius of the bearing hole. 11, for standard 608 bearings
bearing_hole_radius_config = 11;

// depth of the bearing hole. 7, for standard 608 bearings. HOWEVER, you may 
// want to expand this a bit if you don't want to need to perfectly extract
// your supports; it's a tight press fit. If you really trust your tolerances,
// bring this down to 7.
bearing_hole_depth_config = 7.1;

// this bores out the center of the bearing hole, leaving a ledge on the edges for the bearing's
// outer ring to rest on but keeping the inner ring free of obstruction, even with pressure. This means
// you can grip it a little bit tighter while retaining more rotation freedom
central_bore_depth_config = 3; // 0 disables central bore 
central_bore_ledge_width_config = 2; // central bore depth = bearing_hole_radius_config - central_bore_ledge_thickness. 2 is usually about right for 608 bearings

//////
// illustrated hole
//////

// a <-> b: bearing_hole_radius_config
// c: bearing_hole_depth_config
// d: central_bore_depth_config (can be zero)
// e: central_bore_ledge_width_config
// total hole depth at deepest point = bearing_hole_depth_config + central_bore_depth_config

// ------------a                      b----------------
//            ↑|                      |                
//            c|   [bearing hole]     |
//            ↓|___                ___|
//                ↑|              | ←e→
//                d|[central bore]|
//                ↓|              |
//                 ----------------

//////
// constants & computed values
//////
// provides base of our phi-unit dodecahedron
PHI = (1 + sqrt(5)) / 2; // ~1.61803

// angle between two faces, used to rotate around dodec and bore holes
DIHEDRAL_ANGLE = 2 * atan(PHI); // ~116.565

// rotation about the x axis needed to bring us off an edge and onto the flat face
FACE_NORMAL_ANGLE = atan(2); // ~63.4349

// this was revealed to me in a dream (an LLM told me, and I distrust the provenance but the value seems right)
// center to face angle: cos(θ) = sin(π/5)/√3
// this is used to rotate about the z-axis to align edges with the x axis
RADIAN_CONVERSION = 180 / PI; // thanks openscad
ANGLE_BETWEEN_NORMAL_AND_EDGE = acos(sin((PI / 5) * RADIAN_CONVERSION) / sqrt(3)); // ~70.1622

// derived by distance between two vertices, (0, φ, 1/φ) to (1, 1, 1) = √(Δx² + Δy² + Δz²) = √(1 + (1-φ)² + (1-1/φ)²)
// only valid for the phi-unit dodecahedron but we can scale it
EDGE_LENGTH_CONSTANT = sqrt(1 + pow((1 - PHI), 2) + pow((1 - 1 / PHI), 2)); // ~1.2360

// rounded corners inflates our size by a bit, so we need to factor that in
RADIUS_CORRECTED_SCALE_FACTOR = (scale_factor_config * corner_radius_config);

// from https://en.wikipedia.org/wiki/Regular_dodecahedron. The radius of the inscribed sphere
// of a phi-unit dodec, corrected for the inflated size from rounded corners.
INSCRIBED_SPHERE_EDGE_MULTIPLICAND = (pow(PHI, 2) / (2 * sqrt(3 - PHI))) * scale_factor_config; // ~1.11352 * scale 

// assemble it all together to get origin-to-face distance i.e. the correct radius of the inscribed sphere
// and how far from the origin our bore holes should start
DISTANCE_TO_FACE = INSCRIBED_SPHERE_EDGE_MULTIPLICAND * EDGE_LENGTH_CONSTANT + RADIUS_CORRECTED_SCALE_FACTOR;

// generates a phi-unit regular dodecahedron with rounded-corner radius `corner_sphere_radius`, scaled by `scale`
module regularDodecahedron(corner_sphere_radius, scale) {
  scale([scale, scale, scale]){
    hull(){
      // (±1 , ±1 , ±1) unit cube
      for(x = [-1, 1], y = [-1, 1], z = [-1, 1]) {
        translate([x,y,z]) sphere(corner_sphere_radius);
      }
      
      // (0, ±ϕ, ±1/ϕ) vertices
      for(y = [-PHI, PHI], z = [-1 / PHI, 1 / PHI]) {
        translate([0,y,z]) sphere(corner_sphere_radius);
      }
      
      // (±1/ϕ, 0, ±ϕ) vertices
      for(x = [-1 / PHI, 1 / PHI], z = [-PHI, PHI]) {
        translate([x,0,z]) sphere(corner_sphere_radius);
      }
      
      // (±ϕ, ±1/ϕ, 0) vertices
      for(x = [-PHI, PHI], y = [-1/PHI, 1/PHI]) {
        translate([x,y,0]) sphere(corner_sphere_radius);
      }
    }
  }
}

// generates a cylinder at the given offset distance with a given radius and depth
module face_cylinder(offset_distance, cylinder_radius, cylinder_depth) {
  // Position cylinder at the center of each face
  translate([0, 0, offset_distance]) {
    // depth times two so we bore the face out cleanly
    cylinder(h = cylinder_depth * 2, r = cylinder_radius, center = true);
  }
}

// generate bored-out cylinders on each face of the dodec of radius `bearing_hole_radius`
// and depth `bearing_hole_depth`
module offset_spaced_face_cylinders(bearing_hole_radius, bearing_hole_depth) {
  // cutouts for each face
  // top + bottom
  face_cylinder(DISTANCE_TO_FACE, bearing_hole_radius, bearing_hole_depth);
  rotate([180, 0, 0]) {
    face_cylinder(DISTANCE_TO_FACE, bearing_hole_radius, bearing_hole_depth);
  }

  // 360 / 5 divides us into a segment for each face
  // dihedral or (180 - dihedral) gives the tilt angle from plane through the dodec center
  
  // middle faces
  for(i = [1:5]) {
    rotate([DIHEDRAL_ANGLE, 0, 360 / 5 * i]) {
      face_cylinder(DISTANCE_TO_FACE, bearing_hole_radius, bearing_hole_depth);
    }
  }

  // lower middle faces
  // i + 0.5 offsets us by half a face as compared to middle faces since the lower ring is offset
  for(i = [1:5]) {
    rotate([180-DIHEDRAL_ANGLE, 0, 360 / 5 * (i + 0.5)]) {
      face_cylinder(DISTANCE_TO_FACE, bearing_hole_radius, bearing_hole_depth);
    }
  }
}

// putting it all together
module fidget_solid(scale_factor, corner_radius, bearing_hole_radius, bearing_hole_depth) {
  difference() {
    // dodec itself. We rotate this into a position that makes the hole-boring simpler, mathematically,
    // and places us on a good face for printing
    rotate([FACE_NORMAL_ANGLE / 2, 0, ANGLE_BETWEEN_NORMAL_AND_EDGE / 2]) {
      regularDodecahedron(corner_radius, scale_factor);
    }
    
    // standard bearing bore
    offset_spaced_face_cylinders(bearing_hole_radius, bearing_hole_depth);

    // optional central bore
    if (central_bore_depth_config > 0) {
      central_bore_radius = bearing_hole_radius - central_bore_ledge_width_config;
      central_bore_depth = bearing_hole_depth + central_bore_depth_config;
      offset_spaced_face_cylinders(central_bore_radius, central_bore_depth);
    }
  }
}

// invoke the words of power
fidget_solid(scale_factor_config, corner_radius_config, bearing_hole_radius_config, bearing_hole_depth_config);
