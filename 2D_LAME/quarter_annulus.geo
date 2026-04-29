// Gmsh project created on Sat Jan 17 15:40:05 2026
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0.0015, 0, 0, 1.0};
//+
Point(3) = {0.0017, 0, 0, 1.0};
//+
Point(4) = {0, 0.0015, 0, 1.0};
//+
Point(5) = {0, 0.0017, 0, 1.0};
//+
Circle(1) = {4, 1, 2};
//+
Circle(2) = {5, 1, 3};
//+
Line(3) = {5, 4};
//+
Line(4) = {3, 2};
//+
Curve Loop(1) = {2, 4, -1, -3};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {2, 4, -1, -3};
//+
Curve Loop(3) = {1, -4, -2, 3};
//+
Plane Surface(2) = {2, 3};
//+
Physical Surface("quarter_annulus", 5) = {1};
//+
Physical Curve("r_o", 6) = {2};
//+
Physical Curve("r_i", 7) = {1};
