x0 = 0; x1 = 1;
y0 = 0; y1 = 1;
z0 = 0; z1 = 1;

//         ^ z
//         |
// (5)--------(7)
//  |\     |   |\
//  | \    |   | \
//  |  \   |   |  \
//  |  (6)--------(8)  y
//  |   |  +-- |-- | -->
// (1)--|---\-(3)  |
//   \  |    \  \  |
//    \ |     \  \ |
//     \|      \  \|
//     (2)------\-(4)
//             x \

Point(1) = {x0, y0, z0};
Point(2) = {x1, y0, z0};
Point(3) = {x0, y1, z0};
Point(4) = {x1, y1, z0};
Point(5) = {x0, y0, z1};
Point(6) = {x1, y0, z1};
Point(7) = {x0, y1, z1};
Point(8) = {x1, y1, z1};

//         ^ z
//         |
//  +-----7----+
//  |\     |   |\
//  | 3    |   | 4
//  9  \   |  10  \
//  |   +----8-----+   y
//  |   |  +-- |-- | -->
//  +---|-5-\--+   |
//   \ 11    \  \ 12
//    1 |     \  2 |
//     \|      \  \|
//      +----6--\--+
//             x \

Line( 1) = {1, 2};
Line( 2) = {3, 4};
Line( 3) = {5, 6};
Line( 4) = {7, 8};
Line( 5) = {1, 3};
Line( 6) = {2, 4};
Line( 7) = {5, 7};
Line( 8) = {6, 8};
Line( 9) = {1, 5};
Line(10) = {3, 7};
Line(11) = {2, 6};
Line(12) = {4, 8};

Line Loop(1) = { 5, 10, -7,  -9};
Line Loop(2) = { 6, 12, -8, -11};
Line Loop(3) = {-1,  9,  3, -11};
Line Loop(4) = {-2, 10,  4, -12};
Line Loop(5) = { 1,  6, -2,  -5};
Line Loop(6) = { 3,  8, -4,  -7};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};

Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};

Physical Volume(1)  = {1};
Physical Surface(1) = {1, 2, 3, 4, 5, 6};

Transfinite Line "*" = 3;
Transfinite Surface "*";
Recombine Surface   "*";
