static char help[] = "Tests for uniform refinement\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  debug;             /* The debugging level */
  PetscInt  dim;               /* The topological mesh dimension */
  PetscBool refinementUniform; /* Uniformly refine the mesh */
  PetscBool cellHybrid;        /* Use a hybrid mesh */
  PetscBool cellSimplex;       /* Use simplices or hexes */
  PetscInt  testNum;           /* The particular mesh to test */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug             = 0;
  options->dim               = 2;
  options->refinementUniform = PETSC_FALSE;
  options->cellHybrid        = PETSC_TRUE;
  options->cellSimplex       = PETSC_TRUE;
  options->testNum           = 0;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex4.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex4.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-refinement_uniform", "Uniformly refine the mesh", "ex4.c", options->refinementUniform, &options->refinementUniform, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_hybrid", "Use a hyrbid mesh", "ex4.c", options->cellHybrid, &options->cellHybrid, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex4.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-test_num", "The particular mesh to test", "ex4.c", options->testNum, &options->testNum, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateSimplex_2D"
/* Two triangles
        4
      / | \
     8  |  10
    /   |   \
   2  0 7  1 5
    \   |   /
     6  |  9
      \ | /
        3

Becomes
           10
          / | \
        21  |  26
        /   |   \
      14 2 20 4  16
      /|\   |   /|\
    22 | 28 | 32 | 25
    /  |  \ | /  | 6\
   8  29 3 13  7 31  11
    \0 |  / | \  |  /
    17 | 27 | 30 | 24
      \|/   |   \|/
      12 1 19 5  15
        \   |   /
        18  |  23
          \ | /
            9
*/
PetscErrorCode CreateSimplex_2D(MPI_Comm comm, DM *dm)
{
  PetscInt       depth = 2;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    PetscInt    numPoints[3]         = {4, 5, 2};
    PetscInt    coneSize[11]         = {3, 3, 0, 0, 0, 0, 2, 2, 2, 2, 2};
    PetscInt    cones[16]            = {6, 7, 8,  7, 9, 10,  2, 3,  3, 4,  4, 2,  3, 5,  5, 4};
    PetscInt    coneOrientations[16] = {0, 0, 0, -2, 0,  0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0};
    PetscScalar vertexCoords[8]      = {-0.5, 0.0,  0.0, -0.5,  0.0, 0.5,  0.5, 0.0};

    ierr = DMPlexCreateFromDAG(*dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
  } else {
    PetscInt numPoints[3] = {0, 0, 0};

    ierr = DMPlexCreateFromDAG(*dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSimplexHybrid_2D"
/* Two triangles separated by a zero-volume cell with 4 vertices/2 edges
        5--16--8
      / |      | \
    11  |      |  12
    /   |      |   \
   3  0 10  2 14 1  6
    \   |      |   /
     9  |      |  13
      \ |      | /
        4--15--7
*/
PetscErrorCode CreateSimplexHybrid_2D(MPI_Comm comm, DM dm)
{
  Vec            coordinates;
  PetscSection   coordSection;
  PetscScalar    *coords;
  PetscInt       numVertices = 0, numEdges = 0, numCells = 0, cMax = PETSC_DETERMINE, fMax = PETSC_DETERMINE;
  PetscInt       firstVertex, firstEdge, coordSize;
  PetscInt       v, e;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    numVertices = 3 + 3;
    numEdges    = 6 + 2;
    numCells    = 3;
    cMax        = 2;
    fMax        = 15;
  }
  firstVertex = numCells;
  firstEdge   = numCells + numVertices;
  ierr        = DMPlexSetChart(dm, 0, numCells+numEdges+numVertices);CHKERRQ(ierr);
  if (numCells) {
    ierr = DMPlexSetConeSize(dm, 0, 3);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 1, 3);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(dm, 2, 4);CHKERRQ(ierr);
  }
  for (e = firstEdge; e < firstEdge+numEdges; ++e) {
    ierr = DMPlexSetConeSize(dm, e, 2);CHKERRQ(ierr);
  }
  ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
  ierr = DMPlexSetHybridBounds(dm, cMax, PETSC_DETERMINE, fMax, PETSC_DETERMINE);CHKERRQ(ierr); /* Indicate a hybrid mesh */
  /* Build cells */
  if (numCells > 0) {
    const PetscInt cone[3] = {9, 10, 11};
    const PetscInt ornt[3] = {0,  0,  0};
    const PetscInt cell    = 0;

    ierr = DMPlexSetCone(dm, cell, cone);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(dm, cell, ornt);CHKERRQ(ierr);
  }
  if (numCells > 1) {
    const PetscInt cone[3] = {12, 14, 13};
    const PetscInt ornt[3] = { 0, -2,  0};
    const PetscInt cell    = 1;

    ierr = DMPlexSetCone(dm, cell, cone);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(dm, cell, ornt);CHKERRQ(ierr);
  }
  if (numCells > 2) {
    const PetscInt cone[4] = {10, 14, 15, 16};
    const PetscInt ornt[4] = { 0,  0,  0,  0};
    const PetscInt cell    = 2;

    ierr = DMPlexSetCone(dm, cell, cone);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(dm, cell, ornt);CHKERRQ(ierr);
  }
  /* Build edges*/
  if (numEdges > 0) {
    const PetscInt cone[2] = {3, 4};
    const PetscInt edge    = 9;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
    ierr = DMPlexSetLabelValue(dm, "marker", edge, 1);CHKERRQ(ierr);
  }
  if (numEdges > 1) {
    const PetscInt cone[2] = {4, 5};
    const PetscInt edge    = 10;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
  }
  if (numEdges > 2) {
    const PetscInt cone[2] = {5, 3};
    const PetscInt edge    = 11;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
    ierr = DMPlexSetLabelValue(dm, "marker", edge, 1);CHKERRQ(ierr);
  }
  if (numEdges > 3) {
    const PetscInt cone[2] = {6, 8};
    const PetscInt edge    = 12;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
    ierr = DMPlexSetLabelValue(dm, "marker", edge, 1);CHKERRQ(ierr);
  }
  if (numEdges > 4) {
    const PetscInt cone[2] = {7, 6};
    const PetscInt edge    = 13;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
    ierr = DMPlexSetLabelValue(dm, "marker", edge, 1);CHKERRQ(ierr);
  }
  if (numEdges > 5) {
    const PetscInt cone[2] = {7, 8};
    const PetscInt edge    = 14;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
  }
  if (numEdges > 6) {
    const PetscInt cone[2] = {4, 7};
    const PetscInt edge    = 15;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
  }
  if (numEdges > 7) {
    const PetscInt cone[2] = {5, 8};
    const PetscInt edge    = 16;

    ierr = DMPlexSetCone(dm, edge, cone);CHKERRQ(ierr);
  }
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, firstVertex, firstVertex+numVertices);CHKERRQ(ierr);
  for (v = firstVertex; v < firstVertex+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, 2);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject)dm), &coordinates);CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  if (numVertices) {
    coords[0]  = -0.5; coords[1]  = 0.5;
    coords[2]  = -0.2; coords[3]  = 0.0;
    coords[4]  = -0.2; coords[5]  = 1.0;
    coords[6]  =  0.5; coords[7]  = 0.5;
    coords[8]  =  0.2; coords[9]  = 0.0;
    coords[10] =  0.2; coords[11] = 1.0;
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateTensorProduct_2D"
/* Two quadrilaterals

  5----10-----4----14-----7
  |           |           |
  |           |           |
  |           |           |
 11     0     9     1     13
  |           |           |
  |           |           |
  |           |           |
  2-----8-----3----12-----6
*/
PetscErrorCode CreateTensorProduct_2D(MPI_Comm comm, PetscInt testNum, DM *dm)
{
  PetscInt       depth = 2;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    PetscInt    numPoints[3]         = {6, 7, 2};
    PetscInt    coneSize[15]         = {4, 4, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2};
    PetscInt    cones[22]            = {8, 9, 10, 11,  12, 13, 14,  9,  2, 3,  3, 4,  4, 5,  5, 2,  3, 6,  6, 7,  7, 4};
    PetscInt    coneOrientations[22] = {0, 0,  0,  0,   0,  0,  0, -2,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0};
    PetscScalar vertexCoords[12]     = {-1.0, -0.5,  0.0, -0.5,  0.0, 0.5,  -1.0, 0.5,  1.0, -0.5,  1.0, 0.5};

    ierr = DMPlexCreateFromDAG(*dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
  } else {
    PetscInt numPoints[3] = {0, 0, 0};

    ierr = DMPlexCreateFromDAG(*dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSimplex_3D"
/* Two tetrahedrons

 cell   5          5______    cell
 0    / | \        |\      \     1
    17  |  18      | 18 13  21
    /8 19 10\     19  \      \
   2-14-|----4     |   4--22--6
    \ 9 | 7 /      |10 /      /
    16  |  15      | 15  12 20
      \ | /        |/      /
        3          3------
*/
PetscErrorCode CreateSimplex_3D(MPI_Comm comm, PetscInt testNum, DM *dm)
{
  DM             idm;
  PetscInt       depth = 3;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[4]         = {5, 9, 7, 2};
      PetscInt    coneSize[23]         = {4, 4, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2};
      PetscInt    cones[47]            = { 7,  8,  9, 10,  10, 11, 12, 13,  14, 15, 16,  17, 18, 14,  16, 19, 17,  15, 18, 19,  20, 21, 19,  15, 22, 20,  18, 21, 22,  2, 4,  4, 3,  3, 2,  2, 5,  5, 4,  3, 5,  3, 6,  6, 5,  4, 6};
      PetscInt    coneOrientations[47] = { 0,  0,  0,  0,  -3,  0,  0,  0,   0,  0,  0,   0,  0, -2,  -2,  0, -2,  -2, -2, -2,   0,  0, -2,  -2,  0, -2,  -2, -2, -2,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0};
      PetscScalar vertexCoords[15]     = {0.0, 0.0, -0.5,  0.0, -0.5, 0.0,  1.0, 0.0, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.5};

      ierr = DMPlexCreateFromDAG(*dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    }
    break;
    case 1:
    {
      PetscInt    numPoints[2]        = {5, 2};
      PetscInt    coneSize[7]         = {4, 4, 0, 0, 0, 0, 0};
      PetscInt    cones[8]            = {4, 3, 5, 2,  5, 3, 4, 6};
      PetscInt    coneOrientations[8] = {0, 0, 0, 0,  0, 0, 0, 0};
      PetscScalar vertexCoords[15]    = {-1.0, 0.0, 0.0,  0.0, -1.0, 0.0,  0.0, 0.0, 1.0,  0.0, 1.0, 0.0,  1.0, 0.0, 0.0};

      depth = 1;
      ierr = DMPlexCreateFromDAG(*dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
      ierr = DMPlexCopyCoordinates(*dm, idm);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject) idm, "in_");CHKERRQ(ierr);
      ierr = DMSetFromOptions(idm);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = idm;
    }
    break;
    default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[4] = {0, 0, 0, 0};

    ierr = DMPlexCreateFromDAG(*dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
    switch (testNum) {
    case 1:
      ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
      ierr = DMPlexCopyCoordinates(*dm, idm);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject) idm, "in_");CHKERRQ(ierr);
      ierr = DMSetFromOptions(idm);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = idm;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSimplexHybrid_3D"
/* Two tetrahedrons separated by a zero-volume cell with 6 vertices

 cell   6 ___33___10______    cell
 0    / | \        |\      \     1
    21  |  23      | 29     27
    /12 24 14\    30  \      \
   3-20-|----5--32-|---9--26--7
    \ 13| 11/      |18 /      /
    19  |  22      | 28     25
      \ | /        |/      /
        4----31----8------
         cell 2
*/
PetscErrorCode CreateSimplexHybrid_3D(MPI_Comm comm, PetscInt testNum, DM *dm)
{
  DM             idm, hdm;
  DMLabel        faultLabel, hybridLabel;
  PetscInt       p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[2]        = {5, 2};
      PetscInt    coneSize[7]         = {4, 4, 0, 0, 0, 0, 0};
      PetscInt    cones[8]            = {4, 3, 5, 2,  5, 3, 4, 6};
      PetscInt    coneOrientations[8] = {0, 0, 0, 0,  0, 0, 0, 0};
      PetscScalar vertexCoords[15]    = {-1.0, 0.0, 0.0,  0.0, -1.0, 0.0,  0.0, 0.0, 1.0,  0.0, 1.0, 0.0,  1.0, 0.0, 0.0};
      PetscInt    faultPoints[3]      = {3, 4, 5};

      ierr = DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for(p = 0; p < 3; ++p) {ierr = DMPlexSetLabelValue(*dm, "fault", faultPoints[p], 1);CHKERRQ(ierr);}
    }
    break;
    case 1:
    {
      /* Tets 0,3,5 and 1,2,4 */
      PetscInt    numPoints[2]         = {9, 6};
      PetscInt    coneSize[15]         = {4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      PetscInt    cones[24]            = { 7, 9,  8, 6,  11,  9, 13, 14,  10, 13, 11, 9,
                                          10, 9, 11, 7,   9, 13, 14, 12,   7, 11,  8, 9};
      PetscInt    coneOrientations[24] = { 0, 0,  0, 0,   0,  0,  0,  0,   0,  0,  0, 0,
                                           0, 0,  0, 0,   0,  0,  0,  0,   0,  0,  0, 0};
      PetscScalar vertexCoords[27]     = {-2.0, -1.0,  0.0,  -2.0,  0.0,  0.0,  -2.0,  0.0,  1.0,
                                           0.0, -1.0,  0.0,   0.0,  0.0,  0.0,   0.0,  0.0,  1.0,
                                           2.0, -1.0,  0.0,   2.0,  0.0,  0.0,   2.0,  0.0,  1.0};
      PetscInt    faultPoints[3]       = {9, 10, 11};

      ierr = DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for(p = 0; p < 3; ++p) {ierr = DMPlexSetLabelValue(*dm, "fault", faultPoints[p], 1);CHKERRQ(ierr);}
    }
    break;
    default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
    ierr = DMPlexCheckSymmetry(*dm);CHKERRQ(ierr);
    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(*dm, idm);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) idm, "in_");CHKERRQ(ierr);
    ierr = DMSetFromOptions(idm);CHKERRQ(ierr);
    ierr = DMPlexCheckSymmetry(idm);CHKERRQ(ierr);
    ierr = DMPlexGetLabel(*dm, "fault", &faultLabel);CHKERRQ(ierr);
    ierr = DMPlexCreateHybridMesh(idm, faultLabel, &hybridLabel, &hdm);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&hybridLabel);CHKERRQ(ierr);
    ierr = DMDestroy(&idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = hdm;
  } else {
    PetscInt numPoints[4] = {0, 0, 0, 0};

    ierr = DMPlexCreateFromDAG(*dm, 1, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(*dm, idm);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) idm, "in_");CHKERRQ(ierr);
    ierr = DMSetFromOptions(idm);CHKERRQ(ierr);
    ierr = DMPlexCreateHybridMesh(idm, NULL, NULL, &hdm);CHKERRQ(ierr);
    ierr = DMDestroy(&idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = hdm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateTensorProduct_3D"
PetscErrorCode CreateTensorProduct_3D(MPI_Comm comm, PetscInt testNum, DM *dm)
{
  DM             idm;
  PetscInt       depth = 3;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    PetscInt    numPoints[2]         = {12, 2};
    PetscInt    coneSize[14]         = {8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    PetscInt    cones[16]            = {2, 3, 4, 5, 6, 7, 8, 9,  5, 4, 10, 11, 7, 12, 13, 8};
    PetscInt    coneOrientations[16] = {0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0, 0,  0,  0, 0};
    PetscScalar vertexCoords[36]     = {-1.0, -0.5, -0.5,  -1.0,  0.5, -0.5,  0.0,  0.5, -0.5,   0.0, -0.5, -0.5,
                                        -1.0, -0.5,  0.5,   0.0, -0.5,  0.5,  0.0,  0.5,  0.5,  -1.0,  0.5,  0.5,
                                         1.0,  0.5, -0.5,   1.0, -0.5, -0.5,  1.0, -0.5,  0.5,   1.0,  0.5,  0.5};

    depth = 1;
    ierr = DMPlexCreateFromDAG(*dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(*dm, idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  } else {
    PetscInt numPoints[4] = {0, 0, 0, 0};

    ierr = DMPlexCreateFromDAG(*dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(*dm, idm);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) idm, "in_");CHKERRQ(ierr);
    ierr = DMSetFromOptions(idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateTensorProductHybrid_3D"
PetscErrorCode CreateTensorProductHybrid_3D(MPI_Comm comm, PetscInt testNum, DM *dm)
{
  DM             idm, hdm;
  DMLabel        faultLabel;
  PetscInt       p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[2]         = {12, 2};
      PetscInt    coneSize[14]         = {8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      PetscInt    cones[16]            = {2, 3, 4, 5, 6, 7, 8, 9,  5, 4, 10, 11, 7, 12, 13, 8};
      PetscInt    coneOrientations[16] = {0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0, 0,  0,  0, 0};
      PetscScalar vertexCoords[36]     = {-1.0, -0.5, -0.5,  -1.0,  0.5, -0.5,  0.0,  0.5, -0.5,   0.0, -0.5, -0.5,
                                          -1.0, -0.5,  0.5,   0.0, -0.5,  0.5,  0.0,  0.5,  0.5,  -1.0,  0.5,  0.5,
                                          1.0,  0.5, -0.5,   1.0, -0.5, -0.5,  1.0, -0.5,  0.5,   1.0,  0.5,  0.5};
      PetscInt    faultPoints[4]       = {2, 3, 5, 6};

      ierr = DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for(p = 0; p < 4; ++p) {ierr = DMPlexSetLabelValue(*dm, "fault", faultPoints[p], 1);CHKERRQ(ierr);}
    }
    break;
    case 1:
    {
      PetscInt    numPoints[2]         = {30, 7};
      PetscInt    coneSize[37]         = {8,8,8,8,8,8,8, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
      PetscInt    cones[56]            = { 8,  7, 20, 21, 13, 12, 23, 24,
                                          14,  9, 10, 15, 13,  8, 21, 24,
                                          15, 10, 11, 16, 24, 21, 22, 25,
                                          30, 21, 28, 29, 35, 24, 33, 34,
                                          24, 35, 30, 21, 25, 36, 31, 22,
                                          27, 28, 21, 20, 32, 33, 24, 23,
                                           8, 14, 13, 24, 19, 18, 17, 26};
      PetscInt    coneOrientations[56] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
      PetscScalar vertexCoords[90]     = {-2.0, -2.0, -2.0,  -2.0, -1.0, -2.0,  -3.0,  0.0, -2.0,  -2.0,  1.0, -2.0,  -2.0,  2.0, -2.0,  -2.0, -2.0,  0.0,
                                          -2.0, -1.0,  0.0,  -3.0,  0.0,  0.0,  -2.0,  1.0,  0.0,  -2.0,  2.0,  0.0,  -2.0, -1.0,  2.0,  -3.0,  0.0,  2.0,
                                          -2.0,  1.0,  2.0,   0.0, -2.0, -2.0,   0.0,  0.0, -2.0,   0.0,  2.0, -2.0,   0.0, -2.0,  0.0,   0.0,  0.0,  0.0,
                                           0.0,  2.0,  0.0,   0.0,  0.0,  2.0,   2.0, -2.0, -2.0,   2.0, -1.0, -2.0,   3.0,  0.0, -2.0,   2.0,  1.0, -2.0,
                                           2.0,  2.0, -2.0,   2.0, -2.0,  0.0,   2.0, -1.0,  0.0,   3.0,  0.0,  0.0,   2.0,  1.0,  0.0,   2.0,  2.0,  0.0};
      PetscInt    faultPoints[6]       = {20, 21, 22, 23, 24, 25};

      ierr = DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for(p = 0; p < 6; ++p) {ierr = DMPlexSetLabelValue(*dm, "fault", faultPoints[p], 1);CHKERRQ(ierr);}
    }
    break;
    default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
    ierr = DMPlexCheckSymmetry(*dm);CHKERRQ(ierr);
    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(*dm, idm);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) idm, "in_");CHKERRQ(ierr);
    ierr = DMSetFromOptions(idm);CHKERRQ(ierr);
    ierr = DMPlexCheckSymmetry(idm);CHKERRQ(ierr);
    ierr = DMPlexGetLabel(*dm, "fault", &faultLabel);CHKERRQ(ierr);
    ierr = DMPlexCreateHybridMesh(idm, faultLabel, NULL, &hdm);CHKERRQ(ierr);
    ierr = DMDestroy(&idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = hdm;
  } else {
    PetscInt numPoints[4] = {0, 0, 0, 0};

    ierr = DMPlexCreateFromDAG(*dm, 1, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(*dm, idm);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) idm, "in_");CHKERRQ(ierr);
    ierr = DMSetFromOptions(idm);CHKERRQ(ierr);
    ierr = DMPlexCreateHybridMesh(idm, NULL, NULL, &hdm);CHKERRQ(ierr);
    ierr = DMDestroy(&idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = hdm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim               = user->dim;
  PetscBool      refinementUniform = user->refinementUniform;
  PetscBool      cellHybrid        = user->cellHybrid;
  PetscBool      cellSimplex       = user->cellSimplex;
  const char     *partitioner      = "chaco";
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(*dm, dim);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    if (cellSimplex) {
      if (cellHybrid) {
        ierr = CreateSimplexHybrid_2D(comm, *dm);CHKERRQ(ierr);
      } else {
        ierr = CreateSimplex_2D(comm, dm);CHKERRQ(ierr);
      }
    } else {
      if (cellHybrid) {
        SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make hybrid meshes for quadrilaterals");
      } else {
        ierr = CreateTensorProduct_2D(comm, user->testNum, dm);CHKERRQ(ierr);
      }
    }
    break;
  case 3:
    if (cellSimplex) {
      if (cellHybrid) {
        ierr = CreateSimplexHybrid_3D(comm, user->testNum, dm);CHKERRQ(ierr);
      } else {
        ierr = CreateSimplex_3D(comm, user->testNum, dm);CHKERRQ(ierr);
      }
    } else {
      if (cellHybrid) {
        ierr = CreateTensorProductHybrid_3D(comm, user->testNum, dm);CHKERRQ(ierr);
      } else {
        ierr = CreateTensorProduct_3D(comm, user->testNum, dm);CHKERRQ(ierr);
      }
    }
    break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make hybrid meshes for dimension %d", dim);
  }
  {
    DM refinedMesh     = NULL;
    DM distributedMesh = NULL;

    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, partitioner, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
    if (refinementUniform) {
      ierr = PetscObjectSetOptionsPrefix((PetscObject) *dm, "orig_");CHKERRQ(ierr);
      ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
      ierr = DMPlexCheckSymmetry(*dm);CHKERRQ(ierr);
      ierr = DMPlexSetRefinementUniform(*dm, refinementUniform);CHKERRQ(ierr);
      ierr = DMRefine(*dm, comm, &refinedMesh);CHKERRQ(ierr);
      if (refinedMesh) {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm  = refinedMesh;
      }
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Hybrid Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexEqualReordered"
/*@C
  DMPlexEqualReordered - Determine if two DMs have the same topology, perhaps with a renumbering of the points

  Not Collective

  Input Parameters:
+ dmA - A DMPlex object
- dmB - A DMPlex object

  Output Parameters:
. equal - PETSC_TRUE if the topologies are identical

  Level: intermediate

  Notes:
  I know that this is graph isomorphism, so of course this is only a partial solution.

.keywords: mesh
.seealso: DMPlexGetCone(), DMPlexEqual()
@*/
PetscErrorCode DMPlexEqualReordered(DM dmA, DM dmB, PetscBool *equal)
{
  PetscInt      *perm;
  PetscInt       depth, depthB, pStart, pEnd, pStartB, pEndB, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *equal = PETSC_FALSE;
  ierr = DMPlexGetDepth(dmA, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dmB, &depthB);CHKERRQ(ierr);
  if (depth != depthB) PetscFunctionReturn(0);
  ierr = DMPlexGetChart(dmA, &pStart,  &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dmB, &pStartB, &pEndB);CHKERRQ(ierr);
  if ((pStart != pStartB) || (pEnd != pEndB)) PetscFunctionReturn(0);
  ierr = PetscMalloc((pEnd - pStart) * sizeof(PetscInt), &perm);CHKERRQ(ierr);
  perm -= pStart;
  for (p = pStart; p < pEnd; ++p) perm[p] = -1;
  for (p = pStart; p < pEnd; ++p) {
    const PetscInt *cone, *coneB, *ornt, *orntB, *support, *supportB;
    PetscInt        coneSize, coneSizeB, c, supportSize, supportSizeB, s;

    perm[p] = p;
    ierr = DMPlexGetConeSize(dmA, p, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dmA, p, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dmA, p, &ornt);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dmB, perm[p], &coneSizeB);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dmB, perm[p], &coneB);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dmB, perm[p], &orntB);CHKERRQ(ierr);
    if (coneSize != coneSizeB) {ierr = PetscPrintf(PETSC_COMM_SELF, "Invalid cone size %d != %d for point %d (%d)", coneSize, coneSizeB, p, perm[p]);CHKERRQ(ierr); goto end;}
    for (c = 0; c < coneSize; ++c) {
      if (perm[coneB[c]] < 0) perm[coneB[c]] = cone[c];
      if (cone[c] != perm[coneB[c]]) {ierr = PetscPrintf(PETSC_COMM_SELF, "Invalid cone %d point %d != %d (%d) for point %d (%d)", c, cone[c], coneB[c], perm[coneB[c]], p, perm[p]);CHKERRQ(ierr); goto end;}
      if (ornt[c] != orntB[c])       {ierr = PetscPrintf(PETSC_COMM_SELF, "Invalid cone %d orientation %d != %d for point %d (%d)", c, ornt[c], orntB[c], p, perm[p]);CHKERRQ(ierr); goto end;}
    }
    ierr = DMPlexGetSupportSize(dmA, p, &supportSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dmA, p, &support);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dmB, perm[p], &supportSizeB);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dmB, perm[p], &supportB);CHKERRQ(ierr);
    if (supportSize != supportSizeB) {ierr = PetscPrintf(PETSC_COMM_SELF, "Invalid support size %d != %d for point %d (%d)", supportSize, supportSizeB, p, perm[p]);CHKERRQ(ierr); goto end;}
    for (s = 0; s < supportSize; ++s) {
      if (perm[supportB[s]] < 0) perm[supportB[s]] = support[s];
      if (support[s] != perm[supportB[s]]) goto end;
    }
  }
  *equal = PETSC_TRUE;
  end:
  perm += pStart;
  ierr = PetscFree(perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckOrientation"
PetscErrorCode CheckOrientation(DM dm)
{
  DM             udm, idm;
  PetscBool      equal;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(udm);CHKERRQ(ierr);
  ierr = DMPlexInterpolate(udm, &idm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(idm);CHKERRQ(ierr);
  ierr = DMDestroy(&udm);CHKERRQ(ierr);
  ierr = DMPlexEqualReordered(dm, idm, &equal);CHKERRQ(ierr);
  ierr = DMDestroy(&idm);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid orientation in refined mesh");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = DMPlexCheckSymmetry(dm);CHKERRQ(ierr);
  ierr = DMPlexCheckSkeleton(dm, user.cellSimplex);CHKERRQ(ierr);
#if 0
  ierr = CheckOrientation(dm);CHKERRQ(ierr);
#endif
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
