static char help[] = "Tests for refinement of meshes created by hand\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  debug;          /* The debugging level */
  PetscInt  dim;            /* The topological mesh dimension */
  PetscBool cellHybrid;     /* Use a hybrid mesh */
  PetscBool cellSimplex;    /* Use simplices or hexes */
  PetscBool testPartition;  /* Use a fixed partitioning for testing */
  PetscInt  testNum;        /* The particular mesh to test */
  PetscBool uninterpolate;  /* Uninterpolate the mesh at the end */
  PetscBool reinterpolate;  /* Reinterpolate the mesh at the end */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug          = 0;
  options->dim            = 2;
  options->cellHybrid     = PETSC_TRUE;
  options->cellSimplex    = PETSC_TRUE;
  options->testPartition  = PETSC_TRUE;
  options->testNum        = 0;
  options->uninterpolate  = PETSC_FALSE;
  options->reinterpolate  = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBoundedInt("-debug", "The debugging level", "ex4.c", options->debug, &options->debug, NULL,0));
  CHKERRQ(PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex4.c", options->dim, &options->dim, NULL,1,3));
  CHKERRQ(PetscOptionsBool("-cell_hybrid", "Use a hybrid mesh", "ex4.c", options->cellHybrid, &options->cellHybrid, NULL));
  CHKERRQ(PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex4.c", options->cellSimplex, &options->cellSimplex, NULL));
  CHKERRQ(PetscOptionsBool("-test_partition", "Use a fixed partition for testing", "ex4.c", options->testPartition, &options->testPartition, NULL));
  CHKERRQ(PetscOptionsBoundedInt("-test_num", "The particular mesh to test", "ex4.c", options->testNum, &options->testNum, NULL,0));
  CHKERRQ(PetscOptionsBool("-uninterpolate", "Uninterpolate the mesh at the end", "ex4.c", options->uninterpolate, &options->uninterpolate, NULL));
  CHKERRQ(PetscOptionsBool("-reinterpolate", "Reinterpolate the mesh at the end", "ex4.c", options->reinterpolate, &options->reinterpolate, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Two segments

  2-------0-------3-------1-------4

become

  4---0---7---1---5---2---8---3---6

*/
PetscErrorCode CreateSimplex_1D(MPI_Comm comm, DM *dm)
{
  PetscInt       depth = 1;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    PetscInt    numPoints[2]         = {3, 2};
    PetscInt    coneSize[5]          = {2, 2, 0, 0, 0};
    PetscInt    cones[4]             = {2, 3,  3, 4};
    PetscInt    coneOrientations[16] = {0, 0,  0, 0};
    PetscScalar vertexCoords[3]      = {-1.0, 0.0, 1.0};

    CHKERRQ(DMPlexCreateFromDAG(*dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords));
  } else {
    PetscInt numPoints[2] = {0, 0};

    CHKERRQ(DMPlexCreateFromDAG(*dm, depth, numPoints, NULL, NULL, NULL, NULL));
  }
  PetscFunctionReturn(0);
}

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

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    PetscInt    numPoints[3]         = {4, 5, 2};
    PetscInt    coneSize[11]         = {3, 3, 0, 0, 0, 0, 2, 2, 2, 2, 2};
    PetscInt    cones[16]            = {6, 7, 8,  7, 9, 10,  2, 3,  3, 4,  4, 2,  3, 5,  5, 4};
    PetscInt    coneOrientations[16] = {0, 0, 0, -1, 0,  0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0};
    PetscScalar vertexCoords[8]      = {-0.5, 0.0,  0.0, -0.5,  0.0, 0.5,  0.5, 0.0};

    CHKERRQ(DMPlexCreateFromDAG(*dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords));
  } else {
    PetscInt numPoints[3] = {0, 0, 0};

    CHKERRQ(DMPlexCreateFromDAG(*dm, depth, numPoints, NULL, NULL, NULL, NULL));
  }
  PetscFunctionReturn(0);
}

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
PetscErrorCode CreateSimplexHybrid_2D(MPI_Comm comm, PetscInt testNum, DM *dm)
{
  DM             idm, hdm = NULL;
  DMLabel        faultLabel, hybridLabel;
  PetscInt       p;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[2]        = {4, 2};
      PetscInt    coneSize[6]         = {3, 3, 0, 0, 0, 0};
      PetscInt    cones[6]            = {2, 3, 4,  5, 4, 3};
      PetscInt    coneOrientations[6] = {0, 0, 0,  0, 0, 0};
      PetscScalar vertexCoords[8]     = {-1.0, -0.5,  0.0, -0.5,  0.0, 0.5,  1.0, 0.5};
      PetscInt    faultPoints[2]      = {3, 4};

      CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
      for (p = 0; p < 2; ++p) CHKERRQ(DMSetLabelValue(*dm, "fault", faultPoints[p], 1));
    }
    break;
    case 1:
    {
      PetscInt    numPoints[2]         = {5, 4};
      PetscInt    coneSize[9]          = {3, 3, 3, 3, 0, 0, 0, 0, 0};
      PetscInt    cones[12]            = {4, 5, 6,  6, 7, 4,  6, 5, 8,  6, 8, 7};
      PetscInt    coneOrientations[12] = {0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0};
      PetscScalar vertexCoords[10]     = {-1.0, 0.0,  0.0, -1.0,  0.0, 0.0,  0.0, 1.0,  1.0, 0.0};
      PetscInt    faultPoints[3]       = {5, 6, 7};

      CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
      for (p = 0; p < 3; ++p) CHKERRQ(DMSetLabelValue(*dm, "fault", faultPoints[p], 1));
    }
    break;
    default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
    CHKERRQ(DMPlexInterpolate(*dm, &idm));
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) idm, "in_"));
    CHKERRQ(DMPlexDistributeSetDefault(idm, PETSC_FALSE));
    CHKERRQ(DMSetFromOptions(idm));
    CHKERRQ(DMViewFromOptions(idm, NULL, "-dm_view"));
    CHKERRQ(DMGetLabel(*dm, "fault", &faultLabel));
    CHKERRQ(DMPlexCreateHybridMesh(idm, faultLabel, NULL, &hybridLabel, NULL, NULL, &hdm));
    CHKERRQ(DMLabelDestroy(&hybridLabel));
    CHKERRQ(DMDestroy(&idm));
    CHKERRQ(DMDestroy(dm));
    *dm  = hdm;
  } else {
    PetscInt numPoints[2] = {0, 0};

    CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, NULL, NULL, NULL, NULL));
    CHKERRQ(DMPlexInterpolate(*dm, &idm));
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) idm, "in_"));
    CHKERRQ(DMPlexDistributeSetDefault(idm, PETSC_FALSE));
    CHKERRQ(DMSetFromOptions(idm));
    CHKERRQ(DMViewFromOptions(idm, NULL, "-dm_view"));
    CHKERRQ(DMPlexCreateHybridMesh(idm, NULL, NULL, NULL, NULL, NULL, &hdm));
    CHKERRQ(DMDestroy(&idm));
    CHKERRQ(DMDestroy(dm));
    *dm  = hdm;
  }
  PetscFunctionReturn(0);
}

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

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    PetscInt    numPoints[3]         = {6, 7, 2};
    PetscInt    coneSize[15]         = {4, 4, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2};
    PetscInt    cones[22]            = {8, 9, 10, 11,  12, 13, 14,  9,  2, 3,  3, 4,  4, 5,  5, 2,  3, 6,  6, 7,  7, 4};
    PetscInt    coneOrientations[22] = {0, 0,  0,  0,   0,  0,  0, -1,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0};
    PetscScalar vertexCoords[12]     = {-1.0, -0.5,  0.0, -0.5,  0.0, 0.5,  -1.0, 0.5,  1.0, -0.5,  1.0, 0.5};

    CHKERRQ(DMPlexCreateFromDAG(*dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords));
  } else {
    PetscInt numPoints[3] = {0, 0, 0};

    CHKERRQ(DMPlexCreateFromDAG(*dm, depth, numPoints, NULL, NULL, NULL, NULL));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateTensorProductHybrid_2D(MPI_Comm comm, PetscInt testNum, DM *dm)
{
  DM             idm, hdm = NULL;
  DMLabel        faultLabel, hybridLabel;
  PetscInt       p;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    PetscInt    numPoints[2]        = {6, 2};
    PetscInt    coneSize[8]         = {4, 4, 0, 0, 0, 0, 0, 0};
    PetscInt    cones[8]            = {2, 3, 4, 5,  3, 6, 7, 4,};
    PetscInt    coneOrientations[8] = {0, 0, 0, 0,  0, 0, 0, 0};
    PetscScalar vertexCoords[12]    = {-1.0, -0.5,  0.0, -0.5,  0.0, 0.5,  -1.0, 0.5,  1.0, -0.5,  1.0, 0.5};
    PetscInt    faultPoints[2]      = {3, 4};

    CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
    for (p = 0; p < 2; ++p) CHKERRQ(DMSetLabelValue(*dm, "fault", faultPoints[p], 1));
    CHKERRQ(DMPlexInterpolate(*dm, &idm));
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) idm, "in_"));
    CHKERRQ(DMPlexDistributeSetDefault(idm, PETSC_FALSE));
    CHKERRQ(DMSetFromOptions(idm));
    CHKERRQ(DMViewFromOptions(idm, NULL, "-dm_view"));
    CHKERRQ(DMGetLabel(*dm, "fault", &faultLabel));
    CHKERRQ(DMPlexCreateHybridMesh(idm, faultLabel, NULL, &hybridLabel, NULL, NULL, &hdm));
    CHKERRQ(DMLabelDestroy(&hybridLabel));
  } else {
    PetscInt numPoints[3] = {0, 0, 0};

    CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, NULL, NULL, NULL, NULL));
    CHKERRQ(DMPlexInterpolate(*dm, &idm));
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) idm, "in_"));
    CHKERRQ(DMPlexDistributeSetDefault(idm, PETSC_FALSE));
    CHKERRQ(DMSetFromOptions(idm));
    CHKERRQ(DMViewFromOptions(idm, NULL, "-dm_view"));
    CHKERRQ(DMPlexCreateHybridMesh(idm, NULL, NULL, NULL, NULL, NULL, &hdm));
  }
  CHKERRQ(DMDestroy(&idm));
  CHKERRQ(DMDestroy(dm));
  *dm  = hdm;
  PetscFunctionReturn(0);
}

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

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[4]         = {5, 9, 7, 2};
      PetscInt    coneSize[23]         = {4, 4, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2};
      PetscInt    cones[47]            = { 7,  8,  9, 10,  10, 11, 12, 13,  14, 15, 16,  17, 18, 14,  16, 19, 17,  15, 18, 19,  20, 21, 19,  15, 22, 20,  18, 21, 22,  2, 4,  4, 3,  3, 2,  2, 5,  5, 4,  3, 5,  3, 6,  6, 5,  4, 6};
      PetscInt    coneOrientations[47] = { 0,  0,  0,  0,  -2,  0,  0,  0,   0,  0,  0,   0,  0, -1,  -1,  0, -1,  -1, -1, -1,   0,  0, -1,  -1,  0, -1,  -1, -1, -1,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0};
      PetscScalar vertexCoords[15]     = {0.0, 0.0, -0.5,  0.0, -0.5, 0.0,  1.0, 0.0, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.5};

      CHKERRQ(DMPlexCreateFromDAG(*dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords));
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
      CHKERRQ(DMPlexCreateFromDAG(*dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords));
      CHKERRQ(DMPlexInterpolate(*dm, &idm));
      CHKERRQ(DMViewFromOptions(idm, NULL, "-in_dm_view"));
      CHKERRQ(DMDestroy(dm));
      *dm  = idm;
    }
    break;
    case 2:
    {
      PetscInt    numPoints[2]        = {4, 1};
      PetscInt    coneSize[5]         = {4, 0, 0, 0, 0};
      PetscInt    cones[4]            = {2, 3, 4, 1};
      PetscInt    coneOrientations[4] = {0, 0, 0, 0};
      PetscScalar vertexCoords[12]    = {0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0};

      depth = 1;
      CHKERRQ(DMPlexCreateFromDAG(*dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords));
      CHKERRQ(DMPlexInterpolate(*dm, &idm));
      CHKERRQ(DMViewFromOptions(idm, NULL, "-in_dm_view"));
      CHKERRQ(DMDestroy(dm));
      *dm  = idm;
    }
    break;
    default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[4] = {0, 0, 0, 0};

    CHKERRQ(DMPlexCreateFromDAG(*dm, depth, numPoints, NULL, NULL, NULL, NULL));
    switch (testNum) {
    case 1:
      CHKERRQ(DMPlexInterpolate(*dm, &idm));
      CHKERRQ(DMViewFromOptions(idm, NULL, "-in_dm_view"));
      CHKERRQ(DMDestroy(dm));
      *dm  = idm;
      break;
    }
  }
  PetscFunctionReturn(0);
}

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
  DM             idm, hdm = NULL;
  DMLabel        faultLabel, hybridLabel;
  PetscInt       p;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[2]        = {5, 2};
      PetscInt    coneSize[7]         = {4, 4, 0, 0, 0, 0, 0};
      PetscInt    cones[8]            = {4, 3, 5, 2,  5, 3, 4, 6};
      PetscInt    coneOrientations[8] = {0, 0, 0, 0,  0, 0, 0, 0};
      PetscScalar vertexCoords[15]    = {-1.0, 0.0, 0.0,  0.0, -1.0, 0.0,  0.0, 0.0, 1.0,  0.0, 1.0, 0.0,  1.0, 0.0, 0.0};
      PetscInt    faultPoints[3]      = {3, 4, 5};

      CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
      for (p = 0; p < 3; ++p) CHKERRQ(DMSetLabelValue(*dm, "fault", faultPoints[p], 1));
    }
    break;
    case 1:
    {
      /* Tets 0,3,5 and 1,2,4 */
      PetscInt    numPoints[2]         = {9, 6};
      PetscInt    coneSize[15]         = {4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      PetscInt    cones[24]            = { 7,  8,  9, 6,  11, 13,  9, 14,  10, 11, 13, 9,
                                           9, 10, 11, 7,   9, 14, 13, 12,   7,  8, 11, 9};
      PetscInt    coneOrientations[24] = { 0, 0,  0, 0,   0,  0,  0,  0,   0,  0,  0, 0,
                                           0, 0,  0, 0,   0,  0,  0,  0,   0,  0,  0, 0};
      PetscScalar vertexCoords[27]     = {-2.0, -1.0,  0.0,  -2.0,  0.0,  0.0,  -2.0,  0.0,  1.0,
                                           0.0, -1.0,  0.0,   0.0,  0.0,  0.0,   0.0,  0.0,  1.0,
                                           2.0, -1.0,  0.0,   2.0,  0.0,  0.0,   2.0,  0.0,  1.0};
      PetscInt    faultPoints[3]       = {9, 10, 11};

      CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
      for (p = 0; p < 3; ++p) CHKERRQ(DMSetLabelValue(*dm, "fault", faultPoints[p], 1));
    }
    break;
    default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
    CHKERRQ(DMPlexInterpolate(*dm, &idm));
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) idm, "in_"));
    CHKERRQ(DMPlexDistributeSetDefault(idm, PETSC_FALSE));
    CHKERRQ(DMSetFromOptions(idm));
    CHKERRQ(DMViewFromOptions(idm, NULL, "-dm_view"));
    CHKERRQ(DMGetLabel(*dm, "fault", &faultLabel));
    CHKERRQ(DMPlexCreateHybridMesh(idm, faultLabel, NULL, &hybridLabel, NULL, NULL, &hdm));
    CHKERRQ(DMLabelDestroy(&hybridLabel));
    CHKERRQ(DMDestroy(&idm));
    CHKERRQ(DMDestroy(dm));
    *dm  = hdm;
  } else {
    PetscInt numPoints[4] = {0, 0, 0, 0};

    CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, NULL, NULL, NULL, NULL));
    CHKERRQ(DMPlexInterpolate(*dm, &idm));
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) idm, "in_"));
    CHKERRQ(DMPlexDistributeSetDefault(idm, PETSC_FALSE));
    CHKERRQ(DMSetFromOptions(idm));
    CHKERRQ(DMViewFromOptions(idm, NULL, "-dm_view"));
    CHKERRQ(DMPlexCreateHybridMesh(idm, NULL, NULL, NULL, NULL, NULL, &hdm));
    CHKERRQ(DMDestroy(&idm));
    CHKERRQ(DMDestroy(dm));
    *dm  = hdm;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateTensorProduct_3D(MPI_Comm comm, PetscInt testNum, DM *dm)
{
  DM             idm;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
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

      CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
    }
    break;
    case 1:
    {
      PetscInt    numPoints[2]        = {8, 1};
      PetscInt    coneSize[9]         = {8, 0, 0, 0, 0, 0, 0, 0, 0};
      PetscInt    cones[8]            = {1, 2, 3, 4, 5, 6, 7, 8};
      PetscInt    coneOrientations[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      PetscScalar vertexCoords[24]    = {-1.0, -1.0, -1.0,  -1.0,  1.0, -1.0,  1.0,  1.0, -1.0,   1.0, -1.0, -1.0,
                                         -1.0, -1.0,  1.0,   1.0, -1.0,  1.0,  1.0,  1.0,  1.0,  -1.0,  1.0,  1.0};

      CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
    }
    break;
    default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[4] = {0, 0, 0, 0};

    CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, NULL, NULL, NULL, NULL));
  }
  CHKERRQ(DMPlexInterpolate(*dm, &idm));
  CHKERRQ(DMViewFromOptions(idm, NULL, "-in_dm_view"));
  CHKERRQ(DMDestroy(dm));
  *dm  = idm;
  PetscFunctionReturn(0);
}

PetscErrorCode CreateTensorProductHybrid_3D(MPI_Comm comm, PetscInt testNum, DM *dm)
{
  DM             idm, hdm = NULL;
  DMLabel        faultLabel;
  PetscInt       p;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
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

      CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
      for (p = 0; p < 4; ++p) CHKERRQ(DMSetLabelValue(*dm, "fault", faultPoints[p], 1));
    }
    break;
    case 1:
    {
      PetscInt    numPoints[2]         = {30, 7};
      PetscInt    coneSize[37]         = {8,8,8,8,8,8,8, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
      PetscInt    cones[56]            = { 8, 21, 20,  7, 13, 12, 23, 24,
                                          14, 15, 10,  9, 13,  8, 21, 24,
                                          15, 16, 11, 10, 24, 21, 22, 25,
                                          30, 29, 28, 21, 35, 24, 33, 34,
                                          24, 21, 30, 35, 25, 36, 31, 22,
                                          27, 20, 21, 28, 32, 33, 24, 23,
                                          15, 24, 13, 14, 19, 18, 17, 26};
      PetscInt    coneOrientations[56] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
      PetscScalar vertexCoords[90]     = {-2.0, -2.0, -2.0,  -2.0, -1.0, -2.0,  -3.0,  0.0, -2.0,  -2.0,  1.0, -2.0,  -2.0,  2.0, -2.0,  -2.0, -2.0,  0.0,
                                          -2.0, -1.0,  0.0,  -3.0,  0.0,  0.0,  -2.0,  1.0,  0.0,  -2.0,  2.0,  0.0,  -2.0, -1.0,  2.0,  -3.0,  0.0,  2.0,
                                          -2.0,  1.0,  2.0,   0.0, -2.0, -2.0,   0.0,  0.0, -2.0,   0.0,  2.0, -2.0,   0.0, -2.0,  0.0,   0.0,  0.0,  0.0,
                                           0.0,  2.0,  0.0,   0.0,  0.0,  2.0,   2.0, -2.0, -2.0,   2.0, -1.0, -2.0,   3.0,  0.0, -2.0,   2.0,  1.0, -2.0,
                                           2.0,  2.0, -2.0,   2.0, -2.0,  0.0,   2.0, -1.0,  0.0,   3.0,  0.0,  0.0,   2.0,  1.0,  0.0,   2.0,  2.0,  0.0};
      PetscInt    faultPoints[6]       = {20, 21, 22, 23, 24, 25};

      CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords));
      for (p = 0; p < 6; ++p) CHKERRQ(DMSetLabelValue(*dm, "fault", faultPoints[p], 1));
    }
    break;
    default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
    CHKERRQ(DMPlexInterpolate(*dm, &idm));
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) idm, "in_"));
    CHKERRQ(DMPlexDistributeSetDefault(idm, PETSC_FALSE));
    CHKERRQ(DMSetFromOptions(idm));
    CHKERRQ(DMViewFromOptions(idm, NULL, "-dm_view"));
    CHKERRQ(DMGetLabel(*dm, "fault", &faultLabel));
    CHKERRQ(DMPlexCreateHybridMesh(idm, faultLabel, NULL, NULL, NULL, NULL, &hdm));
    CHKERRQ(DMDestroy(&idm));
    CHKERRQ(DMDestroy(dm));
    *dm  = hdm;
  } else {
    PetscInt numPoints[4] = {0, 0, 0, 0};

    CHKERRQ(DMPlexCreateFromDAG(*dm, 1, numPoints, NULL, NULL, NULL, NULL));
    CHKERRQ(DMPlexInterpolate(*dm, &idm));
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) idm, "in_"));
    CHKERRQ(DMPlexDistributeSetDefault(idm, PETSC_FALSE));
    CHKERRQ(DMSetFromOptions(idm));
    CHKERRQ(DMViewFromOptions(idm, NULL, "-dm_view"));
    CHKERRQ(DMPlexCreateHybridMesh(idm, NULL, NULL, NULL, NULL, NULL, &hdm));
    CHKERRQ(DMDestroy(&idm));
    CHKERRQ(DMDestroy(dm));
    *dm  = hdm;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim         = user->dim;
  PetscBool      cellHybrid  = user->cellHybrid;
  PetscBool      cellSimplex = user->cellSimplex;
  PetscMPIInt    rank, size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetDimension(*dm, dim));
  switch (dim) {
  case 1:
    PetscCheck(!cellHybrid,comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make hybrid meshes for dimension %d", dim);
    CHKERRQ(CreateSimplex_1D(comm, dm));
    break;
  case 2:
    if (cellSimplex) {
      if (cellHybrid) {
        CHKERRQ(CreateSimplexHybrid_2D(comm, user->testNum, dm));
      } else {
        CHKERRQ(CreateSimplex_2D(comm, dm));
      }
    } else {
      if (cellHybrid) {
        CHKERRQ(CreateTensorProductHybrid_2D(comm, user->testNum, dm));
      } else {
        CHKERRQ(CreateTensorProduct_2D(comm, user->testNum, dm));
      }
    }
    break;
  case 3:
    if (cellSimplex) {
      if (cellHybrid) {
        CHKERRQ(CreateSimplexHybrid_3D(comm, user->testNum, dm));
      } else {
        CHKERRQ(CreateSimplex_3D(comm, user->testNum, dm));
      }
    } else {
      if (cellHybrid) {
        CHKERRQ(CreateTensorProductHybrid_3D(comm, user->testNum, dm));
      } else {
        CHKERRQ(CreateTensorProduct_3D(comm, user->testNum, dm));
      }
    }
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %d", dim);
  }
  if (user->testPartition && size > 1) {
    PetscPartitioner part;
    PetscInt  *sizes  = NULL;
    PetscInt  *points = NULL;

    if (rank == 0) {
      if (dim == 2 && cellSimplex && !cellHybrid && size == 2) {
        switch (user->testNum) {
        case 0: {
          PetscInt triSizes_p2[2]  = {1, 1};
          PetscInt triPoints_p2[2] = {0, 1};

          CHKERRQ(PetscMalloc2(2, &sizes, 2, &points));
          CHKERRQ(PetscArraycpy(sizes,  triSizes_p2, 2));
          CHKERRQ(PetscArraycpy(points, triPoints_p2, 2));break;}
        default:
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Could not find matching test number %d for triangular mesh on 2 procs", user->testNum);
        }
      } else if (dim == 2 && cellSimplex && cellHybrid && size == 2) {
        switch (user->testNum) {
        case 0: {
          PetscInt triSizes_p2[2]  = {1, 2};
          PetscInt triPoints_p2[3] = {0, 1, 2};

          CHKERRQ(PetscMalloc2(2, &sizes, 3, &points));
          CHKERRQ(PetscArraycpy(sizes,  triSizes_p2, 2));
          CHKERRQ(PetscArraycpy(points, triPoints_p2, 3));break;}
        default:
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Could not find matching test number %d for triangular hybrid mesh on 2 procs", user->testNum);
        }
      } else if (dim == 2 && !cellSimplex && !cellHybrid && size == 2) {
        switch (user->testNum) {
        case 0: {
          PetscInt quadSizes_p2[2]  = {1, 1};
          PetscInt quadPoints_p2[2] = {0, 1};

          CHKERRQ(PetscMalloc2(2, &sizes, 2, &points));
          CHKERRQ(PetscArraycpy(sizes,  quadSizes_p2, 2));
          CHKERRQ(PetscArraycpy(points, quadPoints_p2, 2));break;}
        default:
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Could not find matching test number %d for quadrilateral mesh on 2 procs", user->testNum);
        }
      } else if (dim == 2 && !cellSimplex && cellHybrid && size == 2) {
        switch (user->testNum) {
        case 0: {
          PetscInt quadSizes_p2[2]  = {1, 2};
          PetscInt quadPoints_p2[3] = {0, 1, 2};

          CHKERRQ(PetscMalloc2(2, &sizes, 3, &points));
          CHKERRQ(PetscArraycpy(sizes,  quadSizes_p2, 2));
          CHKERRQ(PetscArraycpy(points, quadPoints_p2, 3));break;}
        default:
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Could not find matching test number %d for quadrilateral hybrid mesh on 2 procs", user->testNum);
        }
      } else if (dim == 3 && cellSimplex && !cellHybrid && size == 2) {
        switch (user->testNum) {
        case 0: {
          PetscInt tetSizes_p2[2]  = {1, 1};
          PetscInt tetPoints_p2[2] = {0, 1};

          CHKERRQ(PetscMalloc2(2, &sizes, 2, &points));
          CHKERRQ(PetscArraycpy(sizes,  tetSizes_p2, 2));
          CHKERRQ(PetscArraycpy(points, tetPoints_p2, 2));break;}
        case 1: {
          PetscInt tetSizes_p2[2]  = {1, 1};
          PetscInt tetPoints_p2[2] = {0, 1};

          CHKERRQ(PetscMalloc2(2, &sizes, 2, &points));
          CHKERRQ(PetscArraycpy(sizes,  tetSizes_p2, 2));
          CHKERRQ(PetscArraycpy(points, tetPoints_p2, 2));break;}
        default:
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Could not find matching test number %d for tetrahedral mesh on 2 procs", user->testNum);
        }
      } else if (dim == 3 && cellSimplex && cellHybrid && size == 2) {
        switch (user->testNum) {
        case 0: {
          PetscInt tetSizes_p2[2]  = {1, 2};
          PetscInt tetPoints_p2[3] = {0, 1, 2};

          CHKERRQ(PetscMalloc2(2, &sizes, 3, &points));
          CHKERRQ(PetscArraycpy(sizes,  tetSizes_p2, 2));
          CHKERRQ(PetscArraycpy(points, tetPoints_p2, 3));break;}
        case 1: {
          PetscInt tetSizes_p2[2]  = {3, 4};
          PetscInt tetPoints_p2[7] = {0, 3, 5, 1, 2, 4, 6};

          CHKERRQ(PetscMalloc2(2, &sizes, 7, &points));
          CHKERRQ(PetscArraycpy(sizes,  tetSizes_p2, 2));
          CHKERRQ(PetscArraycpy(points, tetPoints_p2, 7));break;}
        default:
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Could not find matching test number %d for tetrahedral hybrid mesh on 2 procs", user->testNum);
        }
      } else if (dim == 3 && !cellSimplex && !cellHybrid && size == 2) {
        switch (user->testNum) {
        case 0: {
          PetscInt hexSizes_p2[2]  = {1, 1};
          PetscInt hexPoints_p2[2] = {0, 1};

          CHKERRQ(PetscMalloc2(2, &sizes, 2, &points));
          CHKERRQ(PetscArraycpy(sizes,  hexSizes_p2, 2));
          CHKERRQ(PetscArraycpy(points, hexPoints_p2, 2));break;}
        default:
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Could not find matching test number %d for hexahedral mesh on 2 procs", user->testNum);
        }
      } else if (dim == 3 && !cellSimplex && cellHybrid && size == 2) {
        switch (user->testNum) {
        case 0: {
          PetscInt hexSizes_p2[2]  = {1, 1};
          PetscInt hexPoints_p2[2] = {0, 1};

          CHKERRQ(PetscMalloc2(2, &sizes, 2, &points));
          CHKERRQ(PetscArraycpy(sizes,  hexSizes_p2, 2));
          CHKERRQ(PetscArraycpy(points, hexPoints_p2, 2));break;}
        case 1: {
          PetscInt hexSizes_p2[2]  = {5, 4};
          PetscInt hexPoints_p2[9] = {3, 4, 5, 7, 8, 0, 1, 2, 6};

          CHKERRQ(PetscMalloc2(2, &sizes, 9, &points));
          CHKERRQ(PetscArraycpy(sizes,  hexSizes_p2, 2));
          CHKERRQ(PetscArraycpy(points, hexPoints_p2, 9));break;}
        default:
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Could not find matching test number %d for hexahedral hybrid mesh on 2 procs", user->testNum);
        }
      } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Could not find matching test partition");
    }
    CHKERRQ(DMPlexGetPartitioner(*dm, &part));
    CHKERRQ(PetscPartitionerSetType(part, PETSCPARTITIONERSHELL));
    CHKERRQ(PetscPartitionerShellSetPartition(part, size, sizes, points));
    CHKERRQ(PetscFree2(sizes, points));
  } else {
    PetscPartitioner part;

    CHKERRQ(DMPlexGetPartitioner(*dm,&part));
    CHKERRQ(PetscPartitionerSetFromOptions(part));
  }
  {
    DM pdm = NULL;

    CHKERRQ(DMPlexDistribute(*dm, 0, NULL, &pdm));
    if (pdm) {
      CHKERRQ(DMViewFromOptions(pdm, NULL, "-dm_view"));
      CHKERRQ(DMDestroy(dm));
      *dm  = pdm;
    }
  }
  CHKERRQ(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view_pre"));
  CHKERRQ(DMSetFromOptions(*dm));
  if (user->uninterpolate || user->reinterpolate) {
    DM udm = NULL;

    CHKERRQ(DMPlexUninterpolate(*dm, &udm));
    CHKERRQ(DMPlexCopyCoordinates(*dm, udm));
    CHKERRQ(DMDestroy(dm));
    *dm  = udm;
  }
  if (user->reinterpolate) {
    DM idm = NULL;

    CHKERRQ(DMPlexInterpolate(*dm, &idm));
    CHKERRQ(DMPlexCopyCoordinates(*dm, idm));
    CHKERRQ(DMDestroy(dm));
    *dm  = idm;
  }
  CHKERRQ(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
  CHKERRQ(PetscObjectSetName((PetscObject) *dm, "Hybrid Mesh"));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) *dm, "hyb_"));
  CHKERRQ(DMSetFromOptions(*dm));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  # 1D Simplex 29-31
  testset:
    args: -dim 1 -cell_hybrid 0 -hyb_dm_plex_check_all -dm_plex_check_all
    test:
      suffix: 29
    test:
      suffix: 30
      args: -dm_refine 1
    test:
      suffix: 31
      args: -dm_refine 5

  # 2D Simplex 0-3
  testset:
    args: -dim 2 -cell_hybrid 0 -hyb_dm_plex_check_all -dm_plex_check_all
    test:
      suffix: 0
      args: -hyb_dm_plex_check_faces
    test:
      suffix: 1
      args: -dm_refine 1 -hyb_dm_plex_check_faces
    test:
      suffix: 2
      nsize: 2
      args: -hyb_dm_plex_check_faces
    test:
      suffix: 3
      nsize: 2
      args: -dm_refine 1 -hyb_dm_plex_check_faces
    test:
      suffix: 32
      args: -dm_refine 1 -uninterpolate
    test:
      suffix: 33
      nsize: 2
      args: -dm_refine 1 -uninterpolate
    test:
      suffix: 34
      nsize: 2
      args: -dm_refine 3 -uninterpolate

  # 2D Hybrid Simplex 4-7
  testset:
    args: -dim 2 -hyb_dm_plex_check_all -in_dm_plex_check_all -dm_plex_check_all
    test:
      suffix: 4
    test:
      suffix: 5
      args: -dm_refine 1
    test:
      suffix: 6
      nsize: 2
    test:
      suffix: 7
      nsize: 2
      args: -dm_refine 1
    test:
      suffix: 24
      args: -test_num 1 -dm_refine 1

  # 2D Quad 12-13
  testset:
    args: -dim 2 -cell_simplex 0 -cell_hybrid 0 -hyb_dm_plex_check_all -dm_plex_check_all
    test:
      suffix: 12
      args: -dm_refine 1
    test:
      suffix: 13
      nsize: 2
      args: -dm_refine 1

  # 2D Hybrid Quad 27-28
  testset:
    args: -dim 2 -cell_simplex 0 -hyb_dm_plex_check_all -in_dm_plex_check_all -dm_plex_check_all
    test:
      suffix: 27
      args: -dm_refine 1
    test:
      suffix: 28
      nsize: 2
      args: -dm_refine 1

  # 3D Simplex 8-11
  testset:
    args: -dim 3 -cell_hybrid 0 -hyb_dm_plex_check_all -dm_plex_check_all
    test:
      suffix: 8
      args: -dm_refine 1
    test:
      suffix: 9
      nsize: 2
      args: -dm_refine 1
    test:
      suffix: 10
      args: -test_num 1 -dm_refine 1
    test:
      suffix: 11
      nsize: 2
      args: -test_num 1 -dm_refine 1
    test:
      suffix: 25
      args: -test_num 2 -dm_refine 2

  # 3D Hybrid Simplex 16-19
  testset:
    args: -dim 3 -hyb_dm_plex_check_all -in_dm_plex_check_all -dm_plex_check_all
    test:
      suffix: 16
      args: -dm_refine 1
    test:
      suffix: 17
      nsize: 2
      args: -dm_refine 1
    test:
      suffix: 18
      args: -test_num 1 -dm_refine 1
    test:
      suffix: 19
      nsize: 2
      args: -test_num 1 -dm_refine 1

  # 3D Hex 14-15
  testset:
    args: -dim 3 -cell_simplex 0 -cell_hybrid 0 -hyb_dm_plex_check_all -dm_plex_check_all
    test:
      suffix: 14
      args: -dm_refine 1
    test:
      suffix: 15
      nsize: 2
     args: -dm_refine 1
    test:
      suffix: 26
      args: -test_num 1 -dm_refine 2

  # 3D Hybrid Hex 20-23
  testset:
    args: -dim 3 -cell_simplex 0 -hyb_dm_plex_check_all -in_dm_plex_check_all -dm_plex_check_all
    test:
      suffix: 20
      args: -dm_refine 1
    test:
      suffix: 21
      nsize: 2
      args: -dm_refine 1
    test:
      suffix: 22
      args: -test_num 1 -dm_refine 1
    test:
      suffix: 23
      nsize: 2
      args: -test_num 1 -dm_refine 1

  # Hybrid interpolation
  #   TODO Setup new tests (like -reinterpolate) that interpolate hybrid cells
  testset:
    nsize: 2
    args: -test_partition 0 -petscpartitioner_type simple -dm_view -hyb_dm_plex_check_all -in_dm_plex_check_all -dm_plex_check_all
    test:
      suffix: hybint_2d_0
      args: -dim 2 -dm_refine 2
    test:
      suffix: hybint_2d_1
      args: -dim 2 -dm_refine 2 -test_num 1
    test:
      suffix: hybint_3d_0
      args: -dim 3 -dm_refine 1
    test:
      suffix: hybint_3d_1
      args: -dim 3 -dm_refine 1 -test_num 1

TEST*/
