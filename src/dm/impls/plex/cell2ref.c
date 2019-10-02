# include <petscdmplex.h>
# include <petscviewer.h>
# include <petsc/private/dmpleximpl.h>

# define ANSI_RED "\033[1;31m"
# define ANSI_GREEN "\033[1;32m"
# define ANSI_RESET "\033[0m"

PetscErrorCode DMPlexComputeCellOrthogonalQuality(DM dm, Vec *OrthogonalQuality)
{
  MPI_Comm              comm;
  PetscObject    	cellgeomobj, facegeomobj;
  PetscErrorCode        ierr;
  IS			centIS, fcentIS, fnormIS, subCellIS, subFaceIS;
  Vec                   cellGeom, faceGeom, subCell, subFace, subCellCent, subFaceCent, subFaceNormal;
  PetscInt		lo, celliter, faceiter, i, j, cellHeight, cStart, cEnd, fStart, numFaces;
  PetscInt		*ltogdx, *cdx, *fdx, *centdx, *fcentdx, *fnormdx;
  size_t		subCellVecSize = 4, subFaceVecSize = 12, centVecSize = 3, normalVecSize = 3;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);

  ierr = VecCreate(comm, OrthogonalQuality);CHKERRQ(ierr);
  ierr = VecSetType(*OrthogonalQuality, VECSTANDARD);CHKERRQ(ierr);
  ierr = VecSetSizes(*OrthogonalQuality, cEnd-cStart, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetUp(*OrthogonalQuality);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(*OrthogonalQuality, &lo, NULL);CHKERRQ(ierr);

  ierr = PetscMalloc1(cEnd-cStart, &ltogdx);CHKERRQ(ierr);
  ierr = PetscMalloc1(subFaceVecSize, &fdx);CHKERRQ(ierr);
  ierr = PetscMalloc1(subCellVecSize, &cdx);CHKERRQ(ierr);
  ierr = PetscMalloc1(centVecSize, &centdx);CHKERRQ(ierr);
  ierr = PetscMalloc1(centVecSize, &fcentdx);CHKERRQ(ierr);
  ierr = PetscMalloc1(normalVecSize, &fnormdx);CHKERRQ(ierr);

  for (i = 0; i < cEnd-cStart; i++) {
    ltogdx[i] = lo+i;
  }
  ierr = ISLocalToGlobalMappingCreate(comm, 1, cEnd-cStart, ltogdx, PETSC_COPY_VALUES, &ltog);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(*OrthogonalQuality, ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);

  ierr = PetscObjectQuery((PetscObject) dm, "DMPlex_cellgeom_fvm", &cellgeomobj);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "DMPlex_facegeom_fvm", &facegeomobj);CHKERRQ(ierr);
  if ((!cellgeomobj) || (!facegeomobj)) {
    ierr = DMPlexComputeGeometryFVM(dm, &cellGeom, &faceGeom);CHKERRQ(ierr);
  } else {
    cellGeom = (Vec) cellgeomobj;
    faceGeom = (Vec) facegeomobj;
  }

  centdx[0]  = 0; centdx[1]  = 1; centdx[2]  = 2;
  fcentdx[0] = 3; fcentdx[1] = 4; fcentdx[2] = 5;
  fnormdx[0] = 0; fnormdx[1] = 1; fnormdx[2] = 2;
  ierr = ISCreateGeneral(PETSC_COMM_SELF, centVecSize, centdx, PETSC_COPY_VALUES, &centIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, centVecSize, fcentdx, PETSC_COPY_VALUES, &fcentIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, normalVecSize, fnormdx, PETSC_COPY_VALUES, &fnormIS);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight+1, &fStart, NULL);CHKERRQ(ierr);

  for (celliter = cStart; celliter < cEnd; celliter++) {
    PetscScalar		OrthQualPerFace, OrthQualPerCell = 2.0, Anorm, Fnorm, DotProd;
    PetscInt		*coneCopy;
    const PetscInt	*cone;

    ierr = DMPlexGetConeSize(dm, celliter, &numFaces);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, celliter, &cone);CHKERRQ(ierr);
    ierr = PetscMalloc1(numFaces, &coneCopy);CHKERRQ(ierr);
    for (i = 0; i < numFaces; i++) {
      coneCopy[i] = cone[i] - fStart;
    }

    for (j = 0; j < subCellVecSize; j++) {
      cdx[j] = (subCellVecSize*celliter)+j;
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF, subCellVecSize, cdx, PETSC_COPY_VALUES, &subCellIS);CHKERRQ(ierr);

    ierr = VecGetSubVector(cellGeom, subCellIS, &subCell);CHKERRQ(ierr);
    ierr = VecGetSubVector(subCell, centIS, &subCellCent);CHKERRQ(ierr);
    for (faceiter = 0; faceiter < numFaces; faceiter++) {
      Vec		cent2face;
      PetscScalar	tempCalc;

      for (j = 0; j < subFaceVecSize; j++) {
        fdx[j] = (subFaceVecSize*(coneCopy[faceiter]))+j;
      }

      ierr = ISCreateGeneral(PETSC_COMM_SELF, subFaceVecSize, fdx, PETSC_COPY_VALUES, &subFaceIS);CHKERRQ(ierr);

      ierr = VecGetSubVector(faceGeom, subFaceIS, &subFace);CHKERRQ(ierr);
      ierr = VecGetSubVector(subFace, fcentIS, &subFaceCent);CHKERRQ(ierr);
      ierr = VecGetSubVector(subFace, fnormIS, &subFaceNormal);CHKERRQ(ierr);
      ierr = VecDuplicate(subFaceCent, &cent2face);CHKERRQ(ierr);

      ierr = VecWAXPY(cent2face, -1.0, subCellCent, subFaceCent);CHKERRQ(ierr);
      ierr = VecDot(cent2face, subFaceNormal, &DotProd);CHKERRQ(ierr);
      ierr = VecNorm(subFaceNormal, NORM_2, &Anorm);CHKERRQ(ierr);
      ierr = VecNorm(cent2face, NORM_2, &Fnorm);CHKERRQ(ierr);
      tempCalc = DotProd/(Anorm*Fnorm);
      tempCalc = PetscAbs(tempCalc);
      OrthQualPerCell = PetscMin(tempCalc, OrthQualPerCell);

      ierr = VecDestroy(&cent2face);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(subFace, fnormIS, &subFaceNormal);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(subFace, fcentIS, &subFaceCent);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(faceGeom, subFaceIS, &subFace);CHKERRQ(ierr);
      ierr = ISDestroy(&subFaceIS);CHKERRQ(ierr);
    }
    ierr = VecSetValueLocal(*OrthogonalQuality, celliter-cStart, OrthQualPerCell, INSERT_VALUES);CHKERRQ(ierr);
    ierr = PetscFree(coneCopy);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(subCell, centIS, &subCellCent);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(cellGeom, subCellIS, &subCell);CHKERRQ(ierr);
    ierr = ISDestroy(&subCellIS);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(*OrthogonalQuality);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*OrthogonalQuality);CHKERRQ(ierr);
  ierr = VecDestroy(&cellGeom);CHKERRQ(ierr);
  ierr = VecDestroy(&faceGeom);CHKERRQ(ierr);
  ierr = ISDestroy(&fnormIS);CHKERRQ(ierr);
  ierr = ISDestroy(&fcentIS);CHKERRQ(ierr);
  ierr = ISDestroy(&centIS);CHKERRQ(ierr);
  ierr = PetscFree(centdx);CHKERRQ(ierr);
  ierr = PetscFree(fcentdx);CHKERRQ(ierr);
  ierr = PetscFree(fnormdx);CHKERRQ(ierr);
  ierr = PetscFree(ltogdx);CHKERRQ(ierr);
  ierr = PetscFree(fdx);CHKERRQ(ierr);
  ierr = PetscFree(cdx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCreatePointNumField(DM dm, Vec *PointNumbering)
{
  PetscErrorCode	ierr;
  PetscFE        	fe;
  PetscScalar    	*vArray;
  PetscInt       	dim, vStart, vEnd, v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, PETSC_TRUE, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "PointNumbering");CHKERRQ(ierr);
  ierr = DMSetField(dm, 1, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, PointNumbering);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *PointNumbering, "PNum");CHKERRQ(ierr);
  ierr = VecGetArray(*PointNumbering, &vArray);CHKERRQ(ierr);
  for (v = 0; v < vEnd-vStart; ++v) {
    vArray[v] = v+vStart;
  }
  ierr = VecRestoreArray(*PointNumbering, &vArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCreateCellNumField(DM dm, Vec *CellNumbering)
{
  PetscErrorCode	ierr;
  PetscFE        	fe;
  PetscScalar    	*cArray;
  PetscInt       	dim, cStart, cEnd, c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, PETSC_TRUE, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "CellNumbering");CHKERRQ(ierr);
  ierr = DMSetField(dm, 1, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, CellNumbering);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *CellNumbering, "CNum");CHKERRQ(ierr);
  ierr = VecGetArray(*CellNumbering, &cArray);CHKERRQ(ierr);
  for (c = 0; c < cEnd-cStart; ++c) {
    cArray[c] = c;
  }
  ierr = VecRestoreArray(*CellNumbering, &cArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StretchArray2D(DM dm, PetscScalar lx, PetscScalar ly)
{
  PetscErrorCode          ierr;
  PetscInt                i, nCoords;
  Vec                     coordsLocal;
  PetscScalar             *coordArray;

  PetscFunctionBegin;
  ierr = DMGetCoordinates(dm, &coordsLocal);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordsLocal, &nCoords);CHKERRQ(ierr);
  ierr = VecGetArray(coordsLocal, &coordArray);CHKERRQ(ierr);

  // Order in coordarray is [x1,y1,z1....]
  for (i = 0; i < nCoords; i++) {
    //if ((i < 6) || (i > 11)) {
    if (i % 2) {
      coordArray[i-1] = lx*coordArray[i-1];
      coordArray[i] = ly*coordArray[i];
    }
    // }
  }
  ierr = VecRestoreArray(coordsLocal, &coordArray);CHKERRQ(ierr);
  ierr = DMSetCoordinates(dm, coordsLocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SkewArray2D(DM dm, PetscScalar omega)
{
  PetscErrorCode          ierr;
  PetscInt                i, nCoords;
  Vec                     coordsLocal;
  PetscScalar             *coordArray;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordsLocal, &nCoords);CHKERRQ(ierr);
  ierr = VecGetArray(coordsLocal, &coordArray);CHKERRQ(ierr);

  // Order in coordarray is [x1,y1,z1....]
  for (i = 0; i < nCoords; i++) {
    if (i % 2) {
      coordArray[i] = coordArray[i] + coordArray[i-1]*PetscSinReal(omega);
      coordArray[i-1] = coordArray[i-1]*PetscCosReal(omega);
      // reversing order sice "y" is changed first
    }
  }
  ierr = VecRestoreArray(coordsLocal, &coordArray);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordsLocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode Matvis(const char prefix[], PetscScalar mat[])
{
  PetscErrorCode	ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_WORLD, "%s ->\t[%2.2f, %2.2f, %2.2f]\n\t\t[%2.2f, %2.2f, %2.2f]\n\t\t[%2.2f, %2.2f, %2.2f]\n", prefix, mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AngleBetweenConnectedEdges(DM dm, PetscInt *foundcells, PetscInt numCells, PetscInt vertex, PetscScalar *angles[], PetscInt *startEdge)
{
  PetscErrorCode	ierr;
  MPI_Comm		comm;
  const PetscInt	*edges, *vertsOnEdge;
  PetscInt		i, j, numEdges, numVerts, dim, vStart, vEnd, refVert, compVert;
  PetscScalar		refx, refy, compx, compy, centerx, centery, det, dot, x;
  PetscScalar		*carr, *angles_;
  Vec			coordinates;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "--------------------- ANGLES --------------------");CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,  &dim);CHKERRQ(ierr);
  ierr = DMPlexGetSupport(dm, vertex, &edges);CHKERRQ(ierr);
  ierr = DMPlexGetSupportSize(dm, vertex, &numEdges);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "\nNUMBER OF EDGES: %2d\n", numEdges);CHKERRQ(ierr);
  ierr = PetscCalloc1(numEdges, &angles_);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, edges[0], &vertsOnEdge);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, edges[0], &numVerts);CHKERRQ(ierr);
  for (i = 0; i < numVerts; i++) {
    if (vertsOnEdge[i] != vertex) { refVert = vertsOnEdge[i];}
  }
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &carr);CHKERRQ(ierr);
  centerx = carr[dim*(vertex-vStart)]; centery = carr[dim*(vertex-vStart)+1];
  refx = carr[dim*(refVert-vStart)]-centerx; refy = carr[dim*(refVert-vStart)+1]-centery;
  ierr = PetscPrintf(comm, "REFERENCE VERTEX: %d -> (%2.2f,%2.2f)\n\n", refVert, refx, refy);CHKERRQ(ierr);
  for (i = 1; i < numEdges; i++) {
    ierr = PetscPrintf(comm, "EDGE: %2d\n", edges[i]);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, edges[i], &vertsOnEdge);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, edges[i], &numVerts);CHKERRQ(ierr);
    for (j = 0; j < numVerts; j++) {
      //printf("CURRENT %2d --- COMPARE %2d\n", vertsOnEdge[j], vertex);
      if (vertsOnEdge[j] != vertex) { compVert = vertsOnEdge[j];}
    }
    compx = carr[dim*(compVert-vStart)]-centerx; compy = carr[dim*(compVert-vStart)+1]-centery;
    ierr = PetscPrintf(comm, "Chosen Vertex:\t  %2.d -> (%2.2f,%2.2f)\n", compVert, compx, compy);CHKERRQ(ierr);
    dot = (refx*compx) + (refy*compy);
    det = (refx*compy) - (refy*compx);
    ierr = PetscPrintf(comm, "DOT: %2.2f\nDET: %2.2f\n", dot, det);CHKERRQ(ierr);
    x = PetscAtan2Real(det, dot);
    angles_[i-1] = (x > 0 ? x : (2*PETSC_PI + x)) * 360 / (2*PETSC_PI);
    ierr = PetscPrintf(comm, "COMPUTED ANGLE: %f\n", angles_[i-1]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &carr);CHKERRQ(ierr);
  ierr = PetscSortReal(numEdges, angles_);CHKERRQ(ierr);
  for (i = 0; i < numEdges-1; i++) {
    angles_[i] = angles_[i+1]-angles_[i];
  }
  angles_[numEdges-1] = 360-angles_[numEdges-2];
  ierr = PetscArraycpy(*angles, angles_, numEdges);CHKERRQ(ierr);
  *startEdge = edges[0];
  ierr = PetscFree(angles_);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "-------------------------------------------------\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RemoveDupsArray(const PetscInt unsortarr[], PetscInt noduparr[], PetscInt ntotal, PetscInt n, PetscInt search, PetscInt *loc)
{
  PetscInt	i, j, k = 0;

  PetscFunctionBegin;
  for (i = 0; i < ntotal; i++) {
    PetscInt 	key = unsortarr[i];
    PetscBool	found = PETSC_FALSE;
    for (j = 0; j < n; j++) {
      if (noduparr[j] == key) {
        found = PETSC_TRUE;
      }
    }
    if (!found) {
      noduparr[k] = key;
      if (key == search) { *loc = k;}
      k++;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeR2X2RMapping(DM dm, PetscInt vertex, PetscInt cell, PetscScalar R2Xmat[], PetscScalar X2Rmat[], PetscScalar realC_[], PetscScalar refC_[])
{
  PetscErrorCode	ierr;
  MPI_Comm		comm;
  IS      		singleCellIS, vertsIS, vertsISfake;
  Vec			coords;
  PetscInt		idx[1] = {cell}, *nodupidx;
  PetscInt		dim, i, nverts, ntotal, vStart, vEnd, loc = 0, tempi, tempi2, tempi3;
  const PetscInt	*ptr;
  PetscScalar		*xtilde, *rtilde, *invR, *coordArray;
  PetscScalar		detR2X, detR;
  PetscBool		USE_ROTATION = PETSC_FALSE;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-rot", &USE_ROTATION, NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "USING ROTATION:\t%s%s%s\n", USE_ROTATION ? ANSI_GREEN : ANSI_RED , USE_ROTATION ? "PETSC_TRUE" : "PETSC_FALSE", ANSI_RESET);CHKERRQ(ierr);

  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  dim = dim+1;
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(coords, &coordArray);CHKERRQ(ierr);
  ierr = PetscMalloc1(dim*dim, &xtilde);CHKERRQ(ierr);
  ierr = PetscMalloc1(dim*dim, &rtilde);CHKERRQ(ierr);
  rtilde[0] = 0.0; rtilde[1] = 0.0; rtilde[2] = 1.0;
  rtilde[3] = 0.0; rtilde[4] = 1.0; rtilde[5] = 1.0;
  rtilde[6] = 1.0; rtilde[7] = 1.0; rtilde[8] = 1.0;
  xtilde[6] = 1.0; xtilde[7] = 1.0; xtilde[8] = 1.0;

  ierr = ISCreateGeneral(comm, 1, idx, PETSC_COPY_VALUES, &singleCellIS);CHKERRQ(ierr);
  ierr = DMPlexGetConeRecursiveVertices(dm, singleCellIS, &vertsIS);CHKERRQ(ierr);
  ierr = ISDuplicate(vertsIS, &vertsISfake);CHKERRQ(ierr);
  ierr = ISSortRemoveDups(vertsISfake);CHKERRQ(ierr);
  ierr = ISGetSize(vertsISfake, &nverts);CHKERRQ(ierr);
  ierr = ISGetSize(vertsIS, &ntotal);CHKERRQ(ierr);
  ierr = ISDestroy(&vertsISfake);CHKERRQ(ierr);
  ierr = PetscCalloc1(nverts, &nodupidx);CHKERRQ(ierr);
  ierr = ISGetIndices(vertsIS, &ptr);CHKERRQ(ierr);
  ierr = RemoveDupsArray(ptr, nodupidx, ntotal, nverts, vertex, &loc);CHKERRQ(ierr);
  ierr = ISRestoreIndices(vertsIS, &ptr);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "LOC: %d\n", loc);CHKERRQ(ierr);
  PetscIntView(nverts, nodupidx, 0);
  for (i = nverts-1; i > 0; i--) {
    PetscScalar	xval, yval, detX;
    PetscBool	SUCESS = PETSC_FALSE;

    tempi = (loc+i+1)%nverts;
    if (tempi-1 < 0) 	{ tempi2 = nverts-1;} else { tempi2 = tempi-1;}
    if (tempi2-1 < 0) 	{ tempi3 = nverts-1;} else { tempi3 = tempi2-1;}
    xval = coordArray[(dim-1)*(nodupidx[tempi]-vStart)];
    yval = coordArray[(dim-1)*(nodupidx[tempi]-vStart)+1];

    ierr = PetscPrintf(comm, "CURRENT %d\t -> [%.1f %.1f]\nNEXT \t%d\t -> [%.1f %.1f]\nNEXT \t%d\t -> [%.1f %.1f]\n", nodupidx[tempi], xval, yval, nodupidx[tempi2], coordArray[(dim-1)*(nodupidx[tempi2]-vStart)], coordArray[(dim-1)*(nodupidx[tempi2]-vStart)+1], nodupidx[tempi3], coordArray[(dim-1)*(nodupidx[tempi3]-vStart)], coordArray[(dim-1)*(nodupidx[tempi3]-vStart)+1]);CHKERRQ(ierr);

    xtilde[0] = coordArray[(dim-1)*(nodupidx[tempi]-vStart)];
    xtilde[1] = coordArray[(dim-1)*(nodupidx[tempi2]-vStart)];
    xtilde[2] = coordArray[(dim-1)*(nodupidx[tempi3]-vStart)];
    xtilde[3] = coordArray[(dim-1)*(nodupidx[tempi]-vStart)+1];
    xtilde[4] = coordArray[(dim-1)*(nodupidx[tempi2]-vStart)+1];
    xtilde[5] = coordArray[(dim-1)*(nodupidx[tempi3]-vStart)+1];
    ierr = PetscPrintf(comm, "But wait! Theres more! Check DETERMINANT\n");
    DMPlex_Det3D_Internal(&detX, xtilde);
    ierr = PetscPrintf(comm, "%sDETX%s:\t\t%f\n", (PetscAbs(detX) > 0) ? ANSI_GREEN : ANSI_RED, ANSI_RESET, PetscAbs(detX));CHKERRQ(ierr);
    if (PetscAbs(detX) > 0) {
      ierr = PetscPrintf(comm, "USING:\t\t%d %d %d\n\n", nodupidx[tempi], nodupidx[tempi2], nodupidx[tempi3]);CHKERRQ(ierr);
      SUCESS = PETSC_TRUE;
    } else {
      ierr = PetscPrintf(comm, "%sZERO DETERMINANT%s: %d %d %d -> %.1f\n", ANSI_RED, ANSI_RESET, nodupidx[tempi], nodupidx[tempi2], nodupidx[tempi3], detX);CHKERRQ(ierr);
      i--;
    }
    if (SUCESS) { i = 0;} else {
      ierr = PetscPrintf(comm, "%sNO SUITABLE TRANSFORM FOR CELL%s: %d\n", ANSI_RED, ANSI_RESET, cell);CHKERRQ(ierr);
      return (0);
    }
  }
  ierr = PetscCalloc1(dim*dim, &invR);CHKERRQ(ierr);
  DMPlex_Det3D_Internal(&detR, rtilde);
  DMPlex_Invert3D_Internal(invR, rtilde, detR);
  DMPlex_MatMult3D_Internal(xtilde, dim, dim, invR, R2Xmat);
  ierr = Matvis("XTmat", xtilde);CHKERRQ(ierr);
  ierr = Matvis("RTmat", rtilde);CHKERRQ(ierr);
  DMPlex_Det3D_Internal(&detR2X, R2Xmat);
  DMPlex_Invert3D_Internal(X2Rmat, R2Xmat, detR2X);

  ierr = PetscPrintf(comm, "\n");CHKERRQ(ierr);
  for (i = 0; i < nverts; i++) {
    PetscScalar x, y;
    PetscScalar	*realC, *refC;

    ierr = PetscCalloc1(dim, &refC);CHKERRQ(ierr);
    ierr = PetscCalloc1(dim,&realC);CHKERRQ(ierr);
    x = coordArray[(dim-1)*(nodupidx[i]-vStart)];
    y = coordArray[(dim-1)*(nodupidx[i]-vStart)+1];
    realC[0] = x; realC[1] = y; realC[2] = 1.0;

    DMPlex_Mult3D_Internal(X2Rmat, 1, realC, refC);
    if (nodupidx[i] == vertex) { ierr = PetscPrintf(comm, "++++++++++++++++++++++++++++++++++++++++++++++++\n");CHKERRQ(ierr);}
    ierr = PetscPrintf(comm, "FOR CELL %3d, VERTEX %3d REALC: (%.3f, %.3f) -> REFC: (%.3f, %.3f)\n", cell, nodupidx[i], realC[0], realC[1], refC[0], refC[1]);CHKERRQ(ierr);

    if ((nodupidx[i] == vertex) && USE_ROTATION) {
      PetscScalar	xc = 0.5, yc = 0.5, theta;
      PetscScalar	*rotMat, *X2Rtemp;
      PetscInt		k;

      ierr = PetscCalloc1(dim*dim, &X2Rtemp);CHKERRQ(ierr);
      for (k = 0; k < dim*dim; k++) {
        X2Rtemp[k] = X2Rmat[k];
      }
      ierr = PetscCalloc1(dim*dim, &rotMat);CHKERRQ(ierr);
      rotMat[0] = 1; rotMat[4] = 1; rotMat[8] = 1;

      if ((PetscAbs(refC[0]) > 0.1) || (PetscAbs(refC[1]) > 0.1)) {
        ierr = PetscPrintf(comm, "%f %f\n", refC[0], refC[1]);CHKERRQ(ierr);
        if (refC[0] == refC[1]) { theta = PETSC_PI;} else { theta = refC[1] > refC[0] ? PETSC_PI/2 : -1.0*PETSC_PI/2;}
        rotMat[0] = PetscCosReal(theta); rotMat[1] = -1.0*PetscSinReal(theta);
        rotMat[2] = (-xc*PetscCosReal(theta)) + (yc*PetscSinReal(theta)) + xc;
        rotMat[3] = PetscSinReal(theta); rotMat[4] = PetscCosReal(theta);
        rotMat[5] = (-xc*PetscSinReal(theta)) - (yc*PetscCosReal(theta)) + yc;
        DMPlex_MatMult3D_Internal(rotMat, dim, dim, X2Rmat, X2Rtemp);
        for (k = 0; k < dim*dim; k++) {
          X2Rmat[k] = X2Rtemp[k];
        }
        ierr = Matvis("X2R + ROT", X2Rmat);CHKERRQ(ierr);
        DMPlex_Mult3D_Internal(X2Rmat, 1, realC, refC);
        ierr = PetscPrintf(comm, "%f, %f, %f\n", theta, refC[0], refC[1]);CHKERRQ(ierr);
        i = -1;
      }
      ierr = PetscFree(rotMat);CHKERRQ(ierr);
      ierr = PetscFree(X2Rtemp);CHKERRQ(ierr);
    }
    if (nodupidx[i] == vertex) { ierr = PetscPrintf(comm, "++++++++++++++++++++++++++++++++++++++++++++++++\n");CHKERRQ(ierr);}

    realC_[(dim-1)*i] = realC[0];
    realC_[((dim-1)*i)+1] = realC[1];
    refC_[(dim-1)*i] = refC[0];
    refC_[((dim-1)*i)+1] = refC[1];
    ierr = PetscFree(realC);CHKERRQ(ierr);
    ierr = PetscFree(refC);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"\n");CHKERRQ(ierr);
  DMPlex_Det3D_Internal(&detR2X, R2Xmat);
  DMPlex_Invert3D_Internal(X2Rmat, R2Xmat, detR2X);
  ierr = Matvis("R2Xmat", R2Xmat);CHKERRQ(ierr);
  ierr = Matvis("X2Rmat", X2Rmat);CHKERRQ(ierr);
  ierr = VecRestoreArray(coords, &coordArray);CHKERRQ(ierr);
  ierr = ISDestroy(&vertsIS);CHKERRQ(ierr);
  ierr = ISDestroy(&singleCellIS);CHKERRQ(ierr);
  ierr = PetscFree(invR);CHKERRQ(ierr);
  ierr = PetscFree(nodupidx);CHKERRQ(ierr);
  ierr = PetscFree(xtilde);CHKERRQ(ierr);
  ierr = PetscFree(rtilde);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm              comm;
  PetscErrorCode        ierr;
  PetscViewer		vtkviewer, textviewer, hdf5viewer, genviewer;
  DM                    dm, dmDist;
  IS                    bcPointsIS, globalCellIS, vertexIS;
  Vec			coords, OrthQual, PointNum, CellNum;
  PetscSection          section;
  PetscInt              overlap = 0, i, dim = 2, numFields = 3, numBC = 1, size, vsize, cStart, cEnd, conesize;
  PetscInt		faces[dim], *bcField, *numComp, *numDOF;
  const PetscInt	*ptr, *vptr;
  PetscScalar		*coordArray, *angles;
  PetscBool             simplex = PETSC_FALSE, dmInterped = PETSC_TRUE, fileflag = PETSC_FALSE;
  char			filename[PETSC_MAX_PATH_LEN];

  ierr = PetscInitialize(&argc, &argv,(char *) 0, NULL);if(ierr){ return ierr;}
  comm = PETSC_COMM_WORLD;
  ierr = PetscViewerVTKOpen(comm, "mesh.vtk", FILE_MODE_WRITE, &vtkviewer);CHKERRQ(ierr);

  ierr = PetscViewerCreate(comm, &genviewer);CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(genviewer);CHKERRQ(ierr);
  ierr = PetscViewerSetUp(genviewer);CHKERRQ(ierr);

  ierr = PetscViewerCreate(comm, &textviewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(textviewer, PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(textviewer, FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(textviewer, "Angles.txt");CHKERRQ(ierr);
  ierr = PetscViewerSetUp(textviewer);CHKERRQ(ierr);

  ierr = PetscOptionsGetString(NULL, NULL, "-f", filename, PETSC_MAX_PATH_LEN, &fileflag); CHKERRQ(ierr);

  if (!fileflag) {
    for (i = 0; i < dim; i++) {
      faces[i] = 2;
    }
    ierr = DMPlexCreateBoxMesh(comm, dim, simplex, faces, NULL, NULL, NULL, dmInterped, &dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, filename, dmInterped, &dm);CHKERRQ(ierr);
  }
  ierr = DMPlexDistribute(dm, overlap, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist) {
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm = dmDist;
  }
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

  ierr = PetscCalloc1(numBC, &bcField);CHKERRQ(ierr);
  ierr = PetscCalloc1(numFields*(dim+1), &numDOF);CHKERRQ(ierr);
  ierr = PetscCalloc1(numFields, &numComp);CHKERRQ(ierr);
  for (i = 0; i < numFields; i++){numComp[i] = 1;}
  numDOF[0] = 1;
  ierr = DMGetStratumIS(dm, "depth", dim, &bcPointsIS);CHKERRQ(ierr);
  ierr = DMSetNumFields(dm, numFields);CHKERRQ(ierr);
  ierr = DMPlexCreateSection(dm, NULL, numComp, numDOF, numBC, bcField, NULL, &bcPointsIS, NULL, &section);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, 0, "Default");CHKERRQ(ierr);
  ierr = DMSetSection(dm, section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = ISDestroy(&bcPointsIS);CHKERRQ(ierr);
  ierr = PetscFree(bcField);CHKERRQ(ierr);
  ierr = PetscFree(numDOF);CHKERRQ(ierr);
  ierr = PetscFree(numComp);CHKERRQ(ierr);

  ierr = DMPlexCreatePointNumField(dm, &PointNum);CHKERRQ(ierr);
  ierr = PetscViewerVTKAddField(vtkviewer, (PetscObject) dm, &DMPlexVTKWriteAll, PETSC_VTK_POINT_FIELD, PETSC_TRUE, (PetscObject) PointNum);CHKERRQ(ierr);

  ierr = DMPlexCreateCellNumField(dm, &CellNum);CHKERRQ(ierr);
  ierr = PetscViewerVTKAddField(vtkviewer, (PetscObject) dm, &DMPlexVTKWriteAll, PETSC_VTK_CELL_FIELD, PETSC_TRUE, (PetscObject) CellNum);CHKERRQ(ierr);
  ierr = DMView(dm, vtkviewer);CHKERRQ(ierr);

  //ierr = StretchArray2D(dm, 2.0, 1.0);CHKERRQ(ierr);
  //ierr = SkewArray2D(dm, 45.0);CHKERRQ(ierr);

  ierr = DMPlexGetCellNumbering(dm, &globalCellIS);CHKERRQ(ierr);
  ierr = DMGetStratumIS(dm, "depth", 0, &vertexIS);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(globalCellIS, &ptr);CHKERRQ(ierr);
  ierr = ISGetIndices(vertexIS, &vptr);CHKERRQ(ierr);
  ierr = ISGetSize(globalCellIS, &size);CHKERRQ(ierr);
  ierr = ISGetSize(vertexIS, &vsize);CHKERRQ(ierr);

  ierr = DMPlexGetConeSize(dm, ptr[i], &conesize);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)coords, "Deformed");CHKERRQ(ierr);
  ierr = VecGetArray(coords, &coordArray);CHKERRQ(ierr);
  if (0) {
    for (i = 0; i < vsize; i++) {
      PetscInt	vertex = vptr[i];
      PetscInt	*points, *foundcells;
      PetscInt	numPoints, numEdges, j, actualj, cell, k = 0, sEdge;

      ierr = DMPlexGetTransitiveClosure(dm, vertex, PETSC_FALSE, &numPoints, &points);CHKERRQ(ierr);
      ierr = PetscPrintf(comm, "VERTEX# : %d -> (%.3f , %.3f) ", vertex, coordArray[dim*i], coordArray[dim*i+1]);CHKERRQ(ierr);
      ierr = PetscCalloc1(conesize, &foundcells);CHKERRQ(ierr);
      for (j = 0; j < numPoints; j++) {
        actualj = dim*j;
        cell = points[actualj];
        if (cell < cEnd) {
          foundcells[k] = cell;
          k++;
        }
      }
      ierr = PetscPrintf(comm, "For Vertex %d found %d cells\n", vertex, k);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, vertex, &numEdges);CHKERRQ(ierr);
      ierr = PetscCalloc1(numEdges, &angles);CHKERRQ(ierr);
      ierr = AngleBetweenConnectedEdges(dm, foundcells, k, vertex, &angles, &sEdge);CHKERRQ(ierr);
      ierr = PetscPrintf(comm, "#NUMEDGE %d VERT %d StartEdge %d\n", numEdges, vertex, sEdge);CHKERRQ(ierr);
      for (j = 0; j < numEdges; j++) {
        ierr = PetscPrintf(comm, "%f\n", angles[j]);CHKERRQ(ierr);
      }
      ierr = PetscFree(angles);CHKERRQ(ierr);
      for (j = 0; j < k; j++) {
        PetscScalar	*R2Xmat, *X2Rmat, *realCtemp, *refCtemp;

        ierr = PetscCalloc1((dim+1)*(dim+1), &R2Xmat);CHKERRQ(ierr);
        ierr = PetscCalloc1((dim+1)*(dim+1), &X2Rmat);CHKERRQ(ierr);
        ierr = PetscCalloc1((dim+1)*conesize, &realCtemp);CHKERRQ(ierr);
        ierr = PetscCalloc1((dim+1)*conesize, &refCtemp);CHKERRQ(ierr);
        ierr = PetscPrintf(comm, "\ncell: %d, vertex: %d\n", foundcells[j], vertex);CHKERRQ(ierr);
        ierr = ComputeR2X2RMapping(dm, vertex, foundcells[j], R2Xmat, X2Rmat, realCtemp, refCtemp);CHKERRQ(ierr);

        ierr = PetscFree(R2Xmat);CHKERRQ(ierr);
        ierr = PetscFree(X2Rmat);CHKERRQ(ierr);
        ierr = PetscFree(realCtemp);CHKERRQ(ierr);
        ierr = PetscFree(refCtemp);CHKERRQ(ierr);
      }
      ierr = PetscPrintf(comm, "=====================================================\n");CHKERRQ(ierr);
      ierr = PetscFree(foundcells);CHKERRQ(ierr);
      ierr = DMPlexRestoreTransitiveClosure(dm, vertex, PETSC_FALSE, &numPoints, &points);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(coords, &coordArray);CHKERRQ(ierr);
  ierr = ISRestoreIndices(vertexIS, &vptr);CHKERRQ(ierr);
  ierr = ISRestoreIndices(globalCellIS, &ptr);CHKERRQ(ierr);
  ierr = ISDestroy(&vertexIS);CHKERRQ(ierr);

  ierr = DMPlexComputeCellOrthogonalQuality(dm, &OrthQual);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm, &textviewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(textviewer, PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(textviewer, FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(textviewer, "Orthqual.txt");CHKERRQ(ierr);
  ierr = PetscViewerSetUp(textviewer);CHKERRQ(ierr);
  ierr = VecView(OrthQual, textviewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&textviewer);CHKERRQ(ierr);
  ierr = VecDestroy(&OrthQual);CHKERRQ(ierr);

  ierr = PetscViewerCreate(comm, &hdf5viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(hdf5viewer, PETSCVIEWERHDF5);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(hdf5viewer, FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(hdf5viewer, "Mesh.H5");CHKERRQ(ierr);
  ierr = PetscViewerSetUp(hdf5viewer);CHKERRQ(ierr);
  ierr = DMView(dm, hdf5viewer);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&hdf5viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&genviewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vtkviewer);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
