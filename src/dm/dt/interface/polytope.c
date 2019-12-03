
#include <petscdt.h>
#include <petsc/private/petscimpl.h>
#include <petscbt.h>

typedef struct _n_PetscPolytopeData *PetscPolytopeData;
typedef struct _n_PetscPolytopeName PetscPolytopeName;

typedef struct _n_PetscPolytopeCone
{
  PetscInt index;
  PetscInt orientation;
} PetscPolytopeCone;

typedef struct _n_PetscPolytopeSupp
{
  PetscInt index;
  PetscInt coneNumber;
} PetscPolytopeSupp;

struct _n_PetscPolytopeData
{
  PetscInt           dim, numFacets, numVertices, numRidges;
  PetscPolytope      *facets;
  PetscBool          *facetsInward;
  PetscInt           *vertexOffsets;
  PetscInt           *facetsToVertices;
  PetscInt           *facetsToVerticesSorted;
  PetscInt           *facetsToVerticesOrder;
  PetscInt           *ridgeOffsets;
  PetscPolytopeCone  *facetsToRidges;
  PetscPolytopeSupp  *ridgesToFacets;
  PetscInt            orientStart, orientEnd, maxFacetSymmetry;
  PetscInt           *vertexPerms;
  PetscInt           *vertexPermsToOrients;
  PetscInt           *orientsToVertexOrders;
  PetscPolytopeCone  *orientsToFacetOrders;
  PetscInt           *orientInverses;
  PetscInt           *orientComps;
  PetscInt           *orbitProjectors;
  PetscInt           *facetOrientations;
};

static PetscErrorCode PetscPolytopeDataDestroy(PetscPolytopeData *pdata)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pdata) PetscFunctionReturn(0);
  ierr = PetscFree((*pdata)->facets);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->facetsInward);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->vertexOffsets);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->facetsToVertices);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->facetsToVerticesSorted);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->facetsToVerticesOrder);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->ridgeOffsets);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->facetsToRidges);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->ridgesToFacets);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->vertexPerms);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->vertexPermsToOrients);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->orientsToVertexOrders);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->orientsToFacetOrders);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->orientInverses);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->orientComps);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->orbitProjectors);CHKERRQ(ierr);
  ierr = PetscFree((*pdata)->facetOrientations);CHKERRQ(ierr);
  ierr = PetscFree(*pdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeDataCreate(PetscInt dim, PetscInt numFacets, PetscInt numVertices, const PetscPolytope facets[], const PetscInt vertexOffsets[], const PetscInt facetsToVertices[], PetscBool firstFacetInward, PetscPolytopeData *pData)
{
  PetscPolytopeData pd;
  PetscInt          i, j;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&pd);CHKERRQ(ierr);
  pd->dim         = dim;
  pd->numFacets   = numFacets;
  pd->numVertices = numVertices;
  ierr = PetscMalloc1(numFacets, &(pd->facets));CHKERRQ(ierr);
  ierr = PetscArraycpy(pd->facets, facets, numFacets);CHKERRQ(ierr);
  ierr = PetscMalloc1(numFacets + 1, &(pd->vertexOffsets));CHKERRQ(ierr);
  if (numFacets) {ierr = PetscArraycpy(pd->vertexOffsets, vertexOffsets, numFacets + 1);CHKERRQ(ierr);}
  else pd->vertexOffsets[0] = 0;
  ierr = PetscMalloc1(pd->vertexOffsets[numFacets], &(pd->facetsToVertices));CHKERRQ(ierr);
  ierr = PetscArraycpy(pd->facetsToVertices, facetsToVertices, pd->vertexOffsets[numFacets]);CHKERRQ(ierr);
  ierr = PetscMalloc1(pd->vertexOffsets[numFacets], &(pd->facetsToVerticesSorted));CHKERRQ(ierr);
  ierr = PetscMalloc1(pd->vertexOffsets[numFacets], &(pd->facetsToVerticesOrder));CHKERRQ(ierr);
  for (i = 0; i < numFacets; i++) {
    PetscInt *v = &pd->facetsToVertices[pd->vertexOffsets[i]];
    PetscInt *f = &pd->facetsToVerticesSorted[pd->vertexOffsets[i]];
    PetscInt *o = &pd->facetsToVerticesOrder[pd->vertexOffsets[i]];
    PetscInt  n = pd->vertexOffsets[i+1]-pd->vertexOffsets[i];

    for (j = 0; j < n; j++) {
      f[j] = v[j];
      o[j] = j;
    }
    ierr = PetscSortIntWithArray(n, f, o);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(numFacets, &(pd->facetsInward));CHKERRQ(ierr);
  if (numFacets) pd->facetsInward[0] = firstFacetInward;
  *pData = pd;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeDataCompare(PetscPolytopeData tdata, PetscInt numFacets, PetscInt numVertices, const PetscPolytope facets[], const PetscInt vertexOffsets[], const PetscInt facetsToVertices[], PetscBool firstFacetInward, PetscBool *same)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (numFacets != tdata->numFacets || numVertices != tdata->numVertices || (numFacets > 0 && (firstFacetInward != tdata->facetsInward[0]))) {
    *same = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  else {
    ierr = PetscArraycmp(facets, tdata->facets, numFacets, same);CHKERRQ(ierr);
    if (!*same) PetscFunctionReturn(0);
    if (numFacets) {
      ierr = PetscArraycmp(vertexOffsets, tdata->vertexOffsets, numFacets+1, same);CHKERRQ(ierr);
      if (!*same) PetscFunctionReturn(0);
    }
    ierr = PetscArraycmp(facetsToVertices, tdata->facetsToVertices, tdata->vertexOffsets[numFacets], same);CHKERRQ(ierr);
    if (!*same) PetscFunctionReturn(0);
    *same = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}


static PetscErrorCode PetscPolytopeDataFacetsFromOrientation(PetscPolytopeData data, PetscInt orientation, PetscPolytopeCone facets[])
{
  PetscInt i, o, numFacets;

  PetscFunctionBegin;
  if (orientation < data->orientStart || orientation >= data->orientEnd) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Orientation %D is not in [%D, %D)\n", orientation, data->orientStart, data->orientEnd);
  o = orientation - data->orientStart;
  numFacets = data->numFacets;
  for (i = 0; i < numFacets; i++) facets[i] = data->orientsToFacetOrders[o*numFacets + i];
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeDataOrientationInverse(PetscPolytopeData data, PetscInt orientation, PetscInt *inverse)
{
  PetscFunctionBegin;
  if (orientation < data->orientStart || orientation >= data->orientEnd) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Orientation %D is not in [%D, %D)\n", orientation, data->orientStart, data->orientEnd);
  *inverse = data->orientInverses[orientation - data->orientStart];
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeDataFacetSign(PetscPolytopeData data, PetscInt f, PetscBool *sign)
{
  PetscFunctionBegin;
  if (f < 0 || f >= data->numFacets) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid facet %D (not in [0, %D)", f, data->numFacets);
  *sign = data->facetsInward[f];
  PetscFunctionReturn(0);
}

struct _n_PetscPolytopeName
{
  char          *name;
  PetscPolytope polytope;
};

typedef struct _n_PetscPolytopeSet *PetscPolytopeSet;

struct _n_PetscPolytopeSet
{
  int               numPolytopes;
  int               numPolytopesAlloc;
  PetscPolytopeData *polytopes;
  int               numNames;
  int               numNamesAlloc;
  PetscPolytopeName *names;
};

static PetscPolytopeSet PetscPolytopes = NULL;

static PetscErrorCode PetscPolytopeSetCreate(PetscPolytopeSet *pset)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(pset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetDestroy(PetscPolytopeSet *pset)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pset) PetscFunctionReturn(0);
  PetscValidPointer(pset, 1);
  for (i = 0; i < (*pset)->numPolytopes; i++) {
    ierr = PetscPolytopeDataDestroy(&((*pset)->polytopes[i]));CHKERRQ(ierr);
  }
  ierr = PetscFree((*pset)->polytopes);CHKERRQ(ierr);
  for (i = 0; i < (*pset)->numNames; i++) {
    ierr = PetscFree((*pset)->names[i].name);CHKERRQ(ierr);
  }
  ierr = PetscFree((*pset)->names);CHKERRQ(ierr);
  ierr = PetscFree(*pset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetOrientationCompose(PetscPolytopeSet pset, PetscPolytope polytope, PetscInt a, PetscInt b, PetscInt *aafterb)
{
  PetscPolytopeData p, fData;
  PetscInt f0, o0, f1, o1, oComp;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  if (!a) {
    *aafterb = b;
    PetscFunctionReturn(0);
  }
  if (!b) {
    *aafterb = a;
    PetscFunctionReturn(0);
  }
  if (polytope < 0 || polytope > pset->numPolytopes) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"No polytope with id %D\n",polytope);
  p = pset->polytopes[polytope];
  if (a < p->orientStart || a >= p->orientEnd) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Orientation %D is not in [%D, %D)\n", a, p->orientStart, p->orientEnd);
  if (b < p->orientStart || b >= p->orientEnd) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Orientation %D is not in [%D, %D)\n", b, p->orientStart, p->orientEnd);
  if (p->orientComps) {
    *aafterb = p->orientComps[(a - p->orientStart) * (p->orientEnd - p->orientStart) + (b - p->orientStart)];
    PetscFunctionReturn(0);
  }
  f0 = p->orientsToFacetOrders[p->numFacets * (a - p->orientStart)].index;
  o0 = p->orientsToFacetOrders[p->numFacets * (a - p->orientStart)].orientation;
  f1 = p->orientsToFacetOrders[p->numFacets * (b - p->orientStart) + f0].index;
  o1 = p->orientsToFacetOrders[p->numFacets * (b - p->orientStart) + f0].orientation;
  ierr = PetscPolytopeSetOrientationCompose(pset, p->facets[0], o0, o1, &oComp);CHKERRQ(ierr);
  /* find (oComp, f1) */
  fData = pset->polytopes[p->facets[0]];
  *aafterb = p->facetOrientations[f1*p->maxFacetSymmetry + (oComp-fData->orientStart)];
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetOrientationFromFacet(PetscPolytopeSet pset, PetscPolytope polytope, PetscInt facetOrigin, PetscInt facetImage, PetscInt imageOrientation,
                                                           PetscBool *isOrient, PetscInt *orientation)
{
  PetscPolytopeData p, b;
  PetscPolytope  f;
  PetscInt       originProj, originSect, imageProj, imageSect;
  PetscInt       oBaseOrigin, oImageBase, oComp;
  PetscInt       baseFacet;
  PetscInt       oBase;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscValidPointer(pset,1);
  if (polytope < 0 || polytope > pset->numPolytopes) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"No polytope with id %D\n",polytope);
  p = pset->polytopes[polytope];
  if (facetOrigin < 0 || facetOrigin > p->numFacets) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Origin %D not a facet number in [0,%D)\n",facetOrigin,p->numFacets);
  if (facetImage < 0 || facetImage > p->numFacets) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Image %D not a facet number in [0,%D)\n",facetImage,p->numFacets);
  if (p->facets[facetOrigin] != p->facets[facetImage]) { /* not the same facet type, can't be in the same orbit */
    *isOrient = PETSC_FALSE;
    *orientation = PETSC_MIN_INT;
    PetscFunctionReturn(0);
  }
  f = p->facets[facetOrigin];
  originProj = p->orbitProjectors[facetOrigin];
  originSect = p->orientInverses[originProj - p->orientStart];
  imageProj = p->orbitProjectors[facetImage];
  imageSect = p->orientInverses[imageProj - p->orientStart];
  baseFacet = p->orientsToFacetOrders[p->numFacets * (originProj - p->orientStart) + facetOrigin].index;
  if (p->orientsToFacetOrders[p->numFacets * (imageSect - p->orientStart) + baseFacet].index != facetImage) { /* origin and image are not in the same orbit */
    *isOrient = PETSC_FALSE;
    *orientation = PETSC_MIN_INT;
    PetscFunctionReturn(0);
  }
  oBaseOrigin = p->orientsToFacetOrders[p->numFacets * (originSect - p->orientStart) + baseFacet].orientation;
  oImageBase  = p->orientsToFacetOrders[p->numFacets * (imageProj - p->orientStart) + facetImage].orientation;
  ierr = PetscPolytopeSetOrientationCompose(pset, f, oBaseOrigin, imageOrientation, &oComp);CHKERRQ(ierr);
  ierr = PetscPolytopeSetOrientationCompose(pset, f, oComp, oImageBase, &oComp);CHKERRQ(ierr);
  /* imageProj o orientation o originSec = oBase , if orientation exists, maps baseFacet to itself with orientation oComp */
  b = pset->polytopes[f];
  oBase = p->facetOrientations[baseFacet*p->maxFacetSymmetry + (oComp-b->orientStart)];
  if (oBase < p->orientStart || oBase >= p->orientEnd) {
    *isOrient = PETSC_FALSE;
    *orientation = PETSC_MIN_INT;
    PetscFunctionReturn(0);
  }
  /* orientation is imageSec o oBase o originProj */
  ierr = PetscPolytopeSetOrientationCompose(pset, polytope, originProj, oBase, &oComp);CHKERRQ(ierr);
  ierr = PetscPolytopeSetOrientationCompose(pset, polytope, oComp, imageSect, orientation);CHKERRQ(ierr);
  *isOrient = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscInt permToID(PetscInt n, const PetscInt perm[], PetscInt work[])
{
  PetscInt j, p = perm[0], subID;

  for (j = 0; j < n; j++) if (work[j] == p) break;
  work[j] = work[0];
  work[0] = p;
  if (n > 2) {
    subID = permToID(n - 1, &perm[1], &work[1]);
    j = n * subID + j;
  }
  return j;
}

static PetscErrorCode PetscPolytopeSetOrientationFromVertices(PetscPolytopeSet pset, PetscPolytope polytope, const PetscInt vertices[], PetscBool *isOrientation, PetscInt *orientation)
{
  PetscInt       numVertices;
  PetscPolytopeData data;
  const PetscInt *otov;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  if (polytope < 0 || polytope > pset->numPolytopes) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"No polytope with id %D\n",polytope);
  data = pset->polytopes[polytope];
  if (!data->numVertices) {
    *isOrientation = PETSC_TRUE;
    *orientation = 0;
    PetscFunctionReturn(0);
  }
  numVertices = data->numVertices;
  otov        = data->orientsToVertexOrders;
  if (otov && numVertices <= 12) { /* directly compare each vertex order with the order given */
    PetscInt work[12];
    PetscInt oStart, oEnd;
    PetscInt id, v, idx;

    for (v = 0; v < numVertices; v++) work[v] = v;
    id = permToID(numVertices, vertices, work);
    oStart = data->orientStart;
    oEnd = data->orientEnd;
    ierr = PetscFindInt(id, oEnd - oStart, data->vertexPerms, &idx);CHKERRQ(ierr);
    if (idx < 0) {
      *isOrientation = PETSC_FALSE;
      *orientation = PETSC_MIN_INT;
    }
    *isOrientation = PETSC_TRUE;
    *orientation = data->vertexPermsToOrients[idx];
  } else {
    PetscInt *vwork, *owork, *order;
    PetscInt i, j, n = data->vertexOffsets[1] - data->vertexOffsets[0];
    PetscInt offset, numFacets = data->numFacets, fo;

    ierr = PetscMalloc2(data->numVertices, &vwork, data->numVertices, &owork);CHKERRQ(ierr);
    /* get the vertices of the first facet */
    for (i = 0; i < n; i++) {
      vwork[i] = vertices[i];
      owork[i] = i;
    }
    /* sort them and permute the order in which they were given */
    ierr = PetscSortIntWithArray(n, vwork, owork);CHKERRQ(ierr);
    for (i = 0; i < numFacets; i++) {
      if (data->facets[i] != data->facets[0]) continue;
      /* compare to the sorted vertices to the sorted vertices for each facet */
      offset = data->vertexOffsets[i];
      for (j = 0; j < n; j++) if (data->facetsToVerticesSorted[j+offset] != vwork[j]) break;
      if (j == n) break;
    }
    if (i == numFacets) { /* vertex order does not correspond to an orientation */
      *isOrientation = PETSC_FALSE;
      *orientation   = PETSC_MIN_INT;
      PetscFunctionReturn(0);
    }
    /* compose the order of the vertices around the image facet and the given order the vertices */
    order = &data->facetsToVerticesOrder[offset];
    for (j = 0; j < n; j++) vwork[owork[j]] = order[j];
    ierr = PetscPolytopeSetOrientationFromVertices(pset, data->facets[0], vwork, isOrientation, &fo);CHKERRQ(ierr);
    ierr = PetscFree2(vwork, owork);CHKERRQ(ierr);
    if (*isOrientation) {
      ierr = PetscPolytopeSetOrientationFromFacet(pset, polytope, 0, i, fo, isOrientation, orientation);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetGetPolytope(PetscPolytopeSet pset, const char name[], PetscPolytope *polytope)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i = 0; i < pset->numNames; i++) {
    PetscBool same;

    ierr = PetscStrcmp(name, pset->names[i].name, &same);CHKERRQ(ierr);
    if (same) {
      *polytope = pset->names[i].polytope;
      PetscFunctionReturn(0);
    }
  }
  *polytope = PETSCPOLYTOPE_NONE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetInsertName(PetscPolytopeSet pset, const char name[], PetscPolytope tope)
{
  PetscInt       index;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  index = pset->numNames++;
  if (index >= pset->numNamesAlloc) {
    PetscPolytopeName *names;
    PetscInt newAlloc = PetscMax(8, pset->numNamesAlloc * 2);
    ierr = PetscCalloc1(newAlloc, &names);CHKERRQ(ierr);
    ierr = PetscArraycpy(names, pset->names, index);CHKERRQ(ierr);
    ierr = PetscFree(pset->names);CHKERRQ(ierr);
    pset->names = names;
    pset->numNamesAlloc = newAlloc;
  }
  pset->names[index].polytope = tope;
  ierr = PetscStrallocpy(name, &(pset->names[index].name));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetInsertPolytope(PetscPolytopeSet pset, PetscPolytopeData pData, PetscInt *id)
{
  PetscInt       index;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  index = *id = pset->numPolytopes++;
  if (index >= pset->numPolytopesAlloc) {
    PetscPolytopeData *ptopes;
    PetscInt newAlloc = PetscMax(8, pset->numPolytopesAlloc * 2);
    ierr = PetscCalloc1(newAlloc, &ptopes);CHKERRQ(ierr);
    ierr = PetscArraycpy(ptopes, pset->polytopes, index);CHKERRQ(ierr);
    ierr = PetscFree(pset->polytopes);CHKERRQ(ierr);
    pset->polytopes = ptopes;
    pset->numPolytopesAlloc = newAlloc;
  }
  pset->polytopes[index] = pData;
  PetscFunctionReturn(0);
}

static int RidgeSorterCompare(const void *a, const void *b)
{
  PetscInt i;
  const PetscInt *A = (const PetscInt *) a;
  const PetscInt *B = (const PetscInt *) b;

  /* compare ridge polytope */
  if (A[1] < B[1]) return -1;
  if (B[1] < A[1]) return 1;
  /* the same polytope (so same size): compare sorted vertices */
  for (i = 0; i < A[2]; i++) {
    if (A[3 + i] < B[3 + i]) return -1;
    if (B[3 + i] < A[3 + i]) return 1;
  }
  /* finally sort by facet */
  if (A[0] < B[0]) return -1;
  if (B[0] < A[0]) return 1;
  return 0;
}

static int SuppCompare(const void *a, const void *b)
{
  const PetscPolytopeSupp *A = (const PetscPolytopeSupp *) a;
  const PetscPolytopeSupp *B = (const PetscPolytopeSupp *) b;
  if (A->index < B->index) return -1;
  if (B->index < A->index) return 1;
  if (A->coneNumber < B->coneNumber) return -1;
  if (B->coneNumber < A->coneNumber) return 1;
  return 0;
}

/* facetsToRidges includes orientations */
/* ridgesToFacets includes cone numbers */
static PetscErrorCode PetscPolytopeSetComputeRidges(PetscPolytopeSet pset, PetscPolytopeData pData)
{
  PetscInt          numFacets, numVertices;
  const PetscPolytope *facets;
  const PetscInt *vertexOffsets;
  const PetscInt *facetsToVertices;
  PetscInt          i, r, maxRidgeSize, numFacetRidges, numRidges;
  PetscInt          sorterSize, count;
  PetscInt          *facetRidgeSorter;
  PetscInt          *ftro, *vwork, *rwork;
  PetscPolytopeCone *ftr;
  PetscPolytopeSupp *rtf;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  numFacets        = pData->numFacets;
  numVertices      = pData->numVertices;
  facets           = pData->facets;
  vertexOffsets    = pData->vertexOffsets;
  facetsToVertices = pData->facetsToVertices;
  ierr = PetscMalloc1(numFacets + 1,&ftro);CHKERRQ(ierr);
  pData->ridgeOffsets = ftro;
  ftro[0] = 0;
  for (i = 0, maxRidgeSize = 0, numFacetRidges = 0; i < numFacets; i++) {
    PetscInt          j;
    PetscPolytope     f = facets[i];
    PetscPolytopeData fData = pset->polytopes[f];

    ftro[i+1] = ftro[i] + fData->numFacets;
    numFacetRidges += fData->numFacets;
    for (j = 0; j < fData->numFacets; j++) {
      PetscInt ridgeSize = fData->vertexOffsets[j+1] - fData->vertexOffsets[j];

      maxRidgeSize = PetscMax(maxRidgeSize,ridgeSize);
    }
  }
  if (numFacetRidges % 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No polytope has an odd number of facet ridges");
  pData->numRidges = numRidges = numFacetRidges / 2;
  ierr = PetscMalloc1(numFacetRidges, &ftr);CHKERRQ(ierr);
  pData->facetsToRidges = ftr;
  /* rtf is two (facets, cone number) pairs */
  ierr = PetscMalloc1(2 * numRidges, &rtf);CHKERRQ(ierr);
  pData->ridgesToFacets = rtf;
  sorterSize = 3 + maxRidgeSize; /* facetRidge, ridgePolytope, ridgeSize, ridgeVertices */
  ierr = PetscMalloc1(numFacetRidges * sorterSize, &facetRidgeSorter);CHKERRQ(ierr);
  for (i = 0, r = 0; i < numFacets; i++) {
    PetscInt          j;
    PetscInt          facetOffset = vertexOffsets[i];
    PetscPolytope     f = facets[i];
    PetscPolytopeData fData = pset->polytopes[f];

    for (j = 0; j < fData->numFacets; j++, r++) {
      PetscInt *ridge = &facetRidgeSorter[r*sorterSize];
      PetscInt *ridgeVertices = &ridge[3];
      PetscInt ridgeOffset = fData->vertexOffsets[j];
      PetscInt ridgeSize = fData->vertexOffsets[j+1] - ridgeOffset;
      PetscInt k;

      ridge[0] = r;
      ridge[1] = fData->facets[j];
      ridge[2] = ridgeSize;

      for (k = 0; k < ridgeSize; k++) {
        PetscInt v = fData->facetsToVertices[ridgeOffset + k];

        ridgeVertices[k] = facetsToVertices[facetOffset + v];
      }
      ierr = PetscSortInt(ridgeSize, ridgeVertices);CHKERRQ(ierr);
      for (k = ridgeSize; k < maxRidgeSize; k++) ridgeVertices[k] = -1;
    }
  }
  for (i = 0; i < numFacetRidges; i++) ftr[i].index = ftr[i].orientation = -1;
  for (i = 0; i < 2 * numRidges; i++) rtf[i].index = rtf[i].coneNumber = -1;
  /* all facetridges should occur as pairs: sort and detect them */
  qsort(facetRidgeSorter,(size_t) numFacetRidges, sorterSize * sizeof(PetscInt), RidgeSorterCompare);
  for (i = 1, count = 0; i < numFacetRidges; i++) {
    PetscInt  *ridgeA = &facetRidgeSorter[(i-1)*sorterSize];
    PetscInt  *ridgeB = &facetRidgeSorter[i*sorterSize];
    PetscBool same;

    ierr = PetscArraycmp(&ridgeA[1], &ridgeB[1], sorterSize-1, &same);CHKERRQ(ierr);
    if (!same) continue;
    if (ridgeA[0] == ridgeB[0]) break; /* cannot have one ridge appear twice in the cone of one facet */
    if (ftr[ridgeA[0]].index != -1 || ftr[ridgeB[0]].index != -1) break; /* cannot have more than two facets per ridge */
    ftr[ridgeA[0]].index = count;
    ftr[ridgeB[0]].index = count;
    rtf[2*count+0].index = ridgeA[0];
    rtf[2*count+1].index = ridgeB[0];
    count++;
  }
  ierr = PetscFree(facetRidgeSorter);CHKERRQ(ierr);
  if (i < numFacetRidges || count != numRidges) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Polytope does not generate consistent ridges");
  /* convert facetRidge number to (facet, cone number) */
  for (i = 0, r = 0; i < numFacets; i++) {
    PetscInt          j;
    PetscPolytope     f = facets[i];
    PetscPolytopeData fData = pset->polytopes[f];

    for (j = 0; j < fData->numFacets; j++, r++) {
      PetscInt ridge = ftr[r].index;
      PetscPolytopeSupp *ridgeData;

      if (ridge < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Polytope has unpaired ridge");
      ridgeData = &rtf[2 * ridge];
      if (ridgeData[0].index == r) {
        ridgeData[0].index = i;
        ridgeData[0].coneNumber = j;
      } else {
        if (ridgeData[1].index != r) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "ridgesToFacets / facetsToRidges mismatch");
        ridgeData[1].index = i;
        ridgeData[1].coneNumber = j;
      }
    }
  }
  /* re-sort rtf so that ridges appear in closure order */
  qsort(rtf,(size_t) numRidges, 2 * sizeof(PetscPolytopeSupp), SuppCompare);
  /* re-align ftr with re-sorted rtf */
  for (i = 0; i < numRidges; i++) {
    PetscPolytopeSupp *ridge = &rtf[2*i];
    PetscInt facetA = ridge[0].index;
    PetscInt coneA = ridge[0].coneNumber;
    PetscInt facetB = ridge[1].index;
    PetscInt coneB = ridge[1].coneNumber;

    ftr[ftro[facetA] + coneA].index = i;
    ftr[ftro[facetB] + coneB].index = i;
  }
  /* make sure that ridges line up in a way that makes sense:
   * the two orders of vertices from the opposing facets imply a
   * permutation of those vertices.  check that that symmetry is
   * valid for the polytope of the ridge */
  ierr = PetscMalloc1(numVertices, &vwork);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxRidgeSize, &rwork);CHKERRQ(ierr);
  for (i = 0, r = 0; i < numFacets; i++) {
    PetscInt          j;
    PetscInt          facetOffset = vertexOffsets[i];
    PetscPolytope     f = facets[i];
    PetscPolytopeData fData = pset->polytopes[f];

    for (j = 0; j < fData->numFacets; j++, r++) {
      PetscInt          ridge = ftr[r].index;
      PetscPolytopeSupp *ridgeData;
      PetscInt          oppFacet, oppCone;
      PetscInt          k, orient;
      PetscPolytope     rtope = fData->facets[j];
      PetscPolytopeData rData = pset->polytopes[rtope];
      PetscPolytopeData oppData;
      PetscInt          ridgeOffset = fData->vertexOffsets[j];
      PetscInt          oppFacetOffset, oppRidgeOffset;
      PetscBool         isOrient;

      ridgeData = &rtf[2 * ridge];
      /* only compute symmetries from first side */
      if (ridgeData[0].index != i) continue;
      oppFacet = ridgeData[1].index;
      oppCone  = ridgeData[1].coneNumber;
      oppData  = pset->polytopes[pData->facets[oppFacet]];
      oppFacetOffset = vertexOffsets[oppFacet];
      oppRidgeOffset = oppData->vertexOffsets[oppCone];
      ftr[r].orientation = 0; /* the orientation from the first facet is defined to be the identity */
      /* clear work */
      for (k = 0; k < numVertices; k++) vwork[k] = -1;
      for (k = 0; k < maxRidgeSize; k++) rwork[k] = -1;
      /* number vertices by the order they occur in the first facet numbering */
      for (k = 0; k < rData->numVertices; k++) {
        PetscInt rv = fData->facetsToVertices[ridgeOffset + k];
        PetscInt v = facetsToVertices[facetOffset + rv];

        vwork[v] = k;
      }
      /* gather that numbering from the perspective of the second facet */
      for (k = 0; k < rData->numVertices; k++) {
        PetscInt rv = oppData->facetsToVertices[oppRidgeOffset + k];
        PetscInt v = facetsToVertices[oppFacetOffset + rv];

        rwork[k] = vwork[v];
      }
      /* get the symmetry number */
      ierr = PetscPolytopeSetOrientationFromVertices(pset, rtope, rwork, &isOrient, &orient);CHKERRQ(ierr);
      if (!isOrient) break;
      ftr[ftro[oppFacet] + oppCone].orientation = orient;
    }
    if (j < fData->numFacets) break;
  }
  ierr = PetscFree(rwork);CHKERRQ(ierr);
  ierr = PetscFree(vwork);CHKERRQ(ierr);
  if (i < numFacets) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Vertices of polytope ridge are permuted in a way that is not an orientation of the ridge");
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetComputeSigns(PetscPolytopeSet pset, PetscPolytopeData pData)
{
  PetscInt       numFacets, numRidges;
  const PetscPolytope *facets;
  const PetscInt *ridgeOffsets;
  const PetscPolytopeCone *facetsToRidges;
  const PetscPolytopeSupp *ridgesToFacets;
  PetscInt       i, rcount, fcount;
  PetscInt       *facetQueue, *ridgeQueue;
  PetscBool      *facetSeen, *ridgeSeen;
  PetscBool      *inward;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  numFacets        = pData->numFacets;
  if (!numFacets) PetscFunctionReturn(0);
  numRidges        = pData->numRidges;
  facets           = pData->facets;
  ridgeOffsets     = pData->ridgeOffsets;
  facetsToRidges   = pData->facetsToRidges;
  ridgesToFacets   = pData->ridgesToFacets;
  inward           = pData->facetsInward;
  ierr = PetscMalloc1(numRidges, &ridgeQueue);CHKERRQ(ierr);
  ierr = PetscCalloc1(numRidges, &ridgeSeen);CHKERRQ(ierr);
  ierr = PetscMalloc1(numFacets, &facetQueue);CHKERRQ(ierr);
  ierr = PetscCalloc1(numFacets, &facetSeen);CHKERRQ(ierr);
  fcount = 1;
  facetQueue[0] = 0;
  facetSeen[0] = PETSC_TRUE;
  /* construct a breadth-first queue of ridges between facets,
   * so that we always test sign when at least one side
   * has its sign determined */
  for (i = 0, rcount = 0; i < numFacets; i++) {
    PetscInt f;
    PetscInt j;

    if (i == fcount) {
      ierr = PetscFree(facetSeen);CHKERRQ(ierr);
      ierr = PetscFree(facetQueue);CHKERRQ(ierr);
      ierr = PetscFree(ridgeSeen);CHKERRQ(ierr);
      ierr = PetscFree(ridgeQueue);CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Polytope is not strongly connected");
    }
    f = facetQueue[i];
    for (j = ridgeOffsets[f]; j < ridgeOffsets[f+1]; j++) {
      PetscInt r = facetsToRidges[j].index;
      PetscInt neigh;

      if (ridgeSeen[r]) continue;
      ridgeQueue[rcount++] = r;
      ridgeSeen[r] = PETSC_TRUE;
      if (ridgesToFacets[2*r].index == f) {
        neigh = ridgesToFacets[2*r+1].index;
      } else {
        neigh = ridgesToFacets[2*r].index;
      }
      if (!facetSeen[neigh]) {
        facetQueue[fcount++] = neigh;
        facetSeen[neigh] = PETSC_TRUE;
      }
    }
  }
  for (i = 0; i < numFacets; i++) facetSeen[i] = PETSC_FALSE;
  facetSeen[0] = PETSC_TRUE;
  for (i = 0; i < numRidges; i++) {
    PetscInt  r = ridgeQueue[i];
    PetscInt  f, g;
    PetscInt  fCone, gCone;
    PetscBool fSign, gSign;
    PetscInt  fOrient, gOrient;
    PetscInt  inwardg;
    PetscPolytopeData fData;
    PetscPolytopeData gData;

    f       = ridgesToFacets[2*r].index;
    fCone   = ridgesToFacets[2*r].coneNumber;
    fOrient = facetsToRidges[ridgeOffsets[f] + fCone].orientation;
    g       = ridgesToFacets[2*r+1].index;
    gCone   = ridgesToFacets[2*r+1].coneNumber;
    gOrient = facetsToRidges[ridgeOffsets[g] + gCone].orientation;
    if (!facetSeen[f]) {
      PetscInt swap;

      swap = f;
      f = g;
      g = swap;
      swap = fCone;
      fCone = gCone;
      gCone = swap;
      swap = fOrient;
      fOrient = gOrient;
      gOrient = swap;
    }
    if (!facetSeen[f]) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Ridge between two unseen facets\n");
    fData = pset->polytopes[facets[f]];
    gData = pset->polytopes[facets[g]];
    ierr  = PetscPolytopeDataFacetSign(fData, fCone, &fSign);CHKERRQ(ierr);
    ierr  = PetscPolytopeDataFacetSign(gData, gCone, &gSign);CHKERRQ(ierr);
    fSign = fSign ^ (fOrient < 0);
    fSign = fSign ^ inward[f];
    gSign = gSign ^ (gOrient < 0);
    inwardg = gSign ^ fSign ^ 1;
    if (!facetSeen[g]) {
      facetSeen[g] = PETSC_TRUE;
      inward[g] = inwardg;
    } else if (inward[g] != inwardg) {
      ierr = PetscFree(facetSeen);CHKERRQ(ierr);
      ierr = PetscFree(facetQueue);CHKERRQ(ierr);
      ierr = PetscFree(ridgeSeen);CHKERRQ(ierr);
      ierr = PetscFree(ridgeQueue);CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Polytope is not orientable\n");
    }
  }
  ierr = PetscFree(facetSeen);CHKERRQ(ierr);
  ierr = PetscFree(facetQueue);CHKERRQ(ierr);
  ierr = PetscFree(ridgeSeen);CHKERRQ(ierr);
  ierr = PetscFree(ridgeQueue);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetAddSymmetry_Insert(PetscPolytopeSet pset, PetscPolytopeData pData, const PetscPolytopeCone perm[], PetscBT permOrient)
{
  PetscInt       id, i, numFacets, foStart, numFOs, maxS;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  maxS = pData->maxFacetSymmetry;
  numFacets = pData->numFacets;
  foStart = pset->polytopes[pData->facets[0]]->orientStart;
  numFOs = pset->polytopes[pData->facets[0]]->orientEnd - foStart;
  id = perm[0].index * numFOs + (perm[0].orientation - foStart);
  if (PetscBTLookup(permOrient, id)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "A previously seen permutation slipped through the cracks");
  ierr = PetscBTSet(permOrient, id);CHKERRQ(ierr);
  for (i = 0; i < numFacets; i++) {
    PetscPolytopeData iData;
    if (perm[i].index < 0 || perm[i].index >= numFacets) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "permutation has bad index");
    if (pData->facets[i] != pData->facets[perm[i].index]) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "permutation maps incompatible facets");
    iData = pset->polytopes[pData->facets[i]];
    if (perm[i].orientation < iData->orientStart || perm[i].orientation >= iData->orientEnd) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "permutation has bad orientation");
  }
  for (i = 0; i < numFacets; i++) { /* update representers */
    PetscInt representer;
    PetscInt f;
    PetscInt proj;

    f = perm[i].index;
    proj = pData->orbitProjectors[f];
    if (proj >= 0) {
      representer = pData->orientsToFacetOrders[proj * numFacets + f].index;
    } else {
      representer = -(proj+1);
    }
    if (i < representer) { /* reset the projector and lookup data for f */
      PetscInt j;

      pData->orbitProjectors[f] = -(i+1);
      for (j = 0; j < maxS; j++) pData->facetOrientations[f*maxS + j] = PETSC_MIN_INT;
    }
    proj = pData->orbitProjectors[i];
    if (proj >= 0) {
      representer = pData->orientsToFacetOrders[proj * numFacets + i].index;
    } else {
      representer = -(proj+1);
    }
    if (f < representer) { /* reset the projector and lookup data for i */
      PetscInt j;

      pData->orbitProjectors[i] = -(f+1);
      for (j = 0; j < maxS; j++) pData->facetOrientations[i*maxS + j] = PETSC_MIN_INT;
    }
  }
  for (i = 0; i < numFacets; i++) { /* update inverses and projectors */
    PetscInt representer;
    PetscInt f, o;
    PetscInt proj;

    f = perm[i].index;
    o = perm[i].orientation;
    proj = pData->orbitProjectors[f];
    if (proj >= 0) {
      representer = pData->orientsToFacetOrders[proj * numFacets + f].index;
    } else {
      representer = -(proj+1);
    }
    if (i == representer) {
      PetscInt ioStart;
      PetscPolytopeData iData;

      iData = pset->polytopes[pData->facets[i]];
      ioStart = iData->orientStart;
      pData->facetOrientations[f*maxS + (o-ioStart)] = pData->orientEnd;
    }
    proj = pData->orbitProjectors[i];
    if (proj >= 0) {
      representer = pData->orientsToFacetOrders[proj * numFacets + i].index;
    } else {
      representer = -(proj+1);
    }
    if (f == representer && proj < 0) pData->orbitProjectors[i] = pData->orientEnd;
  }
  pData->orientEnd++;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetAddSymmetry(PetscPolytopeSet pset, PetscPolytopeData pData, const PetscPolytopeCone *perm, const PetscPolytopeCone *permInv, PetscBT permOrient)
{
  PetscInt       numFacets, maxS;
  PetscInt       image0, orient0;
  PetscInt       invimage0, invorient0;
  PetscInt       q;
  PetscInt       foStart, foEnd, numFOs, id, invid, i, p;
  PetscInt       orientEndOrig;
  PetscPolytopeData fData;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  numFacets = pData->numFacets;
  maxS      = pData->maxFacetSymmetry;
  if (!numFacets) PetscFunctionReturn(0);
  fData   = pset->polytopes[pData->facets[0]];
  foStart = fData->orientStart;
  foEnd   = fData->orientEnd;
  numFOs  = foEnd - foStart;

  orientEndOrig = pData->orientEnd;

  image0  = perm[0].index;
  orient0 = perm[0].orientation;
  id = image0 * numFOs + (orient0 - foStart);
  for (i = 0; i < numFacets; i++) pData->orientsToFacetOrders[pData->orientEnd * numFacets + i] = perm[i];
  ierr = PetscPolytopeSetAddSymmetry_Insert(pset, pData, perm, permOrient);CHKERRQ(ierr);

  invimage0  = permInv[0].index;
  invorient0 = permInv[0].orientation;
  invid = invimage0 * numFOs + (invorient0 - foStart);
  if (invid != id) {
    for (i = 0; i < numFacets; i++) pData->orientsToFacetOrders[pData->orientEnd * numFacets + i] = permInv[i];
    ierr = PetscPolytopeSetAddSymmetry_Insert(pset, pData, permInv, permOrient);CHKERRQ(ierr);
  }

  for (i = 0; i < numFacets; i++) {
    PetscPolytopeData iData;
    PetscInt o, oInvInv;
    if (perm[permInv[i].index].index != i) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "incorrect permutation inverse");
    iData = pset->polytopes[pData->facets[i]];
    oInvInv = iData->orientInverses[permInv[i].orientation - iData->orientStart];
    o = perm[permInv[i].index].orientation;
    if (o != oInvInv) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "incorrect permutation inverse orientations");
  }

  for (q = orientEndOrig; q < pData->orientEnd; q++) {
    const PetscPolytopeCone *permNew = &pData->orientsToFacetOrders[q*numFacets];
    for (p = 0; p <= q; p++) {
      const PetscPolytopeCone *permOld = &pData->orientsToFacetOrders[p*numFacets];
      PetscInt fComp = permOld[permNew[0].index].index;
      PetscInt oNew = permNew[0].orientation;
      PetscInt oOld = permOld[permNew[0].index].orientation;
      PetscInt oComp, proj, representer;
      PetscInt prodIdx;
      PetscPolytopeCone *prod;

      if (permOld[0].index == 0 && permOld[0].orientation == 0) continue; /* permOld is the identity */
      ierr = PetscPolytopeSetOrientationCompose(pset, pData->facets[0], oNew, oOld, &oComp);CHKERRQ(ierr);
      proj = pData->orbitProjectors[fComp];
      if (proj >= 0) {
        representer = pData->orientsToFacetOrders[numFacets*proj + fComp].index;
      } else {
        representer = -(proj+1);
      }
      if (representer == 0) { /* there is a chance that this product, sending 0 to fComp, has been seen before */
        prodIdx = pData->facetOrientations[maxS*fComp + oComp-foStart];
        if (p < orientEndOrig && prodIdx >= 0 && prodIdx < orientEndOrig) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "permutation is new but product of existing permutations");
        if (fComp == 0 && oComp == 0) { /* permNew is the inverse of permOld */
          pData->orientInverses[p] = q;
          pData->orientInverses[q] = p;
        }
        if (prodIdx >= 0) continue; /* this product has been found before */
      }
      /* compute the full product */
      prod = &pData->orientsToFacetOrders[pData->orientEnd * numFacets];
      prod[0].index = fComp;
      prod[0].orientation = oComp;
      for (i = 1; i < numFacets; i++) {
        fComp = permOld[permNew[i].index].index;
        oNew  = permNew[i].orientation;
        oOld  = permOld[permNew[i].index].orientation;

        ierr = PetscPolytopeSetOrientationCompose(pset, pData->facets[i], oNew, oOld, &oComp);CHKERRQ(ierr);
        prod[i].index = fComp;
        prod[i].orientation = oComp;
      }
      ierr = PetscPolytopeSetAddSymmetry_Insert(pset, pData, prod, permOrient);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetSignSymmetries(PetscPolytopeSet pset, PetscPolytopeData pData)
{
  PetscInt       o, f, numSyms, numFacets;
  PetscInt       maxS;
  PetscBool      *negative;
  PetscInt       *newIds, *newIdsInv;
  PetscInt       numNegative, count;
  PetscPolytopeCone *orientsToFacetOrders;
  PetscInt       *orientInverses;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  numFacets = pData->numFacets;
  if (!numFacets) PetscFunctionReturn(0);
  numSyms = pData->orientEnd;
  ierr = PetscCalloc1(numSyms, &negative);CHKERRQ(ierr);
  for (o = 0, numNegative = 0; o < pData->orientEnd; o++) {
    PetscInt f = pData->orientsToFacetOrders[o*numFacets].index;
    PetscInt fo = pData->orientsToFacetOrders[o*numFacets].orientation;

    negative[o] = (PetscBool) ((fo < 0) ^ (pData->facetsInward[0] != pData->facetsInward[f]));
    if (negative[o]) numNegative++;
  }
  if (!numNegative) {
    ierr = PetscFree(negative);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  pData->orientStart = -numNegative;
  pData->orientEnd = numSyms + pData->orientStart;
  ierr = PetscMalloc2(numSyms, &newIds, numSyms, &newIdsInv);CHKERRQ(ierr);
  for (o = 0, count = 0; o < numSyms; o++) {
    PetscInt newId = negative[o] ? (-numNegative + count++) : (o - count);

    newIds[o] = newId;
    newIdsInv[newId + numNegative] = o;
  }
  ierr = PetscMalloc1(numSyms * numFacets, &orientsToFacetOrders);CHKERRQ(ierr);
  for (o = 0; o < numSyms; o++) {
    for (f = 0; f < numFacets; f++) {
      orientsToFacetOrders[o * numFacets + f] = pData->orientsToFacetOrders[newIdsInv[o] * numFacets + f];
    }
  }
  ierr = PetscFree(pData->orientsToFacetOrders);CHKERRQ(ierr);
  pData->orientsToFacetOrders = orientsToFacetOrders;
  ierr = PetscMalloc1(numSyms, &orientInverses);CHKERRQ(ierr);
  for (o = 0; o < numSyms; o++) {
    orientInverses[o] = newIds[pData->orientInverses[newIdsInv[o]]];
  }
  ierr = PetscFree(pData->orientInverses);CHKERRQ(ierr);
  pData->orientInverses = orientInverses;
  maxS = pData->maxFacetSymmetry;
  for (f = 0; f < numFacets; f++) {
    pData->orbitProjectors[f] = newIds[pData->orbitProjectors[f]];
    for (o = 0; o < maxS; o++) {
      PetscInt ov = pData->facetOrientations[f*maxS+o];

      if (ov >= 0) {
        pData->facetOrientations[f*maxS+o] = newIds[ov];
      }
    }
  }
  ierr = PetscFree2(newIds, newIdsInv);CHKERRQ(ierr);
  ierr = PetscFree(negative);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetComputeSymmetries(PetscPolytopeSet pset, PetscPolytopeData pData)
{
  PetscInt       numFacets, numVertices;
  PetscInt       foStart, foEnd, numFOs, numFR, maxR, maxS;
  PetscPolytopeData fData;
  PetscInt       i, f, r, o;
  PetscBT        permOrient;
  PetscPolytopeCone *fCone, *perm, *permInv;
  PetscInt       *originQueue;
  PetscInt       fcount;
  PetscBool      *originSeen;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  numFacets = pData->numFacets;
  numVertices = pData->numVertices;
  /* every facet starts out with the identity */
  pData->orientStart = 0;
  pData->orientEnd   = 1;
  for (i = 0, maxS = 0; i < numFacets; i++) {
    fData = pset->polytopes[pData->facets[i]];
    maxS = PetscMax(maxS, fData->orientEnd - fData->orientStart);
  }
  pData->maxFacetSymmetry = maxS;
  ierr = PetscMalloc1(numFacets * numFacets * PetscMax(1, maxS), &(pData->orientsToFacetOrders));CHKERRQ(ierr);
  for (i = 0; i < numFacets; i++) {
    pData->orientsToFacetOrders[i].index       = i;
    pData->orientsToFacetOrders[i].orientation = 0;
  }
  ierr = PetscMalloc1(PetscMax(1, numFacets * maxS), &(pData->orientInverses));CHKERRQ(ierr);
  pData->orientInverses[0] = 0;
  ierr = PetscMalloc1(numFacets, &(pData->orbitProjectors));CHKERRQ(ierr);
  /* with just the identity, each facet is in its own orbit, so the identity is the projector
   * onto the orbit representer */
  for (i = 0; i < numFacets; i++) pData->orbitProjectors[i] = 0;
  ierr = PetscMalloc1(numFacets * maxS, &(pData->facetOrientations));CHKERRQ(ierr);
  /* with just the identity, a (facet, orientation) pair is valid only if the orientation is the
   * identity for that facets group, and it is always the identity */
  for (i = 0; i < numFacets; i++) {
    PetscPolytopeData iData = pset->polytopes[pData->facets[i]];
    PetscInt ioStart = iData->orientStart;
    PetscInt ioEnd = iData->orientEnd;
    PetscInt j;

    for (j = 0; j < ioEnd - ioStart; j++) {
      pData->facetOrientations[i * maxS + j] = (j == 0-ioStart) ? 0 : PETSC_MIN_INT;
    }
    for (j = ioEnd - ioStart; j < maxS; j++) {
      pData->facetOrientations[i * maxS + j] = PETSC_MIN_INT;
    }
  }
  if (numFacets == 0 || numFacets == 1) {
    /* only identity */
    ierr = PetscMalloc1(numVertices, &(pData->orientsToVertexOrders));CHKERRQ(ierr);
    for (i = 0; i < numVertices; i++) pData->orientsToVertexOrders[i] = i;
    ierr = PetscCalloc1(1, &(pData->vertexPerms));CHKERRQ(ierr);
    ierr = PetscCalloc1(1, &(pData->vertexPermsToOrients));CHKERRQ(ierr);
    ierr = PetscCalloc1(1, &(pData->orientComps));CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  fData   = pset->polytopes[pData->facets[0]];
  foStart = fData->orientStart;
  foEnd   = fData->orientEnd;
  numFOs  = foEnd - foStart;
  ierr = PetscBTCreate(numFacets * numFOs, &permOrient);CHKERRQ(ierr);
  ierr = PetscBTMemzero(numFacets * numFOs, permOrient);CHKERRQ(ierr);
  /* we have already seen the identity */
  ierr = PetscBTSet(permOrient,0-foStart);CHKERRQ(ierr);
  for (f = 0, maxR = 0; f < numFacets; f++) maxR = PetscMax(maxR,pData->ridgeOffsets[f+1] - pData->ridgeOffsets[f]);
  ierr = PetscMalloc1(maxR, &fCone);CHKERRQ(ierr);
  ierr = PetscMalloc1(numFacets, &originSeen);CHKERRQ(ierr);
  ierr = PetscMalloc1(numFacets, &originQueue);CHKERRQ(ierr);
  ierr = PetscMalloc2(numFacets, &perm, numFacets, &permInv);CHKERRQ(ierr);
  for (f = 0; f < numFacets; f++) {
    if (pData->facets[f] != pData->facets[0]) continue; /* different facet polytopes, can't be a symmetry */
    for (o = foStart; o < foEnd; o++) {
      PetscInt id = f * numFOs + (o - foStart);
      PetscInt q, oInv;

      if (PetscBTLookup(permOrient, id)) continue; /* this orientation has been found */

      /* reset data */
      for (q = 0; q < numFacets; q++) {
        perm[q].index = -1;
        perm[q].orientation = -1;
        permInv[q].index = -1;
        permInv[q].orientation = -1;
        originSeen[q] = PETSC_FALSE;
      }

      ierr = PetscPolytopeDataOrientationInverse(fData, o, &oInv);CHKERRQ(ierr);
      perm[0].index = f;
      perm[0].orientation = o;
      permInv[f].index = 0;
      permInv[f].orientation = oInv;
      originSeen[0] = PETSC_TRUE;
      originQueue[0] = 0;
      fcount = 1;
      for (q = 0; q < numFacets; q++) {
        const PetscPolytopeCone *ftr0;
        const PetscPolytopeCone *ftrf;
        PetscPolytopeData oData;
        PetscInt origin;
        PetscInt target;
        PetscInt oo;
        if (q >= fcount) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Disconnected facet graph, should have been caught earlier");

        /* apply the orientation at the origin to the ridges, and use those to propagate to the facet neighbors */
        origin = originQueue[q];
        target = perm[origin].index;
        oo     = perm[origin].orientation;
        oData  = pset->polytopes[pData->facets[origin]];
        ierr   = PetscPolytopeDataFacetsFromOrientation(oData, oo, fCone);CHKERRQ(ierr);
        numFR  = oData->numFacets;
        ftr0   = &pData->facetsToRidges[pData->ridgeOffsets[origin]];
        ftrf   = &pData->facetsToRidges[pData->ridgeOffsets[target]];
        for (r = 0; r < numFR; r++) {
          PetscPolytope rtope;
          PetscPolytopeData rData;
          PetscPolytopeData nData;
          PetscInt rOrigin, rTarget;
          PetscInt n, nConeNum;
          PetscInt m, mConeNum;
          PetscInt oOrigin;
          PetscInt oOriginInv;
          PetscInt oTarget;
          PetscInt oComp;
          PetscInt oAction;
          PetscInt oN, oM, oMinv, oNM;
          PetscBool isOrient;

          rtope = oData->facets[r];
          rData = pset->polytopes[rtope];

          rOrigin = ftr0[r].index;
          oOrigin = ftr0[r].orientation;
          n = pData->ridgesToFacets[2*rOrigin].index;
          nConeNum = pData->ridgesToFacets[2*rOrigin].coneNumber;
          if (n == origin) {
            n = pData->ridgesToFacets[2*rOrigin+1].index;
            nConeNum = pData->ridgesToFacets[2*rOrigin+1].coneNumber;
          }

          rTarget = ftrf[fCone[r].index].index;
          oTarget = ftrf[fCone[r].index].orientation;
          m = pData->ridgesToFacets[2*rTarget].index;
          mConeNum = pData->ridgesToFacets[2*rTarget].coneNumber;
          if (m == target) {
            m = pData->ridgesToFacets[2*rTarget+1].index;
            mConeNum = pData->ridgesToFacets[2*rTarget+1].coneNumber;
          }
          if (pData->facets[m] != pData->facets[n]) break; /* this is not a compatible orientation: it maps different types of facets onto each other */
          nData = pset->polytopes[pData->facets[n]];

          oN = pData->facetsToRidges[pData->ridgeOffsets[n] + nConeNum].orientation;
          oM = pData->facetsToRidges[pData->ridgeOffsets[m] + mConeNum].orientation;

          oAction = fCone[r].orientation;
          ierr = PetscPolytopeDataOrientationInverse(rData, oOrigin, &oOriginInv);CHKERRQ(ierr);
          ierr = PetscPolytopeDataOrientationInverse(rData, oM, &oMinv);CHKERRQ(ierr);
          ierr = PetscPolytopeSetOrientationCompose(pset, rtope, oN, oOriginInv, &oComp);CHKERRQ(ierr);
          ierr = PetscPolytopeSetOrientationCompose(pset, rtope, oComp, oAction, &oComp);CHKERRQ(ierr);
          ierr = PetscPolytopeSetOrientationCompose(pset, rtope, oComp, oTarget, &oComp);CHKERRQ(ierr);
          ierr = PetscPolytopeSetOrientationCompose(pset, rtope, oComp, oMinv, &oComp);CHKERRQ(ierr);
          ierr = PetscPolytopeSetOrientationFromFacet(pset, pData->facets[n], nConeNum, mConeNum, oComp, &isOrient, &oNM);CHKERRQ(ierr);
          if (!isOrient) break; /* TODO: can this happen ? */
          if (!originSeen[n]) {
            PetscInt oNMinv;
            originSeen[n] = PETSC_TRUE;
            originQueue[fcount++] = n;

            perm[n].index = m;
            perm[n].orientation = oNM;
            ierr = PetscPolytopeDataOrientationInverse(nData, oNM, &oNMinv);CHKERRQ(ierr);
            permInv[m].index = n;
            permInv[m].orientation = oNMinv;
          } else {
            if (perm[n].index != m || perm[n].orientation != oNM) break; /* different dictates from different ridges, orientation impossible */
          }
        }
        if (r < numFR) break;
      }
      if (q < numFacets) continue;
      ierr = PetscPolytopeSetAddSymmetry(pset, pData, perm, permInv, permOrient);CHKERRQ(ierr);
    }
  }
  for (i = 0; i < numFacets; i++) if (pData->orbitProjectors[i] < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "failed to find orbit projector");
  ierr = PetscPolytopeSetSignSymmetries(pset, pData);CHKERRQ(ierr);
  ierr = PetscFree2(perm,permInv);CHKERRQ(ierr);
  ierr = PetscFree(originQueue);CHKERRQ(ierr);
  ierr = PetscFree(originSeen);CHKERRQ(ierr);
  ierr = PetscFree(fCone);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&permOrient);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetOrientVertices(PetscPolytopeSet, PetscPolytope, PetscInt, PetscInt[]);

static PetscErrorCode PetscPolytopeSetComputeVertexOrders(PetscPolytopeSet pset, PetscPolytope polytope)
{
  PetscInt          i, vfac, oStart, oEnd, numOrients, numVertices,  o;
  PetscInt          *vertexPerms, *vertexPermsToOrients, *orientsToVertexOrders, work[12], perm[12];
  PetscPolytopeData pdata;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  pdata = pset->polytopes[polytope];
  if (pdata->orientsToVertexOrders || pdata->numVertices == 0 || pdata->numVertices > 12) PetscFunctionReturn(0);
  vfac = 1;
  oStart = pdata->orientStart;
  oEnd   = pdata->orientEnd;
  numOrients = oEnd - oStart;
  numVertices = pdata->numVertices;
  for (i = 0; i < pdata->numVertices; i++) vfac *= i;
  ierr = PetscMalloc1(numOrients, &vertexPerms);CHKERRQ(ierr);
  ierr = PetscMalloc1(numOrients, &vertexPermsToOrients);CHKERRQ(ierr);
  ierr = PetscMalloc1(numOrients * numVertices, &orientsToVertexOrders);CHKERRQ(ierr);
  for (o = oStart; o < oEnd; o++) {
    PetscInt id;

    ierr = PetscPolytopeSetOrientVertices(pset, polytope, o, &orientsToVertexOrders[(o-oStart)*numVertices]);CHKERRQ(ierr);
    ierr = PetscArraycpy(perm, &orientsToVertexOrders[(o-oStart)*numVertices], numVertices);CHKERRQ(ierr);
    for (i = 0; i < numVertices; i++) work[i] = i;
    id = permToID(numVertices, perm, work);
    vertexPerms[o - oStart] = id;
    vertexPermsToOrients[o - oStart] = o;
  }
  ierr = PetscSortIntWithArray(numOrients, vertexPerms, vertexPermsToOrients);CHKERRQ(ierr);
  pdata->vertexPerms = vertexPerms;
  pdata->vertexPermsToOrients = vertexPermsToOrients;
  pdata->orientsToVertexOrders = orientsToVertexOrders;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetComputeCompositions(PetscPolytopeSet pset, PetscPolytope polytope)
{
  PetscInt          oStart, oEnd, numOrients, o, w;
  PetscInt          *orientComps;
  PetscPolytopeData pdata;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  pdata = pset->polytopes[polytope];
  oStart = pdata->orientStart;
  oEnd   = pdata->orientEnd;
  numOrients = oEnd - oStart;
  if (pdata->orientComps || numOrients > 64) PetscFunctionReturn(0);
  ierr = PetscMalloc1(numOrients * numOrients, &orientComps);CHKERRQ(ierr);
  for (o = oStart; o < oEnd; o++) {
    for (w = oStart; w < oEnd; w++) {
      PetscInt ow;

      ierr = PetscPolytopeSetOrientationCompose(pset, polytope, o, w, &ow);CHKERRQ(ierr);
      orientComps[(o - oStart) * (numOrients) + (w - oStart)] = ow;
    }
  }
  pdata->orientComps = orientComps;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetGetOrientationRange(PetscPolytopeSet, PetscPolytope, PetscInt *, PetscInt *);

#define CHKPOLYTOPEERRQ(pData,ierr) if (ierr) {PetscErrorCode _ierr = PetscPolytopeDataDestroy(&(pData));CHKERRQ(_ierr);CHKERRQ(ierr);}

static PetscErrorCode PetscPolytopeSetInsert(PetscPolytopeSet pset, const char name[], PetscInt numFacets, PetscInt numVertices, const PetscPolytope facets[], const PetscInt vertexOffsets[], const PetscInt facetsToVertices[], PetscBool firstFacetInward, PetscPolytope *polytope)
{
  PetscInt          i, dim;
  PetscPolytope     existing, id;
  PetscPolytopeData pData;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (numFacets < 0 || numVertices < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Attempt to create polytope with negative sizes (%D facets, %D vertices)\n", numFacets, numVertices);
  ierr = PetscPolytopeSetGetPolytope(pset, name, &existing);CHKERRQ(ierr);
  if (existing != PETSCPOLYTOPE_NONE) {
    PetscBool same;

    ierr = PetscPolytopeDataCompare(pset->polytopes[existing], numFacets, numVertices, facets, vertexOffsets, facetsToVertices, firstFacetInward, &same);CHKERRQ(ierr);
    if (same) {
      *polytope = existing;
    } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Attempt to insert polytope %s with different data than existing polytope with that name\n", name);
  }
  for (id = 0; id < pset->numPolytopes; id++) {
    PetscBool same;

    ierr = PetscPolytopeDataCompare(pset->polytopes[id], numFacets, numVertices, facets, vertexOffsets, facetsToVertices, firstFacetInward, &same);CHKERRQ(ierr);
    if (same) break;
  }
  if (id < pset->numPolytopes) {
    ierr = PetscPolytopeSetInsertName(pset, name, id);CHKERRQ(ierr);
    *polytope = id;
    PetscFunctionReturn(0);
  }
  { /* make sure this polytope has a consistent dimension (recursively) */
    PetscInt minDim = PETSC_MAX_INT;
    PetscInt maxDim = PETSC_MIN_INT;

    for (i = 0; i < numFacets; i++) {
      PetscPolytopeData fdata;
      PetscInt          numVertices;
      PetscPolytope     f = facets[i];

      if (f < 0 || f > pset->numPolytopes) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Polytope facet %D has id %D which is unknown to the polytope set\n", i, f);
      fdata = pset->polytopes[f];
      numVertices = vertexOffsets[i + 1] - vertexOffsets[i];
      if (fdata->dim != 0 && numVertices != fdata->numVertices) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Polytope facet %D has id %D but %D vertices != %D\n", i, f, numVertices, fdata->numVertices);
      minDim = PetscMin(minDim, fdata->dim);
      maxDim = PetscMax(maxDim, fdata->dim);
    }
    if (!numFacets) minDim = maxDim = -2;
    if (minDim != maxDim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Polytope facets have inconsistent dimensions in [%D,%D]", minDim, maxDim);
    dim = minDim + 1;
  }
  { /* make sure the vertex order is consistent with the closure order */
    PetscInt   *closureVertices;
    PetscBool  *seen;
    PetscInt   count;
    PetscBool  inOrder;
    char       suggestion[256] = {0};

    ierr = PetscMalloc1(numVertices, &closureVertices);CHKERRQ(ierr);
    ierr = PetscCalloc1(numVertices, &seen);CHKERRQ(ierr);
    for (i = 0, count = 0; i < numFacets; i++) {
      PetscInt j;

      for (j = vertexOffsets[i]; j < vertexOffsets[i + 1]; j++) {
        PetscInt v = facetsToVertices[j];

        if (!seen[v]) {
          seen[v] = PETSC_TRUE;
          closureVertices[count++] = v;
        }
      }
    }
    if (count != numVertices) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Polytope does not touch all vertices");
    for (i = 0; i < numVertices; i++) if (closureVertices[i] != i) break;
    inOrder = (PetscBool) (i == numVertices);
    if (!inOrder) {
      PetscInt origLength, length;
      char *top;

      length = origLength = sizeof(suggestion);
      top = suggestion;
      for (i = 0; i < numVertices; i++) {
        if (i < numVertices-1) {
          ierr = PetscSNPrintf(top,length,"%D,",closureVertices[i]);CHKERRQ(ierr);
        } else {
          ierr = PetscSNPrintf(top,length,"%D",closureVertices[i]);CHKERRQ(ierr);
        }
        while ((top - suggestion) < (origLength - 1) && *top != '\0') {
          top++;
          length--;
        }
      }
    }
    ierr = PetscFree(seen);CHKERRQ(ierr);
    ierr = PetscFree(closureVertices);CHKERRQ(ierr);
    if (!inOrder) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Polytope vertices not in closure order: suggested reordering [%s]\n", suggestion);
  }
  ierr = PetscPolytopeDataCreate(dim, numFacets, numVertices, facets, vertexOffsets, facetsToVertices, firstFacetInward, &pData);CHKERRQ(ierr);
  /* I want an incorrectly specified polytope to be recoverable, so I try to tear down the partially created polytope
   * if these two routines fail */
  ierr = PetscPolytopeSetComputeRidges(pset, pData);CHKPOLYTOPEERRQ(pData,ierr);
  ierr = PetscPolytopeSetComputeSigns(pset, pData);CHKPOLYTOPEERRQ(pData,ierr);
  /* Any failure in compute symmetries should be a PLIB failure and not recoverable */
  ierr = PetscPolytopeSetComputeSymmetries(pset, pData);CHKERRQ(ierr);
  ierr = PetscPolytopeSetInsertPolytope(pset, pData, polytope);CHKERRQ(ierr);
  ierr = PetscPolytopeSetInsertName(pset, name, *polytope);CHKERRQ(ierr);
  if (numVertices <= 12) {
    ierr = PetscPolytopeSetComputeVertexOrders(pset, *polytope);CHKERRQ(ierr);
  }
  {
    PetscInt oStart, oEnd;
    ierr = PetscPolytopeSetGetOrientationRange(pset, *polytope, &oStart, &oEnd);CHKERRQ(ierr);
    if ((oEnd - oStart) <= 64) {
      ierr = PetscPolytopeSetComputeCompositions(pset, *polytope);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopesDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPolytopeSetDestroy(&PetscPolytopes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopesGet(PetscPolytopeSet *pset)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!PetscPolytopes) {
    PetscPolytope null, vertex, edge, tri, quad, hyquad, tet, hex, prism;

    ierr = PetscPolytopeSetCreate(&PetscPolytopes);CHKERRQ(ierr);
    ierr = PetscRegisterFinalize(PetscPolytopesDestroy);CHKERRQ(ierr);
    ierr = PetscPolytopeSetInsert(PetscPolytopes, "dmplex-null", 0, 0, NULL, NULL, NULL, PETSC_FALSE, &null);CHKERRQ(ierr);
    if (null != 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "dmplex-null != 0");

    {
      PetscInt vertexOffsets[2] = {0,0};

      ierr = PetscPolytopeInsert("dmplex-vertex", 1, 0, &null, vertexOffsets, NULL, PETSC_FALSE, &vertex);CHKERRQ(ierr);
    }
    if (vertex != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "dmplex-vertex != 1");

    {
      PetscPolytope facets[2];
      PetscInt      vertexOffsets[3] = {0,1,2};
      PetscInt      facetsToVertices[2] = {0, 1};

      facets[0] = facets[1] = vertex;

      ierr = PetscPolytopeInsert("dmplex-edge", 2, 2, facets, vertexOffsets, facetsToVertices, PETSC_TRUE, &edge);CHKERRQ(ierr);
    }
    if (edge != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "dmplex-edge != 2");

    {
      PetscPolytope facets[3];
      PetscInt      vertexOffsets[4] = {0,2,4,6};
      PetscInt      facetsToVertices[6] = {0,1, 1,2, 2,0};

      facets[0] = facets[1] = facets[2] = edge;

      ierr = PetscPolytopeInsert("dmplex-triangle", 3, 3, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, &tri);CHKERRQ(ierr);
    }
    if (tri != 3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "dmplex-triangle != 3");

    {
      PetscPolytope facets[4];
      PetscInt      vertexOffsets[5] = {0,2,4,6,8};
      PetscInt      facetsToVertices[8] = {0,1, 1,2, 2,3, 3,0};

      facets[0] = facets[1] = facets[2] = facets[3] = edge;

      ierr = PetscPolytopeInsert("dmplex-quadrilateral", 4, 4, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, &quad);CHKERRQ(ierr);
    }
    if (quad != 4) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "dmplex-quadrilateral != 4");

    {
      PetscPolytope facets[4];
      PetscInt      vertexOffsets[5] = {0,3,6,9,12};
      PetscInt      facetsToVertices[12] = {0,1,2, 0,3,1, 0,2,3, 2,1,3};

      facets[0] = facets[1] = facets[2] = facets[3] = tri;

      ierr = PetscPolytopeInsert("dmplex-tetrahedron", 4, 4, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, &tet);CHKERRQ(ierr);
    }
    if (tet != 5) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "dmplex-tetrahedron != 5");

    {
      PetscPolytope facets[6];
      PetscInt      vertexOffsets[7] = {0,4,8,12,16,20,24};
      PetscInt      facetsToVertices[24] = {0,1,2,3, 4,5,6,7, 0,3,5,4, 2,1,7,6, 3,2,6,5, 0,4,7,1};

      facets[0] = facets[1] = facets[2] = facets[3] = facets[4] = facets[5] = quad;

      ierr = PetscPolytopeInsert("dmplex-hexahedron", 6, 8, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, &hex);CHKERRQ(ierr);
    }
    if (hex != 6) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "dmplex-hexahedron != 6");

    {
      PetscPolytope facets[4];
      PetscInt      vertexOffsets[5] = {0,2,4,6,8};
      PetscInt      facetsToVertices[8] = {0,1, 2,3, 0,2, 1,3};

      facets[0] = facets[1] = facets[2] = facets[3] = edge;

      ierr = PetscPolytopeInsert("dmplex-hybrid-quadrilateral", 4, 4, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, &hyquad);CHKERRQ(ierr);
    }
    if (hyquad != 7) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "dmplex-hybrid-quadrilateral != 7");

    {
      PetscPolytope facets[5];
      PetscInt      vertexOffsets[6] = {0,3,6,10,14,18};
      PetscInt      facetsToVertices[18] = {
                                           0,1,2,
                                           3,4,5,
                                           0,1,3,4,
                                           1,2,4,5,
                                           2,0,5,3,
      };

      facets[0] = facets[1] = tri;
      facets[2] = facets[3] = facets[4] = hyquad;

      ierr = PetscPolytopeInsert("dmplex-hybrid-prism", 5, 6, facets, vertexOffsets, facetsToVertices, PETSC_FALSE, &prism);CHKERRQ(ierr);
    }
    if (prism != 8) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "dmplex-hybrid-prism != 8");
  }

  *pset = PetscPolytopes;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscPolytopeInsert(const char name[], PetscInt numFacets, PetscInt numVertices, const PetscPolytope facets[], const PetscInt vertexOffsets[], const PetscInt facetsToVertices[], PetscBool firstFacetInward, PetscPolytope *polytope)
{
  PetscPolytopeSet pset;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscPolytopesGet(&pset);CHKERRQ(ierr);
  ierr = PetscPolytopeSetInsert(pset, name, numFacets, numVertices, facets, vertexOffsets, facetsToVertices, firstFacetInward, polytope);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscPolytopeGetByName(const char name[], PetscPolytope *polytope)
{
  PetscPolytopeSet pset;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPolytopesGet(&pset);CHKERRQ(ierr);
  ierr = PetscPolytopeSetGetPolytope(pset, name, polytope);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetGetData(PetscPolytopeSet pset, PetscPolytope polytope, PetscInt *numFacets, PetscInt *numVertices, const PetscPolytope *facets[], const PetscInt *vertexOffsets[], const PetscInt *facetsToVertices[], const PetscBool *facetsInward[])
{
  PetscPolytopeData pData;

  PetscFunctionBegin;
  if (polytope > pset->numPolytopes) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No polytope with id %D\n", polytope);
  pData = pset->polytopes[polytope];
  if (numFacets)        *numFacets        = pData->numFacets;
  if (numVertices)      *numVertices      = pData->numVertices;
  if (facets)           *facets           = pData->facets;
  if (vertexOffsets)    *vertexOffsets    = pData->vertexOffsets;
  if (facetsToVertices) *facetsToVertices = pData->facetsToVertices;
  if (facetsInward)     *facetsInward     = pData->facetsInward;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscPolytopeGetData(PetscPolytope polytope, PetscInt *numFacets, PetscInt *numVertices, const PetscPolytope *facets[], const PetscInt *vertexOffsets[], const PetscInt *facetsToVertices[], const PetscBool *facetsInward[])
{
  PetscPolytopeSet pset;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPolytopesGet(&pset);CHKERRQ(ierr);
  ierr = PetscPolytopeSetGetData(pset, polytope, numFacets, numVertices, facets, vertexOffsets, facetsToVertices, facetsInward);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetGetOrientationRange(PetscPolytopeSet pset, PetscPolytope polytope, PetscInt *orientStart, PetscInt *orientEnd)
{
  PetscPolytopeData pData;

  PetscFunctionBegin;
  if (polytope > pset->numPolytopes) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No polytope with id %D\n", polytope);
  pData = pset->polytopes[polytope];
  if (orientStart) *orientStart = pData->orientStart;
  if (orientEnd)   *orientEnd   = pData->orientEnd;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscPolytopeGetOrientationRange(PetscPolytope polytope, PetscInt *orientStart, PetscInt *orientEnd)
{
  PetscPolytopeSet pset;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPolytopesGet(&pset);CHKERRQ(ierr);
  ierr = PetscPolytopeSetGetOrientationRange(pset, polytope, orientStart, orientEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetOrientVertices(PetscPolytopeSet pset, PetscPolytope polytope, PetscInt orientation, PetscInt vertices[])
{
  PetscPolytopeData pData;
  PetscInt i, j;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  if (polytope > pset->numPolytopes) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No polytope with id %D\n", polytope);
  pData = pset->polytopes[polytope];
  if (orientation < pData->orientStart || orientation >= pData->orientEnd) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Orientation %D not in range [%D, %D)\n", orientation, pData->orientStart, pData->orientEnd);
  if (pData->dim == 1) {
    if (orientation == -1) {
      vertices[0] = 1;
      vertices[1] = 0;
    } else {
      vertices[0] = 0;
      vertices[1] = 1;
    }
    PetscFunctionReturn(0);
  }
  if (pData->orientsToVertexOrders) {
    for (i = 0; i < pData->numVertices; i++) {
      vertices[i] = pData->orientsToVertexOrders[(orientation - pData->orientStart) * pData->numVertices + i];
    }
  } else {
    PetscInt vertArray[32];
    PetscInt *facetVertices = &vertArray[0];

    if (pData->numVertices > 32) {ierr = PetscMalloc1(pData->numVertices, &facetVertices);CHKERRQ(ierr);}
    for (i = 0; i < pData->numFacets; i++) {
      PetscInt f = pData->orientsToFacetOrders[(orientation - pData->orientStart) * pData->numFacets + i].index;
      PetscInt o = pData->orientsToFacetOrders[(orientation - pData->orientStart) * pData->numFacets + i].orientation;
      PetscInt ioffset = pData->vertexOffsets[i];
      PetscInt vcount = pData->vertexOffsets[i+1] - ioffset;
      PetscInt foffset = pData->vertexOffsets[f];

      ierr = PetscPolytopeSetOrientVertices(pset, pData->facets[i], o, facetVertices);CHKERRQ(ierr);
      for (j = 0; j < vcount; j++) {
        vertices[pData->facetsToVertices[ioffset + j]] = pData->facetsToVertices[foffset + facetVertices[j]];
      }
    }
    if (pData->numVertices > 32) {ierr = PetscFree(facetVertices);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscPolytopeOrientVertices(PetscPolytope polytope, PetscInt orientation, PetscInt vertices[])
{
  PetscPolytopeSet pset;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPolytopesGet(&pset);CHKERRQ(ierr);
  ierr = PetscPolytopeSetOrientVertices(pset, polytope, orientation, vertices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetOrientFacets(PetscPolytopeSet pset, PetscPolytope polytope, PetscInt orientation, PetscInt facets[], PetscInt facetOrientations[])
{
  PetscPolytopeData pData;
  PetscInt i;

  PetscFunctionBegin;
  if (polytope > pset->numPolytopes) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No polytope with id %D\n", polytope);
  pData = pset->polytopes[polytope];
  if (orientation < pData->orientStart || orientation >= pData->orientEnd) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Orientation %D not in range [%D, %D)\n", orientation, pData->orientStart, pData->orientEnd);
  for (i = 0; i < pData->numFacets; i++) {
    if (facets) facets[i]                       = pData->orientsToFacetOrders[(orientation - pData->orientStart) * pData->numFacets + i].index;
    if (facetOrientations) facetOrientations[i] = pData->orientsToFacetOrders[(orientation - pData->orientStart) * pData->numFacets + i].orientation;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscPolytopeOrientFacets(PetscPolytope polytope, PetscInt orientation, PetscInt facets[], PetscInt facetOrientations[])
{
  PetscPolytopeSet pset;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPolytopesGet(&pset);CHKERRQ(ierr);
  ierr = PetscPolytopeSetOrientFacets(pset, polytope, orientation, facets, facetOrientations);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPolytopeSetOrientationInverse(PetscPolytopeSet pset, PetscPolytope polytope, PetscInt orientation, PetscInt *inverse)
{
  PetscErrorCode    ierr;
  PetscPolytopeData pData;

  PetscFunctionBegin;
  if (polytope > pset->numPolytopes) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No polytope with id %D\n", polytope);
  pData = pset->polytopes[polytope];
  ierr = PetscPolytopeDataOrientationInverse(pData, orientation, inverse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscPolytopeOrientationInverse(PetscPolytope polytope, PetscInt orientation, PetscInt *inverse)
{
  PetscPolytopeSet pset;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPolytopesGet(&pset);CHKERRQ(ierr);
  ierr = PetscPolytopeSetOrientationInverse(pset, polytope, orientation, inverse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscPolytopeOrientationCompose(PetscPolytope polytope, PetscInt a, PetscInt b, PetscInt *aafterb)
{
  PetscPolytopeSet pset;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPolytopesGet(&pset);CHKERRQ(ierr);
  ierr = PetscPolytopeSetOrientationCompose(pset, polytope, a, b, aafterb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscPolytopeOrientationFromVertices(PetscPolytope polytope, const PetscInt vertices[], PetscBool *isOrientation, PetscInt *orientation)
{
  PetscPolytopeSet pset;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPolytopesGet(&pset);CHKERRQ(ierr);
  ierr = PetscPolytopeSetOrientationFromVertices(pset, polytope, vertices, isOrientation, orientation);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscPolytopeOrientationFromFacet(PetscPolytope polytope, PetscInt facet, PetscInt facetImage, PetscInt facetOrientation, PetscBool *isOrientation, PetscInt *orientation)
{
  PetscPolytopeSet pset;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPolytopesGet(&pset);CHKERRQ(ierr);
  ierr = PetscPolytopeSetOrientationFromFacet(pset, polytope, facet, facetImage, facetOrientation, isOrientation, orientation);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
