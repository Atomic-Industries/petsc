#include <petscdmbf.h>                  /*I "petscdmbf.h" I*/
#include <petsc/private/dmforestimpl.h> /*I "petscdmforest.h" I*/
#include <petsc/private/dmimpl.h>       /*I "petscdm.h" I*/
#include "bf_2d_topology.h"
#include "bf_3d_topology.h"
#include "bf_2d_cells.h"
#include "bf_3d_cells.h"
#include "bf_2d_iterate.h"
#include "bf_3d_iterate.h"


#if defined(PETSC_HAVE_P4EST)

//TODO deprecated
#include "petsc_p4est_package.h"
#include <p4est.h>
#include <p4est_extended.h>
#include <p4est_geometry.h>
#include <p4est_ghost.h>
#include <p4est_lnodes.h>
#include <p4est_vtk.h>
#include <p4est_plex.h>
#include <p4est_bits.h>
#include <p4est_algorithms.h>

/******************************************************************************
 * PRIVATE STRUCTURES
 *****************************************************************************/

typedef struct _p_DM_BF {
  /* forest-of-tree objects sequence: topology -> cells -> nodes */
  void       *ftTopology;
  void       *ftCells;
  void       *ftNodes;
  /* DMBF cells */
  DM_BF_Cell *cells;
  PetscBool  ownedCellsSetUpCalled;
  PetscBool  ghostCellsSetUpCalled;
  /* [option] blocks within a cell */
  PetscInt   blockSize[3];
  /* [option] settings for cell data */
  PetscInt   *valsPerElemRead, *valsPerElemReadWrite;
  PetscInt   nValsPerElemRead, nValsPerElemReadWrite;
  PetscInt   valsPerElemReadTotal, valsPerElemReadWriteTotal;
  VecScatter ltog; /* local to global scatter object, for parallel communication of data between local and global vectors */

} DM_BF;

/******************************************************************************
 * PRIVATE FUNCTIONS WITHOUT ERROR CHECKING
 *****************************************************************************/

static inline DM_BF *_p_getBF(DM dm)
{
  return (DM_BF*) ((DM_Forest*) dm->data)->data;
}

/***************************************
 * CELL SIZES
 **************************************/

#define _p_bytesAlign(a) (((a)+(PETSC_MEMALIGN-1)) & ~(PETSC_MEMALIGN-1))

static inline size_t _p_cellSizeOfInfo()
{
  return _p_bytesAlign(sizeof(DM_BF_Cell));
}

#define _p_cellOffsetDataRead _p_cellSizeOfInfo

static inline size_t _p_cellSizeOfDataRead(DM_BF *bf)
{
  return _p_bytesAlign((size_t)(sizeof(PetscScalar)*bf->blockSize[0]*bf->blockSize[1]*bf->blockSize[2]*bf->valsPerElemReadTotal));
}

#define _p_cellOffsetDataReadWrite(bf) (_p_cellSizeOfInfo() + _p_cellSizeOfDataRead(bf))

static inline size_t _p_cellSizeOfDataReadWrite(DM_BF *bf)
{
  return _p_bytesAlign((size_t)(sizeof(PetscScalar)*bf->blockSize[0]*bf->blockSize[1]*bf->blockSize[2]*bf->valsPerElemReadWriteTotal));
}

static inline size_t _p_cellSizeOfData(DM_BF *bf)
{
  return _p_cellSizeOfDataRead(bf) + _p_cellSizeOfDataReadWrite(bf);
}

static inline size_t _p_cellSize(DM_BF *bf)
{
  return _p_cellSizeOfInfo() + _p_cellSizeOfData(bf);
}

/***************************************
 * CELL POINTERS
 **************************************/

static inline DM_BF_Cell *_p_cellGetPtrIndex(DM_BF *bf, PetscInt index)
{
  return (DM_BF_Cell*)(((char*)bf->cells) + _p_cellSize(bf) * ((size_t)index));
}

static inline PetscScalar *_p_cellGetDataRead(DM_BF_Cell *cell)
{
  return (PetscScalar*)(((char*)cell) + _p_cellSizeOfInfo());
}

static inline PetscScalar *_p_cellGetDataReadWrite(DM_BF_Cell *cell, DM_BF *bf)
{
  return (PetscScalar*)(((char*)cell) + _p_cellSizeOfInfo() + _p_cellSizeOfDataRead(bf));
}

/******************************************************************************
 * PRIVATE FUNCTION DEFINITIONS
 *****************************************************************************/

static PetscErrorCode DMBFSetUpOwnedCells(DM);
static PetscErrorCode DMBFSetUpGhostCells(DM);
static PetscErrorCode DMForestDestroy_BF(DM);
static PetscErrorCode DMClone_BF(DM,DM*);

static PetscErrorCode DMCreateLocalVector_BF(DM,Vec*);
static PetscErrorCode DMCreateGlobalVector_BF(DM,Vec*);
static PetscErrorCode DMLocalToGlobalBegin_BF(DM,Vec,InsertMode,Vec);
static PetscErrorCode DMLocalToGlobalEnd_BF(DM,Vec,InsertMode,Vec);
static PetscErrorCode DMGlobalToLocalBegin_BF(DM,Vec,InsertMode,Vec);
static PetscErrorCode DMGlobalToLocalEnd_BF(DM,Vec,InsertMode,Vec);

static PetscErrorCode DMCreateMatrix_BF(DM,Mat*);

static PetscErrorCode DMCoarsen_BF(DM,MPI_Comm,DM*);
static PetscErrorCode DMRefine_BF(DM,MPI_Comm,DM*);
static PetscErrorCode DMView_BF(DM,PetscViewer);
static PetscErrorCode VecView_BF(Vec,PetscViewer);


/******************************************************************************
 * PRIVATE & PUBLIC FUNCTIONS
 *****************************************************************************/

/***************************************
 * SETUP
 **************************************/

static PetscErrorCode DMBFSetUpOwnedCells(DM dm)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2: ierr = DMBF_2D_IterateSetUpCells(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf));CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_IterateSetUpCells(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf));CHKERRQ(ierr); break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unreachable code");
  }
  bf->ownedCellsSetUpCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBFSetUpGhostCells(DM dm)
{
  DM_BF          *bf;
  DM_BF_Cell     *cell;
  PetscInt       dim, offset_cells, ng_cells, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  /* set data pointers of all ghost cells */
  ierr = DMBFGetInfo(dm,&dim,&offset_cells,PETSC_NULL,&ng_cells);CHKERRQ(ierr);
  for (i=offset_cells; i<(offset_cells+ng_cells); i++) {
    cell                = _p_cellGetPtrIndex(bf,i);
    cell->dataRead      = (const PetscScalar*)_p_cellGetDataRead(cell);
    cell->dataReadWrite = _p_cellGetDataReadWrite(cell,bf);
  }
  bf->ghostCellsSetUpCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_CellsCreate(DM dm)
{
  DM_BF          *bf;
  PetscInt       dim, n, ng;
  size_t         n_cells, ng_cells, cell_size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* get number of cells and their size */
  bf   = _p_getBF(dm);
  ierr = DMBFGetInfo(dm,&dim,&n,PETSC_NULL,&ng);CHKERRQ(ierr);
  n_cells   = (size_t)n;
  ng_cells  = (size_t)ng;
  cell_size = _p_cellSize(bf);
  /* create DMBF cells */
  ierr = PetscMalloc((n_cells+ng_cells)*cell_size,&bf->cells);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)dm,(n_cells+ng_cells)*cell_size);CHKERRQ(ierr);
  /* setup cells */
  ierr = DMBFSetUpOwnedCells(dm);CHKERRQ(ierr);
  ierr = DMBFSetUpGhostCells(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_CellsDestroy(DM dm)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  bf = _p_getBF(dm);
  if (!bf->cells) {
    PetscFunctionReturn(0);
  }
  ierr = PetscFree(bf->cells);CHKERRQ(ierr);
  bf->cells = PETSC_NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_LocalToGlobalScatterCreate(DM dm, VecScatter *ltog)
{
  PetscErrorCode   ierr;
  IS               isfrom,isto;
  MPI_Comm         comm;
  PetscInt         rank,n,ng,N,offset;
  PetscInt         i;
  PetscInt         lid,gid;
  p4est_topidx_t   t;
  p4est_quadrant_t *quad;
  Vec              vin,vout;
  p4est_t         *p4est;
  p4est_ghost_t   *ghost;
  PetscInt *from_idx;
  PetscInt *to_idx;

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr); //TODO deprecated

  rank   = (PetscInt)p4est->mpirank;
  n      = (PetscInt)p4est->local_num_quadrants;
  ng     = (PetscInt)ghost->ghosts.elem_count;
  N      = (PetscInt)p4est->global_num_quadrants;
  offset = (PetscInt)p4est->global_first_quadrant[rank];

  ierr = PetscMalloc1(n+ng,&from_idx);CHKERRQ(ierr);
  ierr = PetscMalloc1(n+ng,&to_idx);CHKERRQ(ierr);

  for(i = 0; i < n; i++) {
    from_idx[i] = i;
    to_idx[i]   = (PetscInt)offset + i;
  }

  for(i = 0; i < ng; i++) {
    quad      = sc_array_index(&ghost->ghosts, i);                  /* get ghost quadrant i */
    t         = quad->p.piggy3.which_tree;                              /* get tree # of ghost quadrant i */
    rank      = p4est_quadrant_find_owner(p4est, t, -1, quad);      /* get mpirank of ghost quadrant i */
    lid       = (PetscInt)quad->p.piggy3.local_num;                     /* get local id of ghost quadrant i on mpirank rank */
    gid       = (PetscInt)p4est->global_first_quadrant[rank] + lid; /* translate local id to global id */
    from_idx[n + i] = n + i;
    to_idx[n + i]   = gid;
  }

  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);

  ierr = ISCreateGeneral(comm,n+ng,from_idx,PETSC_COPY_VALUES,&isfrom);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n+ng,to_idx,PETSC_COPY_VALUES,&isto);CHKERRQ(ierr);
  ierr = PetscFree(from_idx);CHKERRQ(ierr);
  ierr = PetscFree(to_idx);CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm,&vin);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&vout);CHKERRQ(ierr);

  ierr = VecScatterCreate(vin,isfrom,vout,isto,ltog);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetUp_BF(DM dm)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf   = _p_getBF(dm);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim == PETSC_DETERMINE) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Topological dimension has to be set before setup");
  if (dim < 2 || 3 < dim)     SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM does not support %d dimensional domains",dim);
  if (bf->ftTopology)         SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Topology exists already");
  if (bf->ftCells)            SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells exist already");
  if (bf->ftNodes)            SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Nodes exist already");
  /* create forest-of-tree topology */
  switch (dim) {
    case 2: ierr = DMBF_2D_TopologyCreate(dm,(DM_BF_2D_Topology**)&bf->ftTopology);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_TopologyCreate(dm,(DM_BF_3D_Topology**)&bf->ftTopology);CHKERRQ(ierr); break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unreachable code");
  }
  if (!bf->ftTopology) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Topology does not exist");
  /* create forest-of-tree cells */
  switch (dim) {
    case 2: ierr = DMBF_2D_CellsCreate(dm,(DM_BF_2D_Topology*)bf->ftTopology,(DM_BF_2D_Cells**)&bf->ftCells);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_CellsCreate(dm,(DM_BF_3D_Topology*)bf->ftTopology,(DM_BF_3D_Cells**)&bf->ftCells);CHKERRQ(ierr); break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unreachable code");
  }
  if (!bf->ftCells) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  /* create forest-of-tree nodes */
  //TODO create nodes
  /* create DMBF cells */
  ierr = DMBF_CellsCreate(dm);CHKERRQ(ierr);
  ierr = DMBF_LocalToGlobalScatterCreate(dm,&bf->ltog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBFClear(DM dm)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf   = _p_getBF(dm);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  /* destroy forest-of-tree objects (in reverse order of creation) */
  switch (dim) {
    case 2:
      //TODO destroy nodes
      if (bf->ftCells)    { ierr = DMBF_2D_CellsDestroy(dm,(DM_BF_2D_Cells*)bf->ftCells);CHKERRQ(ierr); }
      if (bf->ftTopology) { ierr = DMBF_2D_TopologyDestroy(dm,(DM_BF_2D_Topology*)bf->ftTopology);CHKERRQ(ierr); }
      break;
    case 3:
      //TODO destroy nodes
      if (bf->ftCells)    { ierr = DMBF_3D_CellsDestroy(dm,(DM_BF_3D_Cells*)bf->ftCells);CHKERRQ(ierr); }
      if (bf->ftTopology) { ierr = DMBF_3D_TopologyDestroy(dm,(DM_BF_3D_Topology*)bf->ftTopology);CHKERRQ(ierr); }
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unreachable code");
  }
  bf->ftNodes    = PETSC_NULL;
  bf->ftCells    = PETSC_NULL;
  bf->ftTopology = PETSC_NULL;
  /* destroy DMBF cells */
  ierr = DMBF_CellsDestroy(dm);CHKERRQ(ierr);
  /* destroy cell options */
  if (!bf->valsPerElemRead) {
    ierr = PetscFree(bf->valsPerElemRead);CHKERRQ(ierr);
  }
  if (!bf->valsPerElemReadWrite) {
    ierr = PetscFree(bf->valsPerElemReadWrite);CHKERRQ(ierr);
  }
  bf->valsPerElemRead           = PETSC_NULL;
  bf->valsPerElemReadWrite      = PETSC_NULL;
  bf->nValsPerElemRead          = 0;
  bf->nValsPerElemReadWrite     = 0;
  bf->valsPerElemReadTotal      = 0;
  bf->valsPerElemReadWriteTotal = 0;
  PetscFunctionReturn(0);
}

/***************************************
 * OPTIONS
 **************************************/

/*@
  DMBFSetBlockSize - During the pre-setup phase, set the levels of uniform block refinement of each cell in each dimension.

  Logically collective on dm

  Input Parameters:
+ dm        - the DMBF object

  Output Parameters:
+ blockSize - levels of uniform block refinement of each cell in each dimension

  Level: intermediate

.seealso: DMBFGetBlockSize(), DMGetDimension()
@*/
PetscErrorCode DMBFSetBlockSize(DM dm, PetscInt *blockSize)
{
  DM_BF          *bf;
  PetscInt       dim, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidIntPointer(blockSize,2);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the block refinement after setup");
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim == PETSC_DETERMINE) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot set block refinement before topological dimension");
  bf = _p_getBF(dm);
  for (i=0; i<dim; i++) {
    bf->blockSize[i] = (1 <= blockSize[i] ? blockSize[i] : 1);
  }
  PetscFunctionReturn(0);
}

/*@
  DMBFGetBlockSize - Get the levels of uniform block refinement of each cell in each dimension.

  Logically collective on dm

  Input Parameters:
+ dm        - the DMBF object
- blockSize - levels of uniform block refinement of each cell in each dimension

  Level: intermediate

.seealso: DMBFSetBlockSize(), DMGetDimension()
@*/
PetscErrorCode DMBFGetBlockSize(DM dm, PetscInt *blockSize)
{
  DM_BF          *bf;
  PetscInt       dim, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidIntPointer(blockSize,2);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim == PETSC_DETERMINE) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Topological dimension has to be set for block refinement");
  bf = _p_getBF(dm);
  for (i=0; i<dim; i++) {
    blockSize[i] = bf->blockSize[i];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFSetCellDataSize(DM dm, PetscInt *valsPerElemRead, PetscInt nValsPerElemRead, PetscInt *valsPerElemReadWrite, PetscInt nValsPerElemReadWrite)
{
  DM_BF          *bf;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidIntPointer(valsPerElemRead,2);
  PetscValidIntPointer(valsPerElemReadWrite,4);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change cell data after setup");
  bf = _p_getBF(dm);
  /* reset exising settings */
  if (!bf->valsPerElemRead) {
    ierr = PetscFree(bf->valsPerElemRead);CHKERRQ(ierr);
  }
  if (!bf->valsPerElemReadWrite) {
    ierr = PetscFree(bf->valsPerElemReadWrite);CHKERRQ(ierr);
  }
  bf->valsPerElemRead           = PETSC_NULL;
  bf->valsPerElemReadWrite      = PETSC_NULL;
  bf->nValsPerElemRead          = 0;
  bf->nValsPerElemReadWrite     = 0;
  bf->valsPerElemReadTotal      = 0;
  bf->valsPerElemReadWriteTotal = 0;
  /* set new settings */
  if (0 < nValsPerElemRead) {
    bf->nValsPerElemRead = nValsPerElemRead;
    ierr = PetscMalloc1(bf->nValsPerElemRead,&bf->valsPerElemRead);CHKERRQ(ierr);
    for (i=0; i<bf->nValsPerElemRead; i++) {
      bf->valsPerElemRead[i]    = valsPerElemRead[i];
      bf->valsPerElemReadTotal += valsPerElemRead[i];
    }
  }
  if (0 < nValsPerElemReadWrite) {
    bf->nValsPerElemReadWrite = nValsPerElemReadWrite;
    ierr = PetscMalloc1(bf->nValsPerElemReadWrite,&bf->valsPerElemReadWrite);CHKERRQ(ierr);
    for (i=0; i<bf->nValsPerElemReadWrite; i++) {
      bf->valsPerElemReadWrite[i]    = valsPerElemReadWrite[i];
      bf->valsPerElemReadWriteTotal += valsPerElemReadWrite[i];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFGetCellDataSize(DM dm, PetscInt **valsPerElemRead, PetscInt *nValsPerElemRead, PetscInt **valsPerElemReadWrite, PetscInt *nValsPerElemReadWrite)
{
  DM_BF          *bf;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidIntPointer(valsPerElemRead,3);
  PetscValidIntPointer(valsPerElemReadWrite,5);
  bf = _p_getBF(dm);
  *nValsPerElemRead = bf->nValsPerElemRead;
  if (0 < bf->nValsPerElemRead && valsPerElemRead) {
    ierr = PetscMalloc1(bf->nValsPerElemRead,valsPerElemRead);CHKERRQ(ierr);
    for (i=0; i<bf->nValsPerElemRead; i++) {
      (*valsPerElemRead)[i] = bf->valsPerElemRead[i];
    }
  }
  *nValsPerElemReadWrite = bf->nValsPerElemReadWrite;
  if (0 < bf->nValsPerElemReadWrite && valsPerElemReadWrite) {
    ierr = PetscMalloc1(bf->nValsPerElemReadWrite,valsPerElemReadWrite);CHKERRQ(ierr);
    for (i=0; i<bf->nValsPerElemReadWrite; i++) {
      (*valsPerElemReadWrite)[i] = bf->valsPerElemReadWrite[i];
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetFromOptions_BF(PetscOptionItems *PetscOptionsObject,DM dm)
{
  PetscInt          blockSize[3], nBlockDim=3;
  PetscBool         set;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMSetFromOptions_Forest(PetscOptionsObject,dm);CHKERRQ(ierr);
  /* block_size */
  ierr = DMBFGetBlockSize(dm,blockSize);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray(
      "-dm_bf_block_size","set uniform refinement inside each cell in each dimension x,y,z","DMBFSetBlockSize",
      blockSize,&nBlockDim,&set);CHKERRQ(ierr);
  if (set) {
    //TODO if (nBlockDim != dim)
    ierr = DMBFSetBlockSize(dm,blockSize);CHKERRQ(ierr);
  }
//TODO
//char              stringBuffer[256];
//ierr = PetscOptionsHead(PetscOptionsObject,"DM" P4EST_STRING " options");CHKERRQ(ierr);
//ierr = PetscOptionsBool("-dm_p4est_partition_for_coarsening","partition forest to allow for coarsening","DMP4estSetPartitionForCoarsening",pforest->partition_for_coarsening,&(pforest->partition_for_coarsening),NULL);CHKERRQ(ierr);
//ierr = PetscOptionsString("-dm_p4est_ghost_label_name","the name of the ghost label when converting from a DMPlex",NULL,NULL,stringBuffer,sizeof(stringBuffer),&flg);CHKERRQ(ierr);
//ierr = PetscOptionsTail();CHKERRQ(ierr);
//if (flg) {
//  ierr = PetscFree(pforest->ghostName);CHKERRQ(ierr);
//  ierr = PetscStrallocpy(stringBuffer,&pforest->ghostName);CHKERRQ(ierr);
//}
  PetscFunctionReturn(0);
}


/***************************************
 * CREATE/DESTROY
 **************************************/

static PetscErrorCode DMInitialize_BF(DM dm)
{
  PetscFunctionBegin;
  dm->ops->setup              = DMSetUp_BF;
  dm->ops->setfromoptions     = DMSetFromOptions_BF;
  dm->ops->clone              = DMClone_BF;
  dm->ops->createlocalvector  = DMCreateLocalVector_BF;
  dm->ops->createglobalvector = DMCreateGlobalVector_BF;
  dm->ops->creatematrix       = DMCreateMatrix_BF;
  dm->ops->coarsen            = DMCoarsen_BF;
  dm->ops->refine             = DMRefine_BF;
  dm->ops->view               = DMView_BF;
  dm->ops->globaltolocalbegin = DMGlobalToLocalBegin_BF;
  dm->ops->globaltolocalend   = DMGlobalToLocalEnd_BF;
  dm->ops->localtoglobalbegin = DMLocalToGlobalBegin_BF;
  dm->ops->localtoglobalend   = DMLocalToGlobalEnd_BF;

  //TODO
  //dm->ops->createsubdm    = DMCreateSubDM_Forest;
  //dm->ops->adaptlabel     = DMAdaptLabel_Forest;
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreate_BF(DM dm)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscP4estInitialize();CHKERRQ(ierr);
  ierr = DMCreate_Forest(dm);CHKERRQ(ierr);
  ierr = DMInitialize_BF(dm);CHKERRQ(ierr);

  /* set default parameters */
  ierr = DMSetDimension(dm,2);CHKERRQ(ierr);
  ierr = DMForestSetTopology(dm,"unit");CHKERRQ(ierr);
  ierr = DMForestSetMinimumRefinement(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetInitialRefinement(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetMaximumRefinement(dm,18);CHKERRQ(ierr);
  ierr = DMForestSetGradeFactor(dm,2);CHKERRQ(ierr);
  ierr = DMForestSetAdjacencyDimension(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetPartitionOverlap(dm,0);CHKERRQ(ierr);

  /* create BF */
  ierr = PetscNewLog(dm,&bf);CHKERRQ(ierr);
  bf->ftTopology                = PETSC_NULL;
  bf->ftCells                   = PETSC_NULL;
  bf->ftNodes                   = PETSC_NULL;
  bf->cells                     = PETSC_NULL;
  bf->ownedCellsSetUpCalled     = PETSC_FALSE;
  bf->ghostCellsSetUpCalled     = PETSC_FALSE;
  bf->blockSize[0]              = 1;
  bf->blockSize[1]              = 1;
  bf->blockSize[2]              = 1;
  bf->valsPerElemRead           = PETSC_NULL;
  bf->valsPerElemReadWrite      = PETSC_NULL;
  bf->nValsPerElemRead          = 0;
  bf->nValsPerElemReadWrite     = 0;
  bf->valsPerElemReadTotal      = 0;
  bf->valsPerElemReadWriteTotal = 0;

  /* set data & functions of Forest object */
  {
    DM_Forest *forest = (DM_Forest*) dm->data;

    forest->data    = bf;
    forest->destroy = DMForestDestroy_BF;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMForestDestroy_BF(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* destroy contents of BF */
  ierr = DMBFClear(dm);CHKERRQ(ierr);
  /* destroy BF object */
  ierr = PetscFree(((DM_Forest*)dm->data)->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMClone_BF(DM dm, DM *newdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMClone_Forest(dm,newdm);CHKERRQ(ierr);
  ierr = DMInitialize_BF(*newdm);CHKERRQ(ierr);
  //TODO this is likely incomplete
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFGetP4est(DM dm, void *p4est)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->ftCells) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  ierr = DMBF_2D_CellsGetP4est((DM_BF_2D_Cells*)bf->ftCells,p4est);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFGetGhost(DM dm, void *ghost)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->ftCells) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  ierr = DMBF_2D_CellsGetGhost((DM_BF_2D_Cells*)bf->ftCells,ghost);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateLocalVector_BF(DM dm, Vec *vec)
{
  PetscInt       dim, n, ng;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidPointer(vec,2);
  /* get number of entries */
  ierr = DMBFGetInfo(dm,&dim,&n,PETSC_NULL,&ng);CHKERRQ(ierr);
  /* create vector */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n+ng,vec);CHKERRQ(ierr);
  ierr = VecSetDM(*vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateGlobalVector_BF(DM dm, Vec *vec)
{
  PetscInt       dim, n, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidPointer(vec,2);
  /* get number of entries */
  ierr = DMBFGetInfo(dm,&dim,&n,&N,PETSC_NULL);CHKERRQ(ierr);
  /* create vector */
  ierr = VecCreateMPI(PetscObjectComm((PetscObject)dm),n,N,vec);CHKERRQ(ierr);
  ierr = VecSetDM(*vec,dm);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec,VECOP_VIEW,(void (*)(void))VecView_BF);CHKERRQ(ierr);
  //TODO
  //ierr = VecSetOperation(*g,VECOP_VIEW,(void (*)(void))VecView_MPI_DA);CHKERRQ(ierr);
  //ierr = VecSetOperation(*vec, VECOP_VIEWNATIVE, (void (*)(void))VecView_pforest_Native);CHKERRQ(ierr);
  //ierr = VecSetOperation(*g,VECOP_LOAD,(void (*)(void))VecLoad_Default_DA);CHKERRQ(ierr);
  //ierr = VecSetOperation(*vec, VECOP_LOADNATIVE, (void (*)(void))VecLoad_pforest_Native);CHKERRQ(ierr);
  //ierr = VecSetOperation(*g,VECOP_DUPLICATE,(void (*)(void))VecDuplicate_MPI_DA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateMatrix_BF(DM dm, Mat *mat)
{
  void           *appctx;
  PetscInt       dim, n, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidPointer(mat,2);
  /* get number of rows/cols */
  ierr = DMBFGetInfo(dm,&dim,&n,&N,PETSC_NULL);CHKERRQ(ierr);
  /* create matrix */
  ierr = DMGetApplicationContext(dm,&appctx);CHKERRQ(ierr);
  ierr = MatCreateShell(PetscObjectComm((PetscObject)dm),n,n,N,N,appctx,mat);CHKERRQ(ierr);
  ierr = MatSetDM(*mat,dm);CHKERRQ(ierr);
  //TODO set null space?
  PetscFunctionReturn(0);
}

/* take global vector and return local version */
static PetscErrorCode DMGlobalToLocalBegin_BF(DM dm, Vec glo, InsertMode mode, Vec loc)
{
  DM_BF              *bf;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidHeaderSpecific(glo,VEC_CLASSID,2);
  PetscValidHeaderSpecific(loc,VEC_CLASSID,4);

  bf = _p_getBF(dm);
  ierr = VecScatterBegin(bf->ltog,glo,loc,mode,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGlobalToLocalEnd_BF(DM dm, Vec glo, InsertMode mode, Vec loc)
{
  DM_BF              *bf;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidHeaderSpecific(glo,VEC_CLASSID,2);
  PetscValidHeaderSpecific(loc,VEC_CLASSID,4);

  bf = _p_getBF(dm);
  ierr = VecScatterEnd(bf->ltog,glo,loc,mode,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode DMLocalToGlobalBegin_BF(DM dm, Vec loc, InsertMode mode, Vec glo)
{
  DM_BF              *bf;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidHeaderSpecific(loc,VEC_CLASSID,2);
  PetscValidHeaderSpecific(glo,VEC_CLASSID,4);

  bf = _p_getBF(dm);
  ierr = VecScatterBegin(bf->ltog,loc,glo,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLocalToGlobalEnd_BF(DM dm, Vec loc, InsertMode mode, Vec glo)
{
  DM_BF              *bf;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidHeaderSpecific(loc,VEC_CLASSID,2);
  PetscValidHeaderSpecific(glo,VEC_CLASSID,4);

  bf = _p_getBF(dm);
  ierr = VecScatterEnd(bf->ltog,loc,glo,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/***************************************
 * MESH
 **************************************/

/*@
  DMBFGetInfo - Gets information about the DM.

  Not Collective

  Input Parameter:
+ dm      - the DMBF object

  Output Parameters:
+ dim     - spatial dimension (2 or 3)
. nLocal  - number of local cells
. nGlobal - number of global cells
. nGhost  - number of ghost cells

  Level: beginner

.seealso: DMGetDimension(), DMBFGetLocalSize(), DMBFGetGlobalSize(), DMBFGetGhostSize()
@*/
PetscErrorCode DMBFGetInfo(DM dm, PetscInt *dim, PetscInt *nLocal, PetscInt *nGlobal, PetscInt *nGhost)
{
  DM_BF          *bf;
  PetscInt       n, ng, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidIntPointer(dim,2);
  bf = _p_getBF(dm);
  if (!bf->ftCells) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  ierr = DMGetDimension(dm,dim);CHKERRQ(ierr);
  switch (*dim) {
    case 2: ierr = DMBF_2D_GetSizes(dm,(DM_BF_2D_Cells*)bf->ftCells,&n,&N,&ng);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_GetSizes(dm,(DM_BF_3D_Cells*)bf->ftCells,&n,&N,&ng);CHKERRQ(ierr); break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unreachable code");
  }
  if (nLocal)  *nLocal  = n;
  if (nGlobal) *nGlobal = N;
  if (nGhost)  *nGhost  = ng;
  PetscFunctionReturn(0);
}

/*@
  DMBFGetLocalSize - Gets local number of quadrants in the forest.

  Not Collective

  Input Parameters:
+ dm      - the DMBF object

  Output Parameters:
+ nLocal  - number of local cells (does not count ghost cells)

  Level: beginner

.seealso: DMBFGetInfo(), DMBFGetGlobalSize(), DMBFGetGhostSize()
@*/

PetscErrorCode DMBFGetLocalSize(DM dm, PetscInt *nLocal)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMBFGetInfo(dm,&dim,nLocal,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMBFGetGlobalSize - Gets global number of quadrants in the forest.

  Logically collective on DM

  Input Parameters:
+ dm      - the DMBF object

  Output Parameters:
+ nGlobal - number of global cells

  Level: beginner

.seealso: DMBFGetInfo(), DMBFGetLocalSize(), DMBFGetGhostSize()
@*/

PetscErrorCode DMBFGetGlobalSize(DM dm, PetscInt *nGlobal)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMBFGetInfo(dm,&dim,PETSC_NULL,nGlobal,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMBFGetGhostSize - Gets number of quadrants in the ghost layer.

  Not Collective

  Input Parameters:
+ dm      - the DMBF object

  Output Parameters:
+ nGhost  - number of ghost cells

  Level: beginner

.seealso: DMBFGetInfo(), DMBFGetLocalSize(), DMBFGetGlobalSize()
@*/

PetscErrorCode DMBFGetGhostSize(DM dm, PetscInt *nGhost)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMBFGetInfo(dm,&dim,PETSC_NULL,PETSC_NULL,nGhost);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/***************************************
 * AMR
 **************************************/

typedef struct _p_DM_BF_AmrCtx {
  PetscInt  minLevel;
  PetscInt  maxLevel;
} DM_BF_AmrCtx;

static int p4est_coarsen_uniformly(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrants[])
{
  DM_BF_AmrCtx   *amrCtx = p4est->user_pointer;
  const PetscInt minLevel = amrCtx->minLevel;
  const PetscInt l = quadrants[0]->level;

  return (minLevel < l);
}

static int p4est_refine_uniformly(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant)
{
  DM_BF_AmrCtx   *amrCtx = p4est->user_pointer;
  const PetscInt maxLevel = amrCtx->maxLevel;
  const PetscInt l = quadrant->level;

  return (l < maxLevel);
}

/*@
  DMBFCoarsenInPlace - Coarsens the mesh uniformly.

  Logically collective on DM

  Input Parameters:
+ dm      - the DMBF object
- nCycles - number of coarsening cycles

  Level: intermediate

.seealso: DMBFRefineInPlace()
@*/
PetscErrorCode DMBFCoarsenInPlace(DM dm, PetscInt nCycles)
{
  DM_BF_AmrCtx   amrCtx;
  DM_BF          *bf;
  void           *p4est_user_pointer;
  PetscInt       i;
  PetscErrorCode ierr;
  p4est_t        *p4est; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* set AMR parameters */
  ierr = DMForestGetMinimumRefinement(dm,&amrCtx.minLevel);CHKERRQ(ierr);
  /* prepare p4est for AMR */
  bf = _p_getBF(dm);
  p4est_user_pointer  = p4est->user_pointer;
  p4est->user_pointer = (void*) &amrCtx;
  /* coarsen & balance */
  for (i=0; i<nCycles; i++) {
    PetscStackCallP4est(p4est_coarsen,(p4est,0,p4est_coarsen_uniformly,NULL));
  }
  PetscStackCallP4est(p4est_balance,(p4est,P4EST_CONNECT_FULL,NULL));
  /* finalize p4est after AMR */
  p4est->user_pointer = p4est_user_pointer;
  //TODO need to update bf cells
  PetscFunctionReturn(0);
}

/*@
  DMBFRefineInPlace - Refines the mesh uniformly.

  Logically collective on DM

  Input Parameters:
+ dm      - the DMBF object
- nCycles - number of refinement cycles

  Level: intermediate

.seealso: DMBFCoarsenInPlace()
@*/
PetscErrorCode DMBFRefineInPlace(DM dm, PetscInt nCycles)
{
  DM_BF_AmrCtx   amrCtx;
  DM_BF          *bf;
  void           *p4est_user_pointer;
  PetscInt       i;
  PetscErrorCode ierr;
  p4est_t        *p4est; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* set AMR parameters */
  ierr = DMForestGetMaximumRefinement(dm,&amrCtx.maxLevel);CHKERRQ(ierr);
  /* prepare p4est for AMR */
  bf = _p_getBF(dm);
  p4est_user_pointer  = p4est->user_pointer;
  p4est->user_pointer = (void*) &amrCtx;
  /* refine & balance */
  for (i=0; i<nCycles; i++) {
    PetscStackCallP4est(p4est_refine,(p4est,0,p4est_refine_uniformly,NULL));
  }
  PetscStackCallP4est(p4est_balance,(p4est,P4EST_CONNECT_FULL,NULL));
  /* finalize p4est after AMR */
  p4est->user_pointer = p4est_user_pointer;
  //TODO need to update bf cells
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsen_BF(DM dm, MPI_Comm comm, DM *dmc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* coarsen input DM */
  if (!dmc) {
    ierr = DMBFCoarsenInPlace(dm,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* coarsen output DM */
  if (dm != *dmc) { /* if new fine DM */
    //TODO need to clone
  }
  ierr = DMBFCoarsenInPlace(*dmc,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRefine_BF(DM dm, MPI_Comm comm, DM *dmf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* refine input DM */
  if (!dmf) {
    ierr = DMBFRefineInPlace(dm,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* refine output DM */
  if (dm != *dmf) { /* if new fine DM */
    //TODO need to clone
  }
  ierr = DMBFRefineInPlace(*dmf,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/***************************************
 * ITERATORS
 **************************************/

PetscErrorCode DMBFIterateOverCellsVectors(DM dm, PetscErrorCode (*iterCell)(DM,DM_BF_Cell*,void*), void *userIterCtx,
                                           Vec *vecRead, PetscInt nVecsRead, Vec *vecReadWrite, PetscInt nVecsReadWrite)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidFunction(iterCell,2);
  if (nVecsRead)      PetscValidPointer(vecRead,4);
  if (nVecsReadWrite) PetscValidPointer(vecReadWrite,6);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2: ierr = DMBF_2D_IterateOverCellsVectors(dm,bf->cells,_p_cellSize(bf),iterCell,userIterCtx,
                                                   vecRead,nVecsRead,vecReadWrite,nVecsReadWrite);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_IterateOverCellsVectors(dm,bf->cells,_p_cellSize(bf),iterCell,userIterCtx,
                                                   vecRead,nVecsRead,vecReadWrite,nVecsReadWrite);CHKERRQ(ierr); break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unreachable code");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFIterateOverCells(DM dm, PetscErrorCode (*iterCell)(DM,DM_BF_Cell*,void*), void *userIterCtx)
{
  PetscFunctionBegin;
  CHKERRQ( DMBFIterateOverCellsVectors(dm,iterCell,userIterCtx,PETSC_NULL,0,PETSC_NULL,0) );
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFIterateOverFaces(DM dm, PetscErrorCode (*iterFace)(DM,DM_BF_Face*,void*), void *userIterCtx)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidFunction(iterFace,2);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  if (!bf->ghostCellsSetUpCalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Ghost cells not set up");
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2: ierr = DMBF_2D_IterateOverFaces(dm,bf->cells,_p_cellSize(bf),iterFace,userIterCtx);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_IterateOverFaces(dm,bf->cells,_p_cellSize(bf),iterFace,userIterCtx);CHKERRQ(ierr); break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unreachable code");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFSetCellData(DM dm, Vec *vecRead, Vec *vecReadWrite)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  if (vecRead      && bf->nValsPerElemRead)      PetscValidPointer(vecRead,2);
  if (vecReadWrite && bf->nValsPerElemReadWrite) PetscValidPointer(vecReadWrite,3);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2: ierr = DMBF_2D_IterateSetCellData(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf),
                                              bf->valsPerElemRead,bf->nValsPerElemRead,bf->valsPerElemReadWrite,bf->nValsPerElemReadWrite,
                                              vecRead,vecReadWrite);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_IterateSetCellData(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf),
                                              bf->valsPerElemRead,bf->nValsPerElemRead,bf->valsPerElemReadWrite,bf->nValsPerElemReadWrite,
                                              vecRead,vecReadWrite);CHKERRQ(ierr); break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unreachable code");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFGetCellData(DM dm, Vec *vecRead, Vec *vecReadWrite)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  if (vecRead      && bf->nValsPerElemRead)      PetscValidPointer(vecRead,2);
  if (vecReadWrite && bf->nValsPerElemReadWrite) PetscValidPointer(vecReadWrite,3);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2: ierr = DMBF_2D_IterateGetCellData(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf),
                                              bf->valsPerElemRead,bf->nValsPerElemRead,bf->valsPerElemReadWrite,bf->nValsPerElemReadWrite,
                                              vecRead,vecReadWrite);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_IterateGetCellData(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf),
                                              bf->valsPerElemRead,bf->nValsPerElemRead,bf->valsPerElemReadWrite,bf->nValsPerElemReadWrite,
                                              vecRead,vecReadWrite);CHKERRQ(ierr); break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unreachable code");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFCommunicateGhostCells(DM dm)
{
  DM_BF          *bf;
  DM_BF_Cell     *ghostCells;
  PetscErrorCode ierr;
  p4est_t        *p4est; //TODO deprecated
  p4est_ghost_t  *ghost; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  /* run ghost exchange */
  ghostCells = _p_cellGetPtrIndex(bf,p4est->local_num_quadrants);
  PetscStackCallP4est(p4est_ghost_exchange_data,(p4est,ghost,ghostCells));
  bf->ghostCellsSetUpCalled = PETSC_FALSE;
  ierr = DMBFSetUpGhostCells(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/***************
 * VIEWER
 ***************/

PetscErrorCode DMView_BF(DM dm, PetscViewer viewer)
{

  PetscBool      isvtk, ishdf5, isdraw, isglvis;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (!dm) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONG,"No DM provided to view");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS, &isglvis);CHKERRQ(ierr);
  if(isvtk) {
    ierr = DMBFVTKWriteAll((PetscObject)dm,viewer);CHKERRQ(ierr);
  } else if(ishdf5 || isdraw || isglvis) {
    SETERRQ(PetscObjectComm((PetscObject) dm),PETSC_ERR_SUP,"non-VTK viewer currently not supported by BF");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_BF(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscBool      isvtk, ishdf5, isdraw, isglvis;
  PetscErrorCode ierr;
  PetscInt       size,ng;
  PetscInt       vsize;
  PetscViewerVTKFieldType ft;
  Vec            locv;
  const char     *name;


  PetscFunctionBegin;
  ierr = VecGetDM(v,&dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)v),PETSC_ERR_ARG_WRONG,"Vector not generated from a DM");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS, &isglvis);CHKERRQ(ierr);
  if(isvtk) {
    /* create a copy and store it in the viewer */
    ierr = DMCreateLocalVector(dm,&locv);CHKERRQ(ierr);                /* we store local vectors in the viewer. done't know why, since we don't need ghost values */
    ierr = DMGlobalToLocal(dm,v,INSERT_VALUES,locv);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) locv, name);CHKERRQ(ierr);
    ierr = DMBFGetLocalSize(dm,&size);CHKERRQ(ierr);
    ierr = DMBFGetGhostSize(dm,&ng);CHKERRQ(ierr);
    ierr = VecGetSize(locv,&vsize);CHKERRQ(ierr);

    //if(vsize == P4EST_DIM*size)                { ft = PETSC_VTK_CELL_VECTOR_FIELD; } /* right now this is not actually supported (dm local to global is only for cell fields) */
    //else if(vsize == size)                     { ft = PETSC_VTK_CELL_FIELD;        }
    if(vsize == size + ng)                       { ft = PETSC_VTK_CELL_FIELD;        } /* if it's a local vector field, there will be an error before this in the dmlocaltoglobal */
    else  SETERRQ(PetscObjectComm((PetscObject)locv), PETSC_ERR_SUP, "Only vector and scalar cell fields currently supported");

    ierr = PetscViewerVTKAddField(viewer,(PetscObject)dm,DMBFVTKWriteAll,PETSC_DEFAULT,ft,PETSC_TRUE,(PetscObject)locv);CHKERRQ(ierr);
  } else if(ishdf5 || isdraw || isglvis) {
    SETERRQ(PetscObjectComm((PetscObject) dm),PETSC_ERR_SUP,"non-VTK viewer currently not supported by BF");
  }
  PetscFunctionReturn(0);
}


#endif /* defined(PETSC_HAVE_P4EST) */
