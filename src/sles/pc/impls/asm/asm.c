#ifndef lint
static char vcid[] = "$Id: asm.c,v 1.54 1997/02/12 00:37:23 curfman Exp curfman $";
#endif
/*
  This file defines an additive Schwarz preconditioner for any Mat implementation.

  Note that each processor may have any number of subdomains. But in order to 
  deal easily with the VecScatter(), we treat each processor as if it has the
  same number of subdomains.

       n - total number of true subdomains on all processors
       n_local_true - actual number of subdomains on this processor
       n_local = maximum over all processors of n_local_true
*/
#include "src/pc/pcimpl.h"     /*I "pc.h" I*/
#include "sles.h"

typedef struct {
  int        n,n_local,n_local_true;
  int        is_flg;              /* flg set to 1 if the IS created in pcsetup */
  int        overlap;             /* overlap requested by user */
  SLES       *sles;               /* linear solvers for each block */
  VecScatter *scat;               /* mapping to subregion */
  Vec        *x,*y;
  IS         *is;                 /* index set that defines each subdomain */
  Mat        *mat,*pmat;          /* mat is not currently used */
  PCASMType  type;                /* use reduced interpolation, restriction or both */
} PC_ASM;

#undef __FUNC__  
#define __FUNC__ "PCView_ASM"
static int PCView_ASM(PetscObject obj,Viewer viewer)
{
  PC           pc = (PC)obj;
  FILE         *fd;
  PC_ASM       *jac = (PC_ASM *) pc->data;
  int          rank, ierr;
  char         *cstring = 0;
  ViewerType   vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscFPrintf(pc->comm,fd,"    Additive Schwarz: total subdomain blocks = %d, amount of overlap = %d\n",jac->n,jac->overlap);
    if (jac->type == PC_ASM_NONE) cstring = "limited restriction and interpolation (PC_ASM_NONE)";
    else if (jac->type == PC_ASM_RESTRICT) cstring = "full restriction (PC_ASM_RESTRICT)";
    else if (jac->type == PC_ASM_INTERPOLATE) cstring = "full interpolation (PC_ASM_INTERPOLATE)";
    else if (jac->type == PC_ASM_BASIC) cstring = "full restriction and interpolation (PC_ASM_BASIC)";
    else cstring = "Unknown ASM type";
    PetscFPrintf(pc->comm,fd,"    Additive Schwarz: type - %s\n",cstring);
    MPI_Comm_rank(pc->comm,&rank);
    if (jac->sles) {ierr = SLESView(jac->sles[0],VIEWER_STDOUT_SELF); CHKERRQ(ierr);}
  } else if (vtype == STRING_VIEWER) {
    ViewerStringSPrintf(viewer," blks=%d, overlap=%d, type=%d",jac->n,jac->overlap,jac->type);
    if (jac->sles) {ierr = SLESView(jac->sles[0],viewer);}
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCSetUp_ASM"
static int PCSetUp_ASM(PC pc)
{
  PC_ASM              *osm  = (PC_ASM *) pc->data;
  int                 i,ierr,m,n_local = osm->n_local,n_local_true = osm->n_local_true;
  int                 start, start_val, end_val, size, sz, bs;
  MatGetSubMatrixCall scall = MAT_REUSE_MATRIX;
  IS                  isl;
  SLES                sles;
  KSP                 subksp;
  PC                  subpc;
  char                *prefix;

  if (pc->setupcalled == 0) {
    if (osm->n == PETSC_DECIDE && osm->n_local_true == PETSC_DECIDE) { 
      /* no subdomains given, use one per processor */
      osm->n_local_true = osm->n_local = 1;
      MPI_Comm_size(pc->comm,&size);
      osm->n = size;
    } else if (osm->n == PETSC_DECIDE) { /* determine global number of subdomains */
      MPI_Allreduce(&osm->n_local_true,&osm->n,1,MPI_INT,MPI_SUM,pc->comm);
      MPI_Allreduce(&osm->n_local_true,&osm->n_local,1,MPI_INT,MPI_MAX,pc->comm);
    }
    n_local      = osm->n_local;
    n_local_true = osm->n_local_true;  
    if (!osm->is){ /* build the index sets */
      osm->is    = (IS *) PetscMalloc( n_local_true*sizeof(IS **) );CHKPTRQ(osm->is);
      ierr  = MatGetOwnershipRange(pc->pmat,&start_val,&end_val); CHKERRQ(ierr);
      ierr  = MatGetBlockSize(pc->pmat,&bs); CHKERRQ(ierr);
      sz    = end_val - start_val;
      start = start_val;
      if (end_val/bs*bs != end_val || start_val/bs*bs != start_val) {
        SETERRQ(1,0,"Bad distribution for matrix block size");
      }
      for ( i=0; i<n_local_true; i++){
        size       =  ((sz/bs)/n_local_true + (( (sz/bs) % n_local_true) > i))*bs;
        ierr       =  ISCreateStride(MPI_COMM_SELF,size,start,1,&isl);CHKERRQ(ierr);
        start      += size;
        osm->is[i] =  isl;
      }
      osm->is_flg = PETSC_TRUE;
    }

    osm->sles = (SLES *) PetscMalloc(n_local_true*sizeof(SLES **));CHKPTRQ(osm->sles);
    osm->scat = (VecScatter *) PetscMalloc(n_local*sizeof(VecScatter **));CHKPTRQ(osm->scat);
    osm->x    = (Vec *) PetscMalloc(2*n_local*sizeof(Vec **)); CHKPTRQ(osm->x);
    osm->y    = osm->x + n_local;

    /*  Extend the "overlapping" regions by a number of steps  */
    ierr = MatIncreaseOverlap(pc->pmat,n_local_true,osm->is,osm->overlap); CHKERRQ(ierr);
    for (i=0; i< n_local_true; i++) {
      ierr = ISSort(osm->is[i]); CHKERRQ(ierr);
    }

    /* create the local work vectors and scatter contexts */
    for ( i=0; i<n_local_true; i++ ) {
      ierr = ISGetSize(osm->is[i],&m); CHKERRQ(ierr);
      ierr = VecCreateSeq(MPI_COMM_SELF,m,&osm->x[i]); CHKERRQ(ierr);
      ierr = VecDuplicate(osm->x[i],&osm->y[i]); CHKERRQ(ierr);
      ierr = ISCreateStride(MPI_COMM_SELF,m,0,1,&isl); CHKERRQ(ierr);
      ierr = VecScatterCreate(pc->vec,osm->is[i],osm->x[i],isl,&osm->scat[i]); CHKERRQ(ierr);
      ierr = ISDestroy(isl); CHKERRQ(ierr);
    }
    for ( i=n_local_true; i<n_local; i++ ) {
      ierr = VecCreateSeq(MPI_COMM_SELF,0,&osm->x[i]); CHKERRQ(ierr);
      ierr = VecDuplicate(osm->x[i],&osm->y[i]); CHKERRQ(ierr);
      ierr = ISCreateStride(MPI_COMM_SELF,0,0,1,&isl); CHKERRQ(ierr);
      ierr = VecScatterCreate(pc->vec,isl,osm->x[i],isl,&osm->scat[i]); CHKERRQ(ierr);
      ierr = ISDestroy(isl); CHKERRQ(ierr);   
    }

    /* 
       Create the local solvers.
    */
    for ( i=0; i<n_local_true; i++ ) {
      ierr = SLESCreate(MPI_COMM_SELF,&sles); CHKERRQ(ierr);
      PLogObjectParent(pc,sles);
      ierr = SLESGetKSP(sles,&subksp); CHKERRQ(ierr);
      ierr = KSPSetType(subksp,KSPPREONLY); CHKERRQ(ierr);
      ierr = SLESGetPC(sles,&subpc); CHKERRQ(ierr);
      ierr = PCSetType(subpc,PCILU); CHKERRQ(ierr);
      ierr = PCGetOptionsPrefix(pc,&prefix); CHKERRQ(ierr);
      ierr = SLESSetOptionsPrefix(sles,prefix); CHKERRQ(ierr);
      ierr = SLESAppendOptionsPrefix(sles,"sub_"); CHKERRQ(ierr);
      ierr = SLESSetFromOptions(sles); CHKERRQ(ierr);
      osm->sles[i] = sles;
    }
    scall = MAT_INITIAL_MATRIX;
  } else {
    /* 
       Destroy the blocks from the previous iteration
    */
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      ierr = MatDestroyMatrices(osm->n_local_true,&osm->pmat); CHKERRQ(ierr);
      scall = MAT_INITIAL_MATRIX;
    }
  }

  /* extract out the submatrices */
  ierr = MatGetSubMatrices(pc->pmat,osm->n_local_true,osm->is,osm->is,scall,&osm->pmat);CHKERRQ(ierr);

  /* Return control to the user so that the submatrices can be modified (e.g., to apply
     different boundary conditions for the submatrices than for the global problem) */
  ierr = PCModifySubMatrices(pc,osm->n_local,osm->is,osm->is,osm->pmat,pc->modifysubmatricesP);CHKERRQ(ierr);

  /* loop over subdomains putting them into local sles */
  for ( i=0; i<n_local_true; i++ ) {
    PLogObjectParent(pc,osm->pmat[i]);
    ierr = SLESSetOperators(osm->sles[i],osm->pmat[i],osm->pmat[i],pc->flag);CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCSetUpOnBlocks_ASM"
static int PCSetUpOnBlocks_ASM(PC pc)
{
  PC_ASM *osm = (PC_ASM *) pc->data;
  int    i,ierr;

  for ( i=0; i<osm->n_local_true; i++ ) {
    ierr = SLESSetUp(osm->sles[i],osm->x[i],osm->y[i]);CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCApply_ASM"
static int PCApply_ASM(PC pc,Vec x,Vec y)
{
  PC_ASM      *osm = (PC_ASM *) pc->data;
  int         i,n_local = osm->n_local,n_local_true = osm->n_local_true,ierr,its;
  Scalar      zero = 0.0;
  ScatterMode forward = SCATTER_FORWARD, reverse = SCATTER_REVERSE;

  /*
       Support for limiting the restriction or interpolation to only local 
     subdomain values (leaving the other values 0). 
  */
  if (!(osm->type & PC_ASM_RESTRICT)) {
    forward = SCATTER_FORWARD_LOCAL;
    /* have to zero the work RHS since scatter may leave some slots empty */
    for ( i=0; i<n_local; i++ ) {
      ierr = VecSet(&zero,osm->x[i]);CHKERRQ(ierr);
    }
  }
  if (!(osm->type & PC_ASM_INTERPOLATE)) {
    reverse = SCATTER_REVERSE_LOCAL;
  }


  for ( i=0; i<n_local; i++ ) {
    ierr = VecScatterBegin(x,osm->x[i],INSERT_VALUES,forward,osm->scat[i]);CHKERRQ(ierr);
  }
  ierr = VecSet(&zero,y); CHKERRQ(ierr);
  /* do the local solves */
  for ( i=0; i<n_local_true; i++ ) {
    ierr = VecScatterEnd(x,osm->x[i],INSERT_VALUES,forward,osm->scat[i]);CHKERRQ(ierr);
    ierr = SLESSolve(osm->sles[i],osm->x[i],osm->y[i],&its);CHKERRQ(ierr); 
    ierr = VecScatterBegin(osm->y[i],y,ADD_VALUES,reverse,osm->scat[i]);CHKERRQ(ierr);
  }
  /* handle the rest of the scatters that do not have local solves */
  for ( i=n_local_true; i<n_local; i++ ) {
    ierr = VecScatterEnd(x,osm->x[i],INSERT_VALUES,forward,osm->scat[i]);CHKERRQ(ierr);
    ierr = VecScatterBegin(osm->y[i],y,ADD_VALUES,reverse,osm->scat[i]);CHKERRQ(ierr);
  }
  for ( i=0; i<n_local; i++ ) {
    ierr = VecScatterEnd(osm->y[i],y,ADD_VALUES,reverse,osm->scat[i]);CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCDestroy_ASM"
static int PCDestroy_ASM(PetscObject obj)
{
  PC     pc = (PC) obj;
  PC_ASM *osm = (PC_ASM *) pc->data;
  int    i, ierr;

  for ( i=0; i<osm->n_local; i++ ) {
    ierr = VecScatterDestroy(osm->scat[i]);
    ierr = VecDestroy(osm->x[i]);
    ierr = VecDestroy(osm->y[i]);
  }
  if (osm->n_local_true > 0) {
    ierr = MatDestroyMatrices(osm->n_local_true,&osm->pmat); CHKERRQ(ierr);
  }
  for ( i=0; i<osm->n_local_true; i++ ) {
    ierr = SLESDestroy(osm->sles[i]);
  }
  if (osm->is_flg) {
     for ( i=0; i<osm->n_local_true; i++ ) ISDestroy(osm->is[i]);
     PetscFree(osm->is);
  }
  if (osm->sles) PetscFree(osm->sles);
  if (osm->scat) PetscFree(osm->scat);
  if (osm->x) PetscFree(osm->x);
  PetscFree(osm);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp_ASM"
static int PCPrintHelp_ASM(PC pc,char *p)
{
  PetscPrintf(pc->comm," Options for PCASM preconditioner:\n");
  PetscPrintf(pc->comm," %spc_asm_blocks <blks>: total subdomain blocks\n",p);
  PetscPrintf(pc->comm, " %spc_asm_overlap <ovl>: amount of overlap between subdomains, defaults to 1\n",p); 
  PetscPrintf(pc->comm, " %spc_asm_type <basic,restrict,interpolate,none>: type of restriction/interpolation\n",p); 
  PetscPrintf(pc->comm," %ssub : prefix to control options for individual blocks.\
 Add before the \n      usual KSP and PC option names (e.g., %ssub_ksp_type\
 <method>)\n",p,p);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions_ASM"
static int PCSetFromOptions_ASM(PC pc)
{
  int       blocks,flg, ovl,ierr;
  char      buff[16];

  ierr = OptionsGetInt(pc->prefix,"-pc_asm_blocks",&blocks,&flg); CHKERRQ(ierr);
  if (flg) {ierr = PCASMSetTotalSubdomains(pc,blocks,PETSC_NULL); CHKERRQ(ierr); }
  ierr = OptionsGetInt(pc->prefix,"-pc_asm_overlap", &ovl, &flg); CHKERRQ(ierr);
  if (flg) {ierr = PCASMSetOverlap( pc, ovl); CHKERRQ(ierr); }
  ierr = OptionsGetString(pc->prefix,"-pc_asm_type",buff,15,&flg);CHKERRQ(ierr);
  if (flg) {
    PCASMType type = PC_ASM_RESTRICT;
    if (!PetscStrcmp(buff,"basic"))            type = PC_ASM_BASIC;
    else if (!PetscStrcmp(buff,"restrict"))    type = PC_ASM_RESTRICT;
    else if (!PetscStrcmp(buff,"interpolate")) type = PC_ASM_INTERPOLATE;
    else if (!PetscStrcmp(buff,"none"))        type = PC_ASM_NONE;
    else SETERRQ(1,0,"Unknown type");
    ierr = PCASMSetType(pc,type); CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCCreate_ASM"
int PCCreate_ASM(PC pc)
{
  PC_ASM *osm = PetscNew(PC_ASM); CHKPTRQ(osm);

  PetscMemzero(osm,sizeof(PC_ASM)); 
  osm->n            = PETSC_DECIDE;
  osm->n_local_true = PETSC_DECIDE;
  osm->overlap      = 1;
  osm->is_flg       = PETSC_FALSE;
  osm->sles         = 0;
  osm->type         = PC_ASM_RESTRICT;

  pc->apply         = PCApply_ASM;
  pc->setup         = PCSetUp_ASM;
  pc->destroy       = PCDestroy_ASM;
  pc->type          = PCASM;
  pc->printhelp     = PCPrintHelp_ASM;
  pc->setfrom       = PCSetFromOptions_ASM;
  pc->setuponblocks = PCSetUpOnBlocks_ASM;
  pc->data          = (void *) osm;
  pc->view          = PCView_ASM;
  pc->applyrich     = 0;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCASMSetLocalSubdomains"
/*@
    PCASMSetLocalSubdomains - Sets the local subdomains (for this processor
    only) for the additive Schwarz preconditioner.  Either all or no
    processors in the PC communicator must call this routine.

    Input Parameters:
.   pc - the preconditioner context
.   n - the number of subdomains for this processor (default value = 1)
.   is - the index sets that define the subdomains for this processor
         (or PETSC_NULL for PETSc to determine subdomains)

    Note:
    Use PCASMSetTotalSubdomains() to set the subdomains for all processors.

.keywords: PC, ASM, set, local, subdomains, additive Schwarz

.seealso: PCASMSetTotalSubdomains(), PCASMSetOverlap(), PCASMGetSubSLES(),
          PCASMCreateSubdomains2D()
@*/
int PCASMSetLocalSubdomains(PC pc, int n, IS *is)
{
  PC_ASM *osm;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCASM) return 0;  
  if (n <= 0) SETERRQ(1,0,"Each process must have 1+ blocks");
  osm               = (PC_ASM *) pc->data;
  osm->n_local_true = n;
  osm->is           = is;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCASMSetTotalSubdomains"
/*@
    PCASMSetTotalSubdomains - Sets the subdomains for all processor for the 
    additive Schwarz preconditioner.  Either all or no processors in the
    PC communicator must call this routine, with the same index sets.

    Input Parameters:
.   pc - the preconditioner context
.   n - the number of subdomains for all processors
.   is - the index sets that define the subdomains for all processor
         (or PETSC_NULL for PETSc to determine subdomains)

    Options Database Key:
    To set the total number of subdomain blocks rather than specify the
    index sets, use the option
$    -pc_asm_blocks <blks>

    Note:
    Use PCASMSetLocalSubdomains() to set local subdomains.

.keywords: PC, ASM, set, total, global, subdomains, additive Schwarz

.seealso: PCASMSetLocalSubdomains(), PCASMSetOverlap(), PCASMGetSubSLES(),
          PCASMCreateSubdomains2D()
@*/
int PCASMSetTotalSubdomains(PC pc, int N, IS *is)
{
  PC_ASM *osm;
  int    rank,size;

  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCASM) return 0;  

  if (is) SETERRQ(1,0,"Use PCASMSetLocalSubdomains to \
set specific index sets\n they cannot be set globally yet.");

  osm               = (PC_ASM *) pc->data;
  /*
     Split the subdomains equally amoung all processors 
  */
  MPI_Comm_rank(pc->comm,&rank);
  MPI_Comm_size(pc->comm,&size);
  osm->n_local_true = N/size + ((N % size) > rank);
  if (osm->n_local_true <= 0) 
    SETERRQ(1,0,"Each process must have 1+ blocks");
  osm->is           = 0;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCASMSetOverlap"
/*@
    PCASMSetOverlap - Sets the overlap between a pair of subdomains for the
    additive Schwarz preconditioner.  Either all or no processors in the
    PC communicator must call this routine. 

    Input Parameters:
.   pc  - the preconditioner context
.   ovl - the amount of overlap between subdomains (ovl >= 0, default value = 1)

    Options Database Key:
$   -pc_asm_overlap <ovl>

.keywords: PC, ASM, set, overlap

.seealso: PCASMSetTotalSubdomains(), PCASMSetTotalSubdomains(), PCASMGetSubSLES(),
          PCASMCreateSubdomains2D()
@*/
int PCASMSetOverlap(PC pc, int ovl)
{
  PC_ASM *osm;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCASM) return 0;  
  if (ovl < 0 ) SETERRQ(1,0,"Negative overlap value used");

  osm               = (PC_ASM *) pc->data;
  osm->overlap      = ovl;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCASMSetOverlap"
/*@
    PCASMSetType - Sets the type of restriction and interpolation used
    for local problems in the additive Schwarz method.

    Input Parameters:
.   pc  - the preconditioner context
.   type - variant of ASM
$      PC_ASM_BASIC       - full interpolation and restriction
$      PC_ASM_RESTRICT    - full restriction, local processor interpolation
$      PC_ASM_INTERPOLATE - full interpolation, local processor restriction
$      PC_ASM_NONE        - local processor restriction and interpolation

    Options Database Key:
$   -pc_asm_type [basic,restrict,interpolate,none]

.keywords: PC, ASM, set, type

.seealso: PCASMSetTotalSubdomains(), PCASMSetTotalSubdomains(), PCASMGetSubSLES(),
          PCASMCreateSubdomains2D()
@*/
int PCASMSetType(PC pc,PCASMType type)
{
  PC_ASM *osm;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCASM) return 0;  

  osm        = (PC_ASM *) pc->data;
  osm->type  = type;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCASMCreateSubdomains2D"
/*@
   PCASMCreateSubdomains2D - Creates the index sets for the overlapping Schwarz 
   preconditioner for a two-dimensional problem on a regular grid.

   Input Parameters:
.  m, n - the number of mesh points in the x and y directions
.  M, N - the number of subdomains in the x and y directions
.  dof - degrees of freedom per node
.  overlap - overlap in mesh lines

   Output Parameters:
.  Nsub - the number of subdomains created
.  is - the array of index sets defining the subdomains

   Note:
   Presently PCAMSCreateSubdomains2d() is valid only for sequential
   preconditioners.  More general related routines are
   PCASMSetTotalSubdomains() and PCASMSetLocalSubdomains().

.keywords: PC, ASM, additive Schwarz, create, subdomains, 2D, regular grid

.seealso: PCASMSetTotalSubdomains(), PCASMSetLocalSubdomains(), PCASMGetSubSLES(),
          PCASMSetOverlap()
@*/
int PCASMCreateSubdomains2D(int m,int n,int M,int N,int dof,int overlap,int *Nsub,IS **is)
{
  int i,j, height,width,ystart,xstart,yleft,yright,xleft,xright,loc_outter;
  int nidx,*idx,loc,ii,jj,ierr,count;

  if (dof != 1) SETERRQ(PETSC_ERR_SUP,0,"");

  *Nsub = N*M;
  *is = (IS *) PetscMalloc( (*Nsub)*sizeof(IS **) ); CHKPTRQ(is);
  ystart = 0;
  loc_outter = 0;
  for ( i=0; i<N; i++ ) {
    height = n/N + ((n % N) > i); /* height of subdomain */
    if (height < 2) SETERRA(1,0,"Too many N subdomains for mesh dimension n");
    yleft  = ystart - overlap; if (yleft < 0) yleft = 0;
    yright = ystart + height + overlap; if (yright > n) yright = n;
    xstart = 0;
    for ( j=0; j<M; j++ ) {
      width = m/M + ((m % M) > j); /* width of subdomain */
      if (width < 2) SETERRA(1,0,"Too many M subdomains for mesh dimension m");
      xleft  = xstart - overlap; if (xleft < 0) xleft = 0;
      xright = xstart + width + overlap; if (xright > m) xright = m;
      /*            
       printf("subdomain %d %d xstart %d end %d ystart %d end %d\n",i,j,xleft,xright,
              yleft,yright);
      */
      nidx   = (xright - xleft)*(yright - yleft);
      idx    = (int *) PetscMalloc( nidx*sizeof(int) ); CHKPTRQ(idx);
      loc    = 0;
      for ( ii=yleft; ii<yright; ii++ ) {
        count = m*ii + xleft;
        for ( jj=xleft; jj<xright; jj++ ) {
          idx[loc++] = count++;
        }
      }
      ierr = ISCreateGeneral(MPI_COMM_SELF,nidx,idx,(*is)+loc_outter++); CHKERRQ(ierr);
      PetscFree(idx);
      /* ISView((*is)[loc_outter-1],0); */
      xstart += width;
    }
    ystart += height;
  }
  for ( i=0; i<*Nsub; i++ ) { ierr = ISSort((*is)[i]); CHKERRQ(ierr); }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PCASMGetSubSLES"
/*@C
   PCASMGetSubSLES - Gets the local SLES contexts for all blocks on
   this processor.
   
   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
.  n_local - the number of blocks on this processor
.  first_local - the global number of the first block on this processor
.  sles - the array of SLES contexts

   Note:  
   Currently for some matrix implementations only 1 block per processor 
   is supported.
   
   You must call SLESSetUp() before calling PCASMGetSubSLES().

.keywords: PC, ASM, additive Schwarz, get, sub, SLES, context

.seealso: PCASMSetTotalSubdomains(), PCASMSetTotalSubdomains(), PCASMSetOverlap(),
          PCASMCreateSubdomains2D(),
@*/
int PCASMGetSubSLES(PC pc,int *n_local,int *first_local,SLES **sles)
{
  PC_ASM   *jac;

  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCASM) return 0;
  if (!pc->setupcalled) SETERRQ(1,0,"Must call SLESSetUp first");
  jac = (PC_ASM *) pc->data;
  *n_local     = jac->n_local_true;
  *first_local = -1; /* need to determine global number of local blocks*/
  *sles        = jac->sles;
  return 0;
}
