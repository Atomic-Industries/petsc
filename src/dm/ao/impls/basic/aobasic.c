
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aobasic.c,v 1.35 1998/04/13 18:02:03 bsmith Exp bsmith $";
#endif

/*
    The most basic AO application ordering routines. These store the 
  entire orderings on each processor.
*/

#include "src/ao/aoimpl.h"
#include "pinclude/pviewer.h"
#include "sys.h"

typedef struct {
  int N;
  int *app,*petsc;  /* app[i] is the partner for the ith PETSc slot */
                    /* petsc[j] is the partner for the jth app slot */
} AO_Basic;

#undef __FUNC__  
#define __FUNC__ "AOBasicGetIndices_Private" 
int AOBasicGetIndices_Private(AO ao,int **app,int **petsc)
{
  AO_Basic *basic = (AO_Basic *) ao->data;

  PetscFunctionBegin;
  if (app)   *app   = basic->app;
  if (petsc) *petsc = basic->petsc;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODestroy_Basic" 
int AODestroy_Basic(AO ao)
{
  AO_Basic *aodebug = (AO_Basic *) ao->data;

  PetscFunctionBegin;
  PetscFree(aodebug->app);
  PetscFree(ao->data); 
  PLogObjectDestroy(ao);
  PetscHeaderDestroy(ao);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AOView_Basic" 
int AOView_Basic(AO ao,Viewer viewer)
{
  int         rank,ierr,i;
  ViewerType  vtype;
  FILE        *fd;
  AO_Basic    *aodebug = (AO_Basic*) ao->data;

  PetscFunctionBegin;
  MPI_Comm_rank(ao->comm,&rank); if (rank) PetscFunctionReturn(0);

  if (!viewer) {
    viewer = VIEWER_STDOUT_SELF; 
  }

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) { 
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    fprintf(fd,"Number of elements in ordering %d\n",aodebug->N);
    fprintf(fd,"   App.   PETSc\n");
    for ( i=0; i<aodebug->N; i++ ) {
      fprintf(fd,"%d   %d    %d\n",i,aodebug->app[i],aodebug->petsc[i]);
    }
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AOPetscToApplication_Basic"  
int AOPetscToApplication_Basic(AO ao,int n,int *ia)
{
  int      i;
  AO_Basic *aodebug = (AO_Basic *) ao->data;

  PetscFunctionBegin;
  for ( i=0; i<n; i++ ) {
    if (ia[i] >= 0) {ia[i] = aodebug->app[ia[i]];}
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AOApplicationToPetsc_Basic" 
int AOApplicationToPetsc_Basic(AO ao,int n,int *ia)
{
  int      i;
  AO_Basic *aodebug = (AO_Basic *) ao->data;

  PetscFunctionBegin;
  for ( i=0; i<n; i++ ) {
    if (ia[i] >= 0) {ia[i] = aodebug->petsc[ia[i]];}
  }
  PetscFunctionReturn(0);
}

static struct _AOOps myops = {AOPetscToApplication_Basic,
                              AOApplicationToPetsc_Basic};

#undef __FUNC__  
#define __FUNC__ "AOCreateBasic" 
/*@C
   AOCreateBasic - Creates a basic application ordering using two integer arrays.

   Input Parameters:
+  comm - MPI communicator that is to share AO
.  napp - size of integer arrays
.  myapp - integer array that defines an ordering
-  mypetsc - integer array that defines another ordering

   Output Parameter:
.  aoout - the new application ordering

   Collective on MPI_Comm

   Options Database Key:
.   -ao_view - call AOView() at the conclusion of AOCreateBasic()

.keywords: AO, create

.seealso: AOCreateBasicIS(), AODestroy()
@*/
int AOCreateBasic(MPI_Comm comm,int napp,int *myapp,int *mypetsc,AO *aoout)
{
  AO_Basic  *aodebug;
  AO        ao;
  int       *lens,size,rank,N,i,flg1,ierr,*petsc,start;
  int       *allpetsc,*allapp,*disp,ip,ia;

  PetscFunctionBegin;
  *aoout = 0;
  PetscHeaderCreate(ao, _p_AO,struct _AOOps,AO_COOKIE,AO_BASIC,comm,AODestroy,AOView); 
  PLogObjectCreate(ao);
  aodebug            = PetscNew(AO_Basic);
  PLogObjectMemory(ao,sizeof(struct _p_AO) + sizeof(AO_Basic));

  PetscMemcpy(ao->ops,&myops,sizeof(myops));
  ao->ops->destroy = AODestroy_Basic;
  ao->ops->view    = AOView_Basic;
  ao->data    = (void *)aodebug;

  /* transmit all lengths to all processors */
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
  lens = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(lens);
  disp = lens + size;
  ierr = MPI_Allgather(&napp,1,MPI_INT,lens,1,MPI_INT,comm);CHKERRQ(ierr);
  N =  0;
  for ( i=0; i<size; i++ ) {
    disp[i] = N;
    N += lens[i];
  }
  aodebug->N = N;

  /*
     If mypetsc is 0 then use "natural" numbering 
  */
  if (!mypetsc) {
    start = disp[rank];
    petsc = (int *) PetscMalloc((napp+1)*sizeof(int));CHKPTRQ(petsc);
    for ( i=0; i<napp; i++ ) {
      petsc[i] = start + i;
    }
  } else {
    petsc = mypetsc;
  }

  /* get all indices on all processors */
  allpetsc = (int *) PetscMalloc( 2*N*sizeof(int) ); CHKPTRQ(allpetsc);
  allapp   = allpetsc + N;
  ierr = MPI_Allgatherv(petsc,napp,MPI_INT,allpetsc,lens,disp,MPI_INT,comm);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(myapp,napp,MPI_INT,allapp,lens,disp,MPI_INT,comm);CHKERRQ(ierr);
  PetscFree(lens);

  /* generate a list of application and PETSc node numbers */
  aodebug->app = (int *) PetscMalloc(2*N*sizeof(int));CHKPTRQ(aodebug->app);
  PLogObjectMemory(ao,2*N*sizeof(int));
  aodebug->petsc = aodebug->app + N;
  PetscMemzero(aodebug->app,2*N*sizeof(int));
  for ( i=0; i<N; i++ ) {
    ip = allpetsc[i]; ia = allapp[i];
    /* check there are no duplicates */
    if (aodebug->app[ip]) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Duplicate in Application ordering");
    aodebug->app[ip] = ia + 1;
    if (aodebug->petsc[ia]) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Duplicate in PETSc ordering");
    aodebug->petsc[ia] = ip + 1;
  }
  if (!mypetsc) PetscFree(petsc);
  PetscFree(allpetsc);
  /* shift indices down by one */
  for ( i=0; i<N; i++ ) {
    aodebug->app[i]--;
    aodebug->petsc[i]--;
  }

  ierr = OptionsHasName(PETSC_NULL,"-ao_view",&flg1); CHKERRQ(ierr);
  if (flg1) {ierr = AOView(ao,VIEWER_STDOUT_SELF); CHKERRQ(ierr);}

  *aoout = ao; PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AOCreateBasicIS" 
/*@C
   AOCreateBasicIS - Creates a basic application ordering using two index sets.

   Input Parameters:
+  isapp - index set that defines an ordering
-  ispetsc - index set that defines another ordering (may be PETSC_NULL to use the
             natural ordering)

   Output Parameter:
.  aoout - the new application ordering

   Collective on IS

   Options Database Key:
-   -ao_view - call AOView() at the conclusion of AOCreateBasicIS()

.keywords: AO, create

.seealso: AOCreateBasic(),  AODestroy()
@*/
int AOCreateBasicIS(IS isapp,IS ispetsc,AO *aoout)
{
  int       *mypetsc = 0,*myapp,ierr,napp,npetsc;
  MPI_Comm  comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)isapp,&comm);CHKERRQ(ierr);
  ierr = ISGetSize(isapp,&napp); CHKERRQ(ierr);
  if (ispetsc) {
    ierr = ISGetSize(ispetsc,&npetsc); CHKERRQ(ierr);
    if (napp != npetsc) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Local IS lengths must match");
    ierr = ISGetIndices(ispetsc,&mypetsc); CHKERRQ(ierr);
  }
  ierr = ISGetIndices(isapp,&myapp); CHKERRQ(ierr);

  ierr = AOCreateBasic(comm,napp,myapp,mypetsc,aoout); CHKERRQ(ierr);

  ierr = ISRestoreIndices(isapp,&myapp); CHKERRQ(ierr);
  if (ispetsc) {
    ierr = ISRestoreIndices(ispetsc,&mypetsc); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

