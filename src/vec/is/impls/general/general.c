#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: general.c,v 1.67 1998/03/12 23:14:55 bsmith Exp bsmith $";
#endif
/*
     Provides the functions for index sets (IS) defined by a list of integers.
*/
#include "src/is/isimpl.h"
#include "pinclude/pviewer.h"
#include "sys.h"

typedef struct {
  int n;         /* number of indices */ 
  int sorted;    /* indicates the indices are sorted */ 
  int *idx;
} IS_General;

#undef __FUNC__  
#define __FUNC__ "ISDuplicate_General" 
int ISDuplicate_General(IS is, IS *newIS)
{
  int ierr;
  IS_General *sub = (IS_General *)is->data;

  PetscFunctionBegin;
  ierr = ISCreateGeneral(is->comm, sub->n, sub->idx, newIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISDestroy_General" 
int ISDestroy_General(IS is)
{
  IS_General *is_general = (IS_General *) is->data;

  PetscFunctionBegin;
  PetscFree(is_general->idx);
  PetscFree(is_general); 
  PLogObjectDestroy(is);
  PetscHeaderDestroy(is); PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISGetIndices_General" 
int ISGetIndices_General(IS in,int **idx)
{
  IS_General *sub = (IS_General *) in->data;

  PetscFunctionBegin;
  *idx = sub->idx; PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISRestoreIndices_General" 
int ISRestoreIndices_General(IS in,int **idx)
{
  IS_General *sub = (IS_General *) in->data;

  PetscFunctionBegin;
  if (*idx != sub->idx ) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must restore with value from ISGetIndices()");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISGetSize_General" 
int ISGetSize_General(IS is,int *size)
{
  IS_General *sub = (IS_General *)is->data;

  PetscFunctionBegin;
  *size = sub->n; PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISInvertPermutation_General" 
int ISInvertPermutation_General(IS is, IS *isout)
{
  IS_General *sub = (IS_General *)is->data;
  int        i,ierr, *ii,n = sub->n,*idx = sub->idx;

  PetscFunctionBegin;
  ii = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(ii);
  for ( i=0; i<n; i++ ) {
    ii[idx[i]] = i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,ii,isout); CHKERRQ(ierr);
  ISSetPermutation(*isout);
  PetscFree(ii);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISView_General" 
int ISView_General(IS is, Viewer viewer)
{
  IS_General  *sub = (IS_General *)is->data;
  int         i,n = sub->n,*idx = sub->idx,ierr;
  FILE        *fd;
  ViewerType  vtype;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (is->isperm) {
      fprintf(fd,"Index set is permutation\n");
    }
    fprintf(fd,"Number of indices in set %d\n",n);
    for ( i=0; i<n; i++ ) {
      fprintf(fd,"%d %d\n",i,idx[i]);
    }
    fflush(fd);
  } else if (vtype  == ASCII_FILES_VIEWER) {
    MPI_Comm comm;
    int      rank;
    ierr = PetscObjectGetComm((PetscObject)viewer,&comm); CHKERRQ(ierr);
    MPI_Comm_rank(comm,&rank);
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (is->isperm) {
      PetscSynchronizedFPrintf(comm,fd,"[%d] Index set is permutation\n",rank);
    }
    PetscSynchronizedFPrintf(comm,fd,"[%d] Number of indices in set %d\n",rank,n);
    for ( i=0; i<n; i++ ) {
      PetscSynchronizedFPrintf(comm,fd,"[%d] %d %d\n",rank,i,idx[i]);
    }
    PetscSynchronizedFlush(comm);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISSort_General" 
int ISSort_General(IS is)
{
  IS_General *sub = (IS_General *)is->data;
  int        ierr;

  PetscFunctionBegin;
  if (sub->sorted) PetscFunctionReturn(0);
  ierr = PetscSortInt(sub->n, sub->idx); CHKERRQ(ierr);
  sub->sorted = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISSorted_General" 
int ISSorted_General(IS is, PetscTruth *flg)
{
  IS_General *sub = (IS_General *)is->data;

  PetscFunctionBegin;
  *flg = (PetscTruth) sub->sorted;
  PetscFunctionReturn(0);
}

static struct _ISOps myops = { ISGetSize_General,
                               ISGetSize_General,
                               ISGetIndices_General,
                               ISRestoreIndices_General,
                               ISInvertPermutation_General,
                               ISSort_General,
                               ISSorted_General,
                               ISDuplicate_General };

#undef __FUNC__  
#define __FUNC__ "ISCreateGeneral" 
/*@C
   ISCreateGeneral - Creates a data structure for an index set 
   containing a list of integers.

   Input Parameters:
.  n - the length of the index set
.  idx - the list of integers
.  comm - the MPI communicator

   Output Parameter:
.  is - the new index set

.keywords: IS, general, index set, create

.seealso: ISCreateStride(), ISCreateBlock()
@*/
int ISCreateGeneral(MPI_Comm comm,int n,int *idx,IS *is)
{
  int        i, sorted = 1, min, max, flg, ierr;
  IS         Nindex;
  IS_General *sub;

  PetscFunctionBegin;
  PetscValidPointer(is);
  if (n < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"length < 0");
  if (n) {PetscValidIntPointer(idx);}

  *is = 0;
  PetscHeaderCreate(Nindex, _p_IS,struct _ISOps,IS_COOKIE,IS_GENERAL,comm,ISDestroy,ISView); 
  PLogObjectCreate(Nindex);
  sub            = PetscNew(IS_General); CHKPTRQ(sub);
  PLogObjectMemory(Nindex,sizeof(IS_General)+n*sizeof(int)+sizeof(struct _p_IS));
  sub->idx       = (int *) PetscMalloc((n+1)*sizeof(int));CHKPTRQ(sub->idx);
  sub->n         = n;
  for ( i=1; i<n; i++ ) {
    if (idx[i] < idx[i-1]) {sorted = 0; break;}
  }
  if (n) {min = max = idx[0];} else {min = max = 0;}
  for ( i=1; i<n; i++ ) {
    if (idx[i] < min) min = idx[i];
    if (idx[i] > max) max = idx[i];
  }
  PetscMemcpy(sub->idx,idx,n*sizeof(int));
  sub->sorted     = sorted;
  Nindex->min     = min;
  Nindex->max     = max;
  Nindex->data    = (void *) sub;
  PetscMemcpy(Nindex->ops,&myops,sizeof(myops));
  Nindex->ops->destroy = ISDestroy_General;
  Nindex->ops->view    = ISView_General;
  Nindex->isperm  = 0;
  ierr = OptionsHasName(PETSC_NULL,"-is_view",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = ISView(Nindex,VIEWER_STDOUT_(Nindex->comm)); CHKERRQ(ierr);
  }
  *is = Nindex; PetscFunctionReturn(0);
}




