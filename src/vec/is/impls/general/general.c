#ifndef lint
static char vcid[] = "$Id: general.c,v 1.54 1997/01/01 03:35:05 bsmith Exp balay $";
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
#define __FUNC__ "ISDestroy_General"
static int ISDestroy_General(PetscObject obj)
{
  IS         is = (IS) obj;
  IS_General *is_general = (IS_General *) is->data;

  PetscFree(is_general->idx);
  PetscFree(is_general); 
  PLogObjectDestroy(is);
  PetscHeaderDestroy(is); return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISGetIndices_General"
static int ISGetIndices_General(IS in,int **idx)
{
  IS_General *sub = (IS_General *) in->data;
  *idx = sub->idx; return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISRestoreIndices_General"
static int ISRestoreIndices_General(IS in,int **idx)
{
  IS_General *sub = (IS_General *) in->data;
  if (*idx != sub->idx ) {
    SETERRQ(1,0,"Must restore with value from ISGetIndices()");
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISGetSize_General"
static int ISGetSize_General(IS is,int *size)
{
  IS_General *sub = (IS_General *)is->data;
  *size = sub->n; return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISInvertPermutation_General"
static int ISInvertPermutation_General(IS is, IS *isout)
{
  IS_General *sub = (IS_General *)is->data;
  int        i,ierr, *ii,n = sub->n,*idx = sub->idx;

  ii = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(ii);
  for ( i=0; i<n; i++ ) {
    ii[idx[i]] = i;
  }
  ierr = ISCreateGeneral(MPI_COMM_SELF,n,ii,isout); CHKERRQ(ierr);
  ISSetPermutation(*isout);
  PetscFree(ii);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISView_General"
static int ISView_General(PetscObject obj, Viewer viewer)
{
  IS          is = (IS) obj;
  IS_General  *sub = (IS_General *)is->data;
  int         i,n = sub->n,*idx = sub->idx,ierr;
  FILE        *fd;
  ViewerType  vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) { 
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (is->isperm) {
      fprintf(fd,"Index set is permutation\n");
    }
    fprintf(fd,"Number of indices in set %d\n",n);
    for ( i=0; i<n; i++ ) {
      fprintf(fd,"%d %d\n",i,idx[i]);
    }
    fflush(fd);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISSort_General"
static int ISSort_General(IS is)
{
  IS_General *sub = (IS_General *)is->data;
  int        ierr;

  if (sub->sorted) return 0;
  ierr = PetscSortInt(sub->n, sub->idx); CHKERRQ(ierr);
  sub->sorted = 1;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISSorted_General"
static int ISSorted_General(IS is, PetscTruth *flg)
{
  IS_General *sub = (IS_General *)is->data;
  *flg = (PetscTruth) sub->sorted;
  return 0;
}

static struct _ISOps myops = { ISGetSize_General,
                               ISGetSize_General,
                               ISGetIndices_General,
                               ISRestoreIndices_General,
                               ISInvertPermutation_General,
                               ISSort_General,
                               ISSorted_General };

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

  PetscValidPointer(is);
  if (n < 0) SETERRQ(1,0,"length < 0");
  if (n) {PetscValidIntPointer(idx);}

  *is = 0;
  PetscHeaderCreate(Nindex, _IS,IS_COOKIE,IS_GENERAL,comm); 
  PLogObjectCreate(Nindex);
  sub            = PetscNew(IS_General); CHKPTRQ(sub);
  PLogObjectMemory(Nindex,sizeof(IS_General)+n*sizeof(int)+sizeof(struct _IS));
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
  PetscMemcpy(&Nindex->ops,&myops,sizeof(myops));
  Nindex->destroy = ISDestroy_General;
  Nindex->view    = ISView_General;
  Nindex->isperm  = 0;
  ierr = OptionsHasName(PETSC_NULL,"-is_view",&flg); CHKERRQ(ierr);
  if (flg) {
    Viewer viewer;
    ierr = ViewerFileOpenASCII(comm,"stdout",&viewer);CHKERRQ(ierr);
    ierr = ISView(Nindex,viewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
  }
  *is = Nindex; return 0;
}




