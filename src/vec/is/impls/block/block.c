#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: block.c,v 1.17 1997/07/09 20:49:17 balay Exp bsmith $";
#endif
/*
     Provides the functions for index sets (IS) defined by a list of integers.
   These are for blocks of data, each block is indicated with a single integer.
*/
#include "src/is/isimpl.h"               /*I  "is.h"     I*/
#include "pinclude/pviewer.h"
#include "sys.h"

typedef struct {
  int n;            /* number of blocks */
  int sorted;       /* are the blocks sorted? */
  int *idx;
  int bs;           /* blocksize */
} IS_Block;

#undef __FUNC__  
#define __FUNC__ "ISDestroy_Block" 
int ISDestroy_Block(PetscObject obj)
{
  IS       is = (IS) obj;
  IS_Block *is_block = (IS_Block *) is->data;
  PetscFree(is_block->idx); 
  PetscFree(is_block); 
  PLogObjectDestroy(is);
  PetscHeaderDestroy(is); return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISGetIndices_Block" 
int ISGetIndices_Block(IS in,int **idx)
{
  IS_Block *sub = (IS_Block *) in->data;
  int      i,j,k,bs = sub->bs,n = sub->n,*ii,*jj;

  if (sub->bs == 1) {
    *idx = sub->idx; 
  } else {
    jj   = (int *) PetscMalloc(sub->bs*(1+sub->n)*sizeof(int));CHKPTRQ(jj)
    *idx = jj;
    k    = 0;
    ii   = sub->idx;
    for ( i=0; i<n; i++ ) {
      for ( j=0; j<bs; j++ ) {
        jj[k++] = ii[i] + j;
      }
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISRestoreIndices_Block" 
int ISRestoreIndices_Block(IS in,int **idx)
{
  IS_Block *sub = (IS_Block *) in->data;

  if (sub->bs != 1) {
    PetscFree(*idx);
  } else {
    if (*idx !=  sub->idx) {
      SETERRQ(1,0,"Must restore with value from ISGetIndices()");
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISGetSize_Block" 
int ISGetSize_Block(IS is,int *size)
{
  IS_Block *sub = (IS_Block *)is->data;
  *size = sub->bs*sub->n; 
  return 0;
}


#undef __FUNC__  
#define __FUNC__ "ISInvertPermutation_Block" 
int ISInvertPermutation_Block(IS is, IS *isout)
{
  IS_Block *sub = (IS_Block *)is->data;
  int      i,ierr, *ii,n = sub->n,*idx = sub->idx;

  ii = (int *) PetscMalloc( (n+1)*sizeof(int) ); CHKPTRQ(ii);
  for ( i=0; i<n; i++ ) {
    ii[idx[i]] = i;
  }
  ierr = ISCreateBlock(PETSC_COMM_SELF,sub->bs,n,ii,isout); CHKERRQ(ierr);
  ISSetPermutation(*isout);
  PetscFree(ii);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISView_Block" 
int ISView_Block(PetscObject obj, Viewer viewer)
{
  IS          is = (IS) obj;
  IS_Block    *sub = (IS_Block *)is->data;
  int         i,n = sub->n,*idx = sub->idx,ierr;
  FILE        *fd;
  ViewerType  vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) { 
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (is->isperm) {
      fprintf(fd,"Block Index set is permutation\n");
    }
    fprintf(fd,"Block size %d\n",sub->bs);
    fprintf(fd,"Number of block indices in set %d\n",n);
    fprintf(fd,"The first indices of each block are\n");
    for ( i=0; i<n; i++ ) {
      fprintf(fd,"%d %d\n",i,idx[i]);
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISSort_Block" 
int ISSort_Block(IS is)
{
  IS_Block *sub = (IS_Block *)is->data;
  int      ierr;

  if (sub->sorted) return 0;
  ierr = PetscSortInt(sub->n, sub->idx); CHKERRQ(ierr);
  sub->sorted = 1;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISSorted_Block" 
int ISSorted_Block(IS is, PetscTruth *flg)
{
  IS_Block *sub = (IS_Block *)is->data;
  *flg = (PetscTruth) sub->sorted;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISDuplicate_Block" 
int ISDuplicate_Block(IS is, IS *newIS)
{
  IS_Block *sub = (IS_Block *)is->data;

  return ISCreateBlock(is->comm, sub->bs, sub->n, sub->idx, newIS);
}


static struct _ISOps myops = { ISGetSize_Block,
                               ISGetSize_Block,
                               ISGetIndices_Block,
                               ISRestoreIndices_Block,
                               ISInvertPermutation_Block,
                               ISSort_Block,
                               ISSorted_Block,
                               ISDuplicate_Block };
#undef __FUNC__  
#define __FUNC__ "ISCreateBlock" 
/*@C
   ISCreateBlock - Creates a data structure for an index set containing
   a list of integers. The indices are relative to entries, not blocks. 

   Input Parameters:
.  n - the length of the index set
.  bs - number of elements in each block
.  idx - the list of integers
.  comm - the MPI communicator

   Output Parameter:
.  is - the new index set

   Example:
$   If you wish to index {0,1,4,5} then use
$   a block size of 2 and idx of 0,4.

.keywords: IS, index set, create, block

.seealso: ISCreateStride(), ISCreateGeneral()
@*/
int ISCreateBlock(MPI_Comm comm,int bs,int n,int *idx,IS *is)
{
  int      i, sorted = 1, min, max;
  IS       Nindex;
  IS_Block *sub;

  *is = 0;
  PetscHeaderCreate(Nindex, _p_IS,IS_COOKIE,IS_BLOCK,comm); 
  PLogObjectCreate(Nindex);
  sub            = PetscNew(IS_Block); CHKPTRQ(sub);
  PLogObjectMemory(Nindex,sizeof(IS_Block)+n*sizeof(int)+sizeof(struct _p_IS));
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
  sub->bs         = bs;
  Nindex->min     = min;
  Nindex->max     = max;
  Nindex->data    = (void *) sub;
  PetscMemcpy(&Nindex->ops,&myops,sizeof(myops));
  Nindex->destroy = ISDestroy_Block;
  Nindex->view    = ISView_Block;
  Nindex->isperm  = 0;
  *is = Nindex; return 0;
}


#undef __FUNC__  
#define __FUNC__ "ISBlockGetIndices" 
/*@C
   ISBlockGetIndices - Gets the indices associated with each block.

   Input Parameter:
.  is - the index set

   Output Parameter:
.  idx - the integer indices

.keywords: IS, index set, block, get, indices

.seealso: ISGetIndices(), ISBlockRestoreIndices()
@*/
int ISBlockGetIndices(IS in,int **idx)
{
  IS_Block *sub;
  PetscValidHeaderSpecific(in,IS_COOKIE);
  PetscValidPointer(idx);
  if (in->type != IS_BLOCK) SETERRQ(1,0,"Not a block index set");

  sub = (IS_Block *) in->data;
  *idx = sub->idx; 
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISBlockRestoreIndices" 
/*@C
   ISBlockRestoreIndices - Restores the indices associated with each block.

   Input Parameter:
.  is - the index set

   Output Parameter:
.  idx - the integer indices

.keywords: IS, index set, block, restore, indices

.seealso: ISRestoreIndices(), ISBlockGetIndices()
@*/
int ISBlockRestoreIndices(IS is,int **idx)
{
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidPointer(idx);
  if (is->type != IS_BLOCK) SETERRQ(1,0,"Not a block index set");
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISBlockGetBlockSize" 
/*@
   ISBlockGetBlockSize - Returns the number of elements in a block.

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the number of elements in a block

.keywords: IS, index set, block, get, size

.seealso: ISBlockGetSize(), ISGetSize(), ISBlock(), ISCreateBlock()
@*/
int ISBlockGetBlockSize(IS is,int *size)
{
  IS_Block *sub;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidIntPointer(size);
  if (is->type != IS_BLOCK) SETERRQ(1,0,"Not a block index set");

  sub = (IS_Block *)is->data;
  *size = sub->bs; 
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISBlock" 
/*@C
   ISBlock - Checks if an index set is blocked.

   Input Parameter:
.  is - the index set

   Output Parameter:
.  flag - PETSC_TRUE if a block index set, else PETSC_FALSE

.keywords: IS, index set, block

.seealso: ISBlockGetSize(), ISGetSize(), ISBlockGetBlockSize(), ISCreateBlock()
@*/
int ISBlock(IS is,PetscTruth *flag)
{
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidIntPointer(flag);
  if (is->type != IS_BLOCK) *flag = PETSC_FALSE;
  else                          *flag = PETSC_TRUE;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ISBlockGetSize" 
/*@
   ISBlockGetSize - Returns the number of blocks in the index set.

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the number of blocks

.keywords: IS, index set, block, get, size

.seealso: ISBlockGetBlockSize(), ISGetSize(), ISBlock(), ISCreateBlock()
@*/
int ISBlockGetSize(IS is,int *size)
{
  IS_Block *sub;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidIntPointer(size);
  if (is->type != IS_BLOCK) SETERRQ(1,0,"Not a block index set");

  sub = (IS_Block *)is->data;
  *size = sub->n; 
  return 0;
}
