#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: baij2.c,v 1.19 1997/10/19 03:26:08 bsmith Exp bsmith $";
#endif

#include "src/mat/impls/baij/seq/baij.h"
#include "petsc.h"
#include "src/inline/bitarray.h"

#undef __FUNC__  
#define __FUNC__ "MatIncreaseOverlap_SeqBAIJ"
int MatIncreaseOverlap_SeqBAIJ(Mat A,int is_max,IS *is,int ov)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         row, i,j,k,l,m,n, *idx,ierr, *nidx, isz, val, ival;
  int         start, end, *ai, *aj,bs,*nidx2;
  BT          table;

  PetscFunctionBegin;
  m     = a->mbs;
  ai    = a->i;
  aj    = a->j;
  bs    = a->bs;

  if (ov < 0)  SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative overlap specified");

  ierr  = BTCreate(m,table); CHKERRQ(ierr);
  nidx  = (int *) PetscMalloc((m+1)*sizeof(int)); CHKPTRQ(nidx); 
  nidx2 = (int *)PetscMalloc((a->m+1)*sizeof(int)); CHKPTRQ(nidx2);

  for ( i=0; i<is_max; i++ ) {
    /* Initialise the two local arrays */
    isz  = 0;
    BTMemzero(m,table);
                 
    /* Extract the indices, assume there can be duplicate entries */
    ierr = ISGetIndices(is[i],&idx);  CHKERRQ(ierr);
    ierr = ISGetSize(is[i],&n);  CHKERRQ(ierr);

    /* Enter these into the temp arrays i.e mark table[row], enter row into new index */
    for ( j=0; j<n ; ++j){
      ival = idx[j]/bs; /* convert the indices into block indices */
      if (ival>m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"index greater than mat-dim");
      if(!BTLookupSet(table, ival)) { nidx[isz++] = ival;}
    }
    ierr = ISRestoreIndices(is[i],&idx);  CHKERRQ(ierr);
    ierr = ISDestroy(is[i]); CHKERRQ(ierr);
    
    k = 0;
    for ( j=0; j<ov; j++){ /* for each overlap*/
      n = isz;
      for ( ; k<n ; k++){ /* do only those rows in nidx[k], which are not done yet */
        row   = nidx[k];
        start = ai[row];
        end   = ai[row+1];
        for ( l = start; l<end ; l++){
          val = aj[l];
          if (!BTLookupSet(table,val)) {nidx[isz++] = val;}
        }
      }
    }
    /* expand the Index Set */
    for (j=0; j<isz; j++ ) {
      for (k=0; k<bs; k++ )
        nidx2[j*bs+k] = nidx[j]*bs+k;
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF, isz*bs, nidx2, (is+i)); CHKERRQ(ierr);
  }
  BTDestroy(table);
  PetscFree(nidx);
  PetscFree(nidx2);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrix_SeqBAIJ_Private"
int MatGetSubMatrix_SeqBAIJ_Private(Mat A,IS isrow,IS iscol,MatGetSubMatrixCall scall,Mat *B)
{
  Mat_SeqBAIJ  *a = (Mat_SeqBAIJ *) A->data,*c;
  int          nznew, *smap, i, k, kstart, kend, ierr, oldcols = a->nbs,*lens;
  int          row,mat_i,*mat_j,tcol,*mat_ilen;
  int          *irow, *icol, nrows, ncols,*ssmap,bs=a->bs, bs2=a->bs2;
  int          *aj = a->j, *ai = a->i;
  Scalar       *mat_a;
  Mat          C;

  PetscFunctionBegin;
  ierr = ISSorted(iscol,(PetscTruth*)&i);
  if (!i) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"IS is not sorted");

  ierr = ISGetIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol); CHKERRQ(ierr);
  ierr = ISGetSize(isrow,&nrows); CHKERRQ(ierr);
  ierr = ISGetSize(iscol,&ncols); CHKERRQ(ierr);

  smap  = (int *) PetscMalloc((1+oldcols)*sizeof(int)); CHKPTRQ(smap);
  ssmap = smap;
  lens  = (int *) PetscMalloc((1+nrows)*sizeof(int)); CHKPTRQ(lens);
  PetscMemzero(smap,oldcols*sizeof(int));
  for ( i=0; i<ncols; i++ ) smap[icol[i]] = i+1;
  /* determine lens of each row */
  for (i=0; i<nrows; i++) {
    kstart  = ai[irow[i]]; 
    kend    = kstart + a->ilen[irow[i]];
    lens[i] = 0;
      for ( k=kstart; k<kend; k++ ) {
        if (ssmap[aj[k]]) {
          lens[i]++;
        }
      }
    }
  /* Create and fill new matrix */
  if (scall == MAT_REUSE_MATRIX) {
    c = (Mat_SeqBAIJ *)((*B)->data);

    if (c->mbs!=nrows || c->nbs!=ncols || c->bs!=bs) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Submatrix wrong size");
    if (PetscMemcmp(c->ilen,lens, c->mbs *sizeof(int))) {
      SETERRQ(PETSC_ERR_ARG_SIZ,0,"Cannot reuse matrix. wrong no of nonzeros");
    }
    PetscMemzero(c->ilen,c->mbs*sizeof(int));
    C = *B;
  } else {  
    ierr = MatCreateSeqBAIJ(A->comm,bs,nrows*bs,ncols*bs,0,lens,&C);CHKERRQ(ierr);
  }
  c = (Mat_SeqBAIJ *)(C->data);
  for (i=0; i<nrows; i++) {
    row    = irow[i];
    nznew  = 0;
    kstart = ai[row]; 
    kend   = kstart + a->ilen[row];
    mat_i  = c->i[i];
    mat_j  = c->j + mat_i; 
    mat_a  = c->a + mat_i*bs2;
    mat_ilen = c->ilen + i;
    for ( k=kstart; k<kend; k++ ) {
      if ((tcol=ssmap[a->j[k]])) {
        *mat_j++ = tcol - 1;
        PetscMemcpy(mat_a,a->a+k*bs2,bs2*sizeof(Scalar)); mat_a+=bs2;
        (*mat_ilen)++;
        
      }
    }
  }
    
  /* Free work space */
  ierr = ISRestoreIndices(iscol,&icol); CHKERRQ(ierr);
  PetscFree(smap); PetscFree(lens);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  
  ierr = ISRestoreIndices(isrow,&irow); CHKERRQ(ierr);
  *B = C;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrix_SeqBAIJ"
int MatGetSubMatrix_SeqBAIJ(Mat A,IS isrow,IS iscol,MatGetSubMatrixCall scall,Mat *B)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  IS          is1,is2;
  int         *vary,*iary,*irow,*icol,nrows,ncols,i,ierr,bs=a->bs,count;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol); CHKERRQ(ierr);
  ierr = ISGetSize(isrow,&nrows); CHKERRQ(ierr);
  ierr = ISGetSize(iscol,&ncols); CHKERRQ(ierr);
  
  /* Verify if the indices corespond to each elementin a block 
   and form the IS with compressed IS */
  vary = (int *) PetscMalloc(2*(a->mbs+1)*sizeof(int)); CHKPTRQ(vary);
  iary = vary + a->mbs;
  PetscMemzero(vary,(a->mbs)*sizeof(int));
  for ( i=0; i<nrows; i++) vary[irow[i]/bs]++;
  count = 0;
  for (i=0; i<a->mbs; i++) {
    if (vary[i]!=0 && vary[i]!=bs) SETERRA(1,0,"MatGetSubmatrices_SeqBAIJ:");
    if (vary[i]==bs) iary[count++] = i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, count, iary,&is1); CHKERRQ(ierr);
  
  PetscMemzero(vary,(a->mbs)*sizeof(int));
  for ( i=0; i<ncols; i++) vary[icol[i]/bs]++;
  count = 0;
  for (i=0; i<a->mbs; i++) {
    if (vary[i]!=0 && vary[i]!=bs) SETERRA(1,0,"MatGetSubmatrices_SeqBAIJ:");
    if (vary[i]==bs) iary[count++] = i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, count, iary,&is2); CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&icol); CHKERRQ(ierr);
  PetscFree(vary);

  ierr = MatGetSubMatrix_SeqBAIJ_Private(A,is1,is2,scall,B); CHKERRQ(ierr);
  ISDestroy(is1);
  ISDestroy(is2);
  PetscFunctionReturn(0);
}
  
extern int MatGetSubMatrix_SeqBAIJ(Mat,IS,IS,MatGetSubMatrixCall,Mat*);

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrices_SeqBAIJ"
int MatGetSubMatrices_SeqBAIJ(Mat A,int n, IS *irow,IS *icol,MatGetSubMatrixCall scall,
                                    Mat **B)
{
  int ierr,i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    *B = (Mat *) PetscMalloc( (n+1)*sizeof(Mat) ); CHKPTRQ(*B);
  }

  for ( i=0; i<n; i++ ) {
    ierr = MatGetSubMatrix_SeqBAIJ(A,irow[i],icol[i],scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}







