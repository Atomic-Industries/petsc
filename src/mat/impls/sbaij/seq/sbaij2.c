/*$Id: sbaij2.c,v 1.32 2001/08/07 03:03:01 balay Exp $*/

#include "src/mat/impls/baij/seq/baij.h"
#include "src/inline/spops.h"
#include "src/inline/ilu.h"
#include "petscbt.h"
#include "src/mat/impls/sbaij/seq/sbaij.h"

#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_SeqSBAIJ"
int MatIncreaseOverlap_SeqSBAIJ(Mat A,int is_max,IS is[],int ov)
{
  PetscFunctionBegin;
  SETERRQ(1,"Function not yet written for SBAIJ format");
  /* PetscFunctionReturn(0); */
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_SeqSBAIJ_Private"
int MatGetSubMatrix_SeqSBAIJ_Private(Mat A,IS isrow,IS iscol,int cs,MatReuse scall,Mat *B)
{
  Mat_SeqSBAIJ  *a = (Mat_SeqSBAIJ*)A->data,*c;
  int          *smap,i,k,kstart,kend,ierr,oldcols = a->mbs,*lens;
  int          row,mat_i,*mat_j,tcol,*mat_ilen;
  int          *irow,nrows,*ssmap,bs=a->bs,bs2=a->bs2;
  int          *aj = a->j,*ai = a->i;
  MatScalar    *mat_a;
  Mat          C;
  PetscTruth   flag;

  PetscFunctionBegin;
 
  if (isrow != iscol) SETERRQ(1,"MatGetSubmatrices_SeqSBAIJ: For symm. format, iscol must equal isro"); 
  ierr = ISSorted(iscol,(PetscTruth*)&i);CHKERRQ(ierr);
  if (!i) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"IS is not sorted");

  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetSize(isrow,&nrows);CHKERRQ(ierr);
  
  ierr  = PetscMalloc((1+oldcols)*sizeof(int),&smap);CHKERRQ(ierr);
  ssmap = smap;
  ierr  = PetscMalloc((1+nrows)*sizeof(int),&lens);CHKERRQ(ierr);
  ierr  = PetscMemzero(smap,oldcols*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<nrows; i++) smap[irow[i]] = i+1; /* nrows = ncols */
  /* determine lens of each row */
  for (i=0; i<nrows; i++) {
    kstart  = ai[irow[i]]; 
    kend    = kstart + a->ilen[irow[i]];
    lens[i] = 0;
      for (k=kstart; k<kend; k++) {
        if (ssmap[aj[k]]) {
          lens[i]++;
        }
      }
    }
  /* Create and fill new matrix */
  if (scall == MAT_REUSE_MATRIX) {
    c = (Mat_SeqSBAIJ *)((*B)->data);

    if (c->mbs!=nrows || c->bs!=bs) SETERRQ(PETSC_ERR_ARG_SIZ,"Submatrix wrong size");
    ierr = PetscMemcmp(c->ilen,lens,c->mbs *sizeof(int),&flag);CHKERRQ(ierr);
    if (flag == PETSC_FALSE) {
      SETERRQ(PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong no of nonzeros");
    }
    ierr = PetscMemzero(c->ilen,c->mbs*sizeof(int));CHKERRQ(ierr);
    C = *B;
  } else {  
    ierr = MatCreate(A->comm,nrows*bs,nrows*bs,PETSC_DETERMINE,PETSC_DETERMINE,&C);CHKERRQ(ierr);
    ierr = MatSetType(C,A->type_name);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(C,bs,0,lens);CHKERRQ(ierr);
  }
  c = (Mat_SeqSBAIJ *)(C->data);
  for (i=0; i<nrows; i++) {
    row    = irow[i];
    kstart = ai[row]; 
    kend   = kstart + a->ilen[row];
    mat_i  = c->i[i];
    mat_j  = c->j + mat_i; 
    mat_a  = c->a + mat_i*bs2;
    mat_ilen = c->ilen + i;
    for (k=kstart; k<kend; k++) {
      if ((tcol=ssmap[a->j[k]])) {
        *mat_j++ = tcol - 1;
        ierr     = PetscMemcpy(mat_a,a->a+k*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
        mat_a   += bs2;
        (*mat_ilen)++;
      }
    }
  }
    
  /* Free work space */
  ierr = PetscFree(smap);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = ISRestoreIndices(isrow,&irow);CHKERRQ(ierr);
  *B = C;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_SeqSBAIJ"
int MatGetSubMatrix_SeqSBAIJ(Mat A,IS isrow,IS iscol,int cs,MatReuse scall,Mat *B)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;
  IS          is1;
  int         *vary,*iary,*irow,nrows,i,ierr,bs=a->bs,count;

  PetscFunctionBegin;
  if (isrow != iscol) SETERRQ(1,"MatGetSubmatrices_SeqSBAIJ: For symm. format, iscol must equal isro");
 
  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetSize(isrow,&nrows);CHKERRQ(ierr);
  
  /* Verify if the indices corespond to each element in a block 
   and form the IS with compressed IS */
  ierr = PetscMalloc(2*(a->mbs+1)*sizeof(int),&vary);CHKERRQ(ierr);
  iary = vary + a->mbs;
  ierr = PetscMemzero(vary,(a->mbs)*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<nrows; i++) vary[irow[i]/bs]++; 
 
  count = 0;
  for (i=0; i<a->mbs; i++) {
    if (vary[i]!=0 && vary[i]!=bs) SETERRQ(1,"Index set does not match blocks");
    if (vary[i]==bs) iary[count++] = i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,count,iary,&is1);CHKERRQ(ierr);
  
  ierr = ISRestoreIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = PetscFree(vary);CHKERRQ(ierr);

  ierr = MatGetSubMatrix_SeqSBAIJ_Private(A,is1,is1,cs,scall,B);CHKERRQ(ierr);
  ISDestroy(is1);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrices_SeqSBAIJ"
int MatGetSubMatrices_SeqSBAIJ(Mat A,int n,const IS irow[],const IS icol[],MatReuse scall,Mat *B[])
{
  int ierr,i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscMalloc((n+1)*sizeof(Mat),B);CHKERRQ(ierr);
  }

  for (i=0; i<n; i++) {
    ierr = MatGetSubMatrix_SeqSBAIJ(A,irow[i],icol[i],PETSC_DECIDE,scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------*/
/* Should check that shapes of vectors and matrices match */
/* -------------------------------------------------------*/
#include "petscblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqSBAIJ_1"
int MatMult_SeqSBAIJ_1(Mat A,Vec xx,Vec zz)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar     *x,*z,*xb,x1,zero=0.0;
  MatScalar       *v;
  int             mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,ierr,*ib,cval,j,jmin;  

  PetscFunctionBegin;
  ierr = VecSet(&zero,zz);CHKERRQ(ierr);
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr);

  v  = a->a; 
  xb = x;
   
  for (i=0; i<mbs; i++) {
    n  = ai[1] - ai[0];  /* length of i_th row of A */    
    x1 = xb[0];
    ib = aj + *ai;
    jmin = 0;
    if (*ib == i) {      /* (diag of A)*x */
      z[i] += *v++ * x[*ib++]; 
      jmin++;  
    }
    for (j=jmin; j<n; j++) {
      cval    = *ib; 
      z[cval] += *v * x1;      /* (strict lower triangular part of A)*x  */
      z[i] += *v++ * x[*ib++]; /* (strict upper triangular part of A)*x  */
    }
    xb++; ai++; 
  }

  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(2*(a->nz*2 - A->m) - A->m);  /* nz = (nz+m)/2 */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqSBAIJ_2"
int MatMult_SeqSBAIJ_2(Mat A,Vec xx,Vec zz)
{
  Mat_SeqSBAIJ     *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar      *x,*z,*xb,x1,x2,zero=0.0;
  MatScalar        *v;  
  int              mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,ierr,*ib,cval,j,jmin;  


  PetscFunctionBegin;
  ierr = VecSet(&zero,zz);CHKERRQ(ierr);
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr);
   
  v     = a->a;
  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[1] - ai[0]; /* length of i_th block row of A */
    x1 = xb[0]; x2 = xb[1];
    ib = aj + *ai;
    jmin = 0;
    if (*ib == i){     /* (diag of A)*x */
      z[2*i]   += v[0]*x1 + v[2]*x2;
      z[2*i+1] += v[2]*x1 + v[3]*x2;
      v += 4; jmin++;
    }
    for (j=jmin; j<n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval       = ib[j]*2;
      z[cval]     += v[0]*x1 + v[1]*x2;
      z[cval+1]   += v[2]*x1 + v[3]*x2;
      /* (strict upper triangular part of A)*x  */
      z[2*i]   += v[0]*x[cval] + v[2]*x[cval+1];
      z[2*i+1] += v[1]*x[cval] + v[3]*x[cval+1];
      v  += 4;      
    }
    xb +=2; ai++; 
  }

  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(8*(a->nz*2 - A->m) - A->m); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqSBAIJ_3"
int MatMult_SeqSBAIJ_3(Mat A,Vec xx,Vec zz)
{
  Mat_SeqSBAIJ  *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar   *x,*z,*xb,x1,x2,x3,zero=0.0;
  MatScalar     *v;  
  int           mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,ierr,*ib,cval,j,jmin;  


  PetscFunctionBegin;
  ierr = VecSet(&zero,zz);CHKERRQ(ierr);
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr);
   
  v     = a->a;
  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[1] - ai[0]; /* length of i_th block row of A */
    x1 = xb[0]; x2 = xb[1]; x3 = xb[2];
    ib = aj + *ai;
    jmin = 0;
    if (*ib == i){     /* (diag of A)*x */
      z[3*i]   += v[0]*x1 + v[3]*x2 + v[6]*x3;
      z[3*i+1] += v[3]*x1 + v[4]*x2 + v[7]*x3;
      z[3*i+2] += v[6]*x1 + v[7]*x2 + v[8]*x3;
      v += 9; jmin++;
    }
    for (j=jmin; j<n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval       = ib[j]*3;
      z[cval]     += v[0]*x1 + v[1]*x2 + v[2]*x3;
      z[cval+1]   += v[3]*x1 + v[4]*x2 + v[5]*x3;
      z[cval+2]   += v[6]*x1 + v[7]*x2 + v[8]*x3;
      /* (strict upper triangular part of A)*x  */
      z[3*i]   += v[0]*x[cval] + v[3]*x[cval+1]+ v[6]*x[cval+2];
      z[3*i+1] += v[1]*x[cval] + v[4]*x[cval+1]+ v[7]*x[cval+2];
      z[3*i+2] += v[2]*x[cval] + v[5]*x[cval+1]+ v[8]*x[cval+2];
      v  += 9;      
    }
    xb +=3; ai++; 
  }

  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(18*(a->nz*2 - A->m) - A->m);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqSBAIJ_4"
int MatMult_SeqSBAIJ_4(Mat A,Vec xx,Vec zz)
{
  Mat_SeqSBAIJ     *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar      *x,*z,*xb,x1,x2,x3,x4,zero=0.0;
  MatScalar        *v;  
  int              mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,ierr,*ib,cval,j,jmin;  

  PetscFunctionBegin;
  ierr = VecSet(&zero,zz);CHKERRQ(ierr);
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr);
   
  v     = a->a;
  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[1] - ai[0]; /* length of i_th block row of A */
    x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];
    ib = aj + *ai;
    jmin = 0;
    if (*ib == i){     /* (diag of A)*x */
      z[4*i]   += v[0]*x1 + v[4]*x2 +  v[8]*x3 + v[12]*x4;
      z[4*i+1] += v[4]*x1 + v[5]*x2 +  v[9]*x3 + v[13]*x4;
      z[4*i+2] += v[8]*x1 + v[9]*x2 + v[10]*x3 + v[14]*x4;
      z[4*i+3] += v[12]*x1+ v[13]*x2+ v[14]*x3 + v[15]*x4;
      v += 16; jmin++;
    }
    for (j=jmin; j<n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval       = ib[j]*4;
      z[cval]     += v[0]*x1 + v[1]*x2 + v[2]*x3 + v[3]*x4;
      z[cval+1]   += v[4]*x1 + v[5]*x2 + v[6]*x3 + v[7]*x4;
      z[cval+2]   += v[8]*x1 + v[9]*x2 + v[10]*x3 + v[11]*x4;
      z[cval+3]   += v[12]*x1 + v[13]*x2 + v[14]*x3 + v[15]*x4;
      /* (strict upper triangular part of A)*x  */
      z[4*i]   += v[0]*x[cval] + v[4]*x[cval+1]+ v[8]*x[cval+2] + v[12]*x[cval+3];
      z[4*i+1] += v[1]*x[cval] + v[5]*x[cval+1]+ v[9]*x[cval+2] + v[13]*x[cval+3];
      z[4*i+2] += v[2]*x[cval] + v[6]*x[cval+1]+ v[10]*x[cval+2]+ v[14]*x[cval+3];
      z[4*i+3] += v[3]*x[cval] + v[7]*x[cval+1]+ v[11]*x[cval+2]+ v[15]*x[cval+3];
      v  += 16;      
    }
    xb +=4; ai++; 
  }

  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(32*(a->nz*2 - A->m) - A->m);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqSBAIJ_5"
int MatMult_SeqSBAIJ_5(Mat A,Vec xx,Vec zz)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar     *x,*z,*xb,x1,x2,x3,x4,x5,zero=0.0;
  MatScalar       *v;  
  int             mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,ierr,*ib,cval,j,jmin;  

  PetscFunctionBegin;
  ierr = VecSet(&zero,zz);CHKERRQ(ierr);
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr);
   
  v     = a->a;
  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[1] - ai[0]; /* length of i_th block row of A */
    x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5=xb[4];
    ib = aj + *ai;
    jmin = 0;
    if (*ib == i){      /* (diag of A)*x */
      z[5*i]   += v[0]*x1  + v[5]*x2 + v[10]*x3 + v[15]*x4+ v[20]*x5;
      z[5*i+1] += v[5]*x1  + v[6]*x2 + v[11]*x3 + v[16]*x4+ v[21]*x5;
      z[5*i+2] += v[10]*x1 +v[11]*x2 + v[12]*x3 + v[17]*x4+ v[22]*x5;
      z[5*i+3] += v[15]*x1 +v[16]*x2 + v[17]*x3 + v[18]*x4+ v[23]*x5;
      z[5*i+4] += v[20]*x1 +v[21]*x2 + v[22]*x3 + v[23]*x4+ v[24]*x5; 
      v += 25; jmin++;
    }
    for (j=jmin; j<n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval       = ib[j]*5;
      z[cval]     += v[0]*x1 + v[1]*x2 + v[2]*x3 + v[3]*x4 + v[4]*x5;
      z[cval+1]   += v[5]*x1 + v[6]*x2 + v[7]*x3 + v[8]*x4 + v[9]*x5;
      z[cval+2]   += v[10]*x1 + v[11]*x2 + v[12]*x3 + v[13]*x4+ v[14]*x5;
      z[cval+3]   += v[15]*x1 + v[16]*x2 + v[17]*x3 + v[18]*x4+ v[19]*x5;
      z[cval+4]   += v[20]*x1 + v[21]*x2 + v[22]*x3 + v[23]*x4+ v[24]*x5;
      /* (strict upper triangular part of A)*x  */
      z[5*i]   +=v[0]*x[cval]+v[5]*x[cval+1]+v[10]*x[cval+2]+v[15]*x[cval+3]+v[20]*x[cval+4];
      z[5*i+1] +=v[1]*x[cval]+v[6]*x[cval+1]+v[11]*x[cval+2]+v[16]*x[cval+3]+v[21]*x[cval+4];
      z[5*i+2] +=v[2]*x[cval]+v[7]*x[cval+1]+v[12]*x[cval+2]+v[17]*x[cval+3]+v[22]*x[cval+4];
      z[5*i+3] +=v[3]*x[cval]+v[8]*x[cval+1]+v[13]*x[cval+2]+v[18]*x[cval+3]+v[23]*x[cval+4];
      z[5*i+4] +=v[4]*x[cval]+v[9]*x[cval+1]+v[14]*x[cval+2]+v[19]*x[cval+3]+v[24]*x[cval+4];
      v  += 25;      
    }
    xb +=5; ai++; 
  }

  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(50*(a->nz*2 - A->m) - A->m);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqSBAIJ_6"
int MatMult_SeqSBAIJ_6(Mat A,Vec xx,Vec zz)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar     *x,*z,*xb,x1,x2,x3,x4,x5,x6,zero=0.0;
  MatScalar       *v;  
  int             mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,ierr,*ib,cval,j,jmin;  

  PetscFunctionBegin;
  ierr = VecSet(&zero,zz);CHKERRQ(ierr);
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr);
   
  v     = a->a;
  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[1] - ai[0]; /* length of i_th block row of A */
    x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5=xb[4]; x6=xb[5];
    ib = aj + *ai;
    jmin = 0;
    if (*ib == i){      /* (diag of A)*x */
      z[6*i]   += v[0]*x1  + v[6]*x2 + v[12]*x3 + v[18]*x4+ v[24]*x5 + v[30]*x6;
      z[6*i+1] += v[6]*x1  + v[7]*x2 + v[13]*x3 + v[19]*x4+ v[25]*x5 + v[31]*x6;
      z[6*i+2] += v[12]*x1 +v[13]*x2 + v[14]*x3 + v[20]*x4+ v[26]*x5 + v[32]*x6;
      z[6*i+3] += v[18]*x1 +v[19]*x2 + v[20]*x3 + v[21]*x4+ v[27]*x5 + v[33]*x6;
      z[6*i+4] += v[24]*x1 +v[25]*x2 + v[26]*x3 + v[27]*x4+ v[28]*x5 + v[34]*x6; 
      z[6*i+5] += v[30]*x1 +v[31]*x2 + v[32]*x3 + v[33]*x4+ v[34]*x5 + v[35]*x6;
      v += 36; jmin++;
    }
    for (j=jmin; j<n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval       = ib[j]*6;
      z[cval]   += v[0]*x1  + v[1]*x2 + v[2]*x3 + v[3]*x4+ v[4]*x5 + v[5]*x6;
      z[cval+1] += v[6]*x1  + v[7]*x2 + v[8]*x3 + v[9]*x4+ v[10]*x5 + v[11]*x6;
      z[cval+2] += v[12]*x1  + v[13]*x2 + v[14]*x3 + v[15]*x4+ v[16]*x5 + v[17]*x6;
      z[cval+3] += v[18]*x1  + v[19]*x2 + v[20]*x3 + v[21]*x4+ v[22]*x5 + v[23]*x6;
      z[cval+4] += v[24]*x1  + v[25]*x2 + v[26]*x3 + v[27]*x4+ v[28]*x5 + v[29]*x6;
      z[cval+5] += v[30]*x1  + v[31]*x2 + v[32]*x3 + v[33]*x4+ v[34]*x5 + v[35]*x6;
      /* (strict upper triangular part of A)*x  */
      z[6*i]   +=v[0]*x[cval]+v[6]*x[cval+1]+v[12]*x[cval+2]+v[18]*x[cval+3]+v[24]*x[cval+4]+v[30]*x[cval+5];
      z[6*i+1] +=v[1]*x[cval]+v[7]*x[cval+1]+v[13]*x[cval+2]+v[19]*x[cval+3]+v[25]*x[cval+4]+v[31]*x[cval+5];
      z[6*i+2] +=v[2]*x[cval]+v[8]*x[cval+1]+v[14]*x[cval+2]+v[20]*x[cval+3]+v[26]*x[cval+4]+v[32]*x[cval+5];
      z[6*i+3] +=v[3]*x[cval]+v[9]*x[cval+1]+v[15]*x[cval+2]+v[21]*x[cval+3]+v[27]*x[cval+4]+v[33]*x[cval+5];
      z[6*i+4] +=v[4]*x[cval]+v[10]*x[cval+1]+v[16]*x[cval+2]+v[22]*x[cval+3]+v[28]*x[cval+4]+v[34]*x[cval+5];
      z[6*i+5] +=v[5]*x[cval]+v[11]*x[cval+1]+v[17]*x[cval+2]+v[23]*x[cval+3]+v[29]*x[cval+4]+v[35]*x[cval+5];
      v  += 36;      
    }
    xb +=6; ai++; 
  }

  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(72*(a->nz*2 - A->m) - A->m);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqSBAIJ_7"
int MatMult_SeqSBAIJ_7(Mat A,Vec xx,Vec zz)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar     *x,*z,*xb,x1,x2,x3,x4,x5,x6,x7,zero=0.0;
  MatScalar       *v;  
  int             mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,ierr,*ib,cval,j,jmin;  

  PetscFunctionBegin;
  ierr = VecSet(&zero,zz);CHKERRQ(ierr);
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr);
   
  v     = a->a;
  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[1] - ai[0]; /* length of i_th block row of A */
    x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5=xb[4]; x6=xb[5]; x7=xb[6];
    ib = aj + *ai;
    jmin = 0;
    if (*ib == i){      /* (diag of A)*x */
      z[7*i]   += v[0]*x1 + v[7]*x2 + v[14]*x3 + v[21]*x4+ v[28]*x5 + v[35]*x6+ v[42]*x7;
      z[7*i+1] += v[7]*x1 + v[8]*x2 + v[15]*x3 + v[22]*x4+ v[29]*x5 + v[36]*x6+ v[43]*x7;
      z[7*i+2] += v[14]*x1+ v[15]*x2 +v[16]*x3 + v[23]*x4+ v[30]*x5 + v[37]*x6+ v[44]*x7;
      z[7*i+3] += v[21]*x1+ v[22]*x2 +v[23]*x3 + v[24]*x4+ v[31]*x5 + v[38]*x6+ v[45]*x7;
      z[7*i+4] += v[28]*x1+ v[29]*x2 +v[30]*x3 + v[31]*x4+ v[32]*x5 + v[39]*x6+ v[46]*x7;
      z[7*i+5] += v[35]*x1+ v[36]*x2 +v[37]*x3 + v[38]*x4+ v[39]*x5 + v[40]*x6+ v[47]*x7;
      z[7*i+6] += v[42]*x1+ v[43]*x2 +v[44]*x3 + v[45]*x4+ v[46]*x5 + v[47]*x6+ v[48]*x7;
      v += 49; jmin++;
    }
    for (j=jmin; j<n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval       = ib[j]*7;
      z[cval]   += v[0]*x1  + v[1]*x2 + v[2]*x3 + v[3]*x4+ v[4]*x5 + v[5]*x6+ v[6]*x7;
      z[cval+1] += v[7]*x1  + v[8]*x2 + v[9]*x3 + v[10]*x4+ v[11]*x5 + v[12]*x6+ v[13]*x7;
      z[cval+2] += v[14]*x1  + v[15]*x2 + v[16]*x3 + v[17]*x4+ v[18]*x5 + v[19]*x6+ v[20]*x7;
      z[cval+3] += v[21]*x1  + v[22]*x2 + v[23]*x3 + v[24]*x4+ v[25]*x5 + v[26]*x6+ v[27]*x7;
      z[cval+4] += v[28]*x1  + v[29]*x2 + v[30]*x3 + v[31]*x4+ v[32]*x5 + v[33]*x6+ v[34]*x7;
      z[cval+5] += v[35]*x1  + v[36]*x2 + v[37]*x3 + v[38]*x4+ v[39]*x5 + v[40]*x6+ v[41]*x7;
      z[cval+6] += v[42]*x1  + v[43]*x2 + v[44]*x3 + v[45]*x4+ v[46]*x5 + v[47]*x6+ v[48]*x7;
      /* (strict upper triangular part of A)*x  */
      z[7*i]  +=v[0]*x[cval]+v[7]*x[cval+1]+v[14]*x[cval+2]+v[21]*x[cval+3]+v[28]*x[cval+4]+v[35]*x[cval+5]+v[42]*x[cval+6];
      z[7*i+1]+=v[1]*x[cval]+v[8]*x[cval+1]+v[15]*x[cval+2]+v[22]*x[cval+3]+v[29]*x[cval+4]+v[36]*x[cval+5]+v[43]*x[cval+6];
      z[7*i+2]+=v[2]*x[cval]+v[9]*x[cval+1]+v[16]*x[cval+2]+v[23]*x[cval+3]+v[30]*x[cval+4]+v[37]*x[cval+5]+v[44]*x[cval+6];
      z[7*i+3]+=v[3]*x[cval]+v[10]*x[cval+1]+v[17]*x[cval+2]+v[24]*x[cval+3]+v[31]*x[cval+4]+v[38]*x[cval+5]+v[45]*x[cval+6];
      z[7*i+4]+=v[4]*x[cval]+v[11]*x[cval+1]+v[18]*x[cval+2]+v[25]*x[cval+3]+v[32]*x[cval+4]+v[39]*x[cval+5]+v[46]*x[cval+6];
      z[7*i+5]+=v[5]*x[cval]+v[12]*x[cval+1]+v[19]*x[cval+2]+v[26]*x[cval+3]+v[33]*x[cval+4]+v[40]*x[cval+5]+v[47]*x[cval+6];
      z[7*i+6]+=v[6]*x[cval]+v[13]*x[cval+1]+v[20]*x[cval+2]+v[27]*x[cval+3]+v[34]*x[cval+4]+v[41]*x[cval+5]+v[48]*x[cval+6];
      v  += 49;      
    }
    xb +=7; ai++; 
  }
  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(98*(a->nz*2 - A->m) - A->m);
  PetscFunctionReturn(0);
}

/*
    This will not work with MatScalar == float because it calls the BLAS
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqSBAIJ_N"
int MatMult_SeqSBAIJ_N(Mat A,Vec xx,Vec zz)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar     *x,*x_ptr,*z,*z_ptr,*xb,*zb,*work,*workt,zero=0.0;
  MatScalar       *v;
  int             ierr,mbs=a->mbs,i,*idx,*aj,*ii,bs=a->bs,j,n,bs2=a->bs2;
  int             ncols,k;

  PetscFunctionBegin;
  ierr = VecSet(&zero,zz);CHKERRQ(ierr);
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr); x_ptr=x;
  ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr); z_ptr=z;

  aj   = a->j; 
  v    = a->a;
  ii   = a->i;

  if (!a->mult_work) {    
    ierr = PetscMalloc((A->m+1)*sizeof(PetscScalar),&a->mult_work);CHKERRQ(ierr);
  }
  work = a->mult_work; 
    
  for (i=0; i<mbs; i++) {
    n     = ii[1] - ii[0]; ncols = n*bs;
    workt = work; idx=aj+ii[0]; 

    /* upper triangular part */ 
    for (j=0; j<n; j++) {       
      xb = x_ptr + bs*(*idx++);
      for (k=0; k<bs; k++) workt[k] = xb[k];
      workt += bs;
    }
    /* z(i*bs:(i+1)*bs-1) += A(i,:)*x */
    Kernel_w_gets_w_plus_Ar_times_v(bs,ncols,work,v,z); 
    
    /* strict lower triangular part */    
    idx = aj+ii[0];    
    if (*idx == i){
      ncols -= bs; v += bs2; idx++; n--;
    }
   
    if (ncols > 0){
      workt = work;
      ierr  = PetscMemzero(workt,ncols*sizeof(PetscScalar));CHKERRQ(ierr);
      Kernel_w_gets_w_plus_trans_Ar_times_v(bs,ncols,x,v,workt);
      for (j=0; j<n; j++) {
        zb = z_ptr + bs*(*idx++);  
        for (k=0; k<bs; k++) zb[k] += workt[k] ;
        workt += bs;
      }
    }
    x += bs; v += n*bs2; z += bs; ii++;
  }                   
  
  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(2*(a->nz*2 - A->m)*bs2 - A->m);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqSBAIJ_1"
int MatMultAdd_SeqSBAIJ_1(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar     *x,*y,*z,*xb,x1;
  MatScalar       *v;
  int             mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,ierr,*ib,cval,j,jmin;  

  PetscFunctionBegin;
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr);
  if (yy != xx) {
    ierr = VecGetArrayFast(yy,&y);CHKERRQ(ierr);
  } else {
    y = x;
  }
  if (zz != yy) {
    /* ierr = VecCopy(yy,zz);CHKERRQ(ierr); */
    ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr);
    ierr = PetscMemcpy(z,y,yy->n*sizeof(PetscScalar));CHKERRQ(ierr); 
  } else {
    z = y;
  }

  v  = a->a; 
  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[1] - ai[0];  /* length of i_th row of A */    
    x1 = xb[0];
    ib = aj + *ai;
    jmin = 0;
    if (*ib == i) {            /* (diag of A)*x */
      z[i] += *v++ * x[*ib++]; jmin++;  
    }
    for (j=jmin; j<n; j++) {
      cval    = *ib; 
      z[cval] += *v * x1;      /* (strict lower triangular part of A)*x  */
      z[i] += *v++ * x[*ib++]; /* (strict upper triangular part of A)*x  */
    }
    xb++; ai++; 
  }  

  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  if (yy != xx) ierr = VecRestoreArrayFast(yy,&y);CHKERRQ(ierr);
  if (zz != yy) ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);
  
  PetscLogFlops(2*(a->nz*2 - A->m));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqSBAIJ_2"
int MatMultAdd_SeqSBAIJ_2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar     *x,*y,*z,*xb,x1,x2;
  MatScalar       *v;  
  int             mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,ierr,*ib,cval,j,jmin;  

  PetscFunctionBegin;
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr);
  if (yy != xx) {
    ierr = VecGetArrayFast(yy,&y);CHKERRQ(ierr);
  } else {
    y = x;
  }
  if (zz != yy) {
    /* ierr = VecCopy(yy,zz);CHKERRQ(ierr); */
    ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr);
    ierr = PetscMemcpy(z,y,yy->n*sizeof(PetscScalar));CHKERRQ(ierr); 
  } else {
    z = y;
  }

  v     = a->a;
  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[1] - ai[0]; /* length of i_th block row of A */
    x1 = xb[0]; x2 = xb[1];
    ib = aj + *ai;
    jmin = 0;
    if (*ib == i){      /* (diag of A)*x */
      z[2*i]   += v[0]*x1 + v[2]*x2;
      z[2*i+1] += v[2]*x1 + v[3]*x2;
      v += 4; jmin++;
    }
    for (j=jmin; j<n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval       = ib[j]*2;
      z[cval]     += v[0]*x1 + v[1]*x2;
      z[cval+1]   += v[2]*x1 + v[3]*x2;
      /* (strict upper triangular part of A)*x  */
      z[2*i]   += v[0]*x[cval] + v[2]*x[cval+1];
      z[2*i+1] += v[1]*x[cval] + v[3]*x[cval+1];
      v  += 4;      
    }
    xb +=2; ai++; 
  }

  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  if (yy != xx) ierr = VecRestoreArrayFast(yy,&y);CHKERRQ(ierr);
  if (zz != yy) ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);

  PetscLogFlops(4*(a->nz*2 - A->m));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqSBAIJ_3"
int MatMultAdd_SeqSBAIJ_3(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar     *x,*y,*z,*xb,x1,x2,x3;
  MatScalar       *v;  
  int             mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,ierr,*ib,cval,j,jmin; 

  PetscFunctionBegin;
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr);
  if (yy != xx) {
    ierr = VecGetArrayFast(yy,&y);CHKERRQ(ierr);
  } else {
    y = x;
  }
  if (zz != yy) {
    /* ierr = VecCopy(yy,zz);CHKERRQ(ierr); */
    ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr);
    ierr = PetscMemcpy(z,y,yy->n*sizeof(PetscScalar));CHKERRQ(ierr); 
  } else {
    z = y;
  }     

  v     = a->a;
  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[1] - ai[0]; /* length of i_th block row of A */
    x1 = xb[0]; x2 = xb[1]; x3 = xb[2];
    ib = aj + *ai;
    jmin = 0;
    if (*ib == i){     /* (diag of A)*x */
     z[3*i]   += v[0]*x1 + v[3]*x2 + v[6]*x3;
     z[3*i+1] += v[3]*x1 + v[4]*x2 + v[7]*x3;
     z[3*i+2] += v[6]*x1 + v[7]*x2 + v[8]*x3;
     v += 9; jmin++;
    }
    for (j=jmin; j<n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval       = ib[j]*3;
      z[cval]     += v[0]*x1 + v[1]*x2 + v[2]*x3;
      z[cval+1]   += v[3]*x1 + v[4]*x2 + v[5]*x3;
      z[cval+2]   += v[6]*x1 + v[7]*x2 + v[8]*x3;
      /* (strict upper triangular part of A)*x  */
      z[3*i]   += v[0]*x[cval] + v[3]*x[cval+1]+ v[6]*x[cval+2];
      z[3*i+1] += v[1]*x[cval] + v[4]*x[cval+1]+ v[7]*x[cval+2];
      z[3*i+2] += v[2]*x[cval] + v[5]*x[cval+1]+ v[8]*x[cval+2];
      v  += 9;      
    }
    xb +=3; ai++; 
  }

  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  if (yy != xx) ierr = VecRestoreArrayFast(yy,&y);CHKERRQ(ierr);
  if (zz != yy) ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);

  PetscLogFlops(18*(a->nz*2 - A->m));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqSBAIJ_4"
int MatMultAdd_SeqSBAIJ_4(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar     *x,*y,*z,*xb,x1,x2,x3,x4;
  MatScalar       *v;  
  int             mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,ierr,*ib,cval,j,jmin;  

  PetscFunctionBegin;
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr);
  if (yy != xx) {
    ierr = VecGetArrayFast(yy,&y);CHKERRQ(ierr);
  } else {
    y = x;
  }
  if (zz != yy) {
    /* ierr = VecCopy(yy,zz);CHKERRQ(ierr); */
    ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr);
    ierr = PetscMemcpy(z,y,yy->n*sizeof(PetscScalar));CHKERRQ(ierr); 
  } else {
    z = y;
  }   

  v     = a->a;
  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[1] - ai[0]; /* length of i_th block row of A */
    x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];
    ib = aj + *ai;
    jmin = 0;
    if (*ib == i){      /* (diag of A)*x */
      z[4*i]   += v[0]*x1 + v[4]*x2 +  v[8]*x3 + v[12]*x4;
      z[4*i+1] += v[4]*x1 + v[5]*x2 +  v[9]*x3 + v[13]*x4;
      z[4*i+2] += v[8]*x1 + v[9]*x2 + v[10]*x3 + v[14]*x4;
      z[4*i+3] += v[12]*x1+ v[13]*x2+ v[14]*x3 + v[15]*x4;
      v += 16; jmin++;
    }
    for (j=jmin; j<n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval       = ib[j]*4;
      z[cval]     += v[0]*x1 + v[1]*x2 + v[2]*x3 + v[3]*x4;
      z[cval+1]   += v[4]*x1 + v[5]*x2 + v[6]*x3 + v[7]*x4;
      z[cval+2]   += v[8]*x1 + v[9]*x2 + v[10]*x3 + v[11]*x4;
      z[cval+3]   += v[12]*x1 + v[13]*x2 + v[14]*x3 + v[15]*x4;
      /* (strict upper triangular part of A)*x  */
      z[4*i]   += v[0]*x[cval] + v[4]*x[cval+1]+ v[8]*x[cval+2] + v[12]*x[cval+3];
      z[4*i+1] += v[1]*x[cval] + v[5]*x[cval+1]+ v[9]*x[cval+2] + v[13]*x[cval+3];
      z[4*i+2] += v[2]*x[cval] + v[6]*x[cval+1]+ v[10]*x[cval+2]+ v[14]*x[cval+3];
      z[4*i+3] += v[3]*x[cval] + v[7]*x[cval+1]+ v[11]*x[cval+2]+ v[15]*x[cval+3];
      v  += 16;      
    }
    xb +=4; ai++; 
  }

  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  if (yy != xx) ierr = VecRestoreArrayFast(yy,&y);CHKERRQ(ierr);
  if (zz != yy) ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);

  PetscLogFlops(32*(a->nz*2 - A->m));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqSBAIJ_5"
int MatMultAdd_SeqSBAIJ_5(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar     *x,*y,*z,*xb,x1,x2,x3,x4,x5;
  MatScalar       *v;  
  int             mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,ierr,*ib,cval,j,jmin; 

  PetscFunctionBegin;
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr);
  if (yy != xx) {
    ierr = VecGetArrayFast(yy,&y);CHKERRQ(ierr);
  } else {
    y = x;
  }
  if (zz != yy) {
    /* ierr = VecCopy(yy,zz);CHKERRQ(ierr); */
    ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr);
    ierr = PetscMemcpy(z,y,yy->n*sizeof(PetscScalar));CHKERRQ(ierr); 
  } else {
    z = y;
  } 

  v     = a->a;
  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[1] - ai[0]; /* length of i_th block row of A */
    x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5=xb[4];
    ib = aj + *ai;
    jmin = 0;
    if (*ib == i){      /* (diag of A)*x */
      z[5*i]   += v[0]*x1  + v[5]*x2 + v[10]*x3 + v[15]*x4+ v[20]*x5;
      z[5*i+1] += v[5]*x1  + v[6]*x2 + v[11]*x3 + v[16]*x4+ v[21]*x5;
      z[5*i+2] += v[10]*x1 +v[11]*x2 + v[12]*x3 + v[17]*x4+ v[22]*x5;
      z[5*i+3] += v[15]*x1 +v[16]*x2 + v[17]*x3 + v[18]*x4+ v[23]*x5;
      z[5*i+4] += v[20]*x1 +v[21]*x2 + v[22]*x3 + v[23]*x4+ v[24]*x5; 
      v += 25; jmin++;
    }
    for (j=jmin; j<n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval       = ib[j]*5;
      z[cval]     += v[0]*x1 + v[1]*x2 + v[2]*x3 + v[3]*x4 + v[4]*x5;
      z[cval+1]   += v[5]*x1 + v[6]*x2 + v[7]*x3 + v[8]*x4 + v[9]*x5;
      z[cval+2]   += v[10]*x1 + v[11]*x2 + v[12]*x3 + v[13]*x4+ v[14]*x5;
      z[cval+3]   += v[15]*x1 + v[16]*x2 + v[17]*x3 + v[18]*x4+ v[19]*x5;
      z[cval+4]   += v[20]*x1 + v[21]*x2 + v[22]*x3 + v[23]*x4+ v[24]*x5;
      /* (strict upper triangular part of A)*x  */
      z[5*i]   +=v[0]*x[cval]+v[5]*x[cval+1]+v[10]*x[cval+2]+v[15]*x[cval+3]+v[20]*x[cval+4];
      z[5*i+1] +=v[1]*x[cval]+v[6]*x[cval+1]+v[11]*x[cval+2]+v[16]*x[cval+3]+v[21]*x[cval+4];
      z[5*i+2] +=v[2]*x[cval]+v[7]*x[cval+1]+v[12]*x[cval+2]+v[17]*x[cval+3]+v[22]*x[cval+4];
      z[5*i+3] +=v[3]*x[cval]+v[8]*x[cval+1]+v[13]*x[cval+2]+v[18]*x[cval+3]+v[23]*x[cval+4];
      z[5*i+4] +=v[4]*x[cval]+v[9]*x[cval+1]+v[14]*x[cval+2]+v[19]*x[cval+3]+v[24]*x[cval+4];
      v  += 25;      
    }
    xb +=5; ai++; 
  }

  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  if (yy != xx) ierr = VecRestoreArrayFast(yy,&y);CHKERRQ(ierr);
  if (zz != yy) ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);

  PetscLogFlops(50*(a->nz*2 - A->m));
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqSBAIJ_6"
int MatMultAdd_SeqSBAIJ_6(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar     *x,*y,*z,*xb,x1,x2,x3,x4,x5,x6;
  MatScalar       *v;  
  int             mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,ierr,*ib,cval,j,jmin;  

  PetscFunctionBegin;
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr);
  if (yy != xx) {
    ierr = VecGetArrayFast(yy,&y);CHKERRQ(ierr);
  } else {
    y = x;
  }
  if (zz != yy) {
    /* ierr = VecCopy(yy,zz);CHKERRQ(ierr); */
    ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr);
    ierr = PetscMemcpy(z,y,yy->n*sizeof(PetscScalar));CHKERRQ(ierr); 
  } else {
    z = y;
  }      

  v     = a->a;
  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[1] - ai[0]; /* length of i_th block row of A */
    x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5=xb[4]; x6=xb[5];
    ib = aj + *ai;
    jmin = 0;
    if (*ib == i){     /* (diag of A)*x */
      z[6*i]   += v[0]*x1  + v[6]*x2 + v[12]*x3 + v[18]*x4+ v[24]*x5 + v[30]*x6;
      z[6*i+1] += v[6]*x1  + v[7]*x2 + v[13]*x3 + v[19]*x4+ v[25]*x5 + v[31]*x6;
      z[6*i+2] += v[12]*x1 +v[13]*x2 + v[14]*x3 + v[20]*x4+ v[26]*x5 + v[32]*x6;
      z[6*i+3] += v[18]*x1 +v[19]*x2 + v[20]*x3 + v[21]*x4+ v[27]*x5 + v[33]*x6;
      z[6*i+4] += v[24]*x1 +v[25]*x2 + v[26]*x3 + v[27]*x4+ v[28]*x5 + v[34]*x6; 
      z[6*i+5] += v[30]*x1 +v[31]*x2 + v[32]*x3 + v[33]*x4+ v[34]*x5 + v[35]*x6;
      v += 36; jmin++;
    }
    for (j=jmin; j<n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval       = ib[j]*6;
      z[cval]   += v[0]*x1  + v[1]*x2 + v[2]*x3 + v[3]*x4+ v[4]*x5 + v[5]*x6;
      z[cval+1] += v[6]*x1  + v[7]*x2 + v[8]*x3 + v[9]*x4+ v[10]*x5 + v[11]*x6;
      z[cval+2] += v[12]*x1  + v[13]*x2 + v[14]*x3 + v[15]*x4+ v[16]*x5 + v[17]*x6;
      z[cval+3] += v[18]*x1  + v[19]*x2 + v[20]*x3 + v[21]*x4+ v[22]*x5 + v[23]*x6;
      z[cval+4] += v[24]*x1  + v[25]*x2 + v[26]*x3 + v[27]*x4+ v[28]*x5 + v[29]*x6;
      z[cval+5] += v[30]*x1  + v[31]*x2 + v[32]*x3 + v[33]*x4+ v[34]*x5 + v[35]*x6;
      /* (strict upper triangular part of A)*x  */
      z[6*i]   +=v[0]*x[cval]+v[6]*x[cval+1]+v[12]*x[cval+2]+v[18]*x[cval+3]+v[24]*x[cval+4]+v[30]*x[cval+5];
      z[6*i+1] +=v[1]*x[cval]+v[7]*x[cval+1]+v[13]*x[cval+2]+v[19]*x[cval+3]+v[25]*x[cval+4]+v[31]*x[cval+5];
      z[6*i+2] +=v[2]*x[cval]+v[8]*x[cval+1]+v[14]*x[cval+2]+v[20]*x[cval+3]+v[26]*x[cval+4]+v[32]*x[cval+5];
      z[6*i+3] +=v[3]*x[cval]+v[9]*x[cval+1]+v[15]*x[cval+2]+v[21]*x[cval+3]+v[27]*x[cval+4]+v[33]*x[cval+5];
      z[6*i+4] +=v[4]*x[cval]+v[10]*x[cval+1]+v[16]*x[cval+2]+v[22]*x[cval+3]+v[28]*x[cval+4]+v[34]*x[cval+5];
      z[6*i+5] +=v[5]*x[cval]+v[11]*x[cval+1]+v[17]*x[cval+2]+v[23]*x[cval+3]+v[29]*x[cval+4]+v[35]*x[cval+5];
      v  += 36;      
    }
    xb +=6; ai++; 
  }

  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  if (yy != xx) ierr = VecRestoreArrayFast(yy,&y);CHKERRQ(ierr);
  if (zz != yy) ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);

  PetscLogFlops(72*(a->nz*2 - A->m));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqSBAIJ_7"
int MatMultAdd_SeqSBAIJ_7(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar     *x,*y,*z,*xb,x1,x2,x3,x4,x5,x6,x7;
  MatScalar       *v;  
  int             mbs=a->mbs,i,*aj=a->j,*ai=a->i,n,ierr,*ib,cval,j,jmin; 

  PetscFunctionBegin;
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr);
  if (yy != xx) {
    ierr = VecGetArrayFast(yy,&y);CHKERRQ(ierr);
  } else {
    y = x;
  }
  if (zz != yy) {
    /* ierr = VecCopy(yy,zz);CHKERRQ(ierr); */
    ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr);
    ierr = PetscMemcpy(z,y,yy->n*sizeof(PetscScalar));CHKERRQ(ierr); 
  } else {
    z = y;
  }  

  v     = a->a;
  xb = x;

  for (i=0; i<mbs; i++) {
    n  = ai[1] - ai[0]; /* length of i_th block row of A */
    x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5=xb[4]; x6=xb[5]; x7=xb[6];
    ib = aj + *ai;
    jmin = 0;
    if (*ib == i){     /* (diag of A)*x */
      z[7*i]   += v[0]*x1 + v[7]*x2 + v[14]*x3 + v[21]*x4+ v[28]*x5 + v[35]*x6+ v[42]*x7;
      z[7*i+1] += v[7]*x1 + v[8]*x2 + v[15]*x3 + v[22]*x4+ v[29]*x5 + v[36]*x6+ v[43]*x7;
      z[7*i+2] += v[14]*x1+ v[15]*x2 +v[16]*x3 + v[23]*x4+ v[30]*x5 + v[37]*x6+ v[44]*x7;
      z[7*i+3] += v[21]*x1+ v[22]*x2 +v[23]*x3 + v[24]*x4+ v[31]*x5 + v[38]*x6+ v[45]*x7;
      z[7*i+4] += v[28]*x1+ v[29]*x2 +v[30]*x3 + v[31]*x4+ v[32]*x5 + v[39]*x6+ v[46]*x7;
      z[7*i+5] += v[35]*x1+ v[36]*x2 +v[37]*x3 + v[38]*x4+ v[39]*x5 + v[40]*x6+ v[47]*x7;
      z[7*i+6] += v[42]*x1+ v[43]*x2 +v[44]*x3 + v[45]*x4+ v[46]*x5 + v[47]*x6+ v[48]*x7;
      v += 49; jmin++;
    }
    for (j=jmin; j<n; j++) {
      /* (strict lower triangular part of A)*x  */
      cval       = ib[j]*7;
      z[cval]   += v[0]*x1  + v[1]*x2 + v[2]*x3 + v[3]*x4+ v[4]*x5 + v[5]*x6+ v[6]*x7;
      z[cval+1] += v[7]*x1  + v[8]*x2 + v[9]*x3 + v[10]*x4+ v[11]*x5 + v[12]*x6+ v[13]*x7;
      z[cval+2] += v[14]*x1  + v[15]*x2 + v[16]*x3 + v[17]*x4+ v[18]*x5 + v[19]*x6+ v[20]*x7;
      z[cval+3] += v[21]*x1  + v[22]*x2 + v[23]*x3 + v[24]*x4+ v[25]*x5 + v[26]*x6+ v[27]*x7;
      z[cval+4] += v[28]*x1  + v[29]*x2 + v[30]*x3 + v[31]*x4+ v[32]*x5 + v[33]*x6+ v[34]*x7;
      z[cval+5] += v[35]*x1  + v[36]*x2 + v[37]*x3 + v[38]*x4+ v[39]*x5 + v[40]*x6+ v[41]*x7;
      z[cval+6] += v[42]*x1  + v[43]*x2 + v[44]*x3 + v[45]*x4+ v[46]*x5 + v[47]*x6+ v[48]*x7;
      /* (strict upper triangular part of A)*x  */
      z[7*i]  +=v[0]*x[cval]+v[7]*x[cval+1]+v[14]*x[cval+2]+v[21]*x[cval+3]+v[28]*x[cval+4]+v[35]*x[cval+5]+v[42]*x[cval+6];
      z[7*i+1]+=v[1]*x[cval]+v[8]*x[cval+1]+v[15]*x[cval+2]+v[22]*x[cval+3]+v[29]*x[cval+4]+v[36]*x[cval+5]+v[43]*x[cval+6];
      z[7*i+2]+=v[2]*x[cval]+v[9]*x[cval+1]+v[16]*x[cval+2]+v[23]*x[cval+3]+v[30]*x[cval+4]+v[37]*x[cval+5]+v[44]*x[cval+6];
      z[7*i+3]+=v[3]*x[cval]+v[10]*x[cval+1]+v[17]*x[cval+2]+v[24]*x[cval+3]+v[31]*x[cval+4]+v[38]*x[cval+5]+v[45]*x[cval+6];
      z[7*i+4]+=v[4]*x[cval]+v[11]*x[cval+1]+v[18]*x[cval+2]+v[25]*x[cval+3]+v[32]*x[cval+4]+v[39]*x[cval+5]+v[46]*x[cval+6];
      z[7*i+5]+=v[5]*x[cval]+v[12]*x[cval+1]+v[19]*x[cval+2]+v[26]*x[cval+3]+v[33]*x[cval+4]+v[40]*x[cval+5]+v[47]*x[cval+6];
      z[7*i+6]+=v[6]*x[cval]+v[13]*x[cval+1]+v[20]*x[cval+2]+v[27]*x[cval+3]+v[34]*x[cval+4]+v[41]*x[cval+5]+v[48]*x[cval+6];
      v  += 49;      
    }
    xb +=7; ai++; 
  }

  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  if (yy != xx) ierr = VecRestoreArrayFast(yy,&y);CHKERRQ(ierr);
  if (zz != yy) ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);

  PetscLogFlops(98*(a->nz*2 - A->m));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqSBAIJ_N"
int MatMultAdd_SeqSBAIJ_N(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqSBAIJ    *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar     *x,*x_ptr,*y,*z,*z_ptr=0,*xb,*zb,*work,*workt;
  MatScalar       *v;
  int             ierr,mbs=a->mbs,i,*idx,*aj,*ii,bs=a->bs,j,n,bs2=a->bs2;
  int             ncols,k;

  PetscFunctionBegin;
  ierr = VecGetArrayFast(xx,&x);CHKERRQ(ierr); x_ptr=x;
  if (yy != xx) {
    ierr = VecGetArrayFast(yy,&y);CHKERRQ(ierr);
  } else {
    y = x;
  }
  if (zz != yy) {
    /* ierr = VecCopy(yy,zz);CHKERRQ(ierr); */
    ierr = VecGetArrayFast(zz,&z);CHKERRQ(ierr); z_ptr=z;
    ierr = PetscMemcpy(z,y,yy->n*sizeof(PetscScalar));CHKERRQ(ierr); 
  } else {
    z = y;
  } 

  aj   = a->j; 
  v    = a->a;
  ii   = a->i;

  if (!a->mult_work) {    
    ierr = PetscMalloc((A->m+1)*sizeof(PetscScalar),&a->mult_work);CHKERRQ(ierr);
  }
  work = a->mult_work; 
  
  
  for (i=0; i<mbs; i++) {
    n     = ii[1] - ii[0]; ncols = n*bs;
    workt = work; idx=aj+ii[0]; 

    /* upper triangular part */ 
    for (j=0; j<n; j++) { 
      xb = x_ptr + bs*(*idx++);
      for (k=0; k<bs; k++) workt[k] = xb[k];
      workt += bs;
    }
    /* z(i*bs:(i+1)*bs-1) += A(i,:)*x */
    Kernel_w_gets_w_plus_Ar_times_v(bs,ncols,work,v,z); 

    /* strict lower triangular part */
    idx = aj+ii[0];
    if (*idx == i){
      ncols -= bs; v += bs2; idx++; n--;
    }
    if (ncols > 0){
      workt = work;
      ierr  = PetscMemzero(workt,ncols*sizeof(PetscScalar));CHKERRQ(ierr);
      Kernel_w_gets_w_plus_trans_Ar_times_v(bs,ncols,x,v,workt);
      for (j=0; j<n; j++) {
        zb = z_ptr + bs*(*idx++); 
        /* idx++; */
        for (k=0; k<bs; k++) zb[k] += workt[k] ;
        workt += bs;
      }
    }

    x += bs; v += n*bs2; z += bs; ii++;
  }  

  ierr = VecRestoreArrayFast(xx,&x);CHKERRQ(ierr);
  if (yy != xx) ierr = VecRestoreArrayFast(yy,&y);CHKERRQ(ierr);
  if (zz != yy) ierr = VecRestoreArrayFast(zz,&z);CHKERRQ(ierr);

  PetscLogFlops(2*(a->nz*2 - A->m));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqSBAIJ"
int MatMultTranspose_SeqSBAIJ(Mat A,Vec xx,Vec zz)
{
  int ierr;

  PetscFunctionBegin;
  ierr = MatMult(A,xx,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqSBAIJ"
int MatMultTransposeAdd_SeqSBAIJ(Mat A,Vec xx,Vec yy,Vec zz)

{
  int ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd(A,xx,yy,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_SeqSBAIJ"
int MatScale_SeqSBAIJ(const PetscScalar *alpha,Mat inA)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)inA->data;
  int         one = 1,totalnz = a->bs2*a->nz;

  PetscFunctionBegin;
  BLscal_(&totalnz,(PetscScalar*)alpha,a->a,&one);
  PetscLogFlops(totalnz);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNorm_SeqSBAIJ"
int MatNorm_SeqSBAIJ(Mat A,NormType type,PetscReal *norm)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;
  MatScalar   *v = a->a;
  PetscReal   sum_diag = 0.0, sum_off = 0.0, *sum;
  int         i,j,k,bs = a->bs,bs2=a->bs2,k1,mbs=a->mbs,*aj=a->j;
  int         *jl,*il,jmin,jmax,ierr,nexti,ik,*col;
  
  PetscFunctionBegin;
  if (type == NORM_FROBENIUS) {
    for (k=0; k<mbs; k++){
      jmin = a->i[k]; jmax = a->i[k+1];
      col  = aj + jmin;
      if (*col == k){         /* diagonal block */
        for (i=0; i<bs2; i++){
#if defined(PETSC_USE_COMPLEX)
          sum_diag += PetscRealPart(PetscConj(*v)*(*v)); v++;
#else
          sum_diag += (*v)*(*v); v++;
#endif
        }
        jmin++;
      }
      for (j=jmin; j<jmax; j++){  /* off-diagonal blocks */
        for (i=0; i<bs2; i++){
#if defined(PETSC_USE_COMPLEX)
          sum_off += PetscRealPart(PetscConj(*v)*(*v)); v++;
#else
          sum_off += (*v)*(*v); v++;
#endif  
        }        
      }        
    }
    *norm = sqrt(sum_diag + 2*sum_off);

  }  else if (type == NORM_INFINITY) { /* maximum row sum */
    ierr = PetscMalloc(mbs*sizeof(int),&il);CHKERRQ(ierr); 
    ierr = PetscMalloc(mbs*sizeof(int),&jl);CHKERRQ(ierr);
    ierr = PetscMalloc(bs*sizeof(PetscReal),&sum);CHKERRQ(ierr);
    for (i=0; i<mbs; i++) {
      jl[i] = mbs; il[0] = 0;
    }

    *norm = 0.0;
    for (k=0; k<mbs; k++) { /* k_th block row */   
      for (j=0; j<bs; j++) sum[j]=0.0;

      /*-- col sum --*/
      i = jl[k]; /* first |A(i,k)| to be added */
      /* jl[k]=i: first nozero element in row i for submatrix A(1:k,k:n) (active window)
                  at step k */
      while (i<mbs){
        nexti = jl[i];  /* next block row to be added */
        ik    = il[i];  /* block index of A(i,k) in the array a */
        for (j=0; j<bs; j++){
          v = a->a + ik*bs2 + j*bs;
          for (k1=0; k1<bs; k1++) {
            sum[j] += PetscAbsScalar(*v); v++;
          } 
        }
        /* update il, jl */
        jmin = ik + 1; /* block index of array a: points to the next nonzero of A in row i */
        jmax = a->i[i+1];
        if (jmin < jmax){
          il[i] = jmin; 
          j   = a->j[jmin];
          jl[i] = jl[j]; jl[j]=i;
        }
        i = nexti; 
      }
      
      /*-- row sum --*/
      jmin = a->i[k]; jmax = a->i[k+1];
      for (i=jmin; i<jmax; i++) {
        for (j=0; j<bs; j++){
          v = a->a + i*bs2 + j; 
          for (k1=0; k1<bs; k1++){
            sum[j] += PetscAbsScalar(*v); 
            v   += bs;
          }
        }
      }
      /* add k_th block row to il, jl */
      col = aj+jmin;
      if (*col == k) jmin++;
      if (jmin < jmax){
        il[k] = jmin; 
        j   = a->j[jmin];
        jl[k] = jl[j]; jl[j] = k;
      }
      for (j=0; j<bs; j++){
        if (sum[j] > *norm) *norm = sum[j];
      } 
    }
    ierr = PetscFree(il);CHKERRQ(ierr);
    ierr = PetscFree(jl);CHKERRQ(ierr); 
    ierr = PetscFree(sum);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,"No support for this norm yet");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatEqual_SeqSBAIJ"
int MatEqual_SeqSBAIJ(Mat A,Mat B,PetscTruth* flg)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ *)A->data,*b = (Mat_SeqSBAIJ *)B->data;
  int          ierr;

  PetscFunctionBegin;

  /* If the  matrix/block dimensions are not equal, or no of nonzeros or shift */
  if ((A->m != B->m) || (A->n != B->n) || (a->bs != b->bs)|| (a->nz != b->nz)) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0); 
  }
  
  /* if the a->i are the same */
  ierr = PetscMemcmp(a->i,b->i,(a->mbs+1)*sizeof(int),flg);CHKERRQ(ierr);
  if (*flg == PETSC_FALSE) {
    PetscFunctionReturn(0);
  }
  
  /* if a->j are the same */
  ierr = PetscMemcmp(a->j,b->j,(a->nz)*sizeof(int),flg);CHKERRQ(ierr);
  if (*flg == PETSC_FALSE) {
    PetscFunctionReturn(0);
  }  
  /* if a->a are the same */
  ierr = PetscMemcmp(a->a,b->a,(a->nz)*(a->bs)*(a->bs)*sizeof(PetscScalar),flg);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_SeqSBAIJ"
int MatGetDiagonal_SeqSBAIJ(Mat A,Vec v)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;
  int          ierr,i,j,k,n,row,bs,*ai,*aj,ambs,bs2;
  PetscScalar  *x,zero = 0.0;
  MatScalar    *aa,*aa_j;

  PetscFunctionBegin;
  bs   = a->bs;
  if (A->factor && bs>1) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix with bs>1");   
  
  aa   = a->a;
  ai   = a->i;
  aj   = a->j;
  ambs = a->mbs;
  bs2  = a->bs2;  

  ierr = VecSet(&zero,v);CHKERRQ(ierr);
  ierr = VecGetArrayFast(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != A->m) SETERRQ(PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<ambs; i++) {
    j=ai[i];              
    if (aj[j] == i) {             /* if this is a diagonal element */
      row  = i*bs;      
      aa_j = aa + j*bs2;  
      if (A->factor && bs==1){
        for (k=0; k<bs2; k+=(bs+1),row++) x[row] = 1.0/aa_j[k];
      } else {
        for (k=0; k<bs2; k+=(bs+1),row++) x[row] = aa_j[k];  
      }     
    }
  }
  
  ierr = VecRestoreArrayFast(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalScale_SeqSBAIJ"
int MatDiagonalScale_SeqSBAIJ(Mat A,Vec ll,Vec rr)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;
  PetscScalar  *l,*r,x,*li,*ri;
  MatScalar    *aa,*v;
  int          ierr,i,j,k,lm,rn,M,m,*ai,*aj,mbs,tmp,bs,bs2;

  PetscFunctionBegin;
  ai  = a->i;
  aj  = a->j;
  aa  = a->a;
  m   = A->m;
  bs  = a->bs;
  mbs = a->mbs;
  bs2 = a->bs2;

  if (ll != rr) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"For symmetric format, left and right scaling vectors must be same\n");
  }
  if (ll) { 
    ierr = VecGetArrayFast(ll,&l);CHKERRQ(ierr);
    ierr = VecGetLocalSize(ll,&lm);CHKERRQ(ierr);
    if (lm != m) SETERRQ(PETSC_ERR_ARG_SIZ,"Left scaling vector wrong length");
    for (i=0; i<mbs; i++) { /* for each block row */
      M  = ai[i+1] - ai[i];
      li = l + i*bs;      
      v  = aa + bs2*ai[i];
      for (j=0; j<M; j++) { /* for each block */
        for (k=0; k<bs2; k++) {
          (*v++) *= li[k%bs];
        } 
#ifdef CONT
        /* will be used to replace the above loop */
        ri = l + bs*aj[ai[i]+j];
        for (k=0; k<bs; k++) { /* column value */
          x = ri[k];          
          for (tmp=0; tmp<bs; tmp++) (*v++) *= li[tmp]*x;
        } 
#endif

      }  
    }
    ierr = VecRestoreArrayFast(ll,&l);CHKERRQ(ierr);
    PetscLogFlops(2*a->nz);
  }
  /* will be deleted */
  if (rr) {
    ierr = VecGetArrayFast(rr,&r);CHKERRQ(ierr);
    ierr = VecGetLocalSize(rr,&rn);CHKERRQ(ierr);
    if (rn != m) SETERRQ(PETSC_ERR_ARG_SIZ,"Right scaling vector wrong length");
    for (i=0; i<mbs; i++) { /* for each block row */
      M  = ai[i+1] - ai[i];
      v  = aa + bs2*ai[i];
      for (j=0; j<M; j++) { /* for each block */
        ri = r + bs*aj[ai[i]+j];
        for (k=0; k<bs; k++) {
          x = ri[k];
          for (tmp=0; tmp<bs; tmp++) (*v++) *= x;
        } 
      }  
    }
    ierr = VecRestoreArrayFast(rr,&r);CHKERRQ(ierr);
    PetscLogFlops(a->nz);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_SeqSBAIJ"
int MatGetInfo_SeqSBAIJ(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;

  PetscFunctionBegin;
  info->rows_global    = (double)A->m;
  info->columns_global = (double)A->m;
  info->rows_local     = (double)A->m;
  info->columns_local  = (double)A->m;
  info->block_size     = a->bs2;
  info->nz_allocated   = a->maxnz; /*num. of nonzeros in upper triangular part */
  info->nz_used        = a->bs2*a->nz; /*num. of nonzeros in upper triangular part */ 
  info->nz_unneeded    = (double)(info->nz_allocated - info->nz_used);
  info->assemblies   = A->num_ass;
  info->mallocs      = a->reallocs;
  info->memory       = A->mem;
  if (A->factor) {
    info->fill_ratio_given  = A->info.fill_ratio_given;
    info->fill_ratio_needed = A->info.fill_ratio_needed;
    info->factor_mallocs    = A->info.factor_mallocs;
  } else {
    info->fill_ratio_given  = 0;
    info->fill_ratio_needed = 0;
    info->factor_mallocs    = 0;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries_SeqSBAIJ"
int MatZeroEntries_SeqSBAIJ(Mat A)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data; 
  int         ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(a->a,a->bs2*a->i[a->mbs]*sizeof(MatScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowMax_SeqSBAIJ"
int MatGetRowMax_SeqSBAIJ(Mat A,Vec v)
{
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;
  int          ierr,i,j,n,row,col,bs,*ai,*aj,mbs;
  PetscReal    atmp;
  MatScalar    *aa;
  PetscScalar  zero = 0.0,*x;
  int          ncols,brow,bcol,krow,kcol; 

  PetscFunctionBegin;
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");  
  bs   = a->bs;
  aa   = a->a;
  ai   = a->i;
  aj   = a->j;
  mbs = a->mbs;

  ierr = VecSet(&zero,v);CHKERRQ(ierr);
  ierr = VecGetArrayFast(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != A->m) SETERRQ(PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<mbs; i++) {
    ncols = ai[1] - ai[0]; ai++;
    brow  = bs*i;
    for (j=0; j<ncols; j++){
      bcol = bs*(*aj); 
      for (kcol=0; kcol<bs; kcol++){
        col = bcol + kcol;      /* col index */
        for (krow=0; krow<bs; krow++){         
          atmp = PetscAbsScalar(*aa); aa++;         
          row = brow + krow;    /* row index */
          /* printf("val[%d,%d]: %g\n",row,col,atmp); */
          if (PetscRealPart(x[row]) < atmp) x[row] = atmp;
          if (*aj > i && PetscRealPart(x[col]) < atmp) x[col] = atmp;
        }
      }
      aj++;
    }   
  }
  ierr = VecRestoreArrayFast(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
