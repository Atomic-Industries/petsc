
#ifndef lint
static char vcid[] = "$Id: essl.c,v 1.18 1997/01/06 20:24:23 balay Exp bsmith $";
#endif

/* 
        Provides an interface to the IBM RS6000 Essl sparse solver

*/
#include "src/mat/impls/aij/seq/aij.h"
#include "src/vec/vecimpl.h"

#if defined(HAVE_ESSL) && !defined(__cplusplus)
/* #include <essl.h> This doesn't work!  */
#include <math.h>

typedef struct {
   int    n,nz;
   Scalar *a;
   int    *ia;
   int    *ja;
   int    lna;
   int    iparm[5];
   double rparm[5];
   double oparm[5];
   Scalar *aux;
   int    naux;
} Mat_SeqAIJ_Essl;


extern int MatDestroy_SeqAIJ(PetscObject);

#undef __FUNC__  
#define __FUNC__ "MatDestroy_SeqAIJ_Essl" /* ADIC Ignore */
static int MatDestroy_SeqAIJ_Essl(PetscObject obj)
{
  Mat             A = (Mat) obj;
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*) A->data;
  Mat_SeqAIJ_Essl *essl = (Mat_SeqAIJ_Essl*) a->spptr;

  /* free the Essl datastructures */
  PetscFree(essl->a);
  return MatDestroy_SeqAIJ(obj);
}

#undef __FUNC__  
#define __FUNC__ "MatSolve_SeqAIJ_Essl"
static int MatSolve_SeqAIJ_Essl(Mat A,Vec b,Vec x)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*) A->data;
  Mat_SeqAIJ_Essl *essl = (Mat_SeqAIJ_Essl*) a->spptr;
  Scalar          *xx;
  int             ierr,m, zero = 0;

  VecGetLocalSize_Fast(b,m);
  ierr = VecCopy(b,x); CHKERRQ(ierr);
  VecGetArray_Fast(x,xx);

  dgss(&zero, &a->n, essl->a, essl->ia, essl->ja,&essl->lna,xx,essl->aux,&essl->naux);

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatLUFactorSymbolic_SeqAIJ_Essl"
static int MatLUFactorSymbolic_SeqAIJ_Essl(Mat A,IS r,IS c,double f,Mat *F)
{
  Mat             B;
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*) A->data, *b;
  int             ierr, *ridx, *cidx,i, len;
  Mat_SeqAIJ_Essl *essl;

  if (a->m != a->n) SETERRQ(1,0,"matrix must be square"); 
  ierr          = MatCreateSeqAIJ(A->comm,a->m,a->n,0,PETSC_NULL,F); CHKERRQ(ierr);
  B             = *F;
  B->ops.solve  = MatSolve_SeqAIJ_Essl;
  B->destroy    = MatDestroy_SeqAIJ_Essl;
  B->factor     = FACTOR_LU;
  b             = (Mat_SeqAIJ*) B->data;
  essl          = PetscNew(Mat_SeqAIJ_Essl); CHKPTRQ(essl);
  b->spptr      = (void*) essl;

  /* allocate the work arrays required by ESSL */
  essl->nz   = a->nz;
  essl->lna  = (int) a->nz*f;
  essl->naux = 100 + 10*a->m;

  /* since malloc is slow on IBM we try a single malloc */
  len        = essl->lna*(2*sizeof(int)+sizeof(Scalar)) + 
               essl->naux*sizeof(Scalar);
  essl->a    = (Scalar*) PetscMalloc(len); CHKPTRQ(essl->a);
  essl->aux  = essl->a + essl->lna;
  essl->ia   = (int*) (essl->aux + essl->naux);
  essl->ja   = essl->ia + essl->lna;

  PLogObjectMemory(B,len+sizeof(Mat_SeqAIJ_Essl));
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatLUFactorNumeric_SeqAIJ_Essl"
static int MatLUFactorNumeric_SeqAIJ_Essl(Mat A,Mat *F)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*) (*F)->data;
  Mat_SeqAIJ      *aa = (Mat_SeqAIJ*) (A)->data;
  Mat_SeqAIJ_Essl *essl = (Mat_SeqAIJ_Essl *) a->spptr;
  int             i,ierr, one = 1;

  /* copy matrix data into silly ESSL data structure */
  if (!a->indexshift) {
    for ( i=0; i<aa->m+1; i++ ) essl->ia[i] = aa->i[i] + 1;
    for ( i=0; i<aa->nz; i++ ) essl->ja[i]  = aa->j[i] + 1;
  }
  else {
    PetscMemcpy(essl->ia,aa->i,(aa->m+1)*sizeof(int));
    PetscMemcpy(essl->ja,aa->j,(aa->nz)*sizeof(int));
  }
  PetscMemcpy(essl->a,aa->a,(aa->nz)*sizeof(Scalar));
  
  /* set Essl options */
  essl->iparm[0] = 1; 
  essl->iparm[1] = 5;
  essl->iparm[2] = 1;
  essl->iparm[3] = 0;
  essl->rparm[0] = 1.e-12;
  essl->rparm[1] = A->lupivotthreshold;

  dgsf(&one,&aa->m,&essl->nz,essl->a,essl->ia,essl->ja,&essl->lna,essl->iparm,
               essl->rparm,essl->oparm,essl->aux,&essl->naux);

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatUseEssl_SeqAIJ" /* ADIC Ignore */
int MatUseEssl_SeqAIJ(Mat A)
{
  PetscValidHeaderSpecific(A,MAT_COOKIE);  
  if (A->type != MATSEQAIJ) return 0;

  A->ops.lufactorsymbolic = MatLUFactorSymbolic_SeqAIJ_Essl;
  A->ops.lufactornumeric  = MatLUFactorNumeric_SeqAIJ_Essl;

  return 0;
}

#else

#undef __FUNC__  
#define __FUNC__ "MatUseEssl_SeqAIJ" /* ADIC Ignore */
int MatUseEssl_SeqAIJ(Mat A)
{
  return 0;
}

#endif

