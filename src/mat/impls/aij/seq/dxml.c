#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dxml.c,v 1.11 1997/08/22 15:13:29 bsmith Exp bsmith $";
#endif

/* 
        Provides an interface to the DEC Alpha DXML library
     At the moment the DXNL library only offers sparse matrix vector product.
*/
#include "src/mat/impls/aij/seq/aij.h"
#include "src/vec/vecimpl.h"

#if defined(HAVE_DXML) && !defined(__cplusplus)

#undef __FUNC__  
#define __FUNC__ "MatMult_SeqAIJ_DXML"
static int MatMult_SeqAIJ_DXML(Mat A,Vec x,Vec y)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*) A->data;
  Scalar             *xx,*yy;
  int                ierr, zero = 0;

  PetscFunctionBegin;
  VecGetArray_Fast(x,xx);
  VecGetArray_Fast(y,yy);
  dmatvec_genr_(&zero,a->a,a->i,a->j,&a->nz,0,xx,yy,&a->m);
  PLogFlops(2*a->nz - a->m);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatUseDXML_SeqAIJ"
int MatUseDXML_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE);  
  if (A->type != MATSEQAIJ) PetscFunctionReturn(0);
  A->ops.mult    = MatMult_SeqAIJ_DXML;
  PetscFunctionReturn(0);
}

#else

#undef __FUNC__  
#define __FUNC__ "MatUseDXML_SeqAIJ"
int MatUseDXML_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}


#endif
