#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: axpy.c,v 1.31 1998/03/12 23:20:13 bsmith Exp bsmith $";
#endif

#include "src/mat/matimpl.h"  /*I   "mat.h"  I*/

#undef __FUNC__  
#define __FUNC__ "MatAXPY"
/*@
   MatAXPY - Computes Y = a*X + Y.

   Input Parameters:
.  X,Y - the matrices
.  a - the scalar multiplier

   Collective on Mat

   Contributed by: Matthew Knepley

.keywords: matrix, add

 @*/
int MatAXPY(Scalar *a,Mat X,Mat Y)
{
  int    m1,m2,n1,n2,i,*row,start,end,j,ncols,ierr;
  Scalar *val,*vals;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,MAT_COOKIE); 
  PetscValidHeaderSpecific(Y,MAT_COOKIE);
  PetscValidScalarPointer(a);

  MatGetSize(X,&m1,&n1);  MatGetSize(Y,&m2,&n2);
  if (m1 != m2 || n1 != n2) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Non conforming matrix add");

  if (X->ops->axpy) {
    ierr = (*X->ops->axpy)(a,X,Y); CHKERRQ(ierr);
  } else {
    ierr = MatGetOwnershipRange(X,&start,&end); CHKERRQ(ierr);
    if (*a == 1.0) {
      for (i = start; i < end; i++) {
        ierr = MatGetRow(X, i, &ncols, &row, &vals);                 CHKERRQ(ierr);
        ierr = MatSetValues(Y, 1, &i, ncols, row, vals, ADD_VALUES); CHKERRQ(ierr);
        ierr = MatRestoreRow(X, i, &ncols, &row, &vals);             CHKERRQ(ierr);
      }
    } else {
      vals = (Scalar *) PetscMalloc( (n1+1)*sizeof(Scalar) ); CHKPTRQ(vals);
      for ( i=start; i<end; i++ ) {
        ierr = MatGetRow(X,i,&ncols,&row,&val); CHKERRQ(ierr);
        for ( j=0; j<ncols; j++ ) {
          vals[j] = (*a)*val[j];
        }
        ierr = MatSetValues(Y,1,&i,ncols,row,vals,ADD_VALUES); CHKERRQ(ierr);
        ierr = MatRestoreRow(X,i,&ncols,&row,&val); CHKERRQ(ierr);
      }
      PetscFree(vals);
    }
    ierr = MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatShift"
/*@
   MatShift - Computes Y =  Y + a I, where a is a scalar and I is the identity
   matrix.

   Input Parameters:
.  Y - the matrices
.  a - the scalar 

   Collective on Mat

.keywords: matrix, add, shift

.seealso: MatDiagonalShift()
 @*/
int MatShift(Scalar *a,Mat Y)
{
  int    i,start,end,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,MAT_COOKIE);
  PetscValidScalarPointer(a);
  if (Y->ops->shift) {
    ierr = (*Y->ops->shift)(a,Y); CHKERRQ(ierr);
  }
  else {
    ierr = MatGetOwnershipRange(Y,&start,&end); CHKERRQ(ierr);
    for ( i=start; i<end; i++ ) {
      ierr = MatSetValues(Y,1,&i,1,&i,a,ADD_VALUES); CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDiagonalShift"
/*@
   MatDiagonalShift - Computes Y = Y + D, where D is a diagonal matrix
   that is represented as a vector.

   Input Parameters:
.  Y - the input matrix
.  D - the diagonal matrix, represented as a vector

   Input Parameters:
.  Y - the shifted ouput matrix

   Collective on Mat and Vec

.keywords: matrix, add, shift, diagonal

.seealso: MatShift()
@*/
int MatDiagonalShift(Mat Y,Vec D)
{
  int    i,start,end,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,MAT_COOKIE);
  PetscValidHeaderSpecific(D,VEC_COOKIE);
  if (Y->ops->shift) {
    ierr = (*Y->ops->diagonalshift)(D,Y); CHKERRQ(ierr);
  }
  else {
    int    vstart,vend;
    Scalar *v;
    ierr = VecGetOwnershipRange(D,&vstart,&vend); CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(Y,&start,&end); CHKERRQ(ierr);
    if (vstart != start || vend != end) {
      SETERRQ(PETSC_ERR_ARG_SIZ,0,"Vector ownership range not compatible with matrix");
    }

    ierr = VecGetArray(D,&v); CHKERRQ(ierr);
    for ( i=start; i<end; i++ ) {
      ierr = MatSetValues(Y,1,&i,1,&i,v+i-start,ADD_VALUES); CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatAYPX"
/*@
   MatAYPX - Computes Y = X + a*Y.

   Input Parameters:
.  X,Y - the matrices
.  a - the scalar multiplier

   Collective on Mat

   Contributed by: Matthew Knepley

.keywords: matrix, add

 @*/
int MatAYPX(Scalar *a,Mat X,Mat Y)
{
  Scalar one = 1.0;
  int    mX, mY,nX, nY,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, MAT_COOKIE);
  PetscValidHeaderSpecific(Y, MAT_COOKIE);
  PetscValidScalarPointer(a);

  MatGetSize(X, &mX, &nX);
  MatGetSize(X, &mY, &nY);
  if (mX != mY || nX != nY) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Non conforming matrices");

  ierr = MatScale(a, Y);      CHKERRQ(ierr);
  ierr = MatAXPY(&one, X, Y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
