
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: vector.c,v 1.158 1999/01/13 21:36:43 bsmith Exp bsmith $";
#endif
/*
     Provides the interface functions for all vector operations.
   These are the vector functions the user calls.
*/
#include "src/vec/vecimpl.h"    /*I "vec.h" I*/

#undef __FUNC__  
#define __FUNC__ "VecSetBlockSize"
/*@
   VecSetBlockSize - Sets the blocksize for future calls to VecSetValuesBlocked()
   and VecSetValuesBlockedLocal().

   Collective on Vec

   Input Parameter:
+  v - the vector
-  bs - the blocksize

   Notes:
   All vectors obtained by VecDuplicate() inherit the same blocksize

.seealso: VecSetValuesBlocked(); VecSetLocalToGlobalMappingBlocked()

.keywords: block size, vectors
@*/
int VecSetBlockSize(Vec v,int bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE); 
  if (v->N % bs) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,1,"Vector length not divisible by blocksize %d %d",v->N,bs);
  if (v->n % bs) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,1,"Local vector length not divisible by blocksize %d %d",v->n,bs);
  
  v->bs = bs;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecValid"
/*@
   VecValid - Checks whether a vector object is valid.

   Not Collective

   Input Parameter:
.  v - the object to check

   Output Parameter:
   flg - flag indicating vector status, either
   PETSC_TRUE if vector is valid, or PETSC_FALSE otherwise.

.keywords: vector, valid
@*/
int VecValid(Vec v,PetscTruth *flg)
{
  PetscFunctionBegin;
  PetscValidIntPointer(flg);
  if (!v)                           *flg = PETSC_FALSE;
  else if (v->cookie != VEC_COOKIE) *flg = PETSC_FALSE;
  else                              *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecDot"
/*@
   VecDot - Computes the vector dot product.

   Collective on Vec

   Input Parameters:
.  x, y - the vectors

   Output Parameter:
.  alpha - the dot product

   Notes for Users of Complex Numbers:
   For complex vectors, VecDot() computes 
$     val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

   Use VecTDot() for the indefinite form
$     val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

.keywords: vector, dot product, inner product

.seealso: VecMDot(), VecTDot()
@*/
int VecDot(Vec x, Vec y, Scalar *val)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE); 
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidScalarPointer(val);
  PetscCheckSameType(x,y);
  PLogEventBegin(VEC_Dot,x,y,0,0);
  ierr = (*x->ops->dot)(x,y,val); CHKERRQ(ierr);
  PLogEventEnd(VEC_Dot,x,y,0,0);
  /*
     The next block is for incremental debugging
  */
  if (PetscCompare) {
    int flag;
    ierr = MPI_Comm_compare(PETSC_COMM_WORLD,x->comm,&flag);CHKERRQ(ierr);
    if (flag != MPI_UNEQUAL) {
      ierr = PetscCompareScalar(*val);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecNorm"
/*@
   VecNorm  - Computes the vector norm.

   Collective on Vec

   Input Parameters:
+  x - the vector
-  type - one of NORM_1, NORM_2, NORM_INFINITY.  Also available
          NORM_1_AND_2, which computes both norms and stores them
          in a two element array.

   Output Parameter:
.  val - the norm 

   Notes:
$     NORM_1 denotes sum_i |x_i|
$     NORM_2 denotes sqrt( sum_i (x_i)^2 )
$     NORM_INFINITY denotes max_i |x_i|

.keywords: vector, norm

.seealso: VecDot(), VecTDot()

@*/
int VecNorm(Vec x,NormType type,double *val)  
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PLogEventBegin(VEC_Norm,x,0,0,0);
  ierr = (*x->ops->norm)(x,type,val); CHKERRQ(ierr);
  PLogEventEnd(VEC_Norm,x,0,0,0);
  /*
     The next block is for incremental debugging
  */
  if (PetscCompare) {
    int flag;
    ierr = MPI_Comm_compare(PETSC_COMM_WORLD,x->comm,&flag);CHKERRQ(ierr);
    if (flag != MPI_UNEQUAL) {
      ierr = PetscCompareDouble(*val);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecMax"
/*@C
   VecMax - Determines the maximum vector component and its location.

   Collective on Vec

   Input Parameter:
.  x - the vector

   Output Parameters:
+  val - the maximum component
-  p - the location of val

   Notes:
   Returns the value PETSC_MIN and p = -1 if the vector is of length 0.

.keywords: vector, maximum

.seealso: VecNorm(), VecMin()
@*/
int VecMax(Vec x,int *p,double *val)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidScalarPointer(val);
  PLogEventBegin(VEC_Max,x,0,0,0);
  ierr = (*x->ops->max)(x,p,val); CHKERRQ(ierr);
  PLogEventEnd(VEC_Max,x,0,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecMin"
/*@
   VecMin - Determines the minimum vector component and its location.

   Collective on Vec

   Input Parameters:
.  x - the vector

   Output Parameter:
+  val - the minimum component
-  p - the location of val

   Notes:
   Returns the value PETSC_MAX and p = -1 if the vector is of length 0.

.keywords: vector, minimum

.seealso: VecMax()
@*/
int VecMin(Vec x,int *p,double *val)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidScalarPointer(val);
  PLogEventBegin(VEC_Min,x,0,0,0);
  ierr = (*x->ops->min)(x,p,val); CHKERRQ(ierr);
  PLogEventEnd(VEC_Min,x,0,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecTDot"
/*@
   VecTDot - Computes an indefinite vector dot product. That is, this
   routine does NOT use the complex conjugate.

   Collective on Vec

   Input Parameters:
.  x, y - the vectors

   Output Parameter:
.  val - the dot product

   Notes for Users of Complex Numbers:
   For complex vectors, VecTDot() computes the indefinite form
$     val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

   Use VecDot() for the inner product
$     val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

.keywords: vector, dot product, inner product

.seealso: VecDot(), VecMTDot()
@*/
int VecTDot(Vec x,Vec y,Scalar *val) 
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidScalarPointer(val);
  PetscCheckSameType(x,y);
  PLogEventBegin(VEC_TDot,x,y,0,0);
  ierr = (*x->ops->tdot)(x,y,val); CHKERRQ(ierr);
  PLogEventEnd(VEC_TDot,x,y,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecScale"
/*@
   VecScale - Scales a vector. 

   Collective on Vec

   Input Parameters:
+  x - the vector
-  alpha - the scalar

   Output Parameter:
.  x - the scaled vector

   Note:
   For a vector with n components, VecScale() computes 
$      x[i] = alpha * x[i], for i=1,...,n.

.keywords: vector, scale
@*/
int VecScale(Scalar *alpha,Vec x)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidScalarPointer(alpha);
  PLogEventBegin(VEC_Scale,x,0,0,0);
  ierr = (*x->ops->scale)(alpha,x); CHKERRQ(ierr);
  PLogEventEnd(VEC_Scale,x,0,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecCopy"
/*@
   VecCopy - Copies a vector. 

   Collective on Vec

   Input Parameter:
.  x - the vector

   Output Parameter:
.  y - the copy

   Notes:
   For default parallel PETSc vectors, both x and y must be distributed in
   the same manner; local copies are done.

.keywords: vector, copy

.seealso: VecDuplicate()
@*/
int VecCopy(Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE); 
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PLogEventBegin(VEC_Copy,x,y,0,0);
  ierr = (*x->ops->copy)(x,y); CHKERRQ(ierr);
  PLogEventEnd(VEC_Copy,x,y,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSet"
/*@
   VecSet - Sets all components of a vector to a single scalar value. 

   Collective on Vec

   Input Parameters:
+  alpha - the scalar
-  x  - the vector

   Output Parameter:
.  x  - the vector

   Note:
   For a vector of dimension n, VecSet() computes
$     x[i] = alpha, for i=1,...,n,
   so that all vector entries then equal the identical
   scalar value, alpha.  Use the more general routine
   VecSetValues() to set different vector entries.

.seealso VecSetValues(), VecSetValuesBlocked(), VecSetRandom()

.keywords: vector, set
@*/
int VecSet(Scalar *alpha,Vec x) 
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidScalarPointer(alpha);
  PLogEventBegin(VEC_Set,x,0,0,0);
  ierr = (*x->ops->set)(alpha,x); CHKERRQ(ierr);
  PLogEventEnd(VEC_Set,x,0,0,0);
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "VecSetRandom"
/*@C
   VecSetRandom - Sets all components of a vector to random numbers.

   Collective on Vec

   Input Parameters:
+  rctx - the random number context, formed by PetscRandomCreate()
-  x  - the vector

   Output Parameter:
.  x  - the vector

   Example of Usage:
.vb
     PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rctx);
     VecSetRandom(rctx,x);
     PetscRandomDestroy(rctx);
.ve

.keywords: vector, set, random

.seealso: VecSet(), VecSetValues(), PetscRandomCreate(), PetscRandomDestroy()
@*/
int VecSetRandom(PetscRandom rctx,Vec x) 
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(rctx,PETSCRANDOM_COOKIE);
  PLogEventBegin(VEC_SetRandom,x,rctx,0,0);
  ierr = (*x->ops->setrandom)(rctx,x); CHKERRQ(ierr);
  PLogEventEnd(VEC_SetRandom,x,rctx,0,0);
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "VecAXPY"
/*@
   VecAXPY - Computes y = alpha x + y. 

   Collective on Vec

   Input Parameters:
+  alpha - the scalar
-  x, y  - the vectors

   Output Parameter:
.  y - output vector

.keywords: vector, saxpy

.seealso: VecAYPX(), VecMAXPY(), VecWAXPY()
@*/
int VecAXPY(Scalar *alpha,Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidScalarPointer(alpha);
  PLogEventBegin(VEC_AXPY,x,y,0,0);
  ierr = (*x->ops->axpy)(alpha,x,y); CHKERRQ(ierr);
  PLogEventEnd(VEC_AXPY,x,y,0,0);
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "VecAXPBY"
/*@
   VecAXPBY - Computes y = alpha x + beta y. 

   Collective on Vec

   Input Parameters:
+  alpha,beta - the scalars
-  x, y  - the vectors

   Output Parameter:
.  y - output vector

.keywords: vector, saxpy

.seealso: VecAYPX(), VecMAXPY(), VecWAXPY(), VecAXPY()
@*/
int VecAXPBY(Scalar *alpha,Scalar *beta,Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidScalarPointer(alpha);
  PetscValidScalarPointer(beta);
  PLogEventBegin(VEC_AXPY,x,y,0,0);
  ierr = (*x->ops->axpby)(alpha,beta,x,y); CHKERRQ(ierr);
  PLogEventEnd(VEC_AXPY,x,y,0,0);
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "VecAYPX"
/*@
   VecAYPX - Computes y = x + alpha y.

   Collective on Vec

   Input Parameters:
+  alpha - the scalar
-  x, y  - the vectors

   Output Parameter:
.  y - output vector

.keywords: vector, saypx

.seealso: VecAXPY(), VecWAXPY()
@*/
int VecAYPX(Scalar *alpha,Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE); 
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidScalarPointer(alpha);
  PLogEventBegin(VEC_AYPX,x,y,0,0);
  ierr =  (*x->ops->aypx)(alpha,x,y); CHKERRQ(ierr);
  PLogEventEnd(VEC_AYPX,x,y,0,0);
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "VecSwap"
/*@
   VecSwap - Swaps the vectors x and y.

   Collective on Vec

   Input Parameters:
.  x, y  - the vectors

.keywords: vector, swap
@*/
int VecSwap(Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);  
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscCheckSameType(x,y);
  PLogEventBegin(VEC_Swap,x,y,0,0);
  ierr = (*x->ops->swap)(x,y); CHKERRQ(ierr);
  PLogEventEnd(VEC_Swap,x,y,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecWAXPY"
/*@
   VecWAXPY - Computes w = alpha x + y.

   Collective on Vec

   Input Parameters:
+  alpha - the scalar
-  x, y  - the vectors

   Output Parameter:
.  w - the result

.keywords: vector, waxpy

.seealso: VecAXPY(), VecAYPX()
@*/
int VecWAXPY(Scalar *alpha,Vec x,Vec y,Vec w)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE); 
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(w,VEC_COOKIE);
  PetscValidScalarPointer(alpha);
  PetscCheckSameType(x,y); PetscCheckSameType(y,w);
  PLogEventBegin(VEC_WAXPY,x,y,w,0);
  ierr =  (*x->ops->waxpy)(alpha,x,y,w); CHKERRQ(ierr);
  PLogEventEnd(VEC_WAXPY,x,y,w,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecPointwiseMult"
/*@
   VecPointwiseMult - Computes the componentwise multiplication w = x*y.

   Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

.keywords: vector, multiply, componentwise, pointwise

.seealso: VecPointwiseDivide()
@*/
int VecPointwiseMult(Vec x,Vec y,Vec w)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE); 
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(w,VEC_COOKIE);
  PLogEventBegin(VEC_PMult,x,y,w,0);
  ierr = (*x->ops->pointwisemult)(x,y,w); CHKERRQ(ierr);
  PLogEventEnd(VEC_PMult,x,y,w,0);
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "VecPointwiseDivide"
/*@
   VecPointwiseDivide - Computes the componentwise division w = x/y.

   Collective on Vec

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

.keywords: vector, divide, componentwise, pointwise

.seealso: VecPointwiseMult()
@*/
int VecPointwiseDivide(Vec x,Vec y,Vec w)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE); 
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(w,VEC_COOKIE);
  ierr = (*x->ops->pointwisedivide)(x,y,w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecDuplicate"
/*@C
   VecDuplicate - Creates a new vector of the same type as an existing vector.

   Collective on Vec

   Input Parameters:
.  v - a vector to mimic

   Output Parameter:
.  newv - location to put new vector

   Notes:
   VecDuplicate() does not copy the vector, but rather allocates storage
   for the new vector.  Use VecCopy() to copy a vector.

   Use VecDestroy() to free the space. Use VecDuplicateVecs() to get several
   vectors. 

.keywords: vector, duplicate, create

.seealso: VecDestroy(), VecDuplicateVecs(), VecCreate(), VecCopy()
@*/
int VecDuplicate(Vec v,Vec *newv) 
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  PetscValidPointer(newv);
  ierr = (*v->ops->duplicate)(v,newv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecDestroy"
/*@C
   VecDestroy - Destroys a vector.

   Collective on Vec

   Input Parameters:
.  v  - the vector

.keywords: vector, destroy

.seealso: VecDuplicate()
@*/
int VecDestroy(Vec v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  if (--v->refct > 0) PetscFunctionReturn(0);
  /* destroy the internal part */
  ierr = (*v->ops->destroy)(v);CHKERRQ(ierr);
  /* destroy the external/common part */
  if (v->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(v->mapping); CHKERRQ(ierr);
  }
  if (v->map) {
    ierr = MapDestroy(v->map);CHKERRQ(ierr);
  }
  PLogObjectDestroy(v);
  PetscHeaderDestroy(v); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecDuplicateVecs"
/*@C
   VecDuplicateVecs - Creates several vectors of the same type as an existing vector.

   Collective on Vec

   Input Parameters:
+  m - the number of vectors to obtain
-  v - a vector to mimic

   Output Parameter:
.  V - location to put pointer to array of vectors

   Notes:
   Use VecDestroyVecs() to free the space. Use VecDuplicate() to form a single
   vector.

   Fortran Note:
   The Fortran interface is slightly different from that given below, it 
   requires one to pass in V a Vec (integer) array of size at least m.
   See the Fortran chapter of the users manual and petsc/src/vec/examples for details.

.keywords: vector, duplicate

.seealso:  VecDestroyVecs(), VecDuplicate(), VecCreate(), VecDuplicateVecsF90()
@*/
int VecDuplicateVecs(Vec v,int m,Vec **V)  
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  PetscValidPointer(V);
  ierr = (*v->ops->duplicatevecs)( v, m,V );CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecDestroyVecs"
/*@C
   VecDestroyVecs - Frees a block of vectors obtained with VecDuplicateVecs().

   Collective on Vec

   Input Parameters:
+  vv - pointer to array of vector pointers
-  m - the number of vectors previously obtained

   Fortran Note:
   The Fortran interface is slightly different from that given below.
   See the Fortran chapter of the users manual and 
   petsc/src/vec/examples for details.

.keywords: vector, destroy

.seealso: VecDuplicateVecs(), VecDestroyVecsF90()
@*/
int VecDestroyVecs(Vec *vv,int m)
{
  int ierr;

  PetscFunctionBegin;
  if (!vv) SETERRQ(PETSC_ERR_ARG_BADPTR,0,"Null vectors");
  PetscValidHeaderSpecific(*vv,VEC_COOKIE);
  ierr = (*(*vv)->ops->destroyvecs)( vv, m );CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSetValues"
/*@
   VecSetValues - Inserts or adds values into certain locations of a vector. 

   Input Parameters:
   Not Collective

+  x - vector to insert in
.  ni - number of elements to add
.  ix - indices where to add
.  y - array of values
-  iora - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes: 
   VecSetValues() sets x[ix[i]] = y[i], for i=0,...,ni-1.

   Calls to VecSetValues() with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so VecAssemblyBegin() and VecAssemblyEnd() 
   MUST be called after all calls to VecSetValues() have been completed.

   VecSetValues() uses 0-based indices in Fortran as well as in C.

.keywords: vector, set, values

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValuesLocal(),
           VecSetValue(), VecSetValuesBlocked()
@*/
int VecSetValues(Vec x,int ni,int *ix,Scalar *y,InsertMode iora) 
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidIntPointer(ix);
  PetscValidScalarPointer(y);
  PLogEventBegin(VEC_SetValues,x,0,0,0);
  ierr = (*x->ops->setvalues)( x, ni,ix, y,iora ); CHKERRQ(ierr);
  PLogEventEnd(VEC_SetValues,x,0,0,0);  
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSetValuesBlocked"
/*@
   VecSetValuesBlocked - Inserts or adds blocks of values into certain locations of a vector. 

   Not Collective

   Input Parameters:
+  x - vector to insert in
.  ni - number of blocks to add
.  ix - indices where to add in block count, rather than element count
.  y - array of values
-  iora - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes: 
   VecSetValuesBlocked() sets x[ix[bs*i]+j] = y[bs*i+j], 
   for j=0,...,bs, for i=0,...,ni-1. where bs was set with VecSetBlockSize().

   Calls to VecSetValuesBlocked() with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so VecAssemblyBegin() and VecAssemblyEnd() 
   MUST be called after all calls to VecSetValuesBlocked() have been completed.

   VecSetValuesBlocked() uses 0-based indices in Fortran as well as in C.

.keywords: vector, set, values

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValuesBlockedLocal(),
           VecSetValues()
@*/
int VecSetValuesBlocked(Vec x,int ni,int *ix,Scalar *y,InsertMode iora) 
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidIntPointer(ix);
  PetscValidScalarPointer(y);
  PLogEventBegin(VEC_SetValues,x,0,0,0);
  ierr = (*x->ops->setvaluesblocked)( x, ni,ix, y,iora ); CHKERRQ(ierr);
  PLogEventEnd(VEC_SetValues,x,0,0,0);  
  PetscFunctionReturn(0);
}

/*MC
   VecSetValue - Set a single entry into a vector.

   Synopsis:
   void VecSetValue(Vec v,int row,Scalar value, InsertMode mode);

   Not Collective

   Input Parameters:
+  v - the vector
.  row - the row location of the entry
.  value - the value to insert
-  mode - either INSERT_VALUES or ADD_VALUES

   Notes:
   For efficiency one should use VecSetValues() and set several or 
   many values simultaneously if possible.

   Note that VecSetValue() does NOT return an error code (since this
   is checked internally).

.seealso: VecSetValues()
M*/

#undef __FUNC__  
#define __FUNC__ "VecSetLocalToGlobalMapping"
/*@
   VecSetLocalToGlobalMapping - Sets a local numbering to global numbering used
   by the routine VecSetValuesLocal() to allow users to insert vector entries
   using a local (per-processor) numbering.

   Collective on Vec

   Input Parameters:
+  x - vector
-  mapping - mapping created with ISLocalToGlobalMappingCreate() or ISLocalToGlobalMappingCreateIS()

   Notes: 
   All vectors obtained with VecDuplicate() from this vector inherit the same mapping.

.keywords: vector, set, values, local ordering

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValues(), VecSetValuesLocal(),
           VecSetLocalToGlobalMappingBlocked(), VecSetValuesBlockedLocal()
@*/
int VecSetLocalToGlobalMapping(Vec x, ISLocalToGlobalMapping mapping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE);

  if (x->mapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Mapping already set for vector");
  }

  x->mapping = mapping;
  PetscObjectReference((PetscObject)mapping);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSetLocalToGlobalMappingBlocked"
/*@
   VecSetLocalToGlobalMappingBlocked - Sets a local numbering to global numbering used
   by the routine VecSetValuesBlockedLocal() to allow users to insert vector entries
   using a local (per-processor) numbering.

   Collective on Vec

   Input Parameters:
+  x - vector
-  mapping - mapping created with ISLocalToGlobalMappingCreate() or ISLocalToGlobalMappingCreateIS()

   Notes: 
   All vectors obtained with VecDuplicate() from this vector inherit the same mapping.

.keywords: vector, set, values, local ordering

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValues(), VecSetValuesLocal(),
           VecSetLocalToGlobalMapping(), VecSetValuesBlockedLocal()
@*/
int VecSetLocalToGlobalMappingBlocked(Vec x, ISLocalToGlobalMapping mapping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(mapping,IS_LTOGM_COOKIE);

  if (x->bmapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Mapping already set for vector");
  }
  x->bmapping = mapping;
  PetscObjectReference((PetscObject)mapping);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSetValuesLocal"
/*@
   VecSetValuesLocal - Inserts or adds values into certain locations of a vector,
   using a local ordering of the nodes. 

   Not Collective

   Input Parameters:
+  x - vector to insert in
.  ni - number of elements to add
.  ix - indices where to add
.  y - array of values
-  iora - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes: 
   VecSetValuesLocal() sets x[ix[i]] = y[i], for i=0,...,ni-1.

   Calls to VecSetValues() with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so VecAssemblyBegin() and VecAssemblyEnd() 
   MUST be called after all calls to VecSetValuesLocal() have been completed.

   VecSetValuesLocal() uses 0-based indices in Fortran as well as in C.

.keywords: vector, set, values, local ordering

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValues(), VecSetLocalToGlobalMapping(),
           VecSetValuesBlockedLocal()
@*/
int VecSetValuesLocal(Vec x,int ni,int *ix,Scalar *y,InsertMode iora) 
{
  int ierr,lixp[128],*lix = lixp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidIntPointer(ix);
  PetscValidScalarPointer(y);
  if (!x->mapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Local to global never set with VecSetLocalToGlobalMapping()");
  }
  if (ni > 128) {
    lix = (int *) PetscMalloc( ni*sizeof(int) );CHKPTRQ(lix);
  }

  PLogEventBegin(VEC_SetValues,x,0,0,0);
  ierr = ISLocalToGlobalMappingApply(x->mapping,ni,ix,lix); CHKERRQ(ierr);
  ierr = (*x->ops->setvalues)( x,ni,lix, y,iora ); CHKERRQ(ierr);
  PLogEventEnd(VEC_SetValues,x,0,0,0);  
  if (ni > 128) {
    PetscFree(lix);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSetValuesBlockedLocal"
/*@
   VecSetValuesBlockedLocal - Inserts or adds values into certain locations of a vector,
   using a local ordering of the nodes. 

   Not Collective

   Input Parameters:
+  x - vector to insert in
.  ni - number of blocks to add
.  ix - indices where to add in block count, not element count
.  y - array of values
-  iora - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes: 
   VecSetValuesBlockedLocal() sets x[bs*ix[i]+j] = y[bs*i+j], 
   for j=0,..bs-1, for i=0,...,ni-1, where bs has been set with VecSetBlockSize().

   Notes:
   Calls to VecSetValuesBlockedLocal() with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so VecAssemblyBegin() and VecAssemblyEnd() 
   MUST be called after all calls to VecSetValuesBlockedLocal() have been completed.

   VecSetValuesBlockedLocal() uses 0-based indices in Fortran as well as in C.

.keywords: vector, set, values, local ordering

.seealso:  VecAssemblyBegin(), VecAssemblyEnd(), VecSetValues(), VecSetValuesBlocked(), 
           VecSetLocalToGlobalMapping(), VecSetLocalToGlobalMappingBlocked()
@*/
int VecSetValuesBlockedLocal(Vec x,int ni,int *ix,Scalar *y,InsertMode iora) 
{
  int ierr,lixp[128],*lix = lixp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidIntPointer(ix);
  PetscValidScalarPointer(y);
  if (!x->bmapping) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Local to global never set with VecSetLocalToGlobalMappingBlocked()");
  }
  if (ni > 128) {
    lix = (int *) PetscMalloc( ni*sizeof(int) );CHKPTRQ(lix);
  }

  PLogEventBegin(VEC_SetValues,x,0,0,0);
  ierr = ISLocalToGlobalMappingApply(x->bmapping,ni,ix,lix); CHKERRQ(ierr);
  ierr = (*x->ops->setvaluesblocked)( x,ni,lix, y,iora ); CHKERRQ(ierr);
  PLogEventEnd(VEC_SetValues,x,0,0,0);  
  if (ni > 128) {
    PetscFree(lix);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecAssemblyBegin"
/*@
   VecAssemblyBegin - Begins assembling the vector.  This routine should
   be called after completing all calls to VecSetValues().

   Collective on Vec

   Input Parameter:
.  vec - the vector

.keywords: vector, begin, assembly, assemble

.seealso: VecAssemblyEnd(), VecSetValues()
@*/
int VecAssemblyBegin(Vec vec)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE);
  PLogEventBegin(VEC_AssemblyBegin,vec,0,0,0);
  if (vec->ops->assemblybegin) {
    ierr = (*vec->ops->assemblybegin)(vec); CHKERRQ(ierr);
  }
  PLogEventEnd(VEC_AssemblyBegin,vec,0,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecAssemblyEnd"
/*@
   VecAssemblyEnd - Completes assembling the vector.  This routine should
   be called after VecAssemblyBegin().

   Collective on Vec

   Input Parameter:
.  vec - the vector

.keywords: vector, end, assembly, assemble

.seealso: VecAssemblyBegin(), VecSetValues()
@*/
int VecAssemblyEnd(Vec vec)
{
  int ierr,flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE);
  PLogEventBegin(VEC_AssemblyEnd,vec,0,0,0);
  if (vec->ops->assemblyend) {
    ierr = (*vec->ops->assemblyend)(vec); CHKERRQ(ierr);
  }
  PLogEventEnd(VEC_AssemblyEnd,vec,0,0,0);
  ierr = OptionsHasName(PETSC_NULL,"-vec_view",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = VecView(vec,VIEWER_STDOUT_(vec->comm)); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-vec_view_matlab",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_(vec->comm),VIEWER_FORMAT_ASCII_MATLAB,"V");CHKERRQ(ierr);
    ierr = VecView(vec,VIEWER_STDOUT_(vec->comm)); CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_(vec->comm));CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-vec_view_draw",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = VecView(vec,VIEWER_DRAW_(vec->comm)); CHKERRQ(ierr);
    ierr = ViewerFlush(VIEWER_DRAW_(vec->comm)); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-vec_view_draw_lg",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = ViewerSetFormat(VIEWER_DRAW_(vec->comm),VIEWER_FORMAT_DRAW_LG,0); CHKERRQ(ierr);
    ierr = VecView(vec,VIEWER_DRAW_(vec->comm)); CHKERRQ(ierr);
    ierr = ViewerFlush(VIEWER_DRAW_(vec->comm)); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-vec_view_socket",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = MatView(mat,VIEWER_SOCKET_(vec->comm)); CHKERRQ(ierr);
    ierr = ViewerFlush(VIEWER_SOCKET_(vec->comm)); CHKERRQ(ierr);
  }
#if defined(HAVE_AMS)
  ierr = OptionsHasName(PETSC_NULL,"-vec_view_ams",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = VecView(vec,VIEWER_AMS_(vec->comm)); CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "VecMTDot"
/*@C
   VecMTDot - Computes indefinite vector multiple dot products. 
   That is, it does NOT use the complex conjugate.

   Collective on Vec

   Input Parameters:
+  nv - number of vectors
.  x - one vector
-  y - array of vectors.  Note that vectors are pointers

   Output Parameter:
.  val - array of the dot products

   Notes for Users of Complex Numbers:
   For complex vectors, VecMTDot() computes the indefinite form
$      val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

   Use VecMDot() for the inner product
$      val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

.keywords: vector, dot product, inner product, non-Hermitian, multiple

.seealso: VecMDot(), VecTDot()
@*/
int VecMTDot(int nv,Vec x,Vec *y,Scalar *val)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(*y,VEC_COOKIE);
  PetscValidScalarPointer(val);
  PetscCheckSameType(x,*y);
  PLogEventBegin(VEC_MTDot,x,*y,0,0);
  ierr = (*x->ops->mtdot)(nv,x,y,val); CHKERRQ(ierr);
  PLogEventEnd(VEC_MTDot,x,*y,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecMDot"
/*@C
   VecMDot - Computes vector multiple dot products. 

   Collective on Vec

   Input Parameters:
+  nv - number of vectors
.  x - one vector
-  y - array of vectors. 

   Output Parameter:
.  val - array of the dot products

   Notes for Users of Complex Numbers:
   For complex vectors, VecMDot() computes 
$     val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

   Use VecMTDot() for the indefinite form
$     val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

.keywords: vector, dot product, inner product, multiple

.seealso: VecMTDot(), VecDot()
@*/
int VecMDot(int nv,Vec x,Vec *y,Scalar *val)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE); 
  PetscValidHeaderSpecific(*y,VEC_COOKIE);
  PetscValidScalarPointer(val);
  PetscCheckSameType(x,*y);
  PLogEventBegin(VEC_MDot,x,*y,0,0);
  ierr = (*x->ops->mdot)(nv,x,y,val); CHKERRQ(ierr);
  PLogEventEnd(VEC_MDot,x,*y,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecMAXPY"
/*@C
   VecMAXPY - Computes x = x + sum alpha[j] y[j]

   Collective on Vec

   Input Parameters:
+  nv - number of scalars and x-vectors
.  alpha - array of scalars
.  x - one vector
-  y - array of vectors

   Output Parameter:
.  y  - array of vectors

.keywords: vector, saxpy, maxpy, multiple

.seealso: VecAXPY(), VecWAXPY(), VecAYPX()
@*/
int  VecMAXPY(int nv,Scalar *alpha,Vec x,Vec *y)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(*y,VEC_COOKIE);
  PetscValidScalarPointer(alpha);
  PetscCheckSameType(x,*y);
  PLogEventBegin(VEC_MAXPY,x,*y,0,0);
  ierr = (*x->ops->maxpy)(nv,alpha,x,y); CHKERRQ(ierr);
  PLogEventEnd(VEC_MAXPY,x,*y,0,0);
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "VecGetArray"
/*@C
   VecGetArray - Returns a pointer to a contiguous array that contains this 
     processor's portion of the vector data. For the standard PETSc
     vectors, VecGetArray() returns a pointer to the local data array and
     does not use any copies. If the underlying vector data is not stored
     in a contiquous array this routine will copy the data to a contiquous
     array and return a pointer to that. You MUST call VecRestoreArray() 
     when you no longer need access to the array.

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameter:
.  a - location to put pointer to the array

   Fortran Note:
   This routine is used differently from Fortran
$    Vec         x
$    Scalar      x_array(1)
$    PetscOffset i_x
$    int         ierr
$       call VecGetArray(x,x_array,i_x,ierr)
$
$   Access first local entry in vector with
$      value = x_array(i_x + 1)
$
$      ...... other code
$       call VecRestoreArray(x,x_array,i_x,ierr)

   See the Fortran chapter of the users manual and 
   petsc/src/snes/examples/tutorials/ex5f.F for details.

.keywords: vector, get, array

.seealso: VecRestoreArray(), VecGetArrays(), VecGetArrayF90(), VecPlaceArray()
@*/
int VecGetArray(Vec x,Scalar **a)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidPointer(a);

  ierr = (*x->ops->getarray)(x,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecGetArrays" 
/*@C
   VecGetArrays - Returns a pointer to the arrays in a set of vectors
   that were created by a call to VecDuplicateVecs().  You MUST call
   VecRestoreArrays() when you no longer need access to the array.

   Not Collective

   Input Parameter:
+  x - the vectors
-  n - the number of vectors

   Output Parameter:
.  a - location to put pointer to the array

   Fortran Note:
   This routine is not supported in Fortran.

.keywords: vector, get, arrays

.seealso: VecGetArray(), VecRestoreArrays()
@*/
int VecGetArrays(Vec *x,int n,Scalar ***a)
{
  int    i,ierr;
  Scalar **q;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(*x,VEC_COOKIE);
  PetscValidPointer(a);
  if (n <= 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Must get at least one array n = %d",n);
  q = (Scalar **) PetscMalloc(n*sizeof(Scalar*)); CHKPTRQ(q);
  for (i=0; i<n; ++i) {
    ierr = VecGetArray(x[i],&q[i]); CHKERRQ(ierr);
  }
  *a = q;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecRestoreArrays"
/*@C
   VecRestoreArrays - Restores a group of vectors after VecGetArrays()
   has been called.

   Not Collective

   Input Parameters:
+  x - the vector
.  n - the number of vectors
-  a - location of pointer to arrays obtained from VecGetArrays()

   Fortran Note:
   This routine is not supported in Fortran.

.keywords: vector, restore, arrays

.seealso: VecGetArrays(), VecRestoreArray()
@*/
int VecRestoreArrays(Vec *x,int n,Scalar ***a)
{
  int    i,ierr;
  Scalar **q = *a;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(*x,VEC_COOKIE);
  PetscValidPointer(a);
  q = *a;
  for(i=0;i<n;++i) {
    ierr = VecRestoreArray(x[i],&q[i]); CHKERRQ(ierr);
  }
  PetscFree(q);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecRestoreArray"
/*@C
   VecRestoreArray - Restores a vector after VecGetArray() has been called.
        For regular PETSc vectors this routine does not do any copies. For
        any special vectors that do not store the vector data in a contiquous
        array this routine will copy the data back into the underlying 
        vector data structure from the array obtained with VecGetArray().

   Not Collective

   Input Parameters:
+  x - the vector
-  a - location of pointer to array obtained from VecGetArray()

   Fortran Note:
   This routine is used differently from Fortran
$    Vec         x
$    Scalar      x_array(1)
$    PetscOffset i_x
$    int         ierr
$       call VecGetArray(x,x_array,i_x,ierr)
$
$   Access first local entry in vector with
$      value = x_array(i_x + 1)
$
$      ...... other code
$       call VecRestoreArray(x,x_array,i_x,ierr)

   See the Fortran chapter of the users manual and 
   petsc/src/snes/examples/tutorials/ex5f.F for details.

.keywords: vector, restore, array

.seealso: VecGetArray(), VecRestoreArrays(), VecRestoreArrayF90(), VecPlaceArray()
@*/
int VecRestoreArray(Vec x,Scalar **a)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidPointer(a);
  if (x->ops->restorearray) {
    ierr = (*x->ops->restorearray)(x,a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecView"
/*@C
   VecView - Views a vector object. 

   Collective on Vec unless Viewer is VIEWER_STDOUT_SELF

   Input Parameters:
+  v - the vector
-  viewer - an optional visualization context

   Notes:
   The available visualization contexts include
+     VIEWER_STDOUT_SELF - standard output (default)
-     VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   You can change the format the vector is printed using the 
   option ViewerSetFormat().

   The user can open alternative visualization contexts with
+    ViewerASCIIOpen() - Outputs vector to a specified file
.    ViewerBinaryOpen() - Outputs vector in binary to a
         specified file; corresponding input uses VecLoad()
.    ViewerDrawOpen() - Outputs vector to an X window display
-    ViewerSocketOpen() - Outputs vector to Socket viewer

   The user can call ViewerSetFormat() to specify the output
   format of ASCII printed objects (when using VIEWER_STDOUT_SELF,
   VIEWER_STDOUT_WORLD and ViewerASCIIOpen).  Available formats include
+    VIEWER_FORMAT_ASCII_DEFAULT - default, prints vector contents
.    VIEWER_FORMAT_ASCII_MATLAB - prints vector contents in Matlab format
.    VIEWER_FORMAT_ASCII_INDEX - prints vector contents, including indices of vector elements
.    VIEWER_FORMAT_ASCII_COMMON - prints vector contents, using a 
         format common among all vector types
.    VIEWER_FORMAT_ASCII_INFO - prints basic information about the matrix
         size and structure (not the matrix entries)
-    VIEWER_FORMAT_ASCII_INFO_LONG - prints more detailed information about
         the matrix structure

.keywords: Vec, view, visualize, output, print, write, draw

.seealso: ViewerASCIIOpen(), ViewerDrawOpen(), DrawLGCreate(),
          ViewerSocketOpen(), ViewerBinaryOpen(), VecLoad()
@*/
int VecView(Vec v,Viewer viewer)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  if (!viewer) {viewer = VIEWER_STDOUT_SELF;}
  else { PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);}
  ierr = (*v->ops->view)(v,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecGetSize"
/*@
   VecGetSize - Returns the global number of elements of the vector.

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameters:
.  size - the global length of the vector

.keywords: vector, get, size, global, dimension

.seealso: VecGetLocalSize()
@*/
int VecGetSize(Vec x,int *size)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidIntPointer(size);
  ierr = (*x->ops->getsize)(x,size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecGetLocalSize"
/*@
   VecGetLocalSize - Returns the number of elements of the vector stored 
   in local memory. This routine may be implementation dependent, so use 
   with care.

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameter:
.  size - the length of the local piece of the vector

.keywords: vector, get, dimension, size, local

.seealso: VecGetSize()
@*/
int VecGetLocalSize(Vec x,int *size)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidIntPointer(size);
  ierr = (*x->ops->getlocalsize)(x,size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecGetOwnershipRange"
/*@
   VecGetOwnershipRange - Returns the range of indices owned by 
   this processor, assuming that the vectors are laid out with the
   first n1 elements on the first processor, next n2 elements on the
   second, etc.  For certain parallel layouts this range may not be 
   well defined. 

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameters:
+  low - the first local element
-  high - one more than the last local element

  Note:
  The high argument is one more than the last element stored locally.

.keywords: vector, get, range, ownership
@*/
int VecGetOwnershipRange(Vec x,int *low,int *high)
{
  int ierr;
  Map map;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidIntPointer(low);
  PetscValidIntPointer(high);
  ierr = (*x->ops->getmap)(x,&map);CHKERRQ(ierr);
  ierr = MapGetLocalRange(map,low,high);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecGetMap"
/*@C
   VecGetMap - Returns the map associated with the vector

   Not Collective

   Input Parameter:
.  x - the vector

   Output Parameters:
.  map - the map

.keywords: vector, get, map
@*/
int VecGetMap(Vec x,Map *map)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  ierr = (*x->ops->getmap)(x,map);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSetOption"
/*@
   VecSetOption - Allows one to set options for a vectors behavior.

   Collective on Vec

   Input Parameter:
+  x - the vector
-  op - the option

  Note: Currently the only option supported is
  VEC_IGNORE_OFF_PROC_ENTRIES, which causes VecSetValues() to ignore 
  entries destined to be stored on a seperate processor.

.keywords: vector, options
@*/
int VecSetOption(Vec x,VecOption op)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (x->ops->setoption) {
    ierr = (*x->ops->setoption)(x,op); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecDuplicateVecs_Default"
/* Default routines for obtaining and releasing; */
/* may be used by any implementation */
int VecDuplicateVecs_Default(Vec w,int m,Vec **V )
{
  int  i,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(w,VEC_COOKIE);
  PetscValidPointer(V);
  if (m <= 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"m must be > 0: m = %d",m);
  *V = (Vec *) PetscMalloc( m * sizeof(Vec *)); CHKPTRQ(*V);
  for (i=0; i<m; i++) {ierr = VecDuplicate(w,*V+i); CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecDestroyVecs_Default"
int VecDestroyVecs_Default( Vec *v, int m )
{
  int i,ierr;

  PetscFunctionBegin;
  PetscValidPointer(v);
  if (m <= 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"m must be > 0: m = %d",m);
  for (i=0; i<m; i++) {ierr = VecDestroy(v[i]); CHKERRQ(ierr);}
  PetscFree( v );
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecPlaceArray"
/*@
   VecPlaceArray - Allows one to replace the array in a vector with an
   array provided by the user. This is useful to avoid copying an array
   into a vector.  FOR EXPERTS ONLY!

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the array

   Notes:
   You should back up the original array by calling VecGetArray() and 
   stashing the value somewhere.  Then when finished using the vector,
   call VecPlaceArray() with that stashed value; otherwise, you may
   lose access to the original array.

.seealso: VecGetArray(), VecRestoreArray()

.keywords: vec, place, array
@*/
int VecPlaceArray(Vec vec,Scalar *array)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE);
  if (vec->ops->placearray) {
    ierr = (*vec->ops->placearray)(vec,array);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Cannot place array in this type of vector");
  }
  PetscFunctionReturn(0);
}

/*MC
    VecDuplicateVecsF90 - Creates several vectors of the same type as an existing vector
    and makes them accessible via a Fortran90 pointer.

    Synopsis:
    VecGetArrayF90(Vec x,int n,{Scalar, pointer :: y(:)},integer ierr)

    Collective on Vec

    Input Parameters:
+   x - a vector to mimic
-   n - the number of vectors to obtain

    Output Parameters:
+   y - Fortran90 pointer to the array of vectors
-   ierr - error code

    Example of Usage: 
.vb
    Vec x
    Vec, pointer :: y(:)
    ....
    call VecDuplicateVecsF90(x,2,y,ierr)
    call VecSet(alpha,y(2),ierr)
    call VecSet(alpha,y(2),ierr)
    ....
    call VecDestroyVecsF90(y,2,ierr)
.ve

    Notes:
     Not yet supported for all F90 compilers

    Use VecDestroyVecsF90() to free the space.

.seealso:  VecDestroyVecsF90(), VecDuplicateVecs()

.keywords:  vector, duplicate, f90
M*/

/*MC
    VecRestoreArrayF90 - Restores a vector to a usable state after a call to
    VecGetArrayF90().

    Synopsis:
    VecRestoreArrayF90(Vec x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Not collective

    Input Parameters:
+   x - vector
-   xx_v - the Fortran90 pointer to the array

    Output Parameter:
.   ierr - error code

    Example of Usage: 
.vb
    Scalar, pointer :: xx_v(:)
    ....
    call VecGetArrayF90(x,xx_v,ierr)
    a = xx_v(3)
    call VecRestoreArrayF90(x,xx_v,ierr)
.ve
   
    Notes:
     Not yet supported for all F90 compilers

.seealso:  VecGetArrayF90(), VecGetArray(), VecRestoreArray()

.keywords:  vector, array, f90
M*/

/*MC
    VecDestroyVecsF90 - Frees a block of vectors obtained with VecDuplicateVecsF90().

    Synopsis:
    VecDestroyVecsF90({Scalar, pointer :: x(:)},integer n,integer ierr)

    Input Parameters:
+   x - pointer to array of vector pointers
-   n - the number of vectors previously obtained

    Output Parameter:
.   ierr - error code

    Notes:
     Not yet supported for all F90 compilers

.seealso:  VecDestroyVecs(), VecDuplicateVecsF90()

.keywords:  vector, destroy, f90
M*/

/*MC
    VecGetArrayF90 - Accesses a vector array from Fortran90. For default PETSc
    vectors, VecGetArrayF90() returns a pointer to the local data array. Otherwise,
    this routine is implementation dependent. You MUST call VecRestoreArrayF90() 
    when you no longer need access to the array.

    Synopsis:
    VecGetArrayF90(Vec x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Not Collective 

    Input Parameter:
.   x - vector

    Output Parameters:
+   xx_v - the Fortran90 pointer to the array
-   ierr - error code

    Example of Usage: 
.vb
    Scalar, pointer :: xx_v(:)
    ....
    call VecGetArrayF90(x,xx_v,ierr)
    a = xx_v(3)
    call VecRestoreArrayF90(x,xx_v,ierr)
.ve

    Notes:
     Not yet supported for all F90 compilers

.seealso:  VecRestoreArrayF90(), VecGetArray(), VecRestoreArray()

.keywords:  vector, array, f90
M*/
