
/* $Id: pvec2.c,v 1.17 1996/12/17 18:00:10 balay Exp balay $ */

/*
     Code for some of the parallel vector primatives.
*/
#include <math.h>
#include "pvecimpl.h" 
#include "src/inline/dot.h"

#undef __FUNCTION__  
#define __FUNCTION__ "VecMDot_MPI"
static int VecMDot_MPI( int nv, Vec xin, Vec *y, Scalar *z )
{
  static Scalar awork[128];
  Scalar        *work = awork;

  if (nv > 128) {
    work = (Scalar *)PetscMalloc(nv * sizeof(Scalar)); CHKPTRQ(work);
  }
  VecMDot_Seq(  nv, xin, y, work );
#if defined(PETSC_COMPLEX)
  MPI_Allreduce( work,z,2*nv,MPI_DOUBLE,MPI_SUM,xin->comm );
#else
  MPI_Allreduce( work,z,nv,MPI_DOUBLE,MPI_SUM,xin->comm );
#endif
  if (nv > 128) {
    PetscFree(work);
  }
  return 0;
}


#undef __FUNCTION__  
#define __FUNCTION__ "VecNorm_MPI"
static int VecNorm_MPI(  Vec xin,NormType type, double *z )
{
  Vec_MPI      *x = (Vec_MPI *) xin->data;
  double       sum, work = 0.0;
  Scalar       *xx = x->array;
  register int n = x->n;

  if (type == NORM_2) {
#if defined(PETSC_COMPLEX)
    int i;
    for (i=0; i<n; i++ ) {
      work += real(conj(xx[i])*xx[i]);
    }
#else
    SQR(work,xx,n);
#endif
    MPI_Allreduce( &work, &sum,1,MPI_DOUBLE,MPI_SUM,xin->comm );
    *z = sqrt( sum );
    PLogFlops(2*x->n);
  }
  else if (type == NORM_1) {
    /* Find the local part */
    VecNorm_Seq( xin, NORM_1, &work );
    /* Find the global max */
    MPI_Allreduce( &work, z,1,MPI_DOUBLE,MPI_SUM,xin->comm );
  }
  else if (type == NORM_INFINITY) {
    /* Find the local max */
    VecNorm_Seq( xin, NORM_INFINITY, &work );
    /* Find the global max */
    MPI_Allreduce( &work, z,1,MPI_DOUBLE,MPI_MAX,xin->comm );
  }
  else if (type == NORM_1) {
    VecNorm_Seq( xin, NORM_1, &work );
    MPI_Allreduce( &work, z,1,MPI_DOUBLE,MPI_SUM,xin->comm );
  }
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "VecMax_MPI"
static int VecMax_MPI( Vec xin, int *idx, double *z )
{
  double work;

  /* Find the local max */
  VecMax_Seq( xin, idx, &work );

  /* Find the global max */
  if (!idx) {
    MPI_Allreduce( &work, z,1,MPI_DOUBLE,MPI_MAX,xin->comm );
  }
  else {
    /* Need to use special linked max */
    SETERRQ( 1, "Parallel max with index not supported" );
  }
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "VecMin_MPI"
static int VecMin_MPI( Vec xin, int *idx, double *z )
{
  double work;

  /* Find the local Min */
  VecMin_Seq( xin, idx, &work );

  /* Find the global Min */
  if (!idx) {
    MPI_Allreduce( &work, z,1,MPI_DOUBLE,MPI_MIN,xin->comm );
  }
  else {
    /* Need to use special linked Min */
    SETERRQ( 1, "Parallel Min with index not supported" );
  }
  return 0;
}
