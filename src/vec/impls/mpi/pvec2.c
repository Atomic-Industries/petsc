
/* $Id: pvec2.c,v 1.28 1997/09/09 19:09:17 curfman Exp bsmith $ */

/*
     Code for some of the parallel vector primatives.
*/
#include <math.h>
#include "src/vec/impls/mpi/pvecimpl.h" 
#include "src/inline/dot.h"

#undef __FUNC__  
#define __FUNC__ "VecMDot_MPI"
int VecMDot_MPI( int nv, Vec xin, Vec *y, Scalar *z )
{
  Scalar awork[128],*work = awork;
  int    ierr;

  if (nv > 128) {
    work = (Scalar *) PetscMalloc(nv * sizeof(Scalar)); CHKPTRQ(work);
  }
  ierr = VecMDot_Seq(  nv, xin, y, work ); CHKERRQ(ierr);
  PLogEventBarrierBegin(VEC_MDotBarrier,0,0,0,0,xin->comm);
#if defined(PETSC_COMPLEX)
  MPI_Allreduce(work,z,2*nv,MPI_DOUBLE,MPI_SUM,xin->comm );
#else
  MPI_Allreduce(work,z,nv,MPI_DOUBLE,MPI_SUM,xin->comm );
  PLogEventBarrierEnd(VEC_MDotBarrier,0,0,0,0,xin->comm);
#endif
  if (nv > 128) {
    PetscFree(work);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "VecMTDot_MPI"
int VecMTDot_MPI( int nv, Vec xin, Vec *y, Scalar *z )
{
  Scalar awork[128],*work = awork;
  int    ierr;

  if (nv > 128) {
    work = (Scalar *) PetscMalloc(nv * sizeof(Scalar)); CHKPTRQ(work);
  }
  ierr = VecMTDot_Seq(  nv, xin, y, work ); CHKERRQ(ierr);
  PLogEventBarrierBegin(VEC_MDotBarrier,0,0,0,0,xin->comm);
#if defined(PETSC_COMPLEX)
  MPI_Allreduce(work,z,2*nv,MPI_DOUBLE,MPI_SUM,xin->comm );
#else
  MPI_Allreduce(work,z,nv,MPI_DOUBLE,MPI_SUM,xin->comm );
#endif
  PLogEventBarrierEnd(VEC_MDotBarrier,0,0,0,0,xin->comm);
  if (nv > 128) {
    PetscFree(work);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "VecNorm_MPI"
int VecNorm_MPI(  Vec xin,NormType type, double *z )
{
  Vec_MPI      *x = (Vec_MPI *) xin->data;
  double       sum, work = 0.0;
  Scalar       *xx = x->array;
  register int n = x->n;

  if (type == NORM_2) {

#if defined(USE_FORTRAN_KERNELS)
  fortrannormsqr_(xx,&n,&work);
#else
    /* int i; for ( i=0; i<n; i++ ) work += xx[i]*xx[i];   */
    switch (n & 0x3) {
      case 3: work += PetscReal(xx[0]*PetscConj(xx[0])); xx++;
      case 2: work += PetscReal(xx[0]*PetscConj(xx[0])); xx++;
      case 1: work += PetscReal(xx[0]*PetscConj(xx[0])); xx++; n -= 4;
    }
    while (n>0) {
      work += PetscReal(xx[0]*PetscConj(xx[0])+xx[1]*PetscConj(xx[1])+
                        xx[2]*PetscConj(xx[2])+xx[3]*PetscConj(xx[3]));
      xx += 4; n -= 4;
    } 
    /*
         On the IBM Power2 Super with four memory cards unrolling to 4
         worked better than unrolling to 8.
    */
    /*
    switch (n & 0x7) {
      case 7: work += PetscReal(xx[0]*PetscConj(xx[0])); xx++;
      case 6: work += PetscReal(xx[0]*PetscConj(xx[0])); xx++;
      case 5: work += PetscReal(xx[0]*PetscConj(xx[0])); xx++;
      case 4: work += PetscReal(xx[0]*PetscConj(xx[0])); xx++;
      case 3: work += PetscReal(xx[0]*PetscConj(xx[0])); xx++;
      case 2: work += PetscReal(xx[0]*PetscConj(xx[0])); xx++;
      case 1: work += PetscReal(xx[0]*PetscConj(xx[0])); xx++; n -= 8;
    }
    while (n>0) {
      work += PetscReal(xx[0]*PetscConj(xx[0])+xx[1]*PetscConj(xx[1])+
                        xx[2]*PetscConj(xx[2])+xx[3]*PetscConj(xx[3])+
                        xx[4]*PetscConj(xx[4])+xx[5]*PetscConj(xx[5])+
                        xx[6]*PetscConj(xx[6])+xx[7]*PetscConj(xx[7]));
      xx += 8; n -= 8;
    } 
    */
#endif
    PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,xin->comm);
    MPI_Allreduce(&work, &sum,1,MPI_DOUBLE,MPI_SUM,xin->comm );
    PLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,xin->comm);
    *z = sqrt( sum );
    PLogFlops(2*x->n);
  } else if (type == NORM_1) {
    /* Find the local part */
    VecNorm_Seq( xin, NORM_1, &work );
    /* Find the global max */
    PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,xin->comm);
    MPI_Allreduce( &work, z,1,MPI_DOUBLE,MPI_SUM,xin->comm );
    PLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,xin->comm);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    VecNorm_Seq( xin, NORM_INFINITY, &work );
    /* Find the global max */
    PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,xin->comm);
    MPI_Allreduce(&work, z,1,MPI_DOUBLE,MPI_MAX,xin->comm );
    PLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,xin->comm);
  } else if (type == NORM_1_AND_2) {
    double temp[2];
    VecNorm_Seq( xin, NORM_1, temp );
    VecNorm_Seq( xin, NORM_2, temp+1 ); 
    temp[1] = temp[1]*temp[1];
    PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,xin->comm);
    MPI_Allreduce(temp, z,2,MPI_DOUBLE,MPI_SUM,xin->comm );
    PLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,xin->comm);
    z[1] = sqrt(z[1]);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "VecMax_MPI"
int VecMax_MPI( Vec xin, int *idx, double *z )
{
  int    ierr;
  double work;

  /* Find the local max */
  ierr = VecMax_Seq( xin, idx, &work ); CHKERRQ(ierr);

  /* Find the global max */
  if (!idx) {
    PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,xin->comm);
    MPI_Allreduce(&work, z,1,MPI_DOUBLE,MPI_MAX,xin->comm );
    PLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,xin->comm);
  }
  else {
    /* Need to use special linked max */
    SETERRQ( 1,0, "Parallel max with index not supported" );
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "VecMin_MPI"
int VecMin_MPI( Vec xin, int *idx, double *z )
{
  int    ierr;
  double work;

  /* Find the local Min */
  ierr = VecMin_Seq( xin, idx, &work ); CHKERRQ(ierr);

  /* Find the global Min */
  if (!idx) {
    PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,xin->comm);
    MPI_Allreduce(&work, z,1,MPI_DOUBLE,MPI_MIN,xin->comm );
    PLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,xin->comm);
  }
  else {
    /* Need to use special linked Min */
    SETERRQ( 1,0, "Parallel Min with index not supported" );
  }
  return 0;
}
