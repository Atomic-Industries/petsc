#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: cholbs.c,v 1.45 1997/01/27 18:16:50 bsmith Exp balay $";
#endif

#if defined(HAVE_BLOCKSOLVE) && !defined(PETSC_COMPLEX)

/* We must define MLOG for BlockSolve logging */ 
#if defined(PETSC_LOG)
#define MLOG
#endif

#include "src/pc/pcimpl.h"
#include "src/mat/impls/rowbs/mpi/mpirowbs.h"



#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_MPIRowbs"
int MatCholeskyFactorNumeric_MPIRowbs(Mat mat,Mat *factp) 
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data;

#if defined(PETSC_LOG)
  double flop1 = BSlocal_flops();
#endif

  /* Do prep work if same nonzero structure as previously factored matrix */
  if (mbs->factor == FACTOR_CHOLESKY) {
    /* Copy the nonzeros */
    BScopy_nz(mbs->pA,mbs->fpA); CHKERRBS(0);
  }
  /* Form incomplete Cholesky factor */
  mbs->ierr = 0; mbs->failures = 0; mbs->alpha = 1.0;
  while ((mbs->ierr = BSfactor(mbs->fpA,mbs->comm_fpA,mbs->procinfo))) {
    CHKERRBS(0); mbs->failures++;
    /* Copy only the nonzeros */
    BScopy_nz(mbs->pA,mbs->fpA); CHKERRBS(0);
    /* Increment the diagonal shift */
    mbs->alpha += 0.1;
    BSset_diag(mbs->fpA,mbs->alpha,mbs->procinfo); CHKERRBS(0);
    PLogInfo(mat,"BlockSolve95: %d failed factor(s), err=%d, alpha=%g\n",
                                 mbs->failures,mbs->ierr,mbs->alpha); 
  }
#if defined(PETSC_LOG)
  PLogFlops((int)(BSlocal_flops()-flop1));
#endif

  mbs->factor = FACTOR_CHOLESKY;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatLUFactorNumeric_MPIRowbs"
int MatLUFactorNumeric_MPIRowbs(Mat mat,Mat *factp) 
{
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) mat->data;

#if defined(PETSC_LOG)
  double flop1 = BSlocal_flops();
#endif

  /* Do prep work if same nonzero structure as previously factored matrix */
  if (mbs->factor == FACTOR_LU) {
    /* Copy the nonzeros */
    BScopy_nz(mbs->pA,mbs->fpA); CHKERRBS(0);
  }
  /* Form incomplete Cholesky factor */
  mbs->ierr = 0; mbs->failures = 0; mbs->alpha = 1.0;
  while ((mbs->ierr = BSfactor(mbs->fpA,mbs->comm_fpA,mbs->procinfo))) {
    CHKERRBS(0); mbs->failures++;
    /* Copy only the nonzeros */
    BScopy_nz(mbs->pA,mbs->fpA); CHKERRBS(0);
    /* Increment the diagonal shift */
    mbs->alpha += 0.1;
    BSset_diag(mbs->fpA,mbs->alpha,mbs->procinfo); CHKERRBS(0);
    PLogInfo(mat,"BlockSolve95: %d failed factor(s), err=%d, alpha=%g\n",
                                       mbs->failures,mbs->ierr,mbs->alpha); 
  }
  mbs->factor = FACTOR_LU;
  (*factp)->assembled = PETSC_TRUE;
#if defined(PETSC_LOG)
  PLogFlops((int)(BSlocal_flops()-flop1));
#endif
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "MatSolve_MPIRowbs"
int MatSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat          submat = (Mat) mat->data;
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) submat->data;
  int          ierr;
  Scalar       *ya, *xa, *xworka;

#if defined(PETSC_LOG)
  double flop1 = BSlocal_flops();
#endif
  /* Permute and apply diagonal scaling to vector, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecGetArray(x,&xa); CHKERRQ(ierr);
    ierr = VecGetArray(mbs->xwork,&xworka); CHKERRQ(ierr);
    BSperm_dvec(xa,xworka,mbs->pA->perm); CHKERRBS(0);
    ierr = VecPointwiseMult(mbs->diag,mbs->xwork,y); CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x,y); CHKERRQ(ierr);
  }
  ierr = VecGetArray(y,&ya); CHKERRQ(ierr);

  if (mbs->procinfo->single)
    /* Use BlockSolve routine for no cliques/inodes */
    BSfor_solve1(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  else
    BSfor_solve(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  CHKERRBS(0);

  if (mbs->procinfo->single)
    /* Use BlockSolve routine for no cliques/inodes */
    BSback_solve1(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  else
    BSback_solve(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  CHKERRBS(0);

  /* Apply diagonal scaling and unpermute, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecPointwiseMult(y,mbs->diag,mbs->xwork);  CHKERRQ(ierr);
    BSiperm_dvec(xworka,ya,mbs->pA->perm); CHKERRBS(0);
    ierr = VecRestoreArray(x,&xa); CHKERRQ(ierr);
    ierr = VecRestoreArray(mbs->xwork,&xworka); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(y,&ya); CHKERRQ(ierr);
#if defined(PETSC_LOG)
  PLogFlops((int)(BSlocal_flops()-flop1));
#endif
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "MatForwardSolve_MPIRowbs"
int MatForwardSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat          submat = (Mat) mat->data;
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) submat->data;
  int          ierr;
  Scalar       *ya, *xa, *xworka;

#if defined(PETSC_LOG)
  double flop1 = BSlocal_flops();
#endif
  /* Permute and apply diagonal scaling to vector, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecGetArray(x,&xa); CHKERRQ(ierr);
    ierr = VecGetArray(mbs->xwork,&xworka); CHKERRQ(ierr);
    BSperm_dvec(xa,xworka,mbs->pA->perm); CHKERRBS(0);
    ierr = VecPointwiseMult(mbs->diag,mbs->xwork,y); CHKERRQ(ierr);
    ierr = VecRestoreArray(x,&xa); CHKERRQ(ierr);
    ierr = VecRestoreArray(mbs->xwork,&xworka); CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x,y); CHKERRQ(ierr);
  }
  ierr = VecGetArray(y,&ya); CHKERRQ(ierr);

  if (mbs->procinfo->single)
    /* Use BlockSolve routine for no cliques/inodes */
    BSfor_solve1(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  else
    BSfor_solve(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  CHKERRBS(0);
  ierr = VecRestoreArray(y,&ya); CHKERRQ(ierr);
#if defined(PETSC_LOG)
  PLogFlops((int)(BSlocal_flops()-flop1));
#endif

  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "MatBackwardSolve_MPIRowbs"
int MatBackwardSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat          submat = (Mat) mat->data;
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) submat->data;
  int          ierr;
  Scalar       *ya, *xworka;

#if defined (PETSC_LOG)
  double flop1 = BSlocal_flops();
#endif
  ierr = VecCopy(x,y); CHKERRQ(ierr);
  ierr = VecGetArray(y,&ya);   CHKERRQ(ierr);
  ierr = VecGetArray(mbs->xwork,&xworka); CHKERRQ(ierr);

  if (mbs->procinfo->single)
    /* Use BlockSolve routine for no cliques/inodes */
    BSback_solve1(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  else
    BSback_solve(mbs->fpA,ya,mbs->comm_pA,mbs->procinfo);
  CHKERRBS(0);

  /* Apply diagonal scaling and unpermute, where D^{-1/2} is stored */
  if (!mbs->vecs_permscale) {
    ierr = VecPointwiseMult(y,mbs->diag,mbs->xwork);  CHKERRQ(ierr);
    BSiperm_dvec(xworka,ya,mbs->pA->perm); CHKERRBS(0);
  }
  ierr = VecRestoreArray(y,&ya);   CHKERRQ(ierr);
  ierr = VecRestoreArray(mbs->xwork,&xworka); CHKERRQ(ierr);
#if defined (PETSC_LOG)
  PLogFlops((int)(BSlocal_flops()-flop1));
#endif
  return 0;
}

#else
#undef __FUNC__  
#define __FUNC__ "MatNullMPIRowbs"
int MatNullMPIRowbs()
{
  return 0;
}
#endif

/* 
    The logging variables required by BlockSolve, 

    This is an ugly hack that allows PETSc to run properly with BlockSolve regardless
  of whether PETSc or BlockSolve is compiled with logging turned on. 

    It is bad because it relys on BlockSolve's internals not changing related to 
  logging but we have no choice, plus it is unlikely BlockSolve will be developed
  in the near future anyways.
*/
double MLOG_flops;
double MLOG_event_flops;
double MLOG_time_stamp;
int    MLOG_sequence_num;
#if defined (MLOG_MAX_EVNTS) 
MLOG_log_type MLOG_event_log[MLOG_MAX_EVNTS];
MLOG_log_type MLOG_accum_log[MLOG_MAX_ACCUM];
#else
int    MLOG_event_log[300];
int    MLOG_accum_log[75];
#endif
