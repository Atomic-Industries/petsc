#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.49 1998/06/11 15:14:18 balay Exp bsmith $";
#endif

/* Program usage:  mpirun ex1 [-help] [all PETSc options] */

static char help[] = "Demonstrates various vector routines.\n\n";

/*T
   Concepts: Vectors^Using basic vector routines;
   Routines: VecCreate(); VecDuplicate(); VecSet(); VecValid(); 
   Routines: VecDot(); VecMDot(); VecScale(); VecNorm(); VecCopy(); VecAXPY(); 
   Routines: VecAYPX(); VecWAXPY(); VecPointwiseMult(); VecPointwiseDivide(); 
   Routines: VecSwap(); VecMAXPY(); VecDestroy(); VecDestroyVecs(); VecDuplicateVecs();
   Processors: n
T*/

/* 
  Include "vec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   is.h     - index sets
     sys.h    - system routines       viewer.h - viewers
*/

#include "vec.h"
#include <math.h>

int main(int argc,char **argv)
{
  Vec      x, y, w;               /* vectors */
  Vec      *z;                    /* array of vectors */
  double   norm, v, v1, v2;
  int      n = 20, ierr, flg;
  Scalar   one = 1.0, two = 2.0, three = 3.0, dots[3], dot;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);

  /* 
     Create a vector, specifying only its global dimension.
     When using VecCreate(), the vector format (currently parallel,
     shared, or sequential) is determined at runtime.  Also, the parallel
     partitioning of the vector is determined by PETSc at runtime.

     Routines for creating particular vector types directly are:
        VecCreateSeq() - uniprocessor vector
        VecCreateMPI() - distributed vector, where the user can
                         determine the parallel partitioning
        VecCreateShared() - parallel vector that uses shared memory
                            (available only on the SGI); otherwise,
                            is the same as VecCreateMPI()

     With VecCreate() the option -vec_type mpi or -vec_type shared causes the 
     particular type of vector to be formed.

  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x); CHKERRA(ierr);

  /*
     Duplicate some work vectors (of the same format and
     partitioning as the initial vector).
  */
  ierr = VecDuplicate(x,&y); CHKERRA(ierr);
  ierr = VecDuplicate(x,&w); CHKERRA(ierr);

  /*
     Duplicate more work vectors (of the same format and
     partitioning as the initial vector).  Here we duplicate
     an array of vectors, which is often more convenient than
     duplicating individual ones.
  */
  ierr = VecDuplicateVecs(x,3,&z); CHKERRA(ierr); 

  /*
     Set the vectors to entries to a constant value.
  */
  ierr = VecSet(&one,x); CHKERRA(ierr);
  ierr = VecSet(&two,y); CHKERRA(ierr);
  ierr = VecSet(&one,z[0]); CHKERRA(ierr);
  ierr = VecSet(&two,z[1]); CHKERRA(ierr);
  ierr = VecSet(&three,z[2]); CHKERRA(ierr);

  /*
     Demonstrate various basic vector routines.
  */
  ierr = VecDot(x,x,&dot); CHKERRA(ierr);
  ierr = VecMDot(3,x,z,dots); CHKERRA(ierr);

  /* 
     Note: If using a complex numbers version of PETSc, then
     USE_PETSC_COMPLEX is defined in the makefiles; otherwise,
     (when using real numbers) it is undefined.
  */
#if defined(USE_PETSC_COMPLEX)
  PetscPrintf(PETSC_COMM_WORLD,"Vector length %d\n", int (PetscReal(dot)));
  PetscPrintf(PETSC_COMM_WORLD,"Vector length %d %d %d\n",(int)PetscReal(dots[0]),
                             (int)PetscReal(dots[1]),(int)PetscReal(dots[2]));
#else
  PetscPrintf(PETSC_COMM_WORLD,"Vector length %d\n",(int) dot);
  PetscPrintf(PETSC_COMM_WORLD,"Vector length %d %d %d\n",(int)dots[0],
                             (int)dots[1],(int)dots[2]);
#endif

  PetscPrintf(PETSC_COMM_WORLD,"All other values should be near zero\n");

  ierr = VecScale(&two,x); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  v = norm-2.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  PetscPrintf(PETSC_COMM_WORLD,"VecScale %g\n",v);

  ierr = VecCopy(x,w); CHKERRA(ierr);
  ierr = VecNorm(w,NORM_2,&norm); CHKERRA(ierr);
  v = norm-2.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  PetscPrintf(PETSC_COMM_WORLD,"VecCopy  %g\n",v);

  ierr = VecAXPY(&three,x,y); CHKERRA(ierr);
  ierr = VecNorm(y,NORM_2,&norm); CHKERRA(ierr);
  v = norm-8.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  PetscPrintf(PETSC_COMM_WORLD,"VecAXPY %g\n",v);

  ierr = VecAYPX(&two,x,y); CHKERRA(ierr);
  ierr = VecNorm(y,NORM_2,&norm); CHKERRA(ierr);
  v = norm-18.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  PetscPrintf(PETSC_COMM_WORLD,"VecAXPY %g\n",v);

  ierr = VecSwap(x,y); CHKERRA(ierr);
  ierr = VecNorm(y,NORM_2,&norm); CHKERRA(ierr);
  v = norm-2.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  PetscPrintf(PETSC_COMM_WORLD,"VecSwap  %g\n",v);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  v = norm-18.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  PetscPrintf(PETSC_COMM_WORLD,"VecSwap  %g\n",v);

  ierr = VecWAXPY(&two,x,y,w); CHKERRA(ierr);
  ierr = VecNorm(w,NORM_2,&norm); CHKERRA(ierr);
  v = norm-38.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  PetscPrintf(PETSC_COMM_WORLD,"VecWAXPY %g\n",v);

  ierr = VecPointwiseMult(y,x,w); CHKERRA(ierr);
  ierr = VecNorm(w,NORM_2,&norm); CHKERRA(ierr); 
  v = norm-36.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  PetscPrintf(PETSC_COMM_WORLD,"VecPointwiseMult %g\n",v);

  ierr = VecPointwiseDivide(x,y,w); CHKERRA(ierr);
  ierr = VecNorm(w,NORM_2,&norm); CHKERRA(ierr);
  v = norm-9.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  PetscPrintf(PETSC_COMM_WORLD,"VecPointwiseDivide %g\n",v);

  dots[0] = one;
  dots[1] = three;
  dots[2] = two;
  ierr = VecSet(&one,x); CHKERRA(ierr);
  ierr = VecMAXPY(3,dots,x,z); CHKERRA(ierr);
  ierr = VecNorm(z[0],NORM_2,&norm); CHKERRA(ierr);
  v = norm-sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  ierr = VecNorm(z[1],NORM_2,&norm); CHKERRA(ierr);
  v1 = norm-2.0*sqrt((double) n); if (v1 > -1.e-10 && v1 < 1.e-10) v1 = 0.0; 
  ierr = VecNorm(z[2],NORM_2,&norm); CHKERRA(ierr);
  v2 = norm-3.0*sqrt((double) n); if (v2 > -1.e-10 && v2 < 1.e-10) v2 = 0.0; 
  PetscPrintf(PETSC_COMM_WORLD,"VecMAXPY %g %g %g \n",v,v1,v2);

  /* 
     Test whether vector has been corrupted (just to demonstrate this
     routine) not needed in most application codes.
  */
  ierr = VecValid(x,(PetscTruth*)&flg); CHKERRA(ierr);
  if (!flg) SETERRA(1,0,"Corrupted vector.");

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);
  ierr = VecDestroy(w); CHKERRA(ierr);
  ierr = VecDestroyVecs(z,3); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
