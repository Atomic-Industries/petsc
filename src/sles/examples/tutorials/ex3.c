#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex3.c,v 1.9 1997/10/19 03:27:22 bsmith Exp bsmith $";
#endif

static char help[] = 
"This example solves a linear system in parallel with SLES.  The matrix\n\
uses simple bilinear elements on the unit square.  To test the parallel\n\
matrix assembly, the matrix is intentionally laid out across processors\n\
differently from the way it is assembled.  Input arguments are:\n\
  -m <size> : problem size\n\n";

/*T
   Concepts: SLES^Solving a system of linear equations (basic parallel example);
   Concepts: Matrices^Inserting matrix elements by blocks
   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
   Routines: SLESSolve(); SLESGetKSP(); SLESGetPC(); PCSetType();
   Routines: KSPSetTolerances(); KSPSetInitialGuessNonzero(); 
   Processors: n
T*/

/* 
  Include "sles.h" so that we can use SLES solvers.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       mat.h - matrices
     is.h     - index sets            ksp.h - Krylov subspace methods
     viewer.h - viewers               pc.h  - preconditioners
*/
#include "sles.h"

/* Declare user-defined routines */
int FormElementStiffness(double,Scalar*);
int FormElementRhs(double,double,double,Scalar*);

int main(int argc,char **args)
{
  Vec     u, b, ustar; /* approx solution, RHS, exact solution */
  Mat     A;           /* linear system matrix */
  SLES    sles;        /* linear solver context */
  KSP     ksp;         /* Krylov subspace method context */
  IS      is;          /* index set - used for boundary conditions */
  int     N;           /* dimension of system (global) */
  int     M;           /* number of elements (global) */
  int     rank;        /* processor rank */
  int     size;        /* size of communicator */
  Scalar  Ke[16];      /* element matrix */
  Scalar  r[4];        /* element vector */
  double  h;           /* mesh width */
  double  norm;        /* norm of solution error */
  double  x,y;
  Scalar  val, zero = 0.0, one = 1.0, none = -1.0;
  int     ierr, idx[4], count, *rows, i, m = 5, start, end, its, flg;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  N = (m+1)*(m+1);
  M = m*m;
  h = 1.0/m;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
         Compute the matrix and right-hand-side vector that define
         the linear system, Au = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create stiffness matrix
  */
  ierr = MatCreate(PETSC_COMM_WORLD,N,N,&A); CHKERRA(ierr);
  start = rank*(M/size) + ((M%size) < rank ? (M%size) : rank);
  end   = start + M/size + ((M%size) > rank); 

  /*
     Assemble matrix
  */
  ierr = FormElementStiffness(h*h,Ke);
  for ( i=start; i<end; i++ ) {
     /* location of lower left corner of element */
     x = h*(i % m); y = h*(i/m); 
     /* node numbers for the four corners of element */
     idx[0] = (m+1)*(i/m) + ( i % m);
     idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
     ierr = MatSetValues(A,4,idx,4,idx,Ke,ADD_VALUES); CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  /*
     Create right-hand-side and solution vectors
  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,N,&u); CHKERRA(ierr); 
  PetscObjectSetName((PetscObject)u,"Approx. Solution");
  ierr = VecDuplicate(u,&b); CHKERRA(ierr);
  PetscObjectSetName((PetscObject)b,"Right hand side");
  ierr = VecDuplicate(b,&ustar); CHKERRA(ierr);
  ierr = VecSet(&zero,u); CHKERRA(ierr);
  ierr = VecSet(&zero,b); CHKERRA(ierr);

  /* 
     Assemble right-hand-side vector
  */
  for ( i=start; i<end; i++ ) {
     /* location of lower left corner of element */
     x = h*(i % m); y = h*(i/m); 
     /* node numbers for the four corners of element */
     idx[0] = (m+1)*(i/m) + ( i % m);
     idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
     ierr = FormElementRhs(x,y,h*h,r); CHKERRA(ierr);
     ierr = VecSetValues(b,4,idx,r,ADD_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(b); CHKERRA(ierr);
  ierr = VecAssemblyEnd(b); CHKERRA(ierr);

  /* 
     Modify matrix and right-hand-side for Dirichlet boundary conditions
  */
  rows = (int *) PetscMalloc( 4*m*sizeof(int) ); CHKPTRQ(rows);
  for ( i=0; i<m+1; i++ ) {
    rows[i] = i; /* bottom */
    rows[3*m - 1 +i] = m*(m+1) + i; /* top */
  }
  count = m+1; /* left side */
  for ( i=m+1; i<m*(m+1); i+= m+1 ) {
    rows[count++] = i;
  }
  count = 2*m; /* left side */
  for ( i=2*m+1; i<m*(m+1); i+= m+1 ) {
    rows[count++] = i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,4*m,rows,&is); CHKERRA(ierr);
  for ( i=0; i<4*m; i++ ) {
     x = h*(rows[i] % (m+1)); y = h*(rows[i]/(m+1)); 
     val = y;
     ierr = VecSetValues(u,1,&rows[i],&val,INSERT_VALUES); CHKERRA(ierr);
     ierr = VecSetValues(b,1,&rows[i],&val,INSERT_VALUES); CHKERRA(ierr);
  }    
  PetscFree(rows);
  ierr = VecAssemblyBegin(u);  CHKERRA(ierr);
  ierr = VecAssemblyEnd(u); CHKERRA(ierr);
  ierr = VecAssemblyBegin(b); CHKERRA(ierr); 
  ierr = VecAssemblyEnd(b); CHKERRA(ierr);

  ierr = MatZeroRows(A,is,&one); CHKERRA(ierr);
  ierr = ISDestroy(is); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SLESCreate(PETSC_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESGetKSP(sles,&ksp); CHKERRA(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp); CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SLESSolve(sles,b,u,&its); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Check error */
  ierr = VecGetOwnershipRange(ustar,&start,&end); CHKERRA(ierr);
  for ( i=start; i<end; i++ ) {
     x = h*(i % (m+1)); y = h*(i/(m+1)); 
     val = y;
     ierr = VecSetValues(ustar,1,&i,&val,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(ustar); CHKERRA(ierr);
  ierr = VecAssemblyEnd(ustar); CHKERRA(ierr);
  ierr = VecAXPY(&none,ustar,u); CHKERRA(ierr);
  ierr = VecNorm(u,NORM_2,&norm); CHKERRA(ierr);
  if (norm*h > 1.e-12) 
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g Iterations %d\n",norm*h,its);
  else
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error < 1.e-12 Iterations %d\n",its);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = SLESDestroy(sles); CHKERRA(ierr); ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(ustar); CHKERRA(ierr); ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */
  PetscFinalize();
  return 0;
}

/* --------------------------------------------------------------------- */
   /* element stiffness for Laplacian */
int FormElementStiffness(double H,Scalar *Ke)
{
  Ke[0]  = H/6.0;    Ke[1]  = -.125*H; Ke[2]  = H/12.0;   Ke[3]  = -.125*H;
  Ke[4]  = -.125*H;  Ke[5]  = H/6.0;   Ke[6]  = -.125*H;  Ke[7]  = H/12.0;
  Ke[8]  = H/12.0;   Ke[9]  = -.125*H; Ke[10] = H/6.0;    Ke[11] = -.125*H;
  Ke[12] = -.125*H;  Ke[13] = H/12.0;  Ke[14] = -.125*H;  Ke[15] = H/6.0;
  return 0;
}
/* --------------------------------------------------------------------- */
int FormElementRhs(double x,double y,double H,Scalar *r)
{
  r[0] = 0.; r[1] = 0.; r[2] = 0.; r[3] = 0.0; 
  return 0;
}
