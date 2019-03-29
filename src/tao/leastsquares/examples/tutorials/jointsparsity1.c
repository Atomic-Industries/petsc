/* XH: todo add jointsparsity1.F90 and asjust makefile */
/* [petsc-users] Reshaping a vector into a matrix   https://lists.mcs.anl.gov/pipermail/petsc-users/2011-January/thread.html#7650
   You most definitely want to use the MAIJ.  MAIJ does not "repeat" X, it uses the original matrix passed in but does efficient multiple matrix-vector products at the same time.
   https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatCreateMAIJ.html
*/
/*
   Include "petsctao.h" so that we can use TAO solvers.  Note that this
   file automatically includes libraries such as:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - sysem routines        petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners

*/

#include <petsctao.h>

/*
Description:   BRGN Joint-Sparsity reconstruction example 1.
Reference:     cs1.c, tomography.c
*/

static char help[] = "Finds the least-squares solution to the under constraint linear model Ax = b, with L1-norm regularizer and multiple observations of same sparsity (joint-sparsity). \n\
            A is a M*N real matrix (M<N), x is sparse. In joint-sparsity, we have L observations: A*x^i = b^i for i=1, ... ,L. \n\
            We find the sparse solution by solving 0.5*sum_i{||Ax^i-b^i||^2} + lambda*sum_j{sum_i{((x^i)_j)^2}, where lambda (by default 1e-4) is a user specified weight.\n\
            In future, include a dictionary matrix if the sparsity is for transformed version y=Dx. D is the K*N transform matrix so that D*x is sparse. By default D is identity matrix, so that D*x = x.\n";
/*T
   Concepts: TAO^Solving a system of nonlinear equations, nonlinear least squares
   Routines: TaoCreate();
   Routines: TaoSetType();
   Routines: TaoSetSeparableObjectiveRoutine();
   Routines: TaoSetJacobianRoutine();
   Routines: TaoSetInitialVector();
   Routines: TaoSetFromOptions();
   Routines: TaoSetConvergenceHistory(); TaoGetConvergenceHistory();
   Routines: TaoSolve();
   Routines: TaoView(); TaoDestroy();
   Processors: 1
T*/

/* User-defined application context */
typedef struct {
  /* Working space. linear least square:  res(x) = A*x - b */
  PetscInt  M,N,K;            /* Problem dimension: A is M*N Matrix, D is K*N Matrix */
  Mat       A,D;              /* Coefficients, Dictionary Transform of size M*N and K*N respectively. For linear least square, Jacobian Matrix J = A. For nonlinear least square, it is different from A */
  Vec       b,xGT,xlb,xub;    /* observation b, ground truth xGT, the lower bound and upper bound of x*/
} AppCtx;

/* User provided Routines */
PetscErrorCode InitializeUserData(AppCtx *);
PetscErrorCode FormStartingPoint(Vec,AppCtx *);
PetscErrorCode EvaluateResidual(Tao,Vec,Vec,void *);
PetscErrorCode EvaluateJacobian(Tao,Vec,Mat,Mat,void *);
PetscErrorCode EvaluateRegularizerObjectiveAndGradient(Tao,Vec,PetscReal *,Vec,void*);
PetscErrorCode EvaluateRegularizerHessian(Tao,Vec,Mat,void*);
PetscErrorCode EvaluateRegularizerHessianProd(Mat,Vec,Vec);

/*--------------------------------------------------------------------*/
int main(int argc,char **argv)
{
  PetscErrorCode ierr;               /* used to check for functions returning nonzeros */
  Vec            x,res;              /* solution, function res(x) = A*x-b */
  Mat            Hreg;               /* regularizer Hessian matrix for user specified regularizer*/
  Tao            tao;                /* Tao solver context */
  PetscReal      hist[100],resid[100],v1,v2;
  PetscInt       lits[100];
  AppCtx         user;               /* user-defined work context */
  PetscViewer    fd;   /* used to save result to file */
  char           resultFile[] = "jointsparsityResult_x";  /* other tutorials generate "tomographyResult_x" to "cs1Result_x" */

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);if (ierr) return ierr;

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBRGN);CHKERRQ(ierr);

  /* User set application context: A, D matrice, and b, xGT vector. */
  ierr = InitializeUserData(&user);CHKERRQ(ierr);

  /* Allocate solution vector x, and residual vectors Ax-b.*/
  ierr = VecCreateSeq(PETSC_COMM_SELF,user.N,&x);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,user.M,&res);CHKERRQ(ierr);

  /* Set initial guess */
  ierr = FormStartingPoint(x,&user);CHKERRQ(ierr);

  /* Bind x to tao->solution. */
  ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);
  /* Sets the upper and lower bounds of x */
  ierr = TaoSetVariableBounds(tao,user.xlb,user.xub);CHKERRQ(ierr);

  /* Bind user.D to tao->data->D */
  ierr = TaoBRGNSetDictionaryMatrix(tao,user.D);CHKERRQ(ierr);

  /* Set the residual function and Jacobian routines for least squares. */
  ierr = TaoSetResidualRoutine(tao,res,EvaluateResidual,(void*)&user);CHKERRQ(ierr);
  /* Jacobian matrix fixed as user.A for Linear least square problem. */
  ierr = TaoSetJacobianResidualRoutine(tao,user.A,user.A,EvaluateJacobian,(void*)&user);CHKERRQ(ierr);

  /* User set the regularizer objective, gradient, and hessian. Set it the same as using l2prox choice, for testing purpose.  */
  ierr = TaoBRGNSetRegularizerObjectiveAndGradientRoutine(tao,EvaluateRegularizerObjectiveAndGradient,(void*)&user);CHKERRQ(ierr);
  /* User defined regularizer Hessian setup, here is identiy shell matrix */
  ierr = MatCreate(PETSC_COMM_SELF,&Hreg);CHKERRQ(ierr);
  ierr = MatSetSizes(Hreg,PETSC_DECIDE,PETSC_DECIDE,user.N,user.N);CHKERRQ(ierr);
  ierr = MatSetType(Hreg,MATSHELL);CHKERRQ(ierr);
  ierr = MatSetUp(Hreg);CHKERRQ(ierr);
  ierr = MatShellSetOperation(Hreg,MATOP_MULT,(void (*)(void))EvaluateRegularizerHessianProd);CHKERRQ(ierr);
  ierr = TaoBRGNSetRegularizerHessianRoutine(tao,Hreg,EvaluateRegularizerHessian,(void*)&user);CHKERRQ(ierr);

  /* Check for any TAO command line arguments */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  ierr = TaoSetConvergenceHistory(tao,hist,resid,NULL,lits,100,PETSC_TRUE);CHKERRQ(ierr);

  /* Perform the Solve */
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  /* Save x (reconstruction of object) vector to a binary file, which maybe read from Matlab and convert to a 2D image for comparison. */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,resultFile,FILE_MODE_WRITE,&fd);CHKERRQ(ierr);
  ierr = VecView(x,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

    /* XH: Debug: View the result, function and Jacobian.  */
  ierr = PetscPrintf(PETSC_COMM_SELF, "-------- result x, residual res =A*x-b. -------- \n");CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = VecView(res,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

    /* compute the error */
  ierr = VecAXPY(x,-1,user.xGT);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&v1);CHKERRQ(ierr);
  ierr = VecNorm(user.xGT,NORM_2,&v2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "relative reconstruction error: ||x-xGT||/||xGT|| = %6.4e.\n", (double)(v1/v2));CHKERRQ(ierr);


  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

   /* Free PETSc data structures */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&res);CHKERRQ(ierr);
  ierr = MatDestroy(&Hreg);CHKERRQ(ierr);
  /* Free user data structures */
  ierr = MatDestroy(&user.A);CHKERRQ(ierr);
  ierr = MatDestroy(&user.D);CHKERRQ(ierr);
  ierr = VecDestroy(&user.b);CHKERRQ(ierr);
  ierr = VecDestroy(&user.xGT);CHKERRQ(ierr);
  ierr = VecDestroy(&user.xlb);CHKERRQ(ierr);
  ierr = VecDestroy(&user.xub);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*--------------------------------------------------------------------*/
/* Evaluate residual function F = A(x)-b in least square problem ||A(x)-b||^2 */
PetscErrorCode EvaluateResidual(Tao tao,Vec X,Vec F,void *ptr)
{
  AppCtx         *user = (AppCtx *)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Compute Ax - b */
  ierr = MatMult(user->A,X,F);CHKERRQ(ierr);
  ierr = VecAXPY(F,-1,user->b);CHKERRQ(ierr);
  PetscLogFlops(user->M*user->N*2);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
PetscErrorCode EvaluateJacobian(Tao tao,Vec X,Mat J,Mat Jpre,void *ptr)
{
  /* Jacobian is not changing here, so use a empty dummy function here.  J[m][n] = df[m]/dx[n] = A[m][n] for linear least square */
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
PetscErrorCode EvaluateRegularizerObjectiveAndGradient(Tao tao,Vec X,PetscReal *f_reg,Vec G_reg,void *ptr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* compute regularizer objective = 0.5*x'*x */
  ierr = VecDot(X,X,f_reg);CHKERRQ(ierr);
  *f_reg *= 0.5;
  /* compute regularizer gradient = x */
  ierr = VecCopy(X,G_reg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EvaluateRegularizerHessianProd(Mat Hreg,Vec in,Vec out)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecCopy(in,out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
PetscErrorCode EvaluateRegularizerHessian(Tao tao,Vec X,Mat Hreg,void *ptr)
{
  /* Hessian for regularizer objective = 0.5*x'*x is identity matrix, and is not changing*/
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
PetscErrorCode FormStartingPoint(Vec X,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------- */
PetscErrorCode InitializeUserData(AppCtx *user)
{
  PetscInt       k,n; /* indices for row and columns of D. */
  PetscInt       dictChoice = 1; /* dictChoice = 0/1/2/3, where 0:identity, 1:gradient1D, 2:gradient2D, 3:DCT etc. */
  PetscBool      isAllowDictNULL = PETSC_TRUE; /* isAllowDictNULL = PETSC_TRUE/PETSC_FALSE, where PETSC_TRUE: user.D = NULL is equivalanent to NULL matrix. */
  /* make jointsparsity1; ./jointsparsity1 -tao_monitor -tao_max_it 10 -tao_brgn_regularization_type l1dict -tao_brgn_regularizer_weight 1e-8 -tao_brgn_l1_smooth_epsilon 1e-6 -tao_gatol 1.e-8
     relative reconstruction error: ||x-xGT||/||xGT|| = 4.1366e-01/2.3728e-01 for dictChoice 0/1. */
  
  char           dataFile[] = "jointsparsity1Data_A_b_xGT";   /* Matrix A and vectors b, xGT(ground truth) binary files generated by Matlab. e.g., "tomographyData_A_b_xGT", "cs1Data_A_b_xGT", "jointsparsity1Data_A_b_xGT". */
  PetscViewer    fd;   /* used to load data from file */
  PetscErrorCode ierr;
  PetscReal      v;

  PetscFunctionBegin;

  /*
  Matrix Vector read and write refer to:
  https://www.mcs.anl.gov/petsc/petsc-current/src/mat/examples/tutorials/ex10.c
  https://www.mcs.anl.gov/petsc/petsc-current/src/mat/examples/tutorials/ex12.c
 */
  /* Load the A matrix, b vector, and xGT vector from a binary file. */
  /* Plan: 1st: just use matlab to creat diagonal block matrix, where each sub-matrix is A for the multiple right hand system
           2nd: after 1 is working, use petsc block matirx MatCreateBlockMat() or more advanced choice for multiple right hand system MRS?? */           
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,dataFile,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&user->A);CHKERRQ(ierr);
  ierr = MatSetType(user->A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatLoad(user->A,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user->b);CHKERRQ(ierr);
  ierr = VecLoad(user->b,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user->xGT);CHKERRQ(ierr);
  ierr = VecLoad(user->xGT,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = VecDuplicate(user->xGT,&(user->xlb));CHKERRQ(ierr);
  ierr = VecSet(user->xlb,PETSC_NINFINITY);CHKERRQ(ierr); /* xlb = PETSC_NINFINITY/0.0, where 0.0 generate more accurate result if x>=0.0 is really true*/
  ierr = VecDuplicate(user->xGT,&(user->xub));CHKERRQ(ierr);
  ierr = VecSet(user->xub,PETSC_INFINITY);CHKERRQ(ierr);

  /* Specify the size */
  ierr = MatGetSize(user->A,&user->M,&user->N);CHKERRQ(ierr);

  /* Speficy D */
  /* (1) Specify D Size */
  switch (dictChoice) {
    case 0: /* 0:identity */
      user->K = user->N;
      break;
    case 1: /* 1:gradient1D */
      user->K = user->N-1;
      break;
  }

  ierr = MatCreate(PETSC_COMM_SELF,&user->D);CHKERRQ(ierr);
  ierr = MatSetSizes(user->D,PETSC_DECIDE,PETSC_DECIDE,user->K,user->N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->D);CHKERRQ(ierr);
  ierr = MatSetUp(user->D);CHKERRQ(ierr);

  /* (2) Specify D Content */
  switch (dictChoice) {
    case 0: /* 0:identity */
      if (isAllowDictNULL) {
        /* shortcut, when dictChoice == 0, D is identity matrix, we may just specify it as NULL, and brgn will treat D*x as x without actually computing D*x */
        user->D = NULL;
        PetscFunctionReturn(0);
      }
      else {
        /* Old way to actually set up a indentity matrix. */
        for (k=0; k<user->K; k++) {
          v = 1.0;
          ierr = MatSetValues(user->D,1,&k,1,&k,&v,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      break;
    case 1: /* 1:gradient1D.  [-1, 1, 0,...; 0, -1, 1, 0, ...] */
      for (k=0; k<user->K; k++) {
        v = 1.0;
        n = k+1;
        ierr = MatSetValues(user->D,1,&k,1,&n,&v,INSERT_VALUES);CHKERRQ(ierr);
        v = -1.0;
        ierr = MatSetValues(user->D,1,&k,1,&k,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
      break;
  }
  ierr = MatAssemblyBegin(user->D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex !single !__float128 !define(PETSC_USE_64BIT_INDICES)

   test:
      localrunfiles: tomographyData_A_b_xGT
      args: -tao_max_it 1000 -tao_brgn_regularization_type l1dict -tao_brgn_regularizer_weight 1e-8 -tao_brgn_l1_smooth_epsilon 1e-6 -tao_gatol 1.e-8

   test:
      suffix: 2
      localrunfiles: tomographyData_A_b_xGT
      args: -tao_monitor -tao_max_it 1000 -tao_brgn_regularization_type l2prox -tao_brgn_regularizer_weight 1e-8 -tao_gatol 1.e-6

   test:
      suffix: 3
      localrunfiles: tomographyData_A_b_xGT
      args: -tao_monitor -tao_max_it 1000 -tao_brgn_regularization_type user -tao_brgn_regularizer_weight 1e-8 -tao_gatol 1.e-6

TEST*/
