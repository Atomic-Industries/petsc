static char help[] = "Simple example to test separable objective optimizers.\n";

#include <petsc.h>
#include <petsctao.h>
#include <petscvec.h>

#define NWORKLEFT 4
#define NWORKRIGHT 12

typedef struct _UserCtx
{
  PetscInt m;      /* The row dimension of F */
  PetscInt n;      /* The column dimension of F */
  PetscReal hStart; /* Starting point for Taylor test */
  PetscReal hFactor;/* Taylor test step factor */
  PetscReal hMin;   /* Taylor test end goal */
  Mat F;           /* matrix in least squares component $(1/2) * || F x - d ||_2^2$ */
  Mat W;           /* Workspace matrix. ATA */
  Mat W1;           /* Workspace matrix. AAT */
  Mat Id;           /* Workspace matrix. Dense. Identity */
  Mat Fp;           /* Workspace matrix.  FFTinv  */
  Mat Fpinv;           /* Workspace matrix.   F*FFTinv */
  Mat P;           /* I - FT*((FFT)^-1 * F) */
  Mat temp;
  Vec d;           /* RHS in least squares component $(1/2) * || F x - d ||_2^2$ */
  Vec workLeft[NWORKLEFT];       /* Workspace for temporary vec */
  Vec workRight[NWORKRIGHT];       /* Workspace for temporary vec */
  PetscReal alpha; /* regularization constant applied to || x ||_p */
  PetscReal relax; /* Overrelaxation parameter  */
  PetscReal rho; /*  Augmented Lagrangian Parameter */
  PetscReal eps; /* small constant for approximating gradient of || x ||_1 */
  PetscInt matops;
  PetscInt iter;
  NormType p;
  PetscRandom    rctx;
  PetscBool taylor; /*Flag to determine whether to run Taylor test or not */
} * UserCtx;

PetscErrorCode CreateRHS(UserCtx ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* build the rhs d in ctx */
  ierr = VecCreate(PETSC_COMM_WORLD,&(ctx->d)); CHKERRQ(ierr);
  ierr = VecSetSizes(ctx->d,PETSC_DECIDE,ctx->m); CHKERRQ(ierr);
  ierr=  VecSetFromOptions(ctx->d); CHKERRQ(ierr);
  ierr = VecSetRandom(ctx->d,ctx->rctx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMatrix(UserCtx ctx)
{
  PetscInt       Istart,Iend,i,j,Ii;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* build the matrix F in ctx */
  ierr = MatCreate(PETSC_COMM_WORLD, &(ctx->F)); CHKERRQ(ierr);
  ierr = MatSetSizes(ctx->F,PETSC_DECIDE, PETSC_DECIDE, ctx->m, ctx->n);CHKERRQ(ierr);
  ierr = MatSetType(ctx->F,MATAIJ); CHKERRQ(ierr); /* TODO: Decide specific SetType other than dummy*/
  ierr = MatMPIAIJSetPreallocation(ctx->F, 5, NULL, 5, NULL); CHKERRQ(ierr); /*TODO: some number other than 5?*/
  ierr = MatSeqAIJSetPreallocation(ctx->F, 5, NULL); CHKERRQ(ierr);
  ierr = MatSetUp(ctx->F); CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(ctx->F,&Istart,&Iend); CHKERRQ(ierr);  

  ierr = PetscLogStageRegister("Assembly", &stage); CHKERRQ(ierr);
  ierr= PetscLogStagePush(stage); CHKERRQ(ierr);


  /* Set matrix elements in  2-D fiveopoint stencil format. */
  if (!(ctx->matops)){
    PetscInt gridN;
    if (ctx->m != ctx->n) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Stencil matrix must be square");

    gridN = (PetscInt) PetscSqrtReal((PetscReal) ctx->m);
    if (gridN * gridN != ctx->m) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of rows must be square");
    for (Ii=Istart; Ii<Iend; Ii++) {
      PetscInt I_n, I_s, I_e, I_w;

      i = Ii / gridN; j = Ii % gridN;

      I_n = i * gridN + j + 1;
      if (j + 1 >= gridN) I_n = -1;
      I_s = i * gridN + j - 1;
      if (j - 1 < 0) I_s = -1;
      I_e = (i + 1) * gridN + j;
      if (i + 1 >= gridN) I_e = -1;
      I_w = (i - 1) * gridN + j;
      if (i - 1 < 0) I_w = -1;

      ierr = MatSetValue(ctx->F, Ii, Ii, 4., INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(ctx->F, Ii, I_n, -1., INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(ctx->F, Ii, I_s, -1., INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(ctx->F, Ii, I_e, -1., INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(ctx->F, Ii, I_w, -1., INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  else {
    ierr = MatSetRandom(ctx->F, ctx->rctx); CHKERRQ(ierr);

  }
  ierr = MatAssemblyBegin(ctx->F, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->F, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = PetscLogStagePop(); CHKERRQ(ierr);

  //TODO if condition for running ADMM?
  ierr = MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, ctx->m, ctx->n, NULL, &(ctx->Id)); CHKERRQ(ierr);
  ierr = MatZeroEntries(ctx->Id); CHKERRQ(ierr);
  ierr = MatShift(ctx->Id,1.0); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(ctx->Id, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->Id, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, ctx->m, ctx->n, NULL, &(ctx->Fp)); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(ctx->Fp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->Fp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = MatDuplicate(ctx->Fp,MAT_DO_NOT_COPY_VALUES,&(ctx->Fpinv)); CHKERRQ(ierr);
  ierr = MatDuplicate(ctx->F,MAT_DO_NOT_COPY_VALUES,&(ctx->P)); CHKERRQ(ierr);
  ierr = MatDuplicate(ctx->F,MAT_DO_NOT_COPY_VALUES,&(ctx->temp)); CHKERRQ(ierr);

  /* Stencil matrix is symmetric. Setting symmetric flag for ICC/CHolesky preconditioner */
  if (!(ctx->matops)){
    ierr = MatSetOption(ctx->F,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(ierr);
  }

  ierr = MatTransposeMatMult(ctx->F,ctx->F, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(ctx->W)); CHKERRQ(ierr);
  ierr = MatMatTransposeMult(ctx->F,ctx->F, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(ctx->W1)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupWorkspace(UserCtx ctx)
{
  PetscInt i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreateVecs(ctx->F, &ctx->workLeft[0], &ctx->workRight[0]);CHKERRQ(ierr);
  for (i = 1; i < NWORKLEFT; i++) {
    ierr = VecDuplicate(ctx->workLeft[0], &(ctx->workLeft[i]));CHKERRQ(ierr);
  }
  for (i = 1; i < NWORKRIGHT; i++) {
    ierr = VecDuplicate(ctx->workRight[0], &(ctx->workRight[i]));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ConfigureContext(UserCtx ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ctx->m = 16;
  ctx->n = 16;
  ctx->alpha = 1.;
  ctx->relax = 1.;
  ctx->rho = 1.;
  ctx->eps = 1.e-3;
  ctx->matops = 0;
  ctx->iter = 10;
  ctx->p = NORM_2;
  ctx->hStart = 1.;
  ctx->hMin = 1.e-3;
  ctx->hFactor = 0.5;
  ctx->taylor = PETSC_TRUE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Configure separable objection example", "ex4.c");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-m", "The row dimension of matrix F", "ex4.c", ctx->m, &(ctx->m), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n", "The column dimension of matrix F", "ex4.c", ctx->n, &(ctx->n), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-matrix_format","Decide format of F matrix. 0 for stencil, 1 for dense random", "ex4.c", ctx->matops, &(ctx->matops), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-iter","Iteration number for ADMM Basic Pursuit", "ex4.c", ctx->iter, &(ctx->iter), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha", "The regularization multiplier. 1 default", "ex4.c", ctx->alpha, &(ctx->alpha), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-relax", "Overrelaxation parameter.", "ex4.c", ctx->relax, &(ctx->relax), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rho", "Augmented Lagrangian Parameter", "ex4.c", ctx->rho, &(ctx->rho), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-epsilon", "The small constant added to |x_i| in the denominator to approximate the gradient of ||x||_1", "ex4.c", ctx->eps, &(ctx->eps), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hStart", "Taylor test starting point. 1 default.", "ex4.c", ctx->hStart, &(ctx->hStart), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hFactor", "Taylor test multiplier factor. 0.5 default", "ex4.c", ctx->hFactor, &(ctx->hFactor), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hMin", "Taylor test ending condition. 1.e-3 default", "ex4.c", ctx->hMin, &(ctx->hMin), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-taylor","Flag for Taylor test. Default is true.", "ex4.c", ctx->taylor, &(ctx->taylor), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-p","Norm type.", "ex4.c", NormTypes,  ctx->p, &(ctx->p), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  /* Creating random ctx */
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&(ctx->rctx));CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(ctx->rctx);CHKERRQ(ierr);
  ierr = CreateMatrix(ctx);CHKERRQ(ierr);
  ierr = CreateRHS(ctx);CHKERRQ(ierr);
  ierr = SetupWorkspace(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DestroyContext(UserCtx *ctx)
{
  PetscInt i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&((*ctx)->F)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->W)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->W1)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->Id)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->Fp)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->Fpinv)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->temp)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->P)); CHKERRQ(ierr);
  ierr = VecDestroy(&((*ctx)->d)); CHKERRQ(ierr);
  for (i = 0; i < NWORKLEFT; i++) {
    ierr = VecDestroy(&((*ctx)->workLeft[i])); CHKERRQ(ierr);
  }
  for (i = 0; i < NWORKRIGHT; i++) {
    ierr = VecDestroy(&((*ctx)->workRight[i])); CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&((*ctx)->rctx)); CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  *ctx = NULL;
  PetscFunctionReturn(0);
}

/* compute (1/2) * ||F x - d||^2 */
PetscErrorCode ObjectiveMisfit(Tao tao, Vec x, PetscReal *J, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  Vec y = ctx->workLeft[0];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMult(ctx->F, x, y);CHKERRQ(ierr);
  ierr = VecAXPY(y, -1., ctx->d);CHKERRQ(ierr);
  ierr = VecDot(y, y, J);CHKERRQ(ierr);
  *J *= 0.5;
  PetscFunctionReturn(0);
}

PetscErrorCode GradientMisfit(Tao tao, Vec x, Vec V, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  PetscErrorCode ierr;

  /* work1 is A^T Ax, work2 is Ab, W is A^T A*/

  PetscFunctionBegin;
  ierr = MatMult(ctx->W,x,ctx->workRight[0]); CHKERRQ(ierr);
  ierr = MatMultTranspose(ctx->F, ctx->d, ctx->workRight[1]);CHKERRQ(ierr);
  ierr = VecWAXPY(V, -1., ctx->workRight[1], ctx->workRight[0]);CHKERRQ(ierr);
//  ierr = VecScale(V,-2.); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ObjectiveRegularization(Tao tao, Vec x, PetscReal *J, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  PetscReal norm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNorm (x, ctx->p, &norm);CHKERRQ(ierr);
  if (ctx->p == NORM_2) {
    norm = 0.5 * norm * norm;
  }
  *J = ctx->alpha * norm;
  PetscFunctionReturn(0);
}

PetscErrorCode GradientRegularization(Tao tao, Vec x, Vec V, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  PetscErrorCode ierr;
  PetscReal      S;

  PetscFunctionBegin;
  if (ctx->p == NORM_2) {
    ierr = VecCopy(x, V);CHKERRQ(ierr);
  }
  else if (ctx->p == NORM_1) {
    PetscReal eps = ctx->eps;

    ierr = VecCopy(x, ctx->workRight[1]);CHKERRQ(ierr);
    ierr = VecAbs(ctx->workRight[1]); CHKERRQ(ierr);
    ierr = VecShift(ctx->workRight[1], eps);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(V, x, ctx->workRight[1]); CHKERRQ(ierr);
  }
  else {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Example only works for NORM_1 and NORM_2");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ObjectiveComplete(Tao tao, Vec x, PetscReal *J, void *ctx)
{
  PetscReal Jm, Jr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ObjectiveMisfit(tao, x, &Jm, ctx);CHKERRQ(ierr);
  ierr = ObjectiveRegularization(tao, x, &Jr, ctx);CHKERRQ(ierr);
  *J = Jm + Jr;
  PetscFunctionReturn(0);
}

PetscErrorCode GradientComplete(Tao tao, Vec x, Vec V, void *ctx)
{
  PetscErrorCode ierr;
  UserCtx cntx = (UserCtx) ctx;

  PetscFunctionBegin;
  ierr = GradientMisfit(tao, x, cntx->workRight[2], ctx);CHKERRQ(ierr);
  ierr = GradientRegularization(tao, x, cntx->workRight[3], ctx);CHKERRQ(ierr);
  ierr = VecWAXPY(V,1,cntx->workRight[2],cntx->workRight[3]); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ADMMBasicPursuit(UserCtx ctx, Tao tao, Vec x, PetscReal *C)
{
  PetscErrorCode ierr;
  PetscInt i;
  PetscReal J,Jn;
  IS perm, iscol;
  MatFactorInfo factinfo;
  MPI_Comm       comm = PetscObjectComm((PetscObject)x);
  Vec z_k, u_k, x_k, max_k;

  PetscFunctionBegin;
  z_k = ctx->workRight[3];
  u_k = ctx->workRight[4];
  x_k = ctx->workRight[11];
  x_k = ctx->workRight[11];
  max_k = ctx->workRight[9];
  ierr = VecSet(z_k,0); CHKERRQ(ierr); /* z_k */
  ierr = VecSet(u_k,0); CHKERRQ(ierr); /* u_k */
  ierr = VecSet(x_k,0); CHKERRQ(ierr); /* x_k */
  ierr = VecSet(max_k,0); CHKERRQ(ierr); // compare zero vector for VecPointWiseMax

//  ierr = MatFactorInfoInitialize(&factinfo); CHKERRQ(ierr); 

  ierr  = MatGetOrdering(ctx->W1,MATORDERINGNATURAL,&perm,&iscol);CHKERRQ(ierr);
  ierr  = ISDestroy(&iscol);CHKERRQ(ierr);

  ierr = PetscMemzero(&factinfo,sizeof(MatFactorInfo));CHKERRQ(ierr);
  ierr = MatFactorInfoInitialize(&factinfo); CHKERRQ(ierr); 
  ierr = MatGetFactor(ctx->W1,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&(ctx->temp));CHKERRQ(ierr);
  ierr = MatCholeskyFactorSymbolic(ctx->temp,ctx->W1,perm,&factinfo);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(ctx->temp,ctx->W1,&factinfo);CHKERRQ(ierr);

//  ierr = MatCholeskyFactor(ctx->W1,NULL,NULL); CHKERRQ(ierr); //Cholesky of AAT
  ierr = MatMatSolve(ctx->temp,ctx->Id, ctx->Fp); CHKERRQ(ierr); // Solve LLT FFTinv = I for FFTinv
  ierr = MatTransposeMatMult(ctx->F, ctx->Fp, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(ctx->Fpinv)); CHKERRQ(ierr); 
  ierr = MatMatMult(ctx->Fpinv, ctx->F, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(ctx->P)); CHKERRQ(ierr);
  ierr = MatScale(ctx->P, -1.0); CHKERRQ(ierr);
  ierr = MatShift(ctx->P, 1.0); CHKERRQ(ierr); /* P = I - FT*(FFT^-1)*F */

  ierr = MatMult(ctx->Fpinv, ctx->d, ctx->workRight[5]); /* q = FT*((FFT)^-1 * b) */

  ierr = TaoComputeObjective(tao, x, &J);CHKERRQ(ierr);

  ierr = PetscPrintf (comm, "ADMMBP: Compute Objective:  %g\n", (double) J); CHKERRQ(ierr);
  for (i=0; i<ctx->iter; i++){

    ierr = VecView(x_k, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    // x update 
    ierr = VecWAXPY(ctx->workRight[6], -1.0, u_k, z_k); CHKERRQ(ierr); // work[6] = z-u
    ierr = MatMultAdd(ctx->P, ctx->workRight[6], ctx->workRight[5], x_k); CHKERRQ(ierr); // x = P(z-u) + q
    ierr = VecAXPBYPCZ(ctx->workRight[7], ctx->alpha, 1.0 - ctx->alpha, 0.0, x_k, z_k); CHKERRQ(ierr); // x_hat = ax + (1-a)z

    /* soft thresholding for z */
    /*
     VecGetArray(z_k, &z_array) // local array of values

     for (i = 0; i < n; i++) { // local size of z_k
       z_array[i] = shrinkage (z_array[i]);
     }

     VecRestoreArray(z_k);
    */
    ierr = VecWAXPY(z_k, 1., ctx->workRight[7], u_k); CHKERRQ(ierr); // xhat + u for shrinkage. 
    ierr = VecCopy(z_k, ctx->workRight[8]);CHKERRQ(ierr);
    ierr = VecScale(ctx->workRight[8], -1.); CHKERRQ(ierr);
    ierr = VecShift(z_k, - 1./(ctx->rho)); CHKERRQ(ierr);
    ierr = VecShift(ctx->workRight[8], - 1./(ctx->rho)); CHKERRQ(ierr);
    ierr = VecPointwiseMax(z_k, max_k, z_k); CHKERRQ(ierr);
    ierr = VecPointwiseMax(ctx->workRight[8], max_k, ctx->workRight[8]); CHKERRQ(ierr);
    ierr = VecAXPY(z_k, -1., ctx->workRight[8]); CHKERRQ(ierr);

    // u update 
    ierr = VecWAXPY(ctx->workRight[10], -1., z_k, ctx->workRight[7]); CHKERRQ(ierr); // work[10] = x_hat - z
    ierr = VecAXPY(u_k, 1., ctx->workRight[10]); CHKERRQ(ierr); // u = u + x_hat - z

    ierr = VecNorm(x_k,NORM_1,C); CHKERRQ(ierr);
    //	 ierr = PetscPrintf (comm, "step: %D, NORM1 of x: %g \n", i, (double) *C); CHKERRQ(ierr);
    Jn = PetscAbsReal(J - *C);
    ierr = PetscPrintf (comm, "step: %D, J(x): %g, predicted: %g, diff %g\n", i, (double) J,
                        (double) *C, (double) Jn);CHKERRQ(ierr);
    //     ierr = PetscPrintf (comm, "ADMMBP: step %D, objective:  %g\n", i, (double) &C); CHKERRQ(ierr);
  }


  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Second order Taylor remainder convergence test */
PetscErrorCode TaylorTest(UserCtx ctx, Tao tao, Vec x, PetscReal *C)
{
  PetscReal h,J,Jp,Jm,dJdm,O,temp;
  PetscInt i, j;
  PetscInt numValues;
  PetscReal Jx;
  PetscReal *Js, *hs;
  PetscReal minrate = PETSC_MAX_REAL;
  PetscReal gdotdx;
  MPI_Comm       comm = PetscObjectComm((PetscObject)x);
  Vec       g, dx, xhat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &xhat);CHKERRQ(ierr);

  /* choose a perturbation direction */
  ierr = VecDuplicate(x, &dx);CHKERRQ(ierr);
  ierr = VecSetRandom(dx,ctx->rctx); CHKERRQ(ierr);
  /* evaluate objective at x: J(x) */
  ierr = TaoComputeObjective(tao, x, &Jx);CHKERRQ(ierr);
  /* evaluate gradient at x, save in vector g */
  ierr = TaoComputeGradient(tao, x, g);CHKERRQ(ierr);

  ierr = VecDot(g, dx, &gdotdx);CHKERRQ(ierr);

  for (numValues = 0, h = ctx->hStart; h >= ctx->hMin; h *= ctx->hFactor) numValues++;
  ierr = PetscCalloc2(numValues, &Js, numValues, &hs);CHKERRQ(ierr);

  for (i = 0, h = ctx->hStart; h >= ctx->hMin; h *= ctx->hFactor, i++) {
    PetscReal Jxhat_comp, Jxhat_pred;

    ierr = VecWAXPY(xhat, h, dx, x);CHKERRQ(ierr);

    ierr = TaoComputeObjective(tao, xhat, &Jxhat_comp);CHKERRQ(ierr);

    /* J(\hat(x)) \approx J(x) + g^T (xhat - x) = J(x) + h * g^T dx */
    Jxhat_pred = Jx + h * gdotdx;

    /* Vector to dJdm scalar? Dot?*/
    J = PetscAbsReal(Jxhat_comp - Jxhat_pred);

    ierr = PetscPrintf (comm, "J(xhat): %g, predicted: %g, diff %g\n", (double) Jxhat_comp,
                        (double) Jxhat_pred, (double) J);CHKERRQ(ierr);
    Js[i] = J;
    hs[i] = h;
  }

  for (j=1; j<numValues; j++){
    temp = PetscLogReal(Js[j] / Js[j - 1]) / PetscLogReal (hs[j] / hs[j - 1]);
    ierr = PetscPrintf (comm, "Convergence rate step %D: %g\n", j - 1, (double) temp);CHKERRQ(ierr);
    minrate = PetscMin(minrate, temp);
  }
  //ierr = VecMin(ctx->workLeft[2],NULL, &O); CHKERRQ(ierr);

  /* If O is not ~2, then the test is wrong */  

  ierr = PetscFree2(Js, hs);CHKERRQ(ierr);
  *C = minrate;
  ierr = VecDestroy(&dx);CHKERRQ(ierr);
  ierr = VecDestroy(&xhat);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int main (int argc, char** argv)
{
  UserCtx        ctx;
  Tao            tao;
  Vec            x;
  PetscErrorCode ierr;


  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ierr = ConfigureContext(ctx);CHKERRQ(ierr);

  /* Define two functions that could pass as objectives to TaoSetObjectiveRoutine(): one
   * for the misfit component, and one for the regularization component */
  /* ObjectiveMisfit() and ObjectiveRegularization() */

  /* Define a single function that calls both components adds them together: the complete objective,
   * in the absence of a Tao implementation that handles separability */
  /* ObjectiveComplete() */

  /* Construct the Tao object */
  ierr = TaoCreate(PETSC_COMM_WORLD, &tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAONM); CHKERRQ(ierr);
  ierr = TaoSetObjectiveRoutine(tao, ObjectiveComplete, (void *) ctx);CHKERRQ(ierr);
  ierr = TaoSetGradientRoutine(tao, GradientComplete, (void *) ctx);CHKERRQ(ierr);
  ierr = MatCreateVecs(ctx->F, NULL, &x);CHKERRQ(ierr);
  ierr = VecSet(x, 0.);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao, x);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  PetscReal temp;
  ierr = ADMMBasicPursuit(ctx, tao, x, &temp); CHKERRQ(ierr);
  /* solve */
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  /* examine solution */
  VecViewFromOptions(x, NULL, "-view_sol");CHKERRQ(ierr);

  if (ctx->taylor) {
    PetscReal rate;

    ierr = TaylorTest(ctx, tao, x, &rate);CHKERRQ(ierr);
  }

  /* cleanup */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = DestroyContext(&ctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    args:

  test:
    suffix: l1_1
    args: -p 1 -tao_type lmvm -alpha 1. -epsilon 1.e-7 -m 64 -n 64 -view_sol -mat_format 1

TEST*/
