static char help[] = "Simple example to test separable objective optimizers.\n";

#include <petsc.h>
#include <petsctao.h>

#define NWORKLEFT 1
#define NWORKRIGHT 4

typedef struct _UserCtx
{
  PetscInt m;      /* The row dimension of F */
  PetscInt n;      /* The column dimension of F */
  Mat F;           /* matrix in least squares component $(1/2) * || F x - d ||_2^2$ */
  Mat W;           /* Workspace matrix */
  Vec d;           /* RHS in least squares component $(1/2) * || F x - d ||_2^2$ */
  Vec workLeft[NWORKLEFT];       /* Workspace for temporary vec */
  Vec workRight[NWORKRIGHT];       /* Workspace for temporary vec */
  PetscReal alpha; /* regularization constant applied to || x ||_p */
  PetscInt matops;
  NormType p;
  PetscRandom    rctx;
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

  /* Set matrix elements in  2-D five-point stencil format. */
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

  /* Stencil matrix is symmetric. Setting symmetric flag for ICC/CHolesky preconditioner */
  if (!(ctx->matops)){
    ierr = MatSetOption(ctx->F,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(ierr);
  }

  ierr = MatTransposeMatMult(ctx->F,ctx->F, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(ctx->W)); CHKERRQ(ierr);
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
  ctx->matops = 0;
  ctx->p = NORM_2;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Configure separable objection example", "ex4.c");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-m", "The row dimension of matrix F", "ex4.c", ctx->m, &(ctx->m), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n", "The column dimension of matrix F", "ex4.c", ctx->n, &(ctx->n), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-matrix_format","Decide format of F matrix. 0 for stencil, 1 for dense random", "ex4.c", ctx->matops, &(ctx->matops), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha", "The regularization multiplier", "ex4.c", ctx->alpha, &(ctx->alpha), NULL);CHKERRQ(ierr);
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
  ierr = VecScale(V,-2.); CHKERRQ(ierr);
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

  PetscFunctionBegin;
  if (ctx->p == NORM_2) {
    ierr = VecCopy(x, V);CHKERRQ(ierr);
  }
  else if (ctx->p == NORM_1) {
    /* TODO: What do we do here? */
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

  /* solve */
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  /* examine solution */

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

TEST*/
