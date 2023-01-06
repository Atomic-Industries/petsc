static char help[] = "Finite element discretization on mesh patches.\n\n\n";

/*
  KNOWN BUGS:

  1) Coordinates created at the beginning in the ephemeral mesh. We really want to create only coordinate patches when we are asked to do so for FEGeom.

  IMPLEMENTATION:

  The corrector $C$ expresses maps the fine space $V$ into the kernel of restriction, or detail space $W$, and the complementary projector $I - C$ is a bijection between the coarse space $V_H$ and the optimized space $V_{vms}$. The example code makes a matrix $G$ whose rows are the optimized basis encoded in terms of fine space basis functions, so that it is $n \times N$. It is composed of the difference between the embedding of the original coarse basis $P_H$ and the corrector $C$, both of which have the same dimensions $n \times N$.

  We should be able to decompose this operation into projection at the element matrix level. I think we can bracket the element matrix with the transformation element matrices.

*/

#include <petscdmplex.h>
#include <petscdmplextransform.h>
#include <petscds.h>
#include <petscsnes.h>
#include <petscconvest.h>

typedef struct {
  DMLabel active; /* Label for transform */
} AppCtx;

static PetscErrorCode trig_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 0.0;
  for (d = 0; d < dim; ++d) *u += PetscSinReal(2.0 * PETSC_PI * x[d]);
  return 0;
}

static PetscErrorCode const_mu(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = 1.0;
  return 0;
}

static void f0_trig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[0] += -4.0 * PetscSqr(PETSC_PI) * PetscSinReal(2.0 * PETSC_PI * x[d]);
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

static void f0_mu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[1] - 1.0;
}

static void g0_mumu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt  cells[1024], Nc = 1024;
  PetscBool flg;

  PetscFunctionBeginUser;
  options->active   = NULL;

  PetscOptionsBegin(comm, "", "Mesh Patch Integration Options", "DMPLEX");
  PetscCall(PetscOptionsIntArray("-cells", "Cells to mark for transformation", "ex57.c", cells, &Nc, &flg));
  if (flg) {
    PetscCall(DMLabelCreate(comm, "active", &options->active));
    for (PetscInt c = 0; c < Nc; ++c) PetscCall(DMLabelSetValue(options->active, cells[c], DM_ADAPT_REFINE));
  }
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

// TODO: Patch should be on PETSC_COMM_SELF
static PetscErrorCode CreatePatch(DM dm, PetscInt cell, DM *patch) {
  DMPlexTransform tr;
  DMLabel         active;
  MPI_Comm        comm;
  PetscInt       *adj     = NULL;
  PetscInt        adjSize = PETSC_DETERMINE;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMLabelCreate(comm, "active", &active));
  PetscCall(DMPlexGetAdjacency(dm, cell, &adjSize, &adj));
  for (PetscInt a = 0; a < adjSize; ++a) PetscCall(DMLabelSetValue(active, adj[a], DM_ADAPT_REFINE));
  PetscCall(PetscObjectViewFromOptions((PetscObject)active, NULL, "-active_view"));
  PetscCall(PetscFree(adj));

  PetscCall(DMPlexTransformCreate(comm, &tr));
  PetscCall(PetscObjectSetName((PetscObject)tr, "Transform"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tr, "select_"));
  PetscCall(DMPlexTransformSetDM(tr, dm));
  PetscCall(DMPlexTransformSetActive(tr, active));
  PetscCall(DMPlexTransformSetFromOptions(tr));
  PetscCall(DMPlexTransformSetUp(tr));
  PetscCall(PetscObjectViewFromOptions((PetscObject)tr, NULL, "-dm_plex_transform_view"));
  PetscCall(DMLabelDestroy(&active));

  PetscCall(DMPlexCreateEphemeral(tr, patch));
  PetscCall(PetscObjectSetName((PetscObject)*patch, "Ephemeral Patch"));
  PetscCall(DMViewFromOptions(*patch, NULL, "-patch_view"));
  PetscCall(DMPlexTransformDestroy(&tr));
  PetscCall(DMPlexSetLocationAlg(*patch, DM_PLEX_LOCATE_GRID_HASH));
  PetscFunctionReturn(0);
}

static PetscErrorCode RefinePatch(DM patch, DM *rpatch) {
  DMPlexTransform tr;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)patch), &tr));
  PetscCall(PetscObjectSetName((PetscObject)tr, "Transform"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tr, "refine_"));
  PetscCall(DMPlexTransformSetDM(tr, patch));
  PetscCall(DMPlexTransformSetFromOptions(tr));
  PetscCall(DMPlexTransformSetUp(tr));
  PetscCall(PetscObjectViewFromOptions((PetscObject)tr, NULL, "-dm_plex_transform_view"));

  PetscCall(DMPlexCreateEphemeral(tr, rpatch));
  PetscCall(PetscObjectSetName((PetscObject)*rpatch, "Ephemeral Refined Patch"));
  PetscCall(DMViewFromOptions(*rpatch, NULL, "-patch_view"));
  PetscCall(DMPlexTransformDestroy(&tr));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(PetscDSSetResidual(ds, 0, f0_trig_u, f1_u));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
  PetscCall(PetscDSSetExactSolution(ds, 0, trig_u, user));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))trig_u, NULL, user, NULL));
  PetscCall(PetscDSSetResidual(ds, 1, f0_mu, NULL));
  PetscCall(PetscDSSetJacobian(ds, 1, 1, g0_mumu, NULL, NULL, NULL));
  PetscCall(PetscDSSetExactSolution(ds, 1, const_mu, user));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, PetscInt Nf, const char *names[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cStart;
  char           prefix[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  for (PetscInt f = 0; f < Nf; ++f) {
    PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", names[f]));
    PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, prefix, -1, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, names[f]));
    PetscCall(DMSetField(dm, f, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscCall(DMCreateDS(dm));
  PetscCall((*setup)(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscFunctionReturn(0);
}

/*
  PatchSolve - Solve the saddle-point system on the refined patch and inject the results into the corrector

  Input Parameters:
+ patch  - The patch from the coarse grid
. c      - The central cell for this patch
. rpatch - The patch from the fine grid
- user   - A user context

  Note:
  Coarse indices are those for the closure of the original seed cell. Fine indices are those for the closure of the entire refined patch, so we just indicate the whole section

  Level; advanced

.seealso: `CreatePatch()`
*/
static PetscErrorCode PatchSolve(DM dm, DM patch, PetscInt c, DM rdm, DM rpatch, Mat dP, AppCtx *user)
{
  SNES            snes;
  Mat             P;
  Vec             u, b, cb;
  IS              subpIS;
  PetscScalar    *elemP;
  PetscSection    gs, gsRef, dgs;
  const PetscInt *points;
  PetscInt       *closure = NULL, *rows, *cols;
  PetscInt        cell, pStart, pEnd, Ncl, Nfine = 0, Ncoarse = 0, j = 0;
  const char     *names[] = {"phi", "mu"};

  PetscFunctionBegin;
  {
    PetscInt n;

    PetscCall(DMPlexGetSubpointIS(patch, &subpIS));
    if (subpIS) PetscCall(PetscObjectViewFromOptions((PetscObject)subpIS, NULL, "-subpoint_is_view"));
    PetscCall(ISGetLocalSize(subpIS, &n));
    PetscCall(ISGetIndices(subpIS, &points));
    for (cell = 0; cell < n; ++cell) if (points[cell] == c) break;
    PetscCheck(cell < n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %" PetscInt_FMT " not found in patch mesh", c);
    PetscCall(ISRestoreIndices(subpIS, &points));
  }
  PetscCall(SetupDiscretization(patch, 2, names, SetupPrimalProblem, user));
  PetscCall(SetupDiscretization(rpatch, 2, names, SetupPrimalProblem, user));
  PetscCall(DMGetGlobalSection(patch, &gs));
  PetscCall(DMGetGlobalSection(rpatch, &gsRef));

  PetscCall(PetscSectionGetChart(gsRef, &pStart, &pEnd));
  for (PetscInt p = pStart, dof; p < pEnd; ++p) {
    PetscCall(PetscSectionGetFieldDof(gsRef, p, 0, &dof));
    Nfine += dof > 0 ? dof : 0;
  }
  PetscCall(DMPlexGetTransitiveClosure(patch, cell, PETSC_TRUE, &Ncl, &closure));
  for (PetscInt cl = 0, dof; cl < Ncl*2; cl += 2) {
    PetscCall(PetscSectionGetFieldDof(gs, closure[cl], 0, &dof));
    Ncoarse += dof > 0 ? dof : 0;
  }
  PetscCall(PetscCalloc3(Nfine, &rows, Ncoarse, &cols, Nfine * Ncoarse, &elemP));
  PetscCall(DMGetGlobalSection(dm, &dgs));
  PetscCall(ISGetIndices(subpIS, &points));
  PetscCall(PetscSectionGetChart(gsRef, &pStart, &pEnd));
  for (PetscInt p = pStart, dof; p < pEnd; ++p) {
    PetscInt dof, off, doff;

    // TODO: dof here must be only for field 1
    PetscCall(PetscSectionGetFieldDof(gsRef, p, 0, &dof));
    // TODO: we need to convert this patch refined point to a domain refined point
    //   first get parent point for this point
    //   convert parent point to domain point
    //   get points produced by domain point
    PetscCall(PetscSectionGetFieldOffset(dgs, points[p], 0, &doff));
    for (PetscInt d = 0; d < dof; ++d) rows[off + d] = doff + d;
  }
  for (PetscInt cl = 0, i = 0; cl < Ncl*2; cl += 2) {
    PetscInt dof, doff;

    PetscCall(PetscSectionGetFieldDof(gs, closure[cl], 0, &dof));
    PetscCall(PetscSectionGetFieldOffset(dgs, points[closure[cl]], 0, &doff));
    for (PetscInt d = 0; d < dof; ++d) cols[i++] = doff + d;
  }
  PetscCall(ISRestoreIndices(subpIS, &points));
  PetscCall(DMPlexRestoreTransitiveClosure(patch, cell, PETSC_TRUE, &Ncl, &closure));
  PetscCall(DMCreateInterpolation(patch, rpatch, &P, NULL));

  PetscCall(SNESCreate(PetscObjectComm((PetscObject)rpatch), &snes));
  PetscCall(SNESSetDM(snes, rpatch));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMPlexSetSNESLocalFEM(rpatch, user, user, user));

  PetscCall(DMGetGlobalVector(rpatch, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "potential"));
  PetscCall(DMGetGlobalVector(rpatch, &b));
  PetscCall(PetscObjectSetName((PetscObject)b, "rhs"));
  PetscCall(DMGetGlobalVector(patch, &cb));
  PetscCall(PetscObjectSetName((PetscObject)cb, "rhs"));
  PetscCall(VecZeroEntries(cb));
  PetscCall(DMSNESCheckFromOptions(snes, u));
  PetscCall(DMPlexGetTransitiveClosure(patch, cell, PETSC_TRUE, &Ncl, &closure));
  for (PetscInt cl = 0; cl < Ncl*2; cl += 2) {
    const PetscScalar *a;
    PetscInt           dof, off;

    PetscCall(PetscSectionGetFieldDof(gs, closure[cl], 0, &dof));
    PetscCall(PetscSectionGetFieldOffset(gs, closure[cl], 0, &off));
    for (PetscInt d = 0; d < dof; ++d, ++j) {
      PetscCall(VecSetValue(cb, off+d, 1., INSERT_VALUES));
      PetscCall(MatMult(P, cb, b));
      PetscCall(SNESSolve(snes, b, u));
      PetscCall(VecSetValue(cb, off+d, 0., INSERT_VALUES));
      // Insert values into element matrix
      PetscCall(VecGetArrayRead(u, &a));
      // TODO Should only copy out first field
      for (PetscInt p = pStart; p < pEnd; ++p) {
        PetscInt dof, off;

        PetscCall(PetscSectionGetFieldDof(gsRef, p, 0, &dof));
        PetscCall(PetscSectionGetFieldDof(gsRef, p, 0, &off));
        for (PetscInt d = 0; d < dof; ++d) {
          if (PetscAbsScalar(a[off + d]) > PETSC_SMALL) elemP[(off + d) * Ncoarse + j] = a[off + d];
        }
      }
      PetscCall(VecRestoreArrayRead(u, &a));
    }
  }
  PetscCall(DMPlexRestoreTransitiveClosure(patch, cell, PETSC_TRUE, &Ncl, &closure));
  PetscCheck(j == Ncoarse, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Number of columns %" PetscInt_FMT " != %" PetscInt_FMT " coarse space size", j, Ncoarse);
  PetscCall(MatSetValues(dP, Nfine, rows, Ncoarse, cols, elemP, INSERT_VALUES));
  PetscCall(DMRestoreGlobalVector(rpatch, &u));
  PetscCall(DMRestoreGlobalVector(rpatch, &b));
  PetscCall(DMRestoreGlobalVector(patch, &cb));
  PetscCall(SNESDestroy(&snes));
  PetscCall(MatDestroy(&P));
  PetscCall(PetscFree3(rows, cols, elemP));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM          dm, rdm;
  Mat         P;
  PetscInt    cStart, cEnd;
  PetscBool   useCone, useClosure;
  AppCtx      user;
  const char *names[1] = {"phi"};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(DMRefine(dm, PETSC_COMM_WORLD, &rdm));
  PetscCall(SetupDiscretization(dm, 1, names, SetupPrimalProblem, &user));
  PetscCall(SetupDiscretization(rdm, 1, names, SetupPrimalProblem, &user));
  // Create global prolongator
  {
    PetscSection gs, rgs;
    PetscInt     m, n;

    PetscCall(DMGetGlobalSection(dm, &gs));
    PetscCall(DMGetGlobalSection(rdm, &rgs));
    PetscCall(PetscSectionGetStorageSize(rgs, &m));
    PetscCall(PetscSectionGetStorageSize(gs, &n));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &P));
    PetscCall(MatSetSizes(P, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetUp(P));
    PetscCall(MatSetOption(P, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));
    PetscCall(MatSetOption(P, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  }
  // Make a patch for each cell
  //   TODO: Just need a patch covering each dof
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMGetBasicAdjacency(dm, &useCone, &useClosure));
  PetscCall(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_TRUE));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    DM patch, rpatch;

    PetscCall(CreatePatch(dm, c, &patch));
    PetscCall(RefinePatch(patch, &rpatch));
    PetscCall(PatchSolve(dm, patch, c, rdm, rpatch, P, &user));
    PetscCall(DMDestroy(&rpatch));
    PetscCall(DMDestroy(&patch));
  }
  PetscCall(DMSetBasicAdjacency(dm, useCone, useClosure));
  PetscCall(MatDestroy(&P));
  PetscCall(DMDestroy(&rdm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -select_dm_plex_transform_type transform_filter \
          -phi_petscspace_degree 1 -mu_petscspace_degree 1 -pc_type lu \
          -snes_converged_reason -snes_monitor

TEST*/
