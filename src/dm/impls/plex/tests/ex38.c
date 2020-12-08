const char help[] = "A test demonstrating stratum-dof grouping methods.\n";

#include <petscconvest.h>
#include <petscdmplex.h>
#include <petscds.h>
#include <petscsnes.h>
#include <petsc/private/petscfeimpl.h>

/* Examples solve the system governed by:
 *
 * \vec{u} = -\grad{p}
 * \div{\vec{u}} = f
 *
 */


/* We label solutions by the form of the potential/pressure, p: i.e. linear_u is the analytical form of u one gets when p is linear. */
/* 2D Linear Exact Functions
   p = x;
   \vec{u} = <-1, 0>;
   f = 0;
   \div{\vec{u}} = 0;
   */
static PetscErrorCode linear_p(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *    u,void *           ctx)
{
  u[0] = x[0];
  return 0;
}
static PetscErrorCode linear_u(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *    u,void *           ctx)
{
  /* Need to set only the x-component i.e. c==0  */
  for (PetscInt c = 0; c < Nc; ++c) u[c] = c ? 0.0 : -1.0;
  return 0;
}
static PetscErrorCode linear_source(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *    u,void *           ctx)
{
  for (PetscInt c = 0; c < Nc; ++c) u[c] = 0;
  return 0;
}

/* 2D Sinusoidal Exact Functions
   p = sin(2*pi*x)*sin(2*pi*y);
   \vec{u} = <2*pi*cos(2*pi*x)*sin(2*pi*y), 2*pi*cos(2*pi*y)*sin(2*pi*x);
   \div{\vec{u}} = -8*pi^2*sin(2*pi*x)*sin(2*pi*y);
   */
static PetscErrorCode sinusoid_p(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *    u,void *           ctx)
{
  u[0] = 1;
  for (PetscInt d = 0; d < dim; ++d) u[0] *= PetscSinReal(2 * PETSC_PI * x[d]);
  return 0;
}

static PetscErrorCode sinusoid_u(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *    u,void *           ctx)
{
  for (PetscInt c = 0; c < Nc; ++c)
  {
    u[c] = 1;
    for (PetscInt d = 0; d < dim; ++d)
    {
      if (d == c) u[c] *= 2 * PETSC_PI * PetscCosReal(2 * PETSC_PI * x[d]);
      else u[c] *= PetscSinReal(2 * PETSC_PI * x[d]);
    }
  }
  return 0;
}

static PetscErrorCode sinusoid_source(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *    u,void *           ctx)
{
  u[0] = -8 * PETSC_PI * PETSC_PI;
  for (PetscInt d = 0; d < dim; ++d) u[0] *= sin(2 * PETSC_PI * x[d]);
  return 0;
}


/* Pointwise function for (v,u) */
static void f0_v(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt i;

  for (i = 0; i < dim; ++i) f0[i] = u[uOff[0] + i];
}

/* This is the pointwise function that represents (\trace(\grad v),p) == (\grad
 * v : I*p) */
static void f1_v(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f1[])
{
  PetscInt c,d;
  for (c = 0; c < dim; ++c)
    for (d = 0; d < dim; ++d)
      if (c == d) f1[c * dim + d] = -u[uOff[1]];
}

/* represents (\div u - f,q). */
static void f0_q_linear(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt    i;
  PetscScalar rhs = 0.0;
  PetscScalar divu;

  (void) linear_source(dim,t,x,dim,&rhs,NULL);
  divu = 0.;
  /* diagonal terms of the gradient */
  for (i = 0; i < dim; ++i) divu += u_x[uOff_x[0] + i * dim + i];
  f0[0] = divu - rhs;
}

static void f0_q_sinusoid(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt    i;
  PetscScalar rhs;
  PetscScalar divu;

  (void) sinusoid_source(dim,t,x,dim,&rhs,NULL);
  divu = 0.;
  for (i = 0; i < dim; ++i) divu += u_x[uOff_x[0] + i * dim + i];
  f0[0] = divu - rhs;
}

static void f0_linear_bd_u(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],const PetscReal n[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscScalar pressure;

  (void) linear_p(dim,t,x,dim,&pressure,NULL);
  for (PetscInt d = 0; d < dim; ++d) f0[d] = pressure * n[d];
}
static void f0_sinusoid_bd_u(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],const PetscReal n[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscScalar pressure;

  (void) sinusoid_p(dim,t,x,dim,&pressure,NULL);
  for (PetscInt d = 0; d < dim; ++d) f0[d] = pressure * n[d];
}

/* <v, u> */
static void g0_vu(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g0[])
{
  PetscInt c;
  for (c = 0; c < dim; ++c) g0[c * dim + c] = 1.0;
}

/* <-p,\nabla\cdot v> = <-pI,\nabla u> */
static void g2_vp(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g2[])
{
  PetscInt c;
  for (c = 0; c < dim; ++c) g2[c * dim + c] = -1.0;
}

/* <q, \nabla\cdot u> */
static void g1_qu(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d * dim + d] = 1.0;
}

typedef enum
{
  NONE = 0,
  RANDOM = 1,
  SKEW = 2,
  SKEWRAND = 3
} Perturbation;
const char* const PerturbationTypes[] =
{"none","random","skew","skewrand","Perturbation","",NULL};

typedef enum
{
  LINEAR = 0,
  SINUSOIDAL = 1
} Solution;
const char* const SolutionTypes[] = {"linear",
                                     "sinusoidal",
                                     "Solution",
                                     "",
                                     NULL};

typedef struct
{
  PetscBool    simplex;
  PetscInt     dim;
  Perturbation mesh_transform;
  Solution     sol_form;
} UserCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm,UserCtx * user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  user->simplex        = PETSC_TRUE;
  user->dim            = 2;
  user->mesh_transform = NONE;
  user->sol_form       = LINEAR;
  // Define/Read in example parameters
  ierr = PetscOptionsBegin(comm,"","Stratum Dof Grouping Options","DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool(
    "-simplex",
    "Whether to use simplices (true) or tensor-product (false) cells in "
    "the mesh",
    "ex38.c",
    user->simplex,
    &user->simplex,
    NULL
    );CHKERRQ(ierr);
  ierr = PetscOptionsInt(
    "-dim",
    "Number of solution dimensions",
    "ex38.c",
    user->dim,
    &user->dim,
    NULL
    );CHKERRQ(ierr);
  ierr = PetscOptionsEnum(
    "-mesh_transform",
    "Method used to perturb the mesh vertices. Options are Skew,Random,"
    "SkewRand,or None",
    "ex38.c",
    PerturbationTypes,
    (PetscEnum) user->mesh_transform,
    (PetscEnum*) &user->mesh_transform,
    NULL
    );CHKERRQ(ierr);
  ierr = PetscOptionsEnum(
    "-sol_form",
    "Form of the exact solution. Options are Linear or Sinusoidal",
    "ex38.c",
    SolutionTypes,
    (PetscEnum) user->sol_form,
    (PetscEnum*) &user->sol_form,
    NULL
    );CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PerturbMesh(DM *          mesh,PetscScalar * coordVals,PetscInt ncoord,PetscInt dim,PetscRandom * ran)
{
  PetscErrorCode ierr;
  PetscReal      minCoords[3],maxCoords[3],maxPert[3],randVal;
  PetscScalar    phase,amp;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDim(*mesh,&dim);CHKERRQ(ierr);
  ierr = DMGetLocalBoundingBox(*mesh,minCoords,maxCoords);CHKERRQ(ierr);

  /* Compute something ~= half an edge length. This is the most we can perturb
   * points and gaurantee that there won't be any topology issues. */
  for (int k = 0; k < dim; ++k) maxPert[k] = 0.5 * (maxCoords[k] - minCoords[k])
                                             / (PetscPowReal(ncoord,1. / dim) - 1);
  for (int i = 0; i < ncoord; ++i)
    for (int j = 0; j < dim; ++j)
    {
      ierr                    = PetscRandomGetValueReal(*ran,&randVal);CHKERRQ(ierr);
      phase                   = PETSC_PI * (randVal - 0.5);
      ierr                    = PetscRandomGetValueReal(*ran,&randVal);CHKERRQ(ierr);
      amp                     = maxPert[j] * (randVal - 0.5);
      coordVals[dim * i + j] +=
        amp
        * PetscSinReal(
          2 * PETSC_PI / maxCoords[j] * coordVals[dim * i + j] + phase
          );
    }
  PetscFunctionReturn(0);
}

PetscErrorCode SkewMesh(DM * mesh,PetscScalar * coordVals,PetscInt ncoord,PetscInt dim)
{
  PetscErrorCode ierr;
  PetscReal      * transMat;

  PetscFunctionBegin;
  ierr = PetscCalloc1(dim * dim,&transMat);CHKERRQ(ierr);

  /* Make a matrix representing a skew transformation */
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j)
    {
      if (i == j) transMat[i * dim + j] = 1;
      else if (j < i) transMat[i * dim + j] = 2 * (j + i);
      else transMat[i * dim + j] = 0;
    }

  /* Multiply each coordinate vector by our tranformation */
  for (int i = 0; i < ncoord; ++i)
  {
    PetscReal tmpcoord[3];
    for (int j = 0; j < dim; ++j)
    {
      tmpcoord[j] = 0;
      for (int k = 0; k < dim; ++k) tmpcoord[j] += coordVals[dim * i + k] * transMat[dim * k + j];
    }
    for (int l = 0; l < dim; ++l) coordVals[dim * i + l] = tmpcoord[l];
  }
  ierr = PetscFree(transMat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TransformMesh(UserCtx * user,DM * mesh,PetscRandom * ran)
{
  PetscErrorCode ierr;
  PetscInt       dim,ncoord;
  PetscScalar    * coordVals;
  Vec            coords;

  PetscFunctionBegin;
  ierr   = DMGetCoordinates(*mesh,&coords);CHKERRQ(ierr);
  ierr   = VecGetArray(coords,&coordVals);CHKERRQ(ierr);
  ierr   = VecGetLocalSize(coords,&ncoord);CHKERRQ(ierr);
  ierr   = DMGetCoordinateDim(*mesh,&dim);CHKERRQ(ierr);
  ncoord = ncoord / dim;

  switch (user->mesh_transform) {
  case NONE:
    break;
  case RANDOM:
    ierr = PerturbMesh(mesh,coordVals,ncoord,dim,ran);CHKERRQ(ierr);
    break;
  case SKEW:
    ierr = SkewMesh(mesh,coordVals,ncoord,dim);CHKERRQ(ierr);
    break;
  case SKEWRAND:
    ierr = SkewMesh(mesh,coordVals,ncoord,dim);CHKERRQ(ierr);
    ierr = PerturbMesh(mesh,coordVals,ncoord,dim,ran);CHKERRQ(ierr);
    break;
  default:
    PetscFunctionReturn(-1);
  }
  ierr = VecRestoreArray(coords,&coordVals);CHKERRQ(ierr);
  ierr = DMSetCoordinates(*mesh,coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEIntegrateJacobian_WY(PetscDS ds, PetscFEJacobianType jtype, PetscInt fieldI, PetscInt fieldJ, PetscInt Ne, PetscFEGeom *cgeom,
                                              const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS dsAux, const
                                              PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[]){
PetscErrorCode ierr;
PetscInt lumpMap[]  = {5,0,1,2,3,4};/* Vertex lumping for reference element, I think*/
PetscInt eOffset = 0, totDim, e,offsetI,offsetJ,dim,f,fc,g,gc;
PetscScalar tempEmat[36] = {};
PetscTabulation *T;

PetscFunctionBegin;

ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
ierr = PetscDSGetFieldOffset(ds, fieldI, &offsetI);CHKERRQ(ierr);
ierr = PetscDSGetFieldOffset(ds, fieldJ, &offsetJ);CHKERRQ(ierr);
ierr = PetscDSGetTabulation(ds, &T);CHKERRQ(ierr); 
ierr = PetscFEIntegrateJacobian_Basic(ds, jtype, fieldI, fieldJ, Ne, cgeom, coefficients, coefficients_t, dsAux, coefficientsAux, t,
    u_tshift, elemMat);CHKERRQ(ierr); 
for(e=0; e < Ne; ++e){
 /* Apply vertex lumping map to  elemMat */
  for (fc = 0; fc < T[fieldI]->Nc; ++fc) {
    for (f = 0; f < T[fieldI]->Nb; ++f) {
      const PetscInt i = offsetI + f*T[fieldI]->Nc + fc;
      for (gc = 0; gc < T[fieldJ]->Nc; ++gc) {
        for (g = 0; g < T[fieldJ]->Nb; ++g) {
          const PetscInt j = offsetJ + g*T[fieldJ]->Nc+gc;
        }
      }
    }
  }
  eOffset += PetscSqr(totDim);
}
PetscFunctionReturn(0);
}



static PetscErrorCode CreateMesh(MPI_Comm comm,UserCtx * user,DM * mesh)
{
  PetscErrorCode   ierr;
  PetscRandom      ran;
  DMLabel          label;
  const char       * name = "marker";
  DM               dmDist = NULL;
  PetscPartitioner part;

  PetscFunctionBegin;
  ierr = PetscRandomCreate(comm,&ran);CHKERRQ(ierr);
  // Create a mesh (2D vs. 3D) and (simplex vs. tensor) as determined by
  // parameters
  // TODO: make either a simplex or tensor-product mesh
  // Desirable: a mesh with skewing element transforms that will stress the
  // Piola transformations involved in assembling H-div finite elements
  /* Create box mesh from user parameters */
  ierr = DMPlexCreateBoxMesh(
    comm,user->dim,user->simplex,NULL,NULL,NULL,NULL,PETSC_TRUE,mesh
    );CHKERRQ(ierr);

  ierr = DMPlexGetPartitioner(*mesh,&part);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
  ierr = DMPlexDistribute(*mesh,0,NULL,&dmDist);CHKERRQ(ierr);
  if (dmDist) {
    ierr  = DMDestroy(mesh);CHKERRQ(ierr);
    *mesh = dmDist;
  }
  ierr = DMCreateLabel(*mesh,name);CHKERRQ(ierr);
  ierr = DMGetLabel(*mesh,name,&label);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(*mesh,1,label);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(*mesh,label);CHKERRQ(ierr);
  ierr = DMLocalizeCoordinates(*mesh);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *mesh,"Mesh");CHKERRQ(ierr);
  ierr = TransformMesh(user,mesh,&ran);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*mesh,user);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*mesh);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*mesh,NULL,"-dm_view");CHKERRQ(ierr);

  ierr = DMDestroy(&dmDist);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&ran);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm,UserCtx * user)
{
  PetscDS        prob;
  PetscErrorCode ierr;
  const PetscInt id = 1;

  PetscFunctionBegin;
  ierr = DMGetDS(dm,&prob);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob,0,f0_v,f1_v);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,0,0,g0_vu,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,0,1,NULL,NULL,g2_vp,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,1,0,NULL,g1_qu,NULL,NULL);CHKERRQ(ierr);

  switch (user->sol_form) {
  case LINEAR:
    ierr = PetscDSSetResidual(prob,1,f0_q_linear,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetBdResidual(prob,0,f0_linear_bd_u,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob,0,linear_u,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob,1,linear_p,NULL);CHKERRQ(ierr);
    break;
  case SINUSOIDAL:
    ierr = PetscDSSetResidual(prob,1,f0_q_sinusoid,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetBdResidual(prob,0,f0_sinusoid_bd_u,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob,0,sinusoid_u,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob,1,sinusoid_p,NULL);CHKERRQ(ierr);
    break;
  default:
    PetscFunctionReturn(-1);
  }
  ierr = PetscDSAddBoundary(
    prob,
    DM_BC_NATURAL,
    "Boundary Integral",
    "marker",
    0,
    0,
    NULL,
    (void (*)(void))NULL,
    (void (*)(void))NULL,
    1,
    &id,
    user
    );CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM mesh,PetscErrorCode (*setup)(DM,UserCtx*),UserCtx * user)
{
  DM             cdm = mesh;
  PetscFE        fevel,fepres;
  const PetscInt dim               = user->dim;
  PetscBool      corner_quadrature = PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFECreateDefault(
    PETSC_COMM_WORLD,//PetscObjectComm((PetscObject) mesh),
    dim,
    dim,
    user->simplex,
    "velocity_",
    PETSC_DEFAULT,
    &fevel
    );CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fevel,"velocity");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(
    PetscObjectComm((PetscObject) mesh),
    dim,
    1,
    user->simplex,
    "pressure_",
    PETSC_DEFAULT,
    &fepres
    );CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fepres,"pressure");CHKERRQ(ierr);
  if (corner_quadrature) {
    PetscInt        dim;
    PetscInt        numPoints;
    PetscBool       simplex;
    PetscReal       * points;
    PetscReal       * weights;
    PetscQuadrature quad;

    dim       = user->dim;
    simplex   = user->simplex;
    numPoints = simplex ? (dim + 1) : 2 * dim;
    ierr      = PetscMalloc1(dim * numPoints,&points);CHKERRQ(ierr);
    ierr      = PetscMalloc1(numPoints,&weights);CHKERRQ(ierr);
    if (simplex)
      switch (dim) {
      case 2:
        points[0]  = -1.;
        points[1]  = -1.;
        points[2]  = 1.;
        points[3]  = -1.;
        points[4]  = -1.;
        points[5]  = 1.;
        weights[0] = weights[1] = weights[2] = 2. / 3.;
        break;
      case 3:
        points[0]  = -1.;
        points[1]  = -1.;
        points[2]  = -1.;
        points[3]  = 1.;
        points[4]  = -1.;
        points[5]  = -1.;
        points[6]  = -1.;
        points[7]  = 1.;
        points[8]  = -1.;
        points[9]  = -1.;
        points[10] = -1.;
        points[11] = 1.;
        weights[0] = weights[1] = weights[2] = 3. / 4.;
        break;
      }

    ierr =
      PetscQuadratureCreate(PetscObjectComm((PetscObject) mesh),&quad);CHKERRQ(ierr);
    ierr = PetscQuadratureSetData(quad,dim,1,numPoints,points,weights);CHKERRQ(ierr);
    ierr = PetscFESetQuadrature(fevel,quad);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&quad);CHKERRQ(ierr);
  }

  ierr = PetscFECopyQuadrature(fevel,fepres);CHKERRQ(ierr);
  fevel->ops->integratejacobian = PetscFEIntegrateJacobian_WY;

  ierr = DMSetField(mesh,0,NULL,(PetscObject) fevel);CHKERRQ(ierr);
  ierr = DMSetField(mesh,1,NULL,(PetscObject) fepres);CHKERRQ(ierr);
  ierr = DMCreateDS(mesh);CHKERRQ(ierr);
  ierr = (*setup)(mesh,user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(mesh,cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm,&cdm);CHKERRQ(ierr);
  }

  ierr = PetscFEDestroy(&fevel);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fepres);CHKERRQ(ierr);
  ierr = DMDestroy(&cdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateVertDofMap(DM mesh,PetscSection * vert2Dof,PetscInt **    vertDofMap,PetscInt *     numVertDof)
{
  PetscErrorCode ierr;
  PetscInt       eStart,eEnd,vStart,vEnd,*vertDofCounter;
  PetscSection   localSec;

  PetscFunctionBegin;
  /* Allocate and initialize some function local variables */
  ierr = DMGetLocalSection(mesh,&localSec);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(mesh,1,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(mesh,0,&vStart,&vEnd);CHKERRQ(ierr);
  const PetscInt numEdge = eEnd - eStart;
  const PetscInt numVert = vEnd - vStart;
  *numVertDof = 2 * numEdge;

  ierr = PetscSectionCreate(PETSC_COMM_WORLD,vert2Dof);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*vert2Dof,vStart,vEnd);CHKERRQ(ierr);

  /*Set the correct number of DoFs for each vertex */
  for (PetscInt iEdge = eStart; iEdge < eEnd; ++iEdge)
  {
    PetscInt       numDoF = -1;
    const PetscInt * verts;
    const PetscInt * vOrient;
    ierr = DMPlexGetCone(mesh,iEdge,&verts);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(mesh,iEdge,&vOrient);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(localSec,iEdge,&numDoF);CHKERRQ(ierr);
    for (PetscInt iDoF = 0; iDoF < numDoF; ++iDoF)
    {
      const PetscInt vInd = (vOrient[iDoF] >= 0) ? vOrient[iDoF] : -(vOrient[iDoF]+1);
      ierr = PetscSectionAddDof(*vert2Dof,verts[vInd],1);CHKERRQ(ierr);
    }
  }

  PetscSectionSetUp(*vert2Dof);
  ierr = PetscCalloc1(numVert,&vertDofCounter);CHKERRQ(ierr);
  ierr = PetscCalloc1(2 * numEdge,vertDofMap);CHKERRQ(ierr);

  /*Assign the DoFs into the map */
  for (PetscInt iEdge = eStart; iEdge < eEnd; ++iEdge)
  {
    const PetscInt * verts, *vOrient;
    PetscInt       numDoF;
    PetscInt       eOff;
    ierr = DMPlexGetCone(mesh,iEdge,&verts);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(localSec,iEdge,&numDoF);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(mesh,iEdge,&vOrient);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(localSec,iEdge,&eOff);CHKERRQ(ierr);
    for (PetscInt iDoF = 0; iDoF < numDoF; ++iDoF)
    {
      PetscInt       vOff;
      const PetscInt vInd = (vOrient[iDoF] >= 0) ? vOrient[iDoF] : -(vOrient[iDoF]+1);
      const PetscInt vInd0 = verts[vInd] - vStart;

      ierr = PetscSectionGetOffset(*vert2Dof,verts[vInd],&vOff);CHKERRQ(ierr);
      PetscInt mapInd = vOff + vertDofCounter[vInd0];
      (*vertDofMap)[mapInd] = eOff + iDoF;
      ++vertDofCounter[vInd0];
    }
  }
  ierr = PetscFree(vertDofCounter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 * PetscSectionGetFieldChart - Get the chart range for points which store the
 * DoFs of a specified filed.
 *
 * Input Parameters:
 * + s - The PetscSection
 * - field - The index of the field of interest
 *
 *   Output Parameters:
 *   + pStart - Start index of the chart
 *   - pEnd - End index of the chart
 *
 *   Level:
 @*/
PetscErrorCode PetscSectionGetFieldChart(PetscSection s,PetscInt field,PetscInt *    pStart,PetscInt *    pEnd)
{
  PetscErrorCode ierr;
  PetscSection   fieldSec;
  PetscInt       cBegin,cEnd,nDof;

  PetscFunctionBegin;
  ierr = PetscSectionGetField(s,field,&fieldSec);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(fieldSec,&cBegin,&cEnd);CHKERRQ(ierr);

  for (PetscInt p = cBegin; p < cEnd; ++p)
  {
    ierr = PetscSectionGetDof(fieldSec,p,&nDof);CHKERRQ(ierr);
    if (nDof > 0) {
      *pStart = p;
      break;
    }
  }

  for (PetscInt p = cEnd - 1; p >= cBegin; --p)
  {
    ierr = PetscSectionGetDof(fieldSec,p,&nDof);CHKERRQ(ierr);
    if (nDof > 0) {
      *pEnd = p + 1;
      break;
    }
  }

  /* TODO: Handle case where no points in the current section have DoFs
   * belonging to the specified field. Possibly by returning negative values for
   * pStart and pEnd */

  PetscFunctionReturn(0);
}

/*@C
 * DMPlexGetFieldDepth - Find the stratum on which the desired
 * field's DoFs are currently assigned.
 *
 * Input Parameters:
 * + dm - The DM
 * - field - Index of the field on the DM
 *
 *   Output Parameters:
 *   - depth - The depth of the stratum that to which field's DoFs are assigned
 @*/
PetscErrorCode DMPlexGetFieldDepth(DM dm,PetscInt field,PetscInt * depth)
{
  PetscErrorCode ierr;
  PetscSection   localSec;
  PetscInt       maxDepth,fStart = -1,fEnd = -1,pStart,pEnd;

  PetscFunctionBegin;
  ierr = DMGetLocalSection(dm,&localSec);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldChart(localSec,field,&fStart,&fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm,&maxDepth);CHKERRQ(ierr);

  for (*depth = 0; *depth <= maxDepth; ++(*depth))
  {
    ierr = DMPlexGetDepthStratum(dm,*depth,&pStart,&pEnd);CHKERRQ(ierr);
    if (pStart == fStart && pEnd == fEnd) break;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionInvertMapping(PetscSection s,IS is,PetscSection * newSec,IS *           newIs)
{
  PetscErrorCode ierr;
  PetscInt       sStart,sEnd,isStart,isEnd;

  /* WIP: May not be needed, but may also be useful to have */
  PetscFunctionBegin;
  ierr = PetscSectionGetChart(s,&sStart,&sEnd);CHKERRQ(ierr);
  ierr = ISGetMinMax(is,&isStart,&isEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,newSec);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*newSec,isStart,isEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
 * DMPlexGetStratumMap - Create a mapping in the form of a section and IS which
 * associates points from one stratum with points from another.
 *
 * Input Parameters:
 * + dm - The DM
 * . source - Depth value of the source stratum
 * - target - Depth value of the target stratum
 *
 *   Output Parameters:
 *   + s - PetscSection which contains the number of target points and offset
 *   for each point in source.
 *   - is - IS containing the indices of points in target stratum
 @*/
PetscErrorCode DMPlexGetStratumMap(DM dm,PetscInt source,PetscInt target,PetscSection * s,IS *           is)
{
  PetscErrorCode ierr;
  PetscInt       pStart,pEnd,tStart,tEnd,nClosurePoints,*closurePoints = NULL,
                 isCount = 0,*idx,pOff,*pCount;
  PetscBool inCone = PETSC_TRUE;

  PetscFunctionBegin;
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,s);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,source,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,target,&tStart,&tEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*s,pStart,pEnd);CHKERRQ(ierr);
  if (source == target) {
    /* Each point maps to itself and only to itself, this is the trivial map
     */

    for (PetscInt p = pStart; p < pEnd; ++p)
    {
      ierr = PetscSectionSetDof(*s,p,1);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(*s);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_WORLD,pEnd - pStart,pStart,1,is);CHKERRQ(ierr);
  }else {
    if (source < target) inCone = PETSC_FALSE;
    /* TODO: This routine currently relies on a number of calls to
     * DMPlexGetTransitiveClosure. Determine whether there is a more efficient
     * method and/or if this is an instance of reinventing the wheel due to
     * existence of PetscSectionGetClosureIndex */

    /* Count the number of target points for each source
     * so that the proper amount of memory can be allocated for the section and
     * IS */
    for (PetscInt p = pStart; p < pEnd; ++p)
    {
      ierr = DMPlexGetTransitiveClosure(
        dm,p,inCone,&nClosurePoints,&closurePoints
        );CHKERRQ(ierr);

      for (PetscInt cp = 0; cp < nClosurePoints; ++cp)
      {
        PetscInt closurePoint = closurePoints[2 * cp];
        /* Check if closure point is in target stratum */
        if (closurePoint >= tStart && closurePoint < tEnd) {
          /* Add a DoF to the section and increment IScount */
          ierr = PetscSectionAddDof(*s,p,1);CHKERRQ(ierr);
          ++isCount;
        }
      }

      ierr = DMPlexRestoreTransitiveClosure(
        dm,p,inCone,&nClosurePoints,&closurePoints
        );CHKERRQ(ierr);
    }

    ierr    = PetscSectionSetUp(*s);CHKERRQ(ierr);
    ierr    = PetscCalloc1(isCount,&idx);CHKERRQ(ierr);
    ierr    = PetscCalloc1(pEnd-pStart,&pCount);CHKERRQ(ierr);

    /* Now that proper space is allocated assign the correct values to the IS */
    /* TODO: Check that this method of construction preserves the orientation */
    for (PetscInt p = pStart; p < pEnd; ++p)
    {
      ierr = DMPlexGetTransitiveClosure(
        dm,p,inCone,&nClosurePoints,&closurePoints
        );CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(*s,p,&pOff);CHKERRQ(ierr);

      for (PetscInt cp = 0; cp < nClosurePoints; ++cp)
      {
        PetscInt closurePoint = closurePoints[2 * cp];
        /* Check if closure point is in target stratum */
        if (closurePoint >= tStart && closurePoint < tEnd) idx[pOff + pCount[p-pStart]++] = closurePoint;

      }

      ierr = DMPlexRestoreTransitiveClosure(
        dm,p,inCone,&nClosurePoints,&closurePoints
        );CHKERRQ(ierr);
    }
    ierr = ISCreateGeneral(
      PETSC_COMM_WORLD,isCount,idx,PETSC_OWN_POINTER,is
      );CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/*@C
 * DMPlexGetStratumDofMap - Create a map consisting of a PetscSection and IS
 * from a specified stratum to the DoFs of the specified field.
 *
 * Input Parameters:
 * + dm - The DM.
 * . stratum - The depth value of the stratum to map from.
 * - field - The index of the field whose DoFs are to be assigned to stratum.
 *
 *   Output Parameters:
 *   + section - The section to be created consisting of points in the stratum.
 *   - is - The IS to be created which will contain indices of the field DoFs.
 @*/
PetscErrorCode DMPlexGetStratumDofMap(DM dm,PetscInt stratum,PetscInt field,PetscSection *section,IS *is)
{
  PetscErrorCode ierr;
  PetscInt
                 fieldDepth,pStart,pEnd,fStart,fEnd,*idx,dofCount=0,numDof,numPoints,pOff,stratSize,*stratDofCount,sOff,mapInd,sInd0,dofOff;
  const PetscInt *stratInds;
  PetscSection   stratumSec,localSec;
  IS             stratum2Stratum;

  PetscFunctionBegin;
  ierr      = DMGetLocalSection(dm,&localSec);CHKERRQ(ierr);
  ierr      = DMPlexGetFieldDepth(dm,field,&fieldDepth);CHKERRQ(ierr);
  ierr      = DMPlexGetDepthStratum(dm,fieldDepth,&fStart,&fEnd);CHKERRQ(ierr);
  ierr      = DMPlexGetDepthStratum(dm,stratum,&pStart,&pEnd);CHKERRQ(ierr);
  ierr      = PetscSectionCreate(PETSC_COMM_WORLD,section);CHKERRQ(ierr);
  ierr      = PetscSectionSetChart(*section,pStart,pEnd);CHKERRQ(ierr);
  stratSize = pEnd-pStart;
  ierr      =
    DMPlexGetStratumMap(dm,fieldDepth,stratum,&stratumSec,&stratum2Stratum);CHKERRQ(ierr);
  ierr = ISGetIndices(stratum2Stratum,&stratInds);CHKERRQ(ierr);

  for (PetscInt i = fStart; i <fEnd; ++i) {
    ierr = PetscSectionGetDof(localSec,i,&numDof);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(stratumSec,i,&numPoints);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(stratumSec,i,&pOff);CHKERRQ(ierr);
    if (numDof==numPoints) {
      for (PetscInt j = 0; j<numDof; ++j) {
        ierr = PetscSectionAddDof(*section,stratInds[pOff + j],1);CHKERRQ(ierr);
      }
      dofCount += numDof;
    }
  }

  ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
  ierr = PetscCalloc1(stratSize,&stratDofCount);CHKERRQ(ierr);
  ierr = PetscCalloc1(dofCount,&idx);CHKERRQ(ierr);

  for (PetscInt i = fStart; i < fEnd; ++i) {
    ierr = PetscSectionGetDof(localSec,i,&numDof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(localSec,i,&dofOff);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(stratumSec,i,&numPoints);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(stratumSec,i,&pOff);CHKERRQ(ierr);
    if (numDof==numPoints)
      for (PetscInt j = 0; j < numDof; ++j) {
        sInd0 = stratInds[pOff+j] - pStart;
        ierr  =
          PetscSectionGetOffset(*section,stratInds[pOff+j],&sOff);CHKERRQ(ierr);
        mapInd      = sOff + stratDofCount[sInd0]++;
        idx[mapInd] = dofOff+j;

      }
  }
  ierr = ISRestoreIndices(stratum2Stratum,&stratInds);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,dofCount,idx,PETSC_OWN_POINTER,is);CHKERRQ(ierr);
  ierr = PetscFree(stratDofCount);CHKERRQ(ierr);
  ierr = ISDestroy(&stratum2Stratum);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&stratumSec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char ** argv)
{
  UserCtx        user;
  DM             mesh;
  SNES           snes;
  PetscErrorCode ierr;
  PetscSection   edgeVertSec,cellVertSec,vertCellSec,vertDofSec,conesection;
  IS             edge2Vert,cell2Vert,vert2Cell,vert2Dof,*fieldIS,lumpPerm,isList[2];
  PetscInt       pDepth,uDepth,*cones;
  Mat            jacobian,permJacobian;
  Vec            u,b;

  ierr = PetscInitialize(&argc,&argv,NULL,help);
  if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD,&user);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD,&user,&mesh);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,mesh);CHKERRQ(ierr);
  ierr = SetupDiscretization(mesh,SetupProblem,&user);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(mesh,&user,&user,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(mesh,&u);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(mesh,&b);CHKERRQ(ierr);
  ierr = DMPlexGetConeSection(mesh, &conesection);CHKERRQ(ierr);
  ierr = DMPlexGetCones(mesh,&cones);CHKERRQ(ierr);

  ierr = DMCreateFieldIS(mesh,NULL,NULL,&fieldIS);CHKERRQ(ierr);
  ierr = DMCreateMatrix(mesh,&jacobian);CHKERRQ(ierr);

  ierr = VecSet(u,0.0);CHKERRQ(ierr);
  ierr = VecSet(b,0.0);CHKERRQ(ierr);

  ierr = SNESSolve(snes,b,u);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes,&jacobian,NULL,NULL,NULL);CHKERRQ(ierr); 

  ierr = MatViewFromOptions(jacobian,NULL,"-jacobian_view");CHKERRQ(ierr);

  /* Example: Getting depth stratum of pressure and velocity fields */
  ierr = DMPlexGetFieldDepth(mesh,1,&pDepth);CHKERRQ(ierr);
  ierr = DMPlexGetFieldDepth(mesh,0,&uDepth);CHKERRQ(ierr);

  /* Example: Mapping from edges to vertices*/
  ierr = DMPlexGetStratumMap(mesh,1,0,&edgeVertSec,&edge2Vert);CHKERRQ(ierr);

  PetscSectionView(edgeVertSec,PETSC_VIEWER_STDOUT_WORLD);
  ISView(edge2Vert,PETSC_VIEWER_STDOUT_WORLD);

  /* Example: Mapping from cells to vertices*/
  ierr = DMPlexGetStratumMap(mesh,user.dim,0,&cellVertSec,&cell2Vert);CHKERRQ(ierr);

  //PetscSectionView(cellVertSec,PETSC_VIEWER_STDOUT_WORLD);
  //ISView(cell2Vert,PETSC_VIEWER_STDOUT_WORLD);

  /* Example: Mapping from vertices to cells*/
  ierr = DMPlexGetStratumMap(mesh,0,user.dim,&vertCellSec,&vert2Cell);CHKERRQ(ierr);

  //PetscSectionView(vertCellSec,PETSC_VIEWER_STDOUT_WORLD);
  //ISView(vert2Cell,PETSC_VIEWER_STDOUT_WORLD);

  /* Example: Mapping vertices to velocity DoFs*/
  ierr = DMPlexGetStratumDofMap(mesh,0,0,&vertDofSec,&vert2Dof);CHKERRQ(ierr);

//  PetscSectionView(vertDofSec,PETSC_VIEWER_STDOUT_WORLD);
//  ISView(vert2Dof,PETSC_VIEWER_STDOUT_WORLD);
  isList[0] = fieldIS[1];
  isList[1] = vert2Dof; 
  ierr = ISConcatenate(PETSC_COMM_WORLD,2,isList,&lumpPerm);CHKERRQ(ierr);

  ierr = MatPermute(jacobian,lumpPerm,lumpPerm,&permJacobian);CHKERRQ(ierr);
  ierr = MatViewFromOptions(permJacobian,NULL,"-permJacobian_view");CHKERRQ(ierr);

  // Tear down
  ierr = ISDestroy(&edge2Vert);CHKERRQ(ierr);
  ierr = ISDestroy(&cell2Vert);CHKERRQ(ierr);
  ierr = ISDestroy(&vert2Cell);CHKERRQ(ierr);
  ierr = ISDestroy(&vert2Dof);CHKERRQ(ierr);

  ierr = PetscSectionDestroy(&edgeVertSec);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cellVertSec);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&vertCellSec);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&vertDofSec);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&mesh);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;

}

/*TEST
  testset:
    suffix: 2d_bdm
    requires: triangle
    args: -dim 2 \
      -velocity_petscspace_degree 1 \
      -velocity_petscdualspace_type bdm \
      -velocity_petscdualspace_lagrange_node_endpoints true 
    test:
      suffix: linear
      args: -sol_form linear -mesh_transform none

  testset:
    suffix: 3d_bdm
    requires: triangle
    args: -dim 3 \
      -velocity_petscspace_degree 1 \
      -velocity_petscdualspace_type bdm 
    test:
      suffix: linear
      args: -sol_form linear -mesh_transform none
  
TEST*/
