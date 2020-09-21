<<<<<<< HEAD
<<<<<<< HEAD
/*
  Note:
    -hratio is the ratio between mesh size of corse grids and fine grids
<<<<<<< HEAD
<<<<<<< HEAD
*/

static const char help[] = "1D periodic Finite Volume solver in slope-limiter form with semidiscrete time stepping.\n"
  "  advect      - Constant coefficient scalar advection\n"
  "                u_t       + (a*u)_x               = 0\n"
  "  shallow     - 1D Shallow water equations (Saint Venant System)\n"
  "                h_t + (q)_x = 0 \n"
  "                q_t + (\frac{q^2}{h} + g/2*h^2)_x = 0  \n"
  "                where, h(x,t) denotes the height of the water and q(x,t) the momentum.\n"
=======
    -ts_rk_dtratio is the ratio between the large stepsize and the small stepsize
=======
>>>>>>> Documentation adjustments to ex4
*/

static const char help[] = "1D periodic Finite Volume solver in slope-limiter form with semidiscrete time stepping.\n"
  "  advection   - Constant coefficient scalar advection\n"
  "                u_t       + (a*u)_x               = 0\n"
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
  "  shallow     - 1D Shallow water equations (Saint Venant System)\n"                                                      
=======
  "  shallow     - 1D Shallow water equations (Saint Venant System)\n"
>>>>>>> trivial edit: remove white spaces
  "                h_t + (q)_x = 0 \n"
<<<<<<< HEAD
<<<<<<< HEAD
  "                q_t + (\frac{q^2}{h} + g/2*h^2)_x = -hg*z_x \n"
  "                where, h(x,t) denotes the height of the water, q(x,t) the momentum, and z(x)\n"
  "                the topography of the river bottom. \n"
>>>>>>> Added Shallow water equations to the top menu. Detailed description of options still required.
=======
  "                q_t + (\frac{q^2}{h} + g/2*h^2)_x =  \n"
=======
  "                q_t + (\frac{q^2}{h} + g/2*h^2)_x = 0  \n"
>>>>>>> Added coupling flux to the junction structure along with dynamic allocation of user set coupling flux functions after network distribution. Added new network, initial 2 to test flexibility of the method
  "                where, h(x,t) denotes the height of the water and q(x,t) the momentum.\n"
>>>>>>> Added preliminary data structures and functions for fvnet. Completly untested at this point
  "  for this toy problem, we choose different meshsizes for different sub-domains (slow-fast-slow), say\n"
  "                hxs  = hratio*hxf \n"
  "  where hxs and hxf are the grid spacings for coarse and fine grids respectively.\n"
=======
static const char help[] = "1D periodic Finite Volume solver by a particular slope limiter with semidiscrete time stepping.\n"
  "  advection   - Constant coefficient scalar advection\n"
  "                u_t       + (a*u)_x               = 0\n"
  "  for this toy problem, we choose different meshsizes for different sub-domains, say\n"
  "                hxs  = (xmax - xmin)/2.0*(hratio+1.0)/Mx, \n"
  "                hxf  = (xmax - xmin)/2.0*(1.0+1.0/hratio)/Mx, \n"
  "  with x belongs to (xmin,xmax), the number of total mesh points is Mx and the ratio between the meshsize of corse\n\n"
  "  grids and fine grids is hratio.\n"
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
/*
  Note:
    -hratio is the ratio between mesh size of corse grids and fine grids
*/

static const char help[] = "1D periodic Finite Volume solver in slope-limiter form with semidiscrete time stepping.\n"
  "  advection   - Constant coefficient scalar advection\n"
  "                u_t       + (a*u)_x               = 0\n"
  "  shallow     - 1D Shallow water equations (Saint Venant System)\n"
  "                h_t + (q)_x = 0 \n"
  "                q_t + (\frac{q^2}{h} + g/2*h^2)_x = 0  \n"
  "                where, h(x,t) denotes the height of the water and q(x,t) the momentum.\n"
  "  for this toy problem, we choose different meshsizes for different sub-domains (slow-fast-slow), say\n"
  "                hxs  = hratio*hxf \n"
  "  where hxs and hxf are the grid spacings for coarse and fine grids respectively.\n"
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  "  exact       - Exact Riemann solver which usually needs to perform a Newton iteration to connect\n"
  "                the states across shocks and rarefactions\n"
  "  simulation  - use reference solution which is generated by smaller time step size to be true solution,\n"
  "                also the reference solution should be generated by user and stored in a binary file.\n"
  "  characteristic - Limit the characteristic variables, this is usually preferred (default)\n"
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Documentation adjustments to ex4
  "  bc_type     - Boundary condition for the problem, options are: periodic, outflow, inflow "
  "Several problem descriptions (initial data, physics specific features, boundary data) can be chosen with -initial N\n\n"
  "The problem size should be set with -da_grid_x M\n\n";

/*
  Example:
<<<<<<< HEAD
     ./ex4 -da_grid_x 40 -initial 1 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 7.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
     ./ex4 -da_grid_x 40 -initial 2 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 2.5 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
     ./ex4 -da_grid_x 40 -initial 3 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 4.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
     ./ex4 -da_grid_x 40 -initial 4 -hratio 1 -limit koren3 -ts_dt 0.01 -ts_max_time 4.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
     ./ex4 -da_grid_x 40 -initial 5 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 5.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0

  Contributed by: Aidan Hamilton <aidan@udel.edu>
*/
=======
  "Several initial conditions can be chosen with -initial N\n\n"
=======
  "Several problem descriptions (initial data, physics specific features, boundary data) can be chosen with -initial N\n\n"
>>>>>>> Added Shallow water equations to the top menu. Detailed description of options still required.
  "The problem size should be set with -da_grid_x M\n\n";
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.

/*
  Example: 
    mpiexec -np 1 ex4 -da_grid_x 40 -initial 1 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 7.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
    mpiexec -np 1 ex4 -da_grid_x 40 -initial 2 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 2.5 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
    mpiexec -np 1 ex4 -da_grid_x 40 -initial 3 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 4.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
    mpiexec -np 1 ex4 -da_grid_x 40 -initial 4 -hratio 1 -limit koren3 -ts_dt 0.01 -ts_max_time 4.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
    mpiexec -np 1 ex4 -da_grid_x 40 -initial 5 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 5.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
=======
    mpiexec -np 1 ./ex4 -da_grid_x 40 -initial 1 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 7.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
    mpiexec -np 1 ./ex4 -da_grid_x 40 -initial 2 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 2.5 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
    mpiexec -np 1 ./ex4 -da_grid_x 40 -initial 3 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 4.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
    mpiexec -np 1 ./ex4 -da_grid_x 40 -initial 4 -hratio 1 -limit koren3 -ts_dt 0.01 -ts_max_time 4.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
    mpiexec -np 1 ./ex4 -da_grid_x 40 -initial 5 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 5.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
<<<<<<< HEAD
>>>>>>> trivial edit: remove white spaces
=======

  Contributed by: Aidan Hamilton <aidan@udel.edu>
>>>>>>> add missing output/ex4_1.out; small edits
*/
=======
  "Several initial conditions can be chosen with -initial N\n\n"
<<<<<<< HEAD
  "The problem size should be set with -da_grid_x M\n\n"
  "This script choose the slope limiter by biased second-order upwind procedure which is proposed by Van Leer in 1994\n"
  "                             u(x_(k+1/2),t) = u(x_k,t) + phi(x_(k+1/2),t)*(u(x_k,t)-u(x_(k-1),t))                 \n"
  "                     limiter phi(x_(k+1/2),t) = max(0,min(r(k+1/2),min(2,gamma(k+1/2)*r(k+1/2)+alpha(k+1/2))))    \n"
  "                             r(k+1/2) = (u(x_(k+1))-u(x_k))/(u(x_k)-u(x_(k-1)))                                   \n"
  "                             alpha(k+1/2) = (h_k*h_(k+1))/(h_(k-1)+h_k)/(h_(k-1)+h_k+h_(k+1))                     \n"
  "                             gamma(k+1/2) = h_k*(h_(k-1)+h_k)/(h_k+h_(k+1))/(h_(k-1)+h_k+h_(k+1))                 \n";
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
=======
=======
  "  bc_type     - Boundary condition for the problem, options are: periodic, outflow, inflow "
>>>>>>> Documentation adjustments to ex4
  "Several problem descriptions (initial data, physics specific features, boundary data) can be chosen with -initial N\n\n"
>>>>>>> Added Shallow water equations to the top menu. Detailed description of options still required.
  "The problem size should be set with -da_grid_x M\n\n";
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.

/*
  Example:
    mpiexec -np 1 ./ex4 -da_grid_x 40 -initial 1 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 7.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
    mpiexec -np 1 ./ex4 -da_grid_x 40 -initial 2 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 2.5 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
    mpiexec -np 1 ./ex4 -da_grid_x 40 -initial 3 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 4.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
    mpiexec -np 1 ./ex4 -da_grid_x 40 -initial 4 -hratio 1 -limit koren3 -ts_dt 0.01 -ts_max_time 4.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
    mpiexec -np 1 ./ex4 -da_grid_x 40 -initial 5 -hratio 1 -limit mc -ts_dt 0.01 -ts_max_time 5.0 -ts_type mprk -ts_mprk_type 2a22 -ts_monitor_draw_solution -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0

  Contributed by: Aidan Hamilton <aidan@udel.edu>
*/

#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdraw.h>
<<<<<<< HEAD
<<<<<<< HEAD
#include "finitevolume1d.h"
<<<<<<< HEAD
#include <petsc/private/kernels/blockinvert.h>
=======

<<<<<<< HEAD
#include <petsc/private/kernels/blockinvert.h> 
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
#include <petsc/private/kernels/blockinvert.h>
>>>>>>> trivial edit: remove white spaces

PETSC_STATIC_INLINE PetscReal RangeMod(PetscReal a,PetscReal xmin,PetscReal xmax) { PetscReal range = xmax-xmin; return xmin +PetscFmodReal(range+PetscFmodReal(a,range),range); }
PETSC_STATIC_INLINE PetscReal MaxAbs(PetscReal a,PetscReal b) { return (PetscAbs(a) > PetscAbs(b)) ? a : b; }
/* --------------------------------- Advection ----------------------------------- */
typedef struct {
  PetscReal a;                  /* advective velocity */
} AdvectCtx;

static PetscErrorCode PhysicsRiemann_Advect(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  AdvectCtx *ctx = (AdvectCtx*)vctx;
  PetscReal speed;

  PetscFunctionBeginUser;
  speed     = ctx->a;
  flux[0]   = PetscMax(0,speed)*uL[0] + PetscMin(0,speed)*uR[0];
  *maxspeed = speed;
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCharacteristic_Advect(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi,PetscReal *speeds)
{
  AdvectCtx *ctx = (AdvectCtx*)vctx;

  PetscFunctionBeginUser;
  X[0]      = 1.;
  Xi[0]     = 1.;
  speeds[0] = ctx->a;
=======
#include <petscmath.h>

PETSC_STATIC_INLINE PetscReal RangeMod(PetscReal a,PetscReal xmin,PetscReal xmax) { PetscReal range = xmax-xmin; return xmin +PetscFmodReal(range+PetscFmodReal(a,range),range); }

/* --------------------------------- Finite Volume data structures ----------------------------------- */

typedef enum {FVBC_PERIODIC, FVBC_OUTFLOW} FVBCType;
static const char *FVBCTypes[] = {"PERIODIC","OUTFLOW","FVBCType","FVBC_",0};

typedef struct {
  PetscErrorCode (*sample)(void*,PetscInt,FVBCType,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*);
  PetscErrorCode (*flux)(void*,const PetscScalar*,PetscScalar*,PetscReal*);
  PetscErrorCode (*destroy)(void*);
  void           *user;
  PetscInt       dof;
  char           *fieldname[16];
} PhysicsCtx;

typedef struct {
  PhysicsCtx  physics;
  MPI_Comm    comm;
  char        prefix[256];

  /* Local work arrays */
  PetscScalar *flux;            /* Flux across interface                                                      */
  PetscReal   *speeds;          /* Speeds of each wave                                                        */
  PetscScalar *u;               /* value at face                                                              */

  PetscReal   cfl_idt;          /* Max allowable value of 1/Delta t                                           */
  PetscReal   cfl;
  PetscReal   xmin,xmax;
  PetscInt    initial;
  PetscBool   exact;
  PetscBool   simulation;
  FVBCType    bctype;
  PetscInt    hratio;           /* hratio = hslow/hfast */
  IS          isf,iss;
  PetscInt    sf,fs;            /* slow-fast and fast-slow interfaces */
} FVCtx;

/* --------------------------------- Physics ----------------------------------- */
static PetscErrorCode PhysicsDestroy_SimpleFree(void *vctx)
{
  PetscErrorCode ierr;
=======
#include "finitevolume1d.h"
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.

#include <petsc/private/kernels/blockinvert.h>

PETSC_STATIC_INLINE PetscReal RangeMod(PetscReal a,PetscReal xmin,PetscReal xmax) { PetscReal range = xmax-xmin; return xmin +PetscFmodReal(range+PetscFmodReal(a,range),range); }
PETSC_STATIC_INLINE PetscReal MaxAbs(PetscReal a,PetscReal b) { return (PetscAbs(a) > PetscAbs(b)) ? a : b; }
/* --------------------------------- Advection ----------------------------------- */
typedef struct {
  PetscReal a;                  /* advective velocity */
} AdvectCtx;

static PetscErrorCode PhysicsRiemann_Advect(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  AdvectCtx *ctx = (AdvectCtx*)vctx;
  PetscReal speed;

  PetscFunctionBeginUser;
  speed     = ctx->a;
  flux[0]   = PetscMax(0,speed)*uL[0] + PetscMin(0,speed)*uR[0];
  *maxspeed = speed;
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCharacteristic_Advect(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi,PetscReal *speeds)
{
  AdvectCtx *ctx = (AdvectCtx*)vctx;

  PetscFunctionBeginUser;
  X[0]      = 1.;
  Xi[0]     = 1.;
  speeds[0] = ctx->a;
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsSample_Advect(void *vctx,PetscInt initial,FVBCType bctype,PetscReal xmin,PetscReal xmax,PetscReal t,PetscReal x,PetscReal *u)
{
  AdvectCtx *ctx = (AdvectCtx*)vctx;
  PetscReal a    = ctx->a,x0;

  PetscFunctionBeginUser;
  switch (bctype) {
    case FVBC_OUTFLOW:   x0 = x-a*t; break;
    case FVBC_PERIODIC: x0 = RangeMod(x-a*t,xmin,xmax); break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown BCType");
  }
  switch (initial) {
    case 0: u[0] = (x0 < 0) ? 1 : -1; break;
    case 1: u[0] = (x0 < 0) ? -1 : 1; break;
    case 2: u[0] = (0 < x0 && x0 < 1) ? 1 : 0; break;
    case 3: u[0] = PetscSinReal(2*PETSC_PI*x0); break;
    case 4: u[0] = PetscAbs(x0); break;
    case 5: u[0] = (x0 < 0 || x0 > 0.5) ? 0 : PetscSqr(PetscSinReal(2*PETSC_PI*x0)); break;
    case 6: u[0] = (x0 < 0) ? 0 : ((x0 < 1) ? x0 : ((x0 < 2) ? 2-x0 : 0)); break;
    case 7: u[0] = PetscPowReal(PetscSinReal(PETSC_PI*x0),10.0);break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCreate_Advect(FVCtx *ctx)
{
  PetscErrorCode ierr;
  AdvectCtx      *user;

  PetscFunctionBeginUser;
  ierr = PetscNew(&user);CHKERRQ(ierr);
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  ctx->physics2.sample2         = PhysicsSample_Advect;
  ctx->physics2.riemann2        = PhysicsRiemann_Advect;
  ctx->physics2.characteristic2 = PhysicsCharacteristic_Advect;
  ctx->physics2.destroy         = PhysicsDestroy_SimpleFree;
  ctx->physics2.user            = user;
  ctx->physics2.dof             = 1;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
  ctx->physics2.issource        = PETSC_FALSE; 
=======
  ctx->physics2.issource        = PETSC_FALSE;
>>>>>>> trivial edit: remove white spaces
=======
>>>>>>> Small fixes to ex4

>>>>>>> Added Shallow water equations to the top menu. Detailed description of options still required.
=======
  ctx->physics2.issource        = PETSC_FALSE; 
=======
  ctx->physics2.issource        = PETSC_FALSE;
>>>>>>> trivial edit: remove white spaces
=======
>>>>>>> Small fixes to ex4

>>>>>>> Added Shallow water equations to the top menu. Detailed description of options still required.
  ierr = PetscStrallocpy("u",&ctx->physics2.fieldname[0]);CHKERRQ(ierr);
  user->a = 1;
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for advection","");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-physics_advect_a","Speed","",user->a,&user->a,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------------------- Shallow Water ----------------------------------- */

typedef struct {
  PetscReal gravity;
} ShallowCtx;

PETSC_STATIC_INLINE void ShallowFlux(ShallowCtx *phys,const PetscScalar *u,PetscScalar *f)
{
  f[0] = u[1];
  f[1] = PetscSqr(u[1])/u[0] + 0.5*phys->gravity*PetscSqr(u[0]);
}

static PetscErrorCode PhysicsRiemann_Shallow_Exact(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  ShallowCtx                *phys = (ShallowCtx*)vctx;
  PetscScalar               g    = phys->gravity,ustar[2],cL,cR,c,cstar;
  struct {PetscScalar h,u;} L = {uL[0],uL[1]/uL[0]},R = {uR[0],uR[1]/uR[0]},star;
  PetscInt                  i;

  PetscFunctionBeginUser;
  if (!(L.h > 0 && R.h > 0)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Reconstructed thickness is negative");
  cL = PetscSqrtScalar(g*L.h);
  cR = PetscSqrtScalar(g*R.h);
  c  = PetscMax(cL,cR);
  {
    /* Solve for star state */
    const PetscInt maxits = 50;
    PetscScalar tmp,res,res0=0,h0,h = 0.5*(L.h + R.h); /* initial guess */
    h0 = h;
    for (i=0; i<maxits; i++) {
      PetscScalar fr,fl,dfr,dfl;
      fl = (L.h < h)
        ? PetscSqrtScalar(0.5*g*(h*h - L.h*L.h)*(1/L.h - 1/h)) /* shock */
        : 2*PetscSqrtScalar(g*h) - 2*PetscSqrtScalar(g*L.h);   /* rarefaction */
      fr = (R.h < h)
        ? PetscSqrtScalar(0.5*g*(h*h - R.h*R.h)*(1/R.h - 1/h)) /* shock */
        : 2*PetscSqrtScalar(g*h) - 2*PetscSqrtScalar(g*R.h);   /* rarefaction */
      res = R.u - L.u + fr + fl;
      if (PetscIsInfOrNanScalar(res)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinity or Not-a-Number generated in computation");
      if (PetscAbsScalar(res) < PETSC_SQRT_MACHINE_EPSILON || (i > 0 && PetscAbsScalar(h-h0) < PETSC_SQRT_MACHINE_EPSILON)) {
        star.h = h;
        star.u = L.u - fl;
        goto converged;
      } else if (i > 0 && PetscAbsScalar(res) >= PetscAbsScalar(res0)) {        /* Line search */
        h = 0.8*h0 + 0.2*h;
        continue;
      }
      /* Accept the last step and take another */
      res0 = res;
      h0 = h;
      dfl = (L.h < h) ? 0.5/fl*0.5*g*(-L.h*L.h/(h*h) - 1 + 2*h/L.h) : PetscSqrtScalar(g/h);
      dfr = (R.h < h) ? 0.5/fr*0.5*g*(-R.h*R.h/(h*h) - 1 + 2*h/R.h) : PetscSqrtScalar(g/h);
      tmp = h - res/(dfr+dfl);
      if (tmp <= 0) h /= 2;   /* Guard against Newton shooting off to a negative thickness */
      else h = tmp;
      if (!((h > 0) && PetscIsNormalScalar(h))) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FP,"non-normal iterate h=%g",(double)h);
    }
    SETERRQ1(PETSC_COMM_SELF,1,"Newton iteration for star.h diverged after %D iterations",i);
  }
converged:
  cstar = PetscSqrtScalar(g*star.h);
  if (L.u-cL < 0 && 0 < star.u-cstar) { /* 1-wave is sonic rarefaction */
    PetscScalar ufan[2];
    ufan[0] = 1/g*PetscSqr(L.u/3 + 2./3*cL);
    ufan[1] = PetscSqrtScalar(g*ufan[0])*ufan[0];
    ShallowFlux(phys,ufan,flux);
  } else if (star.u+cstar < 0 && 0 < R.u+cR) { /* 2-wave is sonic rarefaction */
    PetscScalar ufan[2];
    ufan[0] = 1/g*PetscSqr(R.u/3 - 2./3*cR);
    ufan[1] = -PetscSqrtScalar(g*ufan[0])*ufan[0];
    ShallowFlux(phys,ufan,flux);
  } else if ((L.h >= star.h && L.u-c >= 0) || (L.h<star.h && (star.h*star.u-L.h*L.u)/(star.h-L.h) > 0)) {
    /* 1-wave is right-travelling shock (supersonic) */
    ShallowFlux(phys,uL,flux);
  } else if ((star.h <= R.h && R.u+c <= 0) || (star.h>R.h && (R.h*R.u-star.h*star.h)/(R.h-star.h) < 0)) {
    /* 2-wave is left-travelling shock (supersonic) */
    ShallowFlux(phys,uR,flux);
  } else {
    ustar[0] = star.h;
    ustar[1] = star.h*star.u;
    ShallowFlux(phys,ustar,flux);
  }
  *maxspeed = MaxAbs(MaxAbs(star.u-cstar,star.u+cstar),MaxAbs(L.u-cL,R.u+cR));
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsRiemann_Shallow_Rusanov(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  ShallowCtx                *phys = (ShallowCtx*)vctx;
  PetscScalar               g = phys->gravity,fL[2],fR[2],s;
  struct {PetscScalar h,u;} L = {uL[0],uL[1]/uL[0]},R = {uR[0],uR[1]/uR[0]};
  PetscReal                 tol = 1e-6;

  PetscFunctionBeginUser;
  /* Positivity preserving modification*/
<<<<<<< HEAD
<<<<<<< HEAD
  if (L.h < tol) L.u = 0.0;
  if (R.h < tol) R.u = 0.0;

  /*simple pos preserve limiter*/
  if (L.h < 0) L.h = 0;
  if (R.h < 0) R.h = 0;
=======
  if (L.h < tol) L.u = 0.0; 
  if (R.h < tol) R.u = 0.0; 

  /*simple pos preserve limiter*/
  if (L.h < 0) L.h=0; 
  if (R.h < 0) R.h=0; 
>>>>>>> Added Support for Inflow Boundary Conditions
=======
  if (L.h < tol) L.u = 0.0;
  if (R.h < tol) R.u = 0.0;

  /*simple pos preserve limiter*/
<<<<<<< HEAD
  if (L.h < 0) L.h=0;
  if (R.h < 0) R.h=0;
>>>>>>> trivial edit: remove white spaces
=======
  if (L.h < 0) L.h = 0;
  if (R.h < 0) R.h = 0;
>>>>>>> Small fixes to ex4

  ShallowFlux(phys,uL,fL);
  ShallowFlux(phys,uR,fR);

  s         = PetscMax(PetscAbs(L.u)+PetscSqrtScalar(g*L.h),PetscAbs(R.u)+PetscSqrtScalar(g*R.h));
  flux[0]   = 0.5*(fL[0] + fR[0]) + 0.5*s*(L.h - R.h);
  flux[1]   = 0.5*(fL[1] + fR[1]) + 0.5*s*(uL[1] - uR[1]);
  *maxspeed = s;
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCharacteristic_Conservative(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi,PetscReal *speeds)
{
  PetscInt i,j;

  PetscFunctionBeginUser;
  for (i=0; i<m; i++) {
    for (j=0; j<m; j++) Xi[i*m+j] = X[i*m+j] = (PetscScalar)(i==j);
    speeds[i] = PETSC_MAX_REAL; /* Indicates invalid */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCharacteristic_Shallow(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi,PetscReal *speeds)
{
  ShallowCtx     *phys = (ShallowCtx*)vctx;
  PetscReal      c;
  PetscErrorCode ierr;
  PetscReal      tol = 1e-6;

  PetscFunctionBeginUser;
  c           = PetscSqrtScalar(u[0]*phys->gravity);

  if (u[0] < tol) { /*Use conservative variables*/
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> trivial edit: remove white spaces
    X[0*2+0]  = 1;
    X[0*2+1]  = 0;
    X[1*2+0]  = 0;
    X[1*2+1]  = 1;
<<<<<<< HEAD
=======
  X[0*2+0]  = 1;
  X[0*2+1]  = 0;
  X[1*2+0]  = 0;
  X[1*2+1]  = 1;
>>>>>>> Added Support for Inflow Boundary Conditions
=======
>>>>>>> trivial edit: remove white spaces
  } else {
    speeds[0] = u[1]/u[0] - c;
    speeds[1] = u[1]/u[0] + c;
    X[0*2+0]  = 1;
    X[0*2+1]  = speeds[0];
    X[1*2+0]  = 1;
    X[1*2+1]  = speeds[1];
  }

  ierr = PetscArraycpy(Xi,X,4);CHKERRQ(ierr);
  ierr = PetscKernel_A_gets_inverse_A_2(Xi,0,PETSC_FALSE,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsSample_Shallow(void *vctx,PetscInt initial,FVBCType bctype,PetscReal xmin,PetscReal xmax,PetscReal t,PetscReal x,PetscReal *u)
{
  PetscFunctionBeginUser;
  if (t > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exact solutions not implemented for t > 0");
  switch (initial) {
    case 0:
      u[0] = (x < 0) ? 2 : 1; /* Standard Dam Break Problem */
      u[1] = (x < 0) ? 0 : 0;
      break;
    case 1:
      u[0] = (x < 10) ?   1 : 0.1; /*The Next 5 problems are standard Riemann problem tests */
      u[1] = (x < 10) ? 2.5 : 0;
      break;
    case 2:
      u[0] = (x < 25) ?  1 : 1;
      u[1] = (x < 25) ? -5 : 5;
      break;
    case 3:
      u[0] = (x < 20) ?  1 : 1e-12;
      u[1] = (x < 20) ?  0 : 0;
      break;
    case 4:
      u[0] = (x < 30) ? 1e-12 : 1;
      u[1] = (x < 30) ? 0 : 0;
      break;
    case 5:
      u[0] = (x < 25) ?  0.1 : 0.1;
      u[1] = (x < 25) ? -0.3 : 0.3;
      break;
    case 6:
      u[0] = 1+0.5*PetscSinReal(2*PETSC_PI*x);
      u[1] = 1*u[0];
      break;
    case 7:
      if (x < -0.1) {
       u[0] = 1e-9;
       u[1] = 0.0;
<<<<<<< HEAD
      } else if (x < 0.1) {
=======
      } else if(x < 0.1){
>>>>>>> trivial edit: remove white spaces
       u[0] = 1.0;
       u[1] = 0.0;
      } else {
       u[0] = 1e-9;
       u[1] = 0.0;
      }
      break;
    case 8:
     if (x < -0.1) {
       u[0] = 2;
       u[1] = 0.0;
<<<<<<< HEAD
      } else if (x < 0.1) {
=======
      } else if(x < 0.1){
>>>>>>> trivial edit: remove white spaces
       u[0] = 3.0;
       u[1] = 0.0;
      } else {
       u[0] = 2;
       u[1] = 0.0;
      }
      break;
<<<<<<< HEAD
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}

/* Implements inflow conditions for the given initial conditions. Which conditions are actually enforced depends on
   on the results of PhysicsSetInflowType_Shallow. */
static PetscErrorCode PhysicsInflow_Shallow(void *vctx,PetscReal t,PetscReal x,PetscReal *u)
{
  FVCtx          *ctx = (FVCtx*)vctx;

  PetscFunctionBeginUser;
  if (ctx->bctype == FVBC_INFLOW) {
    switch (ctx->initial) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
        u[0] = 0; u[1] = 0.0; /* Left boundary conditions */
        u[2] = 0; u[3] = 0.0; /* Right boundary conditions */
        break;
      case 6:
        u[0] = 0; u[1] = 0.0; /* Left boundary conditions */
        u[2] = 0; u[3] = 0.0; /* Right boundary conditions */
        break;
      case 7:
        u[0] = 0; u[1] = 0.0; /* Left boundary conditions */
        u[2] = 0; u[3] = 0.0; /* Right boundary conditions */
        break;
      case 8:
        u[0] = 0; u[1] = 1.0; /* Left boundary conditions */
        u[2] = 0; u[3] = -1.0;/* Right boundary conditions */
        break;
      default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
    }
  }
  PetscFunctionReturn(0);
}

/* Selects which boundary conditions are marked as inflow and which as outflow when FVBC_INFLOW is selected. */
static PetscErrorCode PhysicsSetInflowType_Shallow(FVCtx *ctx)
{
  PetscFunctionBeginUser;
  switch (ctx->initial) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8: /* Fix left and right momentum, height is outflow */
      ctx->physics2.bcinflowindex[0] = PETSC_FALSE;
      ctx->physics2.bcinflowindex[1] = PETSC_TRUE;
      ctx->physics2.bcinflowindex[2] = PETSC_FALSE;
      ctx->physics2.bcinflowindex[3] = PETSC_TRUE;
      break;
=======
>>>>>>> trivial edit: remove white spaces
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}

/* Implements inflow conditions for the given initial conditions. Which conditions are actually enforced depends on
   on the results of PhysicsSetInflowType_Shallow. */
static PetscErrorCode PhysicsInflow_Shallow(void *vctx,PetscReal t,PetscReal x,PetscReal *u)
{
  FVCtx          *ctx = (FVCtx*)vctx;

  PetscFunctionBeginUser;
  if (ctx->bctype == FVBC_INFLOW){
    switch (ctx->initial) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
        u[0] = 0; u[1] = 0.0; /* Left boundary conditions */
        u[2] = 0; u[3] = 0.0; /* Right boundary conditions */
        break;
      case 6:
        u[0] = 0; u[1] = 0.0; /* Left boundary conditions */
        u[2] = 0; u[3] = 0.0; /* Right boundary conditions */
        break;
      case 7:
        u[0] = 0; u[1] = 0.0; /* Left boundary conditions */
        u[2] = 0; u[3] = 0.0; /* Right boundary conditions */
        break;
      case 8:
        u[0] = 0; u[1] = 1.0; /* Left boundary conditions */
        u[2] = 0; u[3] = -1.0;/* Right boundary conditions */
        break;
      default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
    }
  }
  PetscFunctionReturn(0);
}

/* Selects which boundary conditions are marked as inflow and which as outflow when FVBC_INFLOW is selected. */
static PetscErrorCode PhysicsSetInflowType_Shallow(FVCtx *ctx)
{
  PetscFunctionBeginUser;
  switch (ctx->initial) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8: /* Fix left and right momentum, height is outflow */
      ctx->physics2.bcinflowindex[0] = PETSC_FALSE;
      ctx->physics2.bcinflowindex[1] = PETSC_TRUE;
      ctx->physics2.bcinflowindex[2] = PETSC_FALSE;
      ctx->physics2.bcinflowindex[3] = PETSC_TRUE;
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCreate_Shallow(FVCtx *ctx)
{
  PetscErrorCode    ierr;
  ShallowCtx        *user;
  PetscFunctionList rlist = 0,rclist = 0;
  char              rname[256] = "rusanov",rcname[256] = "characteristic";

  PetscFunctionBeginUser;
  ierr = PetscNew(&user);CHKERRQ(ierr);
  ctx->physics2.sample2         = PhysicsSample_Shallow;
<<<<<<< HEAD
<<<<<<< HEAD
  ctx->physics2.inflow          = PhysicsInflow_Shallow;
=======
  ctx->physics2.inflow          = PhysicsInflow_Shallow;  
>>>>>>> Added Support for Inflow Boundary Conditions
=======
  ctx->physics2.inflow          = PhysicsInflow_Shallow;
>>>>>>> trivial edit: remove white spaces
  ctx->physics2.destroy         = PhysicsDestroy_SimpleFree;
  ctx->physics2.riemann2        = PhysicsRiemann_Shallow_Rusanov;
  ctx->physics2.characteristic2 = PhysicsCharacteristic_Shallow;
  ctx->physics2.user            = user;
  ctx->physics2.dof             = 2;

  PetscMalloc1(2*(ctx->physics2.dof),&ctx->physics2.bcinflowindex);
<<<<<<< HEAD
<<<<<<< HEAD
  PhysicsSetInflowType_Shallow(ctx);

  ierr = PetscStrallocpy("height",&ctx->physics2.fieldname[0]);CHKERRQ(ierr);
=======
  PhysicsSetInflowType_Shallow(ctx); 
  PhysicsSetSource_Shallow(ctx); 
  
=======
  PhysicsSetInflowType_Shallow(ctx);

<<<<<<< HEAD
>>>>>>> trivial edit: remove white spaces
  ierr = PetscStrallocpy("density",&ctx->physics2.fieldname[0]);CHKERRQ(ierr);
>>>>>>> Added Support for Inflow Boundary Conditions
=======
  ierr = PetscStrallocpy("height",&ctx->physics2.fieldname[0]);CHKERRQ(ierr);
>>>>>>> Added -np 4 test for ex4
  ierr = PetscStrallocpy("momentum",&ctx->physics2.fieldname[1]);CHKERRQ(ierr);

  user->gravity = 9.81;

  ierr = RiemannListAdd_2WaySplit(&rlist,"exact",  PhysicsRiemann_Shallow_Exact);CHKERRQ(ierr);
  ierr = RiemannListAdd_2WaySplit(&rlist,"rusanov",PhysicsRiemann_Shallow_Rusanov);CHKERRQ(ierr);
  ierr = ReconstructListAdd_2WaySplit(&rclist,"characteristic",PhysicsCharacteristic_Shallow);CHKERRQ(ierr);
  ierr = ReconstructListAdd_2WaySplit(&rclist,"conservative",PhysicsCharacteristic_Conservative);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for Shallow","");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-physics_shallow_gravity","Gravity","",user->gravity,&user->gravity,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-physics_shallow_riemann","Riemann solver","",rlist,rname,rname,sizeof(rname),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-physics_shallow_reconstruct","Reconstruction","",rclist,rcname,rcname,sizeof(rcname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = RiemannListFind_2WaySplit(rlist,rname,&ctx->physics2.riemann2);CHKERRQ(ierr);
  ierr = ReconstructListFind_2WaySplit(rclist,rcname,&ctx->physics2.characteristic2);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&rlist);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&rclist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------------------- Shallow Water ----------------------------------- */

<<<<<<< HEAD
typedef struct {
  PetscReal gravity;
} ShallowCtx;

PETSC_STATIC_INLINE void ShallowFlux(ShallowCtx *phys,const PetscScalar *u,PetscScalar *f)
=======
PetscErrorCode FVSample_2WaySplit(FVCtx *ctx,DM da,PetscReal time,Vec U)
{
  PetscErrorCode  ierr;
  PetscScalar     *u,*uj,xj,xi;
  PetscInt        i,j,k,dof,xs,xm,Mx;
  const PetscInt  N = 200;
  PetscReal       hs,hf;

  PetscFunctionBeginUser;
  if (!ctx->physics2.sample2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Physics has not provided a sampling function");
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  ierr = PetscMalloc1(dof,&uj);CHKERRQ(ierr);
  hs   = (ctx->xmax-ctx->xmin)*3.0/8.0/ctx->sf;
  hf   = (ctx->xmax-ctx->xmin)/4.0/(ctx->fs-ctx->sf);
  for (i=xs; i<xs+xm; i++) {
    if (i < ctx->sf) {
      xi = ctx->xmin+0.5*hs+i*hs;
      /* Integrate over cell i using trapezoid rule with N points. */
      for (k=0; k<dof; k++) u[i*dof+k] = 0;
      for (j=0; j<N+1; j++) {
        xj = xi+hs*(j-N/2)/(PetscReal)N;
        ierr = (*ctx->physics2.sample2)(ctx->physics2.user,ctx->initial,ctx->bctype,ctx->xmin,ctx->xmax,time,xj,uj);CHKERRQ(ierr);
        for (k=0; k<dof; k++) u[i*dof+k] += ((j==0 || j==N) ? 0.5 : 1.0)*uj[k]/N;
      }
    } else if (i < ctx->fs) {
      xi = ctx->xmin+ctx->sf*hs+0.5*hf+(i-ctx->sf)*hf;
      /* Integrate over cell i using trapezoid rule with N points. */
      for (k=0; k<dof; k++) u[i*dof+k] = 0;
      for (j=0; j<N+1; j++) {
        xj = xi+hf*(j-N/2)/(PetscReal)N;
        ierr = (*ctx->physics2.sample2)(ctx->physics2.user,ctx->initial,ctx->bctype,ctx->xmin,ctx->xmax,time,xj,uj);CHKERRQ(ierr);
        for (k=0; k<dof; k++) u[i*dof+k] += ((j==0 || j==N) ? 0.5 : 1.0)*uj[k]/N;
      }
    } else {
      xi = ctx->xmin+ctx->sf*hs+(ctx->fs-ctx->sf)*hf+0.5*hs+(i-ctx->fs)*hs;
      /* Integrate over cell i using trapezoid rule with N points. */
      for (k=0; k<dof; k++) u[i*dof+k] = 0;
      for (j=0; j<N+1; j++) {
        xj = xi+hs*(j-N/2)/(PetscReal)N;
        ierr = (*ctx->physics2.sample2)(ctx->physics2.user,ctx->initial,ctx->bctype,ctx->xmin,ctx->xmax,time,xj,uj);CHKERRQ(ierr);
        for (k=0; k<dof; k++) u[i*dof+k] += ((j==0 || j==N) ? 0.5 : 1.0)*uj[k]/N;
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  ierr = PetscFree(uj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SolutionErrorNorms_2WaySplit(FVCtx *ctx,DM da,PetscReal t,Vec X,PetscReal *nrm1)
>>>>>>> Added Support for Inflow Boundary Conditions
{
  f[0] = u[1];
  f[1] = PetscSqr(u[1])/u[0] + 0.5*phys->gravity*PetscSqr(u[0]);
}

static PetscErrorCode PhysicsRiemann_Shallow_Exact(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  ShallowCtx                *phys = (ShallowCtx*)vctx;
  PetscScalar               g    = phys->gravity,ustar[2],cL,cR,c,cstar;
  struct {PetscScalar h,u;} L = {uL[0],uL[1]/uL[0]},R = {uR[0],uR[1]/uR[0]},star;
  PetscInt                  i;

  PetscFunctionBeginUser;
<<<<<<< HEAD
<<<<<<< HEAD
  if (!(L.h > 0 && R.h > 0)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Reconstructed thickness is negative");
  cL = PetscSqrtScalar(g*L.h);
  cR = PetscSqrtScalar(g*R.h);
  c  = PetscMax(cL,cR);
  {
    /* Solve for star state */
    const PetscInt maxits = 50;
    PetscScalar tmp,res,res0=0,h0,h = 0.5*(L.h + R.h); /* initial guess */
    h0 = h;
    for (i=0; i<maxits; i++) {
      PetscScalar fr,fl,dfr,dfl;
      fl = (L.h < h)
        ? PetscSqrtScalar(0.5*g*(h*h - L.h*L.h)*(1/L.h - 1/h)) /* shock */
        : 2*PetscSqrtScalar(g*h) - 2*PetscSqrtScalar(g*L.h);   /* rarefaction */
      fr = (R.h < h)
        ? PetscSqrtScalar(0.5*g*(h*h - R.h*R.h)*(1/R.h - 1/h)) /* shock */
        : 2*PetscSqrtScalar(g*h) - 2*PetscSqrtScalar(g*R.h);   /* rarefaction */
      res = R.u - L.u + fr + fl;
      if (PetscIsInfOrNanScalar(res)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinity or Not-a-Number generated in computation");
      if (PetscAbsScalar(res) < 1e-8 || (i > 0 && PetscAbsScalar(h-h0) < 1e-8)) {
        star.h = h;
        star.u = L.u - fl;
        goto converged;
      } else if (i > 0 && PetscAbsScalar(res) >= PetscAbsScalar(res0)) {        /* Line search */
        h = 0.8*h0 + 0.2*h;
        continue;
      }
      /* Accept the last step and take another */
      res0 = res;
      h0 = h;
      dfl = (L.h < h) ? 0.5/fl*0.5*g*(-L.h*L.h/(h*h) - 1 + 2*h/L.h) : PetscSqrtScalar(g/h);
      dfr = (R.h < h) ? 0.5/fr*0.5*g*(-R.h*R.h/(h*h) - 1 + 2*h/R.h) : PetscSqrtScalar(g/h);
      tmp = h - res/(dfr+dfl);
      if (tmp <= 0) h /= 2;   /* Guard against Newton shooting off to a negative thickness */
      else h = tmp;
      if (!((h > 0) && PetscIsNormalScalar(h))) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FP,"non-normal iterate h=%g",(double)h);
    }
    SETERRQ1(PETSC_COMM_SELF,1,"Newton iteration for star.h diverged after %D iterations",i);
  }
converged:
  cstar = PetscSqrtScalar(g*star.h);
  if (L.u-cL < 0 && 0 < star.u-cstar) { /* 1-wave is sonic rarefaction */
    PetscScalar ufan[2];
    ufan[0] = 1/g*PetscSqr(L.u/3 + 2./3*cL);
    ufan[1] = PetscSqrtScalar(g*ufan[0])*ufan[0];
    ShallowFlux(phys,ufan,flux);
  } else if (star.u+cstar < 0 && 0 < R.u+cR) { /* 2-wave is sonic rarefaction */
    PetscScalar ufan[2];
    ufan[0] = 1/g*PetscSqr(R.u/3 - 2./3*cR);
    ufan[1] = -PetscSqrtScalar(g*ufan[0])*ufan[0];
    ShallowFlux(phys,ufan,flux);
  } else if ((L.h >= star.h && L.u-c >= 0) || (L.h<star.h && (star.h*star.u-L.h*L.u)/(star.h-L.h) > 0)) {
    /* 1-wave is right-travelling shock (supersonic) */
    ShallowFlux(phys,uL,flux);
  } else if ((star.h <= R.h && R.u+c <= 0) || (star.h>R.h && (R.h*R.u-star.h*star.h)/(R.h-star.h) < 0)) {
    /* 2-wave is left-travelling shock (supersonic) */
    ShallowFlux(phys,uR,flux);
  } else {
    ustar[0] = star.h;
    ustar[1] = star.h*star.u;
    ShallowFlux(phys,ustar,flux);
  }
  *maxspeed = MaxAbs(MaxAbs(star.u-cstar,star.u+cstar),MaxAbs(L.u-cL,R.u+cR));
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsRiemann_Shallow_Rusanov(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  ShallowCtx                *phys = (ShallowCtx*)vctx;
  PetscScalar               g = phys->gravity,fL[2],fR[2],s;
  struct {PetscScalar h,u;} L = {uL[0],uL[1]/uL[0]},R = {uR[0],uR[1]/uR[0]};
  PetscReal                 tol = 1e-6;

  PetscFunctionBeginUser;
  /* Positivity preserving modification*/
  if (L.h < tol) L.u = 0.0;
  if (R.h < tol) R.u = 0.0;

  /*simple pos preserve limiter*/
  if (L.h < 0) L.h = 0;
  if (R.h < 0) R.h = 0;

  ShallowFlux(phys,uL,fL);
  ShallowFlux(phys,uR,fR);

  s         = PetscMax(PetscAbs(L.u)+PetscSqrtScalar(g*L.h),PetscAbs(R.u)+PetscSqrtScalar(g*R.h));
  flux[0]   = 0.5*(fL[0] + fR[0]) + 0.5*s*(L.h - R.h);
  flux[1]   = 0.5*(fL[1] + fR[1]) + 0.5*s*(uL[1] - uR[1]);
  *maxspeed = s;
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCharacteristic_Conservative(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi,PetscReal *speeds)
{
  PetscInt i,j;

  PetscFunctionBeginUser;
  for (i=0; i<m; i++) {
    for (j=0; j<m; j++) Xi[i*m+j] = X[i*m+j] = (PetscScalar)(i==j);
    speeds[i] = PETSC_MAX_REAL; /* Indicates invalid */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCharacteristic_Shallow(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi,PetscReal *speeds)
{
  ShallowCtx     *phys = (ShallowCtx*)vctx;
  PetscReal      c;
  PetscErrorCode ierr;
  PetscReal      tol = 1e-6;

  PetscFunctionBeginUser;
  c           = PetscSqrtScalar(u[0]*phys->gravity);

  if (u[0] < tol) { /*Use conservative variables*/
    X[0*2+0]  = 1;
    X[0*2+1]  = 0;
    X[1*2+0]  = 0;
    X[1*2+1]  = 1;
  } else {
    speeds[0] = u[1]/u[0] - c;
    speeds[1] = u[1]/u[0] + c;
    X[0*2+0]  = 1;
    X[0*2+1]  = speeds[0];
    X[1*2+0]  = 1;
    X[1*2+1]  = speeds[1];
  }

  ierr = PetscArraycpy(Xi,X,4);CHKERRQ(ierr);
  ierr = PetscKernel_A_gets_inverse_A_2(Xi,0,PETSC_FALSE,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsSample_Shallow(void *vctx,PetscInt initial,FVBCType bctype,PetscReal xmin,PetscReal xmax,PetscReal t,PetscReal x,PetscReal *u)
{
  PetscFunctionBeginUser;
  if (t > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exact solutions not implemented for t > 0");
  switch (initial) {
    case 0:
      u[0] = (x < 0) ? 2 : 1; /* Standard Dam Break Problem */
      u[1] = (x < 0) ? 0 : 0;
      break;
    case 1:
      u[0] = (x < 10) ?   1 : 0.1; /*The Next 5 problems are standard Riemann problem tests */
      u[1] = (x < 10) ? 2.5 : 0;
      break;
    case 2:
      u[0] = (x < 25) ?  1 : 1;
      u[1] = (x < 25) ? -5 : 5;
      break;
    case 3:
      u[0] = (x < 20) ?  1 : 1e-12;
      u[1] = (x < 20) ?  0 : 0;
      break;
    case 4:
      u[0] = (x < 30) ? 1e-12 : 1;
      u[1] = (x < 30) ? 0 : 0;
      break;
    case 5:
      u[0] = (x < 25) ?  0.1 : 0.1;
      u[1] = (x < 25) ? -0.3 : 0.3;
      break;
    case 6:
      u[0] = 1+0.5*PetscSinReal(2*PETSC_PI*x);
      u[1] = 1*u[0];
      break;
    case 7:
      if (x < -0.1) {
       u[0] = 1e-9;
       u[1] = 0.0;
      } else if(x < 0.1){
       u[0] = 1.0;
       u[1] = 0.0;
      } else {
       u[0] = 1e-9;
       u[1] = 0.0;
      }
      break;
    case 8:
     if (x < -0.1) {
       u[0] = 2;
       u[1] = 0.0;
      } else if(x < 0.1){
       u[0] = 3.0;
       u[1] = 0.0;
      } else {
       u[0] = 2;
       u[1] = 0.0;
      }
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}

/* Implements inflow conditions for the given initial conditions. Which conditions are actually enforced depends on
   on the results of PhysicsSetInflowType_Shallow. */
static PetscErrorCode PhysicsInflow_Shallow(void *vctx,PetscReal t,PetscReal x,PetscReal *u)
{
  FVCtx          *ctx = (FVCtx*)vctx;

  PetscFunctionBeginUser;
  if (ctx->bctype == FVBC_INFLOW){
    switch (ctx->initial) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
        u[0] = 0; u[1] = 0.0; /* Left boundary conditions */
        u[2] = 0; u[3] = 0.0; /* Right boundary conditions */
        break;
      case 6:
        u[0] = 0; u[1] = 0.0; /* Left boundary conditions */
        u[2] = 0; u[3] = 0.0; /* Right boundary conditions */
        break;
      case 7:
        u[0] = 0; u[1] = 0.0; /* Left boundary conditions */
        u[2] = 0; u[3] = 0.0; /* Right boundary conditions */
        break;
      case 8:
        u[0] = 0; u[1] = 1.0; /* Left boundary conditions */
        u[2] = 0; u[3] = -1.0;/* Right boundary conditions */
        break;
      default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
    }
  }
  PetscFunctionReturn(0);
}

/* Selects which boundary conditions are marked as inflow and which as outflow when FVBC_INFLOW is selected. */
static PetscErrorCode PhysicsSetInflowType_Shallow(FVCtx *ctx)
{
  PetscFunctionBeginUser;
  switch (ctx->initial) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6: 
    case 7: 
    case 8: /* Fix left and right momentum, height is outflow */
      ctx->physics2.bcinflowindex[0] = PETSC_FALSE;
      ctx->physics2.bcinflowindex[1] = PETSC_TRUE;
      ctx->physics2.bcinflowindex[2] = PETSC_FALSE;
      ctx->physics2.bcinflowindex[3] = PETSC_TRUE;
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCreate_Shallow(FVCtx *ctx)
{
  PetscErrorCode    ierr;
  ShallowCtx        *user;
  PetscFunctionList rlist = 0,rclist = 0;
  char              rname[256] = "rusanov",rcname[256] = "characteristic";

  PetscFunctionBeginUser;
  ierr = PetscNew(&user);CHKERRQ(ierr);
  ctx->physics2.sample2         = PhysicsSample_Shallow;
  ctx->physics2.inflow          = PhysicsInflow_Shallow;
  ctx->physics2.destroy         = PhysicsDestroy_SimpleFree;
  ctx->physics2.riemann2        = PhysicsRiemann_Shallow_Rusanov;
  ctx->physics2.characteristic2 = PhysicsCharacteristic_Shallow;
  ctx->physics2.user            = user;
  ctx->physics2.dof             = 2;

  PetscMalloc1(2*(ctx->physics2.dof),&ctx->physics2.bcinflowindex);
  PhysicsSetInflowType_Shallow(ctx);

<<<<<<< HEAD
  ierr = PetscStrallocpy("density",&ctx->physics2.fieldname[0]);CHKERRQ(ierr);
=======
  ierr = PetscStrallocpy("height",&ctx->physics2.fieldname[0]);CHKERRQ(ierr);
>>>>>>> Added -np 4 test for ex4
  ierr = PetscStrallocpy("momentum",&ctx->physics2.fieldname[1]);CHKERRQ(ierr);

  user->gravity = 9.81;

  ierr = RiemannListAdd_2WaySplit(&rlist,"exact",  PhysicsRiemann_Shallow_Exact);CHKERRQ(ierr);
  ierr = RiemannListAdd_2WaySplit(&rlist,"rusanov",PhysicsRiemann_Shallow_Rusanov);CHKERRQ(ierr);
  ierr = ReconstructListAdd_2WaySplit(&rclist,"characteristic",PhysicsCharacteristic_Shallow);CHKERRQ(ierr);
  ierr = ReconstructListAdd_2WaySplit(&rclist,"conservative",PhysicsCharacteristic_Conservative);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for Shallow","");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-physics_shallow_gravity","Gravity","",user->gravity,&user->gravity,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-physics_shallow_riemann","Riemann solver","",rlist,rname,rname,sizeof(rname),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-physics_shallow_reconstruct","Reconstruction","",rclist,rcname,rcname,sizeof(rcname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = RiemannListFind_2WaySplit(rlist,rname,&ctx->physics2.riemann2);CHKERRQ(ierr);
  ierr = ReconstructListFind_2WaySplit(rclist,rcname,&ctx->physics2.characteristic2);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&rlist);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&rclist);CHKERRQ(ierr);
<<<<<<< HEAD
  PetscFunctionReturn(0);
}

PetscErrorCode FVSample_2WaySplit(FVCtx *ctx,DM da,PetscReal time,Vec U)
{
  PetscErrorCode  ierr;
  PetscScalar     *u,*uj,xj,xi;
  PetscInt        i,j,k,dof,xs,xm,Mx;
  const PetscInt  N = 200;
  PetscReal       hs,hf;

  PetscFunctionBeginUser;
  if (!ctx->physics2.sample2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Physics has not provided a sampling function");
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  ierr = PetscMalloc1(dof,&uj);CHKERRQ(ierr);
  hs   = (ctx->xmax-ctx->xmin)*3.0/8.0/ctx->sf;
  hf   = (ctx->xmax-ctx->xmin)/4.0/(ctx->fs-ctx->sf);
  for (i=xs; i<xs+xm; i++) {
    if (i < ctx->sf) {
      xi = ctx->xmin+0.5*hs+i*hs;
      /* Integrate over cell i using trapezoid rule with N points. */
      for (k=0; k<dof; k++) u[i*dof+k] = 0;
      for (j=0; j<N+1; j++) {
        xj = xi+hs*(j-N/2)/(PetscReal)N;
        ierr = (*ctx->physics2.sample2)(ctx->physics2.user,ctx->initial,ctx->bctype,ctx->xmin,ctx->xmax,time,xj,uj);CHKERRQ(ierr);
        for (k=0; k<dof; k++) u[i*dof+k] += ((j==0 || j==N) ? 0.5 : 1.0)*uj[k]/N;
      }
    } else if (i < ctx->fs) {
      xi = ctx->xmin+ctx->sf*hs+0.5*hf+(i-ctx->sf)*hf;
      /* Integrate over cell i using trapezoid rule with N points. */
      for (k=0; k<dof; k++) u[i*dof+k] = 0;
      for (j=0; j<N+1; j++) {
        xj = xi+hf*(j-N/2)/(PetscReal)N;
        ierr = (*ctx->physics2.sample2)(ctx->physics2.user,ctx->initial,ctx->bctype,ctx->xmin,ctx->xmax,time,xj,uj);CHKERRQ(ierr);
        for (k=0; k<dof; k++) u[i*dof+k] += ((j==0 || j==N) ? 0.5 : 1.0)*uj[k]/N;
      }
    } else {
      xi = ctx->xmin+ctx->sf*hs+(ctx->fs-ctx->sf)*hf+0.5*hs+(i-ctx->fs)*hs;
      /* Integrate over cell i using trapezoid rule with N points. */
      for (k=0; k<dof; k++) u[i*dof+k] = 0;
      for (j=0; j<N+1; j++) {
        xj = xi+hs*(j-N/2)/(PetscReal)N;
        ierr = (*ctx->physics2.sample2)(ctx->physics2.user,ctx->initial,ctx->bctype,ctx->xmin,ctx->xmax,time,xj,uj);CHKERRQ(ierr);
        for (k=0; k<dof; k++) u[i*dof+k] += ((j==0 || j==N) ? 0.5 : 1.0)*uj[k]/N;
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  ierr = PetscFree(uj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

<<<<<<< HEAD
=======
  PetscFunctionReturn(0); 
}

>>>>>>> Added -np 4 test for ex4
PetscErrorCode FVSample_2WaySplit(FVCtx *ctx,DM da,PetscReal time,Vec U)
{
  PetscErrorCode  ierr;
  PetscScalar     *u,*uj,xj,xi;
  PetscInt        i,j,k,dof,xs,xm,Mx;
  const PetscInt  N = 200;
  PetscReal       hs,hf;

  PetscFunctionBeginUser;
  if (!ctx->physics2.sample2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Physics has not provided a sampling function");
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  ierr = PetscMalloc1(dof,&uj);CHKERRQ(ierr);
  hs   = (ctx->xmax-ctx->xmin)*3.0/8.0/ctx->sf;
  hf   = (ctx->xmax-ctx->xmin)/4.0/(ctx->fs-ctx->sf);
  for (i=xs; i<xs+xm; i++) {
    if (i < ctx->sf) {
      xi = ctx->xmin+0.5*hs+i*hs;
      /* Integrate over cell i using trapezoid rule with N points. */
      for (k=0; k<dof; k++) u[i*dof+k] = 0;
      for (j=0; j<N+1; j++) {
        xj = xi+hs*(j-N/2)/(PetscReal)N;
        ierr = (*ctx->physics2.sample2)(ctx->physics2.user,ctx->initial,ctx->bctype,ctx->xmin,ctx->xmax,time,xj,uj);CHKERRQ(ierr);
        for (k=0; k<dof; k++) u[i*dof+k] += ((j==0 || j==N) ? 0.5 : 1.0)*uj[k]/N;
      }
    } else if (i < ctx->fs) {
      xi = ctx->xmin+ctx->sf*hs+0.5*hf+(i-ctx->sf)*hf;
      /* Integrate over cell i using trapezoid rule with N points. */
      for (k=0; k<dof; k++) u[i*dof+k] = 0;
      for (j=0; j<N+1; j++) {
        xj = xi+hf*(j-N/2)/(PetscReal)N;
        ierr = (*ctx->physics2.sample2)(ctx->physics2.user,ctx->initial,ctx->bctype,ctx->xmin,ctx->xmax,time,xj,uj);CHKERRQ(ierr);
        for (k=0; k<dof; k++) u[i*dof+k] += ((j==0 || j==N) ? 0.5 : 1.0)*uj[k]/N;
      }
    } else {
      xi = ctx->xmin+ctx->sf*hs+(ctx->fs-ctx->sf)*hf+0.5*hs+(i-ctx->fs)*hs;
      /* Integrate over cell i using trapezoid rule with N points. */
      for (k=0; k<dof; k++) u[i*dof+k] = 0;
      for (j=0; j<N+1; j++) {
        xj = xi+hs*(j-N/2)/(PetscReal)N;
        ierr = (*ctx->physics2.sample2)(ctx->physics2.user,ctx->initial,ctx->bctype,ctx->xmin,ctx->xmax,time,xj,uj);CHKERRQ(ierr);
        for (k=0; k<dof; k++) u[i*dof+k] += ((j==0 || j==N) ? 0.5 : 1.0)*uj[k]/N;
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  ierr = PetscFree(uj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SolutionErrorNorms_2WaySplit(FVCtx *ctx,DM da,PetscReal t,Vec X,PetscReal *nrm1)
{
  PetscErrorCode    ierr;
  Vec               Y;
  PetscInt          i,Mx;
  const PetscScalar *ptr_X,*ptr_Y;
  PetscReal         hs,hf;

  PetscFunctionBeginUser;
=======
static PetscErrorCode SolutionErrorNorms_2WaySplit(FVCtx *ctx,DM da,PetscReal t,Vec X,PetscReal *nrm1)
{
  PetscErrorCode    ierr;
  Vec               Y;
  PetscInt          i,Mx;
  const PetscScalar *ptr_X,*ptr_Y;
  PetscReal         hs,hf;

  PetscFunctionBeginUser;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  ierr = VecGetSize(X,&Mx);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);
  ierr = FVSample_2WaySplit(ctx,da,t,Y);CHKERRQ(ierr);
  hs   = (ctx->xmax-ctx->xmin)*3.0/8.0/ctx->sf;
  hf   = (ctx->xmax-ctx->xmin)/4.0/(ctx->fs-ctx->sf);
  ierr = VecGetArrayRead(X,&ptr_X);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Y,&ptr_Y);CHKERRQ(ierr);
  for (i=0; i<Mx; i++) {
    if (i < ctx->sf || i > ctx->fs -1) *nrm1 +=  hs*PetscAbs(ptr_X[i]-ptr_Y[i]);
    else *nrm1 += hf*PetscAbs(ptr_X[i]-ptr_Y[i]);
  }
  ierr = VecRestoreArrayRead(X,&ptr_X);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Y,&ptr_Y);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FVRHSFunction_2WaySplit(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx          *ctx = (FVCtx*)vctx;
  PetscErrorCode ierr;
  PetscInt       i,j,k,Mx,dof,xs,xm,sf = ctx->sf,fs = ctx->fs;
  PetscReal      hxf,hxs,cfl_idt = 0;
  PetscScalar    *x,*f,*slope;
  Vec            Xloc;
  DM             da;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);                          /* Xloc contains ghost points                                     */
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);   /* Mx is the number of center points                              */
  hxs  = (ctx->xmax-ctx->xmin)*3.0/8.0/ctx->sf;
  hxf  = (ctx->xmax-ctx->xmin)/4.0/(ctx->fs-ctx->sf);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);       /* X is solution vector which does not contain ghost points       */
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);

  ierr = VecZeroEntries(F);CHKERRQ(ierr);                                   /* F is the right hand side function corresponds to center points */

  ierr = DMDAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAGetArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);                  /* contains ghost points                                           */
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
>>>>>>> Added Support for Inflow Boundary Conditions
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  if (ctx->bctype == FVBC_OUTFLOW) {
    for (i=xs-2; i<0; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[j];
    }
    for (i=Mx; i<xs+xm+2; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[(xs+xm-1)*dof+j];
    }
  }

  if (ctx->bctype == FVBC_INFLOW) {
    /* See LeVeque, R. (2002). Finite Volume Methods for Hyperbolic Problems. doi:10.1017/CBO9780511791253
    pages 137-138 for the scheme. */
    if(xs==0){ /* Left Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmin,ctx->ub);
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[j]){
          for(i=-2; i<0; i++) x[i*dof+j] = 2.0*ctx->ub[j]-x[-(i+1)*dof+j];
        } else {
          for(i=-2; i<0; i++) x[i*dof+j] = x[j]; /* Outflow */
        }
      }
    }
    if(xs+xm==Mx){ /* Right Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmax,ctx->ub);
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[dof+j]){
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = 2.0*ctx->ub[dof+j]-x[(2*Mx-(i+1))*dof+j];
        } else {
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = x[(Mx-1)*dof+j]; /* Outflow */
        }
      }
    }
  }

  for (i=xs-1; i<xs+xm+1; i++) {
    struct _LimitInfo info;
    PetscScalar       *cjmpL,*cjmpR;
    /* Determine the right eigenvectors R, where A = R \Lambda R^{-1} */
    ierr = (*ctx->physics2.characteristic2)(ctx->physics2.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds);CHKERRQ(ierr);
    /* Evaluate jumps across interfaces (i-1, i) and (i, i+1), put in characteristic basis */
    ierr  = PetscArrayzero(ctx->cjmpLR,2*dof);CHKERRQ(ierr);
    cjmpL = &ctx->cjmpLR[0];
    cjmpR = &ctx->cjmpLR[dof];
    for (j=0; j<dof; j++) {
      PetscScalar jmpL,jmpR;
      jmpL = x[(i+0)*dof+j]-x[(i-1)*dof+j];
      jmpR = x[(i+1)*dof+j]-x[(i+0)*dof+j];
      for (k=0; k<dof; k++) {
        cjmpL[k] += ctx->Rinv[k+j*dof]*jmpL;
        cjmpR[k] += ctx->Rinv[k+j*dof]*jmpR;
      }
    }
    /* Apply limiter to the left and right characteristic jumps */
    info.m  = dof;
    info.hxs = hxs;
    info.hxf = hxf;
    (*ctx->limit2)(&info,cjmpL,cjmpR,ctx->sf,ctx->fs,i,ctx->cslope);
    for (j=0; j<dof; j++) {
      PetscScalar tmp = 0;
      for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof]*ctx->cslope[k];
      slope[i*dof+j] = tmp;
    }
  }

<<<<<<< HEAD
  if (ctx->bctype == FVBC_INFLOW) {
    /* See LeVeque, R. (2002). Finite Volume Methods for Hyperbolic Problems. doi:10.1017/CBO9780511791253
    pages 137-138 for the scheme. */
    if (xs==0) { /* Left Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmin,ctx->ub);
      for (j=0; j<dof; j++) {
        if (ctx->physics2.bcinflowindex[j]) {
          for (i=-2; i<0; i++) x[i*dof+j] = 2.0*ctx->ub[j]-x[-(i+1)*dof+j];
        } else {
          for (i=-2; i<0; i++) x[i*dof+j] = x[j]; /* Outflow */
        }
      }
    }
    if (xs+xm==Mx) { /* Right Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmax,ctx->ub);
      for (j=0; j<dof; j++) {
        if (ctx->physics2.bcinflowindex[dof+j]) {
          for (i=Mx; i<Mx+2; i++) x[i*dof+j] = 2.0*ctx->ub[dof+j]-x[(2*Mx-(i+1))*dof+j];
        } else {
          for (i=Mx; i<Mx+2; i++) x[i*dof+j] = x[(Mx-1)*dof+j]; /* Outflow */
        }
      }
    }
  }

  for (i=xs-1; i<xs+xm+1; i++) {
    struct _LimitInfo info;
    PetscScalar       *cjmpL,*cjmpR;
    /* Determine the right eigenvectors R, where A = R \Lambda R^{-1} */
    ierr = (*ctx->physics2.characteristic2)(ctx->physics2.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds);CHKERRQ(ierr);
    /* Evaluate jumps across interfaces (i-1, i) and (i, i+1), put in characteristic basis */
    ierr  = PetscArrayzero(ctx->cjmpLR,2*dof);CHKERRQ(ierr);
    cjmpL = &ctx->cjmpLR[0];
    cjmpR = &ctx->cjmpLR[dof];
    for (j=0; j<dof; j++) {
      PetscScalar jmpL,jmpR;
      jmpL = x[(i+0)*dof+j]-x[(i-1)*dof+j];
      jmpR = x[(i+1)*dof+j]-x[(i+0)*dof+j];
      for (k=0; k<dof; k++) {
        cjmpL[k] += ctx->Rinv[k+j*dof]*jmpL;
        cjmpR[k] += ctx->Rinv[k+j*dof]*jmpR;
      }
    }
    /* Apply limiter to the left and right characteristic jumps */
    info.m  = dof;
    info.hxs = hxs;
    info.hxf = hxf;
    (*ctx->limit2)(&info,cjmpL,cjmpR,ctx->sf,ctx->fs,i,ctx->cslope);
    for (j=0; j<dof; j++) {
      PetscScalar tmp = 0;
      for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof]*ctx->cslope[k];
      slope[i*dof+j] = tmp;
=======
  ctx->physics.sample         = PhysicsSample_Advect;
  ctx->physics.flux           = PhysicsFlux_Advect;
  ctx->physics.destroy        = PhysicsDestroy_SimpleFree;
  ctx->physics.user           = user;
  ctx->physics.dof            = 1;
  ierr = PetscStrallocpy("u",&ctx->physics.fieldname[0]);CHKERRQ(ierr);
=======
  ierr = PetscStrallocpy("u",&ctx->physics2.fieldname[0]);CHKERRQ(ierr);
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  user->a = 1;
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for advection","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-physics_advect_a","Speed","",user->a,&user->a,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------------------- Shallow Water ----------------------------------- */

typedef struct {
  PetscReal gravity;
} ShallowCtx;

PETSC_STATIC_INLINE void ShallowFlux(ShallowCtx *phys,const PetscScalar *u,PetscScalar *f)
{
  f[0] = u[1];
  f[1] = PetscSqr(u[1])/u[0] + 0.5*phys->gravity*PetscSqr(u[0]);
}

static PetscErrorCode PhysicsRiemann_Shallow_Exact(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  ShallowCtx                *phys = (ShallowCtx*)vctx;
  PetscScalar               g    = phys->gravity,ustar[2],cL,cR,c,cstar;
  struct {PetscScalar h,u;} L = {uL[0],uL[1]/uL[0]},R = {uR[0],uR[1]/uR[0]},star;
  PetscInt                  i;

  PetscFunctionBeginUser;
  if (!(L.h > 0 && R.h > 0)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Reconstructed thickness is negative");
  cL = PetscSqrtScalar(g*L.h);
  cR = PetscSqrtScalar(g*R.h);
  c  = PetscMax(cL,cR);
  {
    /* Solve for star state */
    const PetscInt maxits = 50;
    PetscScalar tmp,res,res0=0,h0,h = 0.5*(L.h + R.h); /* initial guess */
    h0 = h;
    for (i=0; i<maxits; i++) {
      PetscScalar fr,fl,dfr,dfl;
      fl = (L.h < h)
        ? PetscSqrtScalar(0.5*g*(h*h - L.h*L.h)*(1/L.h - 1/h)) /* shock */
        : 2*PetscSqrtScalar(g*h) - 2*PetscSqrtScalar(g*L.h);   /* rarefaction */
      fr = (R.h < h)
        ? PetscSqrtScalar(0.5*g*(h*h - R.h*R.h)*(1/R.h - 1/h)) /* shock */
        : 2*PetscSqrtScalar(g*h) - 2*PetscSqrtScalar(g*R.h);   /* rarefaction */
      res = R.u - L.u + fr + fl;
      if (PetscIsInfOrNanScalar(res)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinity or Not-a-Number generated in computation");
      if (PetscAbsScalar(res) < 1e-8 || (i > 0 && PetscAbsScalar(h-h0) < 1e-8)) {
        star.h = h;
        star.u = L.u - fl;
        goto converged;
      } else if (i > 0 && PetscAbsScalar(res) >= PetscAbsScalar(res0)) {        /* Line search */
        h = 0.8*h0 + 0.2*h;
        continue;
      }
      /* Accept the last step and take another */
      res0 = res;
      h0 = h;
      dfl = (L.h < h) ? 0.5/fl*0.5*g*(-L.h*L.h/(h*h) - 1 + 2*h/L.h) : PetscSqrtScalar(g/h);
      dfr = (R.h < h) ? 0.5/fr*0.5*g*(-R.h*R.h/(h*h) - 1 + 2*h/R.h) : PetscSqrtScalar(g/h);
      tmp = h - res/(dfr+dfl);
      if (tmp <= 0) h /= 2;   /* Guard against Newton shooting off to a negative thickness */
      else h = tmp;
      if (!((h > 0) && PetscIsNormalScalar(h))) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FP,"non-normal iterate h=%g",(double)h);
    }
    SETERRQ1(PETSC_COMM_SELF,1,"Newton iteration for star.h diverged after %D iterations",i);
  }
converged:
  cstar = PetscSqrtScalar(g*star.h);
  if (L.u-cL < 0 && 0 < star.u-cstar) { /* 1-wave is sonic rarefaction */
    PetscScalar ufan[2];
    ufan[0] = 1/g*PetscSqr(L.u/3 + 2./3*cL);
    ufan[1] = PetscSqrtScalar(g*ufan[0])*ufan[0];
    ShallowFlux(phys,ufan,flux);
  } else if (star.u+cstar < 0 && 0 < R.u+cR) { /* 2-wave is sonic rarefaction */
    PetscScalar ufan[2];
    ufan[0] = 1/g*PetscSqr(R.u/3 - 2./3*cR);
    ufan[1] = -PetscSqrtScalar(g*ufan[0])*ufan[0];
    ShallowFlux(phys,ufan,flux);
  } else if ((L.h >= star.h && L.u-c >= 0) || (L.h<star.h && (star.h*star.u-L.h*L.u)/(star.h-L.h) > 0)) {
    /* 1-wave is right-travelling shock (supersonic) */
    ShallowFlux(phys,uL,flux);
  } else if ((star.h <= R.h && R.u+c <= 0) || (star.h>R.h && (R.h*R.u-star.h*star.h)/(R.h-star.h) < 0)) {
    /* 2-wave is left-travelling shock (supersonic) */
    ShallowFlux(phys,uR,flux);
  } else {
    ustar[0] = star.h;
    ustar[1] = star.h*star.u;
    ShallowFlux(phys,ustar,flux);
  }
  *maxspeed = MaxAbs(MaxAbs(star.u-cstar,star.u+cstar),MaxAbs(L.u-cL,R.u+cR));
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsRiemann_Shallow_Rusanov(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  ShallowCtx                *phys = (ShallowCtx*)vctx;
  PetscScalar               g = phys->gravity,fL[2],fR[2],s;
  struct {PetscScalar h,u;} L = {uL[0],uL[1]/uL[0]},R = {uR[0],uR[1]/uR[0]};
  PetscReal                 tol = 1e-6; 

  PetscFunctionBeginUser;
  /* Positivity preserving modification*/
  if (L.h < tol) L.u = tol; 
  if (R.h < tol) R.u = tol; 

  /*simple pos preserv limiter */ 
  if (L.h < 0) L.h=0; 
  if (R.h < 0) R.h=0; 

  ShallowFlux(phys,uL,fL);
  ShallowFlux(phys,uR,fR);
  s         = PetscMax(PetscAbs(L.u)+PetscSqrtScalar(g*L.h),PetscAbs(R.u)+PetscSqrtScalar(g*R.h));
  flux[0]   = 0.5*(fL[0] + fR[0]) + 0.5*s*(L.h - R.h);
  flux[1]   = 0.5*(fL[1] + fR[1]) + 0.5*s*(uL[1] - uR[1]);
  *maxspeed = s;
  PetscFunctionReturn(0);
}
static PetscErrorCode PhysicsCharacteristic_Conservative(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi,PetscReal *speeds)
{
  PetscInt i,j;

  PetscFunctionBeginUser;
  for (i=0; i<m; i++) {
    for (j=0; j<m; j++) Xi[i*m+j] = X[i*m+j] = (PetscScalar)(i==j);
    speeds[i] = PETSC_MAX_REAL; /* Indicates invalid */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCharacteristic_Shallow(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi,PetscReal *speeds)
{
  ShallowCtx     *phys = (ShallowCtx*)vctx;
  PetscReal      c;
  PetscErrorCode ierr;
  PetscReal      tol = 1e-6; 
  PetscFunctionBeginUser;
  c         = PetscSqrtScalar(u[0]*phys->gravity);

  if (u[0] < tol) {
  X[0*2+0]  = 1;
  X[0*2+1]  = 0;
  X[1*2+0]  = 0;
  X[1*2+1]  = 1;
  } else {
  speeds[0] = u[1]/u[0] - c;
  speeds[1] = u[1]/u[0] + c;
  X[0*2+0]  = 1;
  X[0*2+1]  = speeds[0];
  X[1*2+0]  = 1;
  X[1*2+1]  = speeds[1];
  }
  
  ierr = PetscArraycpy(Xi,X,4);CHKERRQ(ierr);
  ierr = PetscKernel_A_gets_inverse_A_2(Xi,0,PETSC_FALSE,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsSample_Shallow(void *vctx,PetscInt initial,FVBCType bctype,PetscReal xmin,PetscReal xmax,PetscReal t,PetscReal x,PetscReal *u)
{
  PetscFunctionBeginUser;
  if (t > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exact solutions not implemented for t > 0");
  switch (initial) {
    case 0:
      u[0] = (x < 0) ? 2 : 1; /* Standard Dam Break Problem */
      u[1] = (x < 0) ? 0 : 0;
      break;
    case 1:
      u[0] = (x < 10) ?   1 : 0.1; 
      u[1] = (x < 10) ? 2.5 : 0;
      break;
    case 2:
      u[0] = (x < 25) ?  1 : 1;
      u[1] = (x < 25) ? -5 : 5;
      break;
    case 3:
      u[0] = (x < 20) ?  1 : 1e-12;
      u[1] = (x < 20) ?  0 : 0;
      break;
    case 4:
      u[0] = (x < 30) ? 1e-12 : 1;
      u[1] = (x < 30) ? 0 : 0;
      break;
    case 5:
      u[0] = (x < 25) ?  0.1 : 0.1;
      u[1] = (x < 25) ? -0.3 : 0.3;
      break;
    case 6:
      u[0] = 1+0.5*PetscSinReal(2*PETSC_PI*x); 
      u[1] = 1*u[0];
      break;
    case 7: 
      if (x < -0.1) {
       u[0] = 1e-9; 
       u[1] = 0.0; 
      } else if(x < 0.1){
       u[0] = 1.0; 
       u[1] = 0.0; 
      } else {
       u[0] = 1e-9; 
       u[1] = 0.0; 
      }
      break; 
    case 8:
     if (x < -0.1) {
       u[0] = 2; 
       u[1] = 0.0; 
      } else if(x < 0.1){
       u[0] = 3.0; 
       u[1] = 0.0; 
      } else {
       u[0] = 2; 
       u[1] = 0.0; 
      }
      break; 
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCreate_Shallow(FVCtx *ctx)
{
  PetscErrorCode    ierr;
  ShallowCtx        *user;
  PetscFunctionList rlist = 0,rclist = 0;
  char              rname[256] = "rusanov",rcname[256] = "characteristic";

  PetscFunctionBeginUser;
  ierr = PetscNew(&user);CHKERRQ(ierr);
  ctx->physics2.sample2         = PhysicsSample_Shallow;
  ctx->physics2.destroy         = PhysicsDestroy_SimpleFree;
  ctx->physics2.riemann2        = PhysicsRiemann_Shallow_Rusanov; 
  ctx->physics2.characteristic2 = PhysicsCharacteristic_Shallow; 
  ctx->physics2.user            = user;
  ctx->physics2.dof             = 2;
  
  ierr = PetscStrallocpy("density",&ctx->physics2.fieldname[0]);CHKERRQ(ierr);
  ierr = PetscStrallocpy("momentum",&ctx->physics2.fieldname[1]);CHKERRQ(ierr);

  user->gravity = 9.81;

  ierr = RiemannListAdd_2WaySplit(&rlist,"exact",  PhysicsRiemann_Shallow_Exact);CHKERRQ(ierr);
  ierr = RiemannListAdd_2WaySplit(&rlist,"rusanov",PhysicsRiemann_Shallow_Rusanov);CHKERRQ(ierr);
  ierr = ReconstructListAdd_2WaySplit(&rclist,"characteristic",PhysicsCharacteristic_Shallow);CHKERRQ(ierr);
  ierr = ReconstructListAdd_2WaySplit(&rclist,"conservative",PhysicsCharacteristic_Conservative);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for Shallow","");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-physics_shallow_gravity","Gravity","",user->gravity,&user->gravity,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-physics_shallow_riemann","Riemann solver","",rlist,rname,rname,sizeof(rname),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-physics_shallow_reconstruct","Reconstruction","",rclist,rcname,rcname,sizeof(rcname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = RiemannListFind_2WaySplit(rlist,rname,&ctx->physics2.riemann2);CHKERRQ(ierr);
  ierr = ReconstructListFind_2WaySplit(rclist,rcname,&ctx->physics2.characteristic2);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&rlist);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&rclist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode FVSample_2WaySplit(FVCtx *ctx,DM da,PetscReal time,Vec U)
{
  PetscErrorCode  ierr;
  PetscScalar     *u,*uj,xj,xi;
  PetscInt        i,j,k,dof,xs,xm,Mx;
  const PetscInt  N = 200;
  PetscReal       hs,hf;

  PetscFunctionBeginUser;
  if (!ctx->physics2.sample2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Physics has not provided a sampling function");
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  ierr = PetscMalloc1(dof,&uj);CHKERRQ(ierr);
  hs   = (ctx->xmax-ctx->xmin)*3.0/8.0/ctx->sf;
  hf   = (ctx->xmax-ctx->xmin)/4.0/(ctx->fs-ctx->sf);
  for (i=xs; i<xs+xm; i++) {
    if (i < ctx->sf) {
      xi = ctx->xmin+0.5*hs+i*hs;
      /* Integrate over cell i using trapezoid rule with N points. */
      for (k=0; k<dof; k++) u[i*dof+k] = 0;
      for (j=0; j<N+1; j++) {
        xj = xi+hs*(j-N/2)/(PetscReal)N;
        ierr = (*ctx->physics2.sample2)(ctx->physics2.user,ctx->initial,ctx->bctype,ctx->xmin,ctx->xmax,time,xj,uj);CHKERRQ(ierr);
        for (k=0; k<dof; k++) u[i*dof+k] += ((j==0 || j==N) ? 0.5 : 1.0)*uj[k]/N;
      }
    } else if (i < ctx->fs) {
      xi = ctx->xmin+ctx->sf*hs+0.5*hf+(i-ctx->sf)*hf;
      /* Integrate over cell i using trapezoid rule with N points. */
      for (k=0; k<dof; k++) u[i*dof+k] = 0;
      for (j=0; j<N+1; j++) {
        xj = xi+hf*(j-N/2)/(PetscReal)N;
        ierr = (*ctx->physics2.sample2)(ctx->physics2.user,ctx->initial,ctx->bctype,ctx->xmin,ctx->xmax,time,xj,uj);CHKERRQ(ierr);
        for (k=0; k<dof; k++) u[i*dof+k] += ((j==0 || j==N) ? 0.5 : 1.0)*uj[k]/N;
      }
    } else {
      xi = ctx->xmin+ctx->sf*hs+(ctx->fs-ctx->sf)*hf+0.5*hs+(i-ctx->fs)*hs;
      /* Integrate over cell i using trapezoid rule with N points. */
      for (k=0; k<dof; k++) u[i*dof+k] = 0;
      for (j=0; j<N+1; j++) {
        xj = xi+hs*(j-N/2)/(PetscReal)N;
        ierr = (*ctx->physics2.sample2)(ctx->physics2.user,ctx->initial,ctx->bctype,ctx->xmin,ctx->xmax,time,xj,uj);CHKERRQ(ierr);
        for (k=0; k<dof; k++) u[i*dof+k] += ((j==0 || j==N) ? 0.5 : 1.0)*uj[k]/N;
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  ierr = PetscFree(uj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SolutionErrorNorms_2WaySplit(FVCtx *ctx,DM da,PetscReal t,Vec X,PetscReal *nrm1)
{
  PetscErrorCode    ierr;
  Vec               Y;
  PetscInt          i,Mx;
  const PetscScalar *ptr_X,*ptr_Y;
  PetscReal         hs,hf;

  PetscFunctionBeginUser;
  ierr = VecGetSize(X,&Mx);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);
  ierr = FVSample_2WaySplit(ctx,da,t,Y);CHKERRQ(ierr);
  hs   = (ctx->xmax-ctx->xmin)*3.0/8.0/ctx->sf;
  hf   = (ctx->xmax-ctx->xmin)/4.0/(ctx->fs-ctx->sf);
  ierr = VecGetArrayRead(X,&ptr_X);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Y,&ptr_Y);CHKERRQ(ierr);
  for (i=0; i<Mx; i++) {
    if (i < ctx->sf || i > ctx->fs -1) *nrm1 +=  hs*PetscAbs(ptr_X[i]-ptr_Y[i]);
    else *nrm1 += hf*PetscAbs(ptr_X[i]-ptr_Y[i]);
  }
  ierr = VecRestoreArrayRead(X,&ptr_X);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Y,&ptr_Y);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FVRHSFunction_2WaySplit(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx          *ctx = (FVCtx*)vctx;
  PetscErrorCode ierr;
  PetscInt       i,j,k,Mx,dof,xs,xm,sf = ctx->sf,fs = ctx->fs;
  PetscReal      hxf,hxs,cfl_idt = 0;
  PetscScalar    *x,*f,*slope;
  Vec            Xloc;
  DM             da;

  PetscFunctionBeginUser;
=======
>>>>>>> Added Support for Inflow Boundary Conditions
=======
>>>>>>> trivial edit: remove white spaces
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);                          /* Xloc contains ghost points                                     */
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);   /* Mx is the number of center points                              */
  hxs  = (ctx->xmax-ctx->xmin)*3.0/8.0/ctx->sf;
  hxf  = (ctx->xmax-ctx->xmin)/4.0/(ctx->fs-ctx->sf);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);       /* X is solution vector which does not contain ghost points       */
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);

  ierr = VecZeroEntries(F);CHKERRQ(ierr);                                   /* F is the right hand side function corresponds to center points */

  ierr = DMDAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAGetArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);                  /* contains ghost points                                           */
<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> trivial edit: remove white spaces
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  if (ctx->bctype == FVBC_OUTFLOW) {
    for (i=xs-2; i<0; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[j];
    }
    for (i=Mx; i<xs+xm+2; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[(xs+xm-1)*dof+j];
<<<<<<< HEAD
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
    }
  }
  for (i=xs-1; i<xs+xm+1; i++) {
    struct _LimitInfo info;
    PetscScalar       *cjmpL,*cjmpR;
    /* Determine the right eigenvectors R, where A = R \Lambda R^{-1} */
    ierr = (*ctx->physics2.characteristic2)(ctx->physics2.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds);CHKERRQ(ierr);
    /* Evaluate jumps across interfaces (i-1, i) and (i, i+1), put in characteristic basis */
    ierr  = PetscArrayzero(ctx->cjmpLR,2*dof);CHKERRQ(ierr);
    cjmpL = &ctx->cjmpLR[0];
    cjmpR = &ctx->cjmpLR[dof];
    for (j=0; j<dof; j++) {
      PetscScalar jmpL,jmpR;
      jmpL = x[(i+0)*dof+j]-x[(i-1)*dof+j];
      jmpR = x[(i+1)*dof+j]-x[(i+0)*dof+j];
      for (k=0; k<dof; k++) {
        cjmpL[k] += ctx->Rinv[k+j*dof]*jmpL;
        cjmpR[k] += ctx->Rinv[k+j*dof]*jmpR;
      }
    }
    /* Apply limiter to the left and right characteristic jumps */
    info.m  = dof;
    info.hxs = hxs;
    info.hxf = hxf;
    (*ctx->limit2)(&info,cjmpL,cjmpR,ctx->sf,ctx->fs,i,ctx->cslope);
    for (j=0; j<dof; j++) {
      PetscScalar tmp = 0;
      for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof]*ctx->cslope[k];
      slope[i*dof+j] = tmp;
    }
  }

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
    PetscScalar *uL,*uR;
    uL = &ctx->uLR[0];
    uR = &ctx->uLR[dof];
    if (i < sf) { /* slow region */
<<<<<<< HEAD
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
=======
  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    uL = &ctx->uLR[0];
    uR = &ctx->uLR[dof];
    if (i < sf) { /* slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxs;
      }
    } else if (i == sf) { /* interface between the slow region and the fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxf/2;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
<<<<<<< HEAD
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxs;
      }
    } else if (i == sf) { /* interface between the slow region and the fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
=======
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxf;
      }
    } else if (i > sf && i < fs) { /* fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxf/2;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxf/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
<<<<<<< HEAD
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxs;
=======
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxf;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxf;
      }
<<<<<<< HEAD
    } else if (i > sf && i < fs) { /* fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxf/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxf/2;
=======
    } else if (i == fs) { /* interface between the fast region and the slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxf/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxf;
      }
      if (i < xs+xm) {
<<<<<<< HEAD
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxf;
      }
    } else if (i == fs) { /* interface between the fast region and the slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxf/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxf;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxs;
      }
    } else { /* slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hxs)); /* Max allowable value of 1/Delta t */
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxs;
=======
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxs;
      }
    } else { /* slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hxs)); /* Max allowable value of 1/Delta t */
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxs;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
    PetscScalar *u;
    if (i < sf || i > fs+1) {
      u = &ctx->u[0];
      alpha[0] = 1.0/6.0;
      gamma[0] = 1.0/3.0;
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxs;
      }
    } else if (i == sf) { /* interface between the slow region and the fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxf/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxf;
      }
    } else if (i > sf && i < fs) { /* fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxf/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxf/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxf;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxf;
      }
    } else if (i == fs) { /* interface between the fast region and the slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxf/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxf;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxs;
      }
    } else { /* slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hxs)); /* Max allowable value of 1/Delta t */
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
<<<<<<< HEAD
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hs;
      }
    } else if (i == fs+1) {
      u = &ctx->u[0];
      alpha[0] = hs*hs/(hf+hs)/(hf+hs+hs);
      gamma[0] = hs*(hf+hs)/(hs+hs)/(hf+hs+hs);
      for (j=0; j<dof; j++) {
        r[j] = (x[i*dof+j] - x[(i-1)*dof+j])/(x[(i-1)*dof+j]-x[(i-2)*dof+j]);
        min[j] = PetscMin(r[j],2.0);
        u[j] = x[(i-1)*dof+j]+PetscMax(0,PetscMin(min[j],alpha[0]+gamma[0]*r[j]))*(x[(i-1)*dof+j]-x[(i-2)*dof+j]);
      }
      ierr = (*ctx->physics.flux)(ctx->physics.user,u,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hs;
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hxs;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
<<<<<<< HEAD
<<<<<<< HEAD
  ierr = DMDARestoreArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&cfl_idt,&ctx->cfl_idt,1,MPIU_SCALAR,MPIU_MAX,PetscObjectComm((PetscObject)da));CHKERRMPI(ierr);
=======
=======
  ierr = DMDARestoreArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&cfl_idt,&ctx->cfl_idt,1,MPIU_SCALAR,MPIU_MAX,PetscObjectComm((PetscObject)da));CHKERRQ(ierr);
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
  if (0) {
    /* We need to a way to inform the TS of a CFL constraint, this is a debugging fragment */
    PetscReal dt,tnow;
    ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tnow);CHKERRQ(ierr);
    if (dt > 0.5/ctx->cfl_idt) {
      if (1) {
        ierr = PetscPrintf(ctx->comm,"Stability constraint exceeded at t=%g, dt %g > %g\n",(double)tnow,(double)dt,(double)(0.5/ctx->cfl_idt));CHKERRQ(ierr);
      } else SETERRQ2(PETSC_COMM_SELF,1,"Stability constraint exceeded, %g > %g",(double)dt,(double)(ctx->cfl/ctx->cfl_idt));
    }
  }
<<<<<<< HEAD
<<<<<<< HEAD
  PetscFunctionReturn(0);
}

/* --------------------------------- Finite Volume Solver for slow components ----------------------------------- */
PetscErrorCode FVRHSFunctionslow_2WaySplit(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx          *ctx = (FVCtx*)vctx;
  PetscErrorCode ierr;
  PetscInt       i,j,k,Mx,dof,xs,xm,islow = 0,sf = ctx->sf,fs = ctx->fs,lsbwidth = ctx->lsbwidth,rsbwidth = ctx->rsbwidth;
  PetscReal      hxs,hxf,cfl_idt = 0;
  PetscScalar    *x,*f,*slope;
  Vec            Xloc;
  DM             da;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  hxs  = (ctx->xmax-ctx->xmin)*3.0/8.0/ctx->sf;
  hxf  = (ctx->xmax-ctx->xmin)/4.0/(ctx->fs-ctx->sf);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = DMDAGetArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
=======
  ierr = PetscFree4(r,min,alpha,gamma);CHKERRQ(ierr);
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  PetscFunctionReturn(0);
}

/* --------------------------------- Finite Volume Solver for slow components ----------------------------------- */
PetscErrorCode FVRHSFunctionslow_2WaySplit(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx          *ctx = (FVCtx*)vctx;
  PetscErrorCode ierr;
  PetscInt       i,j,k,Mx,dof,xs,xm,islow = 0,sf = ctx->sf,fs = ctx->fs,lsbwidth = ctx->lsbwidth,rsbwidth = ctx->rsbwidth;
  PetscReal      hxs,hxf,cfl_idt = 0;
  PetscScalar    *x,*f,*slope;
  Vec            Xloc;
  DM             da;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  hxs  = (ctx->xmax-ctx->xmin)*3.0/8.0/ctx->sf;
  hxf  = (ctx->xmax-ctx->xmin)/4.0/(ctx->fs-ctx->sf);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = DMDAGetArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
<<<<<<< HEAD
  ierr = PetscMalloc4(dof,&r,dof,&min,dof,&alpha,dof,&gamma);CHKERRQ(ierr);
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
>>>>>>> Added Support for Inflow Boundary Conditions

  if (ctx->bctype == FVBC_OUTFLOW) {
    for (i=xs-2; i<0; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[j];
    }
    for (i=Mx; i<xs+xm+2; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[(xs+xm-1)*dof+j];
    }
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  if (ctx->bctype == FVBC_INFLOW) {
    /* See LeVeque, R. (2002). Finite Volume Methods for Hyperbolic Problems. doi:10.1017/CBO9780511791253
    pages 137-138 for the scheme. */
    if (xs==0) { /* Left Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmin,ctx->ub);
      for (j=0; j<dof; j++) {
        if (ctx->physics2.bcinflowindex[j]==PETSC_TRUE) {
          for (i=-2; i<0; i++) x[i*dof+j] = 2.0*ctx->ub[j]-x[-(i+1)*dof+j];
        } else {
          for (i=-2; i<0; i++) x[i*dof+j] = x[j]; /* Outflow */
        }
      }
    }
    if (xs+xm==Mx) { /* Right Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmax,ctx->ub);
      for (j=0; j<dof; j++) {
        if (ctx->physics2.bcinflowindex[dof+j]==PETSC_TRUE) {
          for (i=Mx; i<Mx+2; i++) x[i*dof+j] = 2.0*ctx->ub[dof+j]-x[(2*Mx-(i+1))*dof+j];
        } else {
          for (i=Mx; i<Mx+2; i++) x[i*dof+j] = x[(Mx-1)*dof+j]; /* Outflow */
=======
=======
    }
  }

>>>>>>> trivial edit: remove white spaces
  if (ctx->bctype == FVBC_INFLOW) {
    /* See LeVeque, R. (2002). Finite Volume Methods for Hyperbolic Problems. doi:10.1017/CBO9780511791253
    pages 137-138 for the scheme. */
    if(xs==0){ /* Left Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmin,ctx->ub);
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[j]==PETSC_TRUE){
          for(i=-2; i<0; i++) x[i*dof+j] = 2.0*ctx->ub[j]-x[-(i+1)*dof+j];
<<<<<<< HEAD
=======

  if (ctx->bctype == FVBC_INFLOW) {
    /* See LeVeque, R. (2002). Finite Volume Methods for Hyperbolic Problems. doi:10.1017/CBO9780511791253 
    pages 137-138 for the scheme. */
    if(xs==0){ /* Left Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmin,ctx->ub); 
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[j]==PETSC_TRUE){
          for(i=-2; i<0; i++) x[i*dof+j] = 2.0*ctx->ub[j]-x[-(i+1)*dof+j]; 
>>>>>>> Added Support for Inflow Boundary Conditions
=======
>>>>>>> trivial edit: remove white spaces
        } else {
          for(i=-2; i<0; i++) x[i*dof+j] = x[j]; /* Outflow */
        }
      }
    }
    if(xs+xm==Mx){ /* Right Boundary */
<<<<<<< HEAD
<<<<<<< HEAD
      ctx->physics2.inflow(ctx,time,ctx->xmax,ctx->ub);
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[dof+j]==PETSC_TRUE){
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = 2.0*ctx->ub[dof+j]-x[(2*Mx-(i+1))*dof+j];
        } else {
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = x[(Mx-1)*dof+j]; /* Outflow */
>>>>>>> Added Support for Inflow Boundary Conditions
=======
      ctx->physics2.inflow(ctx,time,ctx->xmax,ctx->ub); 
=======
      ctx->physics2.inflow(ctx,time,ctx->xmax,ctx->ub);
>>>>>>> trivial edit: remove white spaces
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[dof+j]==PETSC_TRUE){
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = 2.0*ctx->ub[dof+j]-x[(2*Mx-(i+1))*dof+j];
        } else {
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = x[(Mx-1)*dof+j]; /* Outflow */
>>>>>>> Added Support for Inflow Boundary Conditions
        }
      }
    }
  }
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
>>>>>>> Added Support for Inflow Boundary Conditions
=======

>>>>>>> Added Support for Inflow Boundary Conditions
  for (i=xs-1; i<xs+xm+1; i++) {
    struct _LimitInfo info;
    PetscScalar       *cjmpL,*cjmpR;
    if (i < sf-lsbwidth+1 || i > fs+rsbwidth-2) { /* slow components and the first and last fast components */
      /* Determine the right eigenvectors R, where A = R \Lambda R^{-1} */
      ierr = (*ctx->physics2.characteristic2)(ctx->physics2.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds);CHKERRQ(ierr);
      /* Evaluate jumps across interfaces (i-1, i) and (i, i+1), put in characteristic basis */
      ierr  = PetscArrayzero(ctx->cjmpLR,2*dof);CHKERRQ(ierr);
      cjmpL = &ctx->cjmpLR[0];
      cjmpR = &ctx->cjmpLR[dof];
      for (j=0; j<dof; j++) {
        PetscScalar jmpL,jmpR;
        jmpL = x[(i+0)*dof+j]-x[(i-1)*dof+j];
        jmpR = x[(i+1)*dof+j]-x[(i+0)*dof+j];
        for (k=0; k<dof; k++) {
          cjmpL[k] += ctx->Rinv[k+j*dof]*jmpL;
          cjmpR[k] += ctx->Rinv[k+j*dof]*jmpR;
        }
      }
      /* Apply limiter to the left and right characteristic jumps */
      info.m  = dof;
      info.hxs = hxs;
      info.hxf = hxf;
      (*ctx->limit2)(&info,cjmpL,cjmpR,ctx->sf,ctx->fs,i,ctx->cslope);
      for (j=0; j<dof; j++) {
        PetscScalar tmp = 0;
        for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof]*ctx->cslope[k];
          slope[i*dof+j] = tmp;
      }
    }
  }

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    uL = &ctx->uLR[0];
    uR = &ctx->uLR[dof];
    if (i < sf-lsbwidth) { /* slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hxs)); /* Max allowable value of 1/Delta t */
      if (i > xs) {
        for (j=0; j<dof; j++) f[(islow-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hxs;
        islow++;
      }
    }
    if (i == sf-lsbwidth) { /* interface between the slow region and the fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(islow-1)*dof+j] -= ctx->flux[j]/hxs;
      }
    }
<<<<<<< HEAD
    if (i == fs+rsbwidth) { /* slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hxs;
        islow++;
      }
    }
    if (i > fs+rsbwidth) { /* slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(islow-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hxs;
        islow++;
=======
=======
  }
  for (i=xs-1; i<xs+xm+1; i++) {
    struct _LimitInfo info;
    PetscScalar       *cjmpL,*cjmpR;
    if (i < sf-lsbwidth+1 || i > fs+rsbwidth-2) { /* slow components and the first and last fast components */
      /* Determine the right eigenvectors R, where A = R \Lambda R^{-1} */
      ierr = (*ctx->physics2.characteristic2)(ctx->physics2.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds);CHKERRQ(ierr);
      /* Evaluate jumps across interfaces (i-1, i) and (i, i+1), put in characteristic basis */
      ierr  = PetscArrayzero(ctx->cjmpLR,2*dof);CHKERRQ(ierr);
      cjmpL = &ctx->cjmpLR[0];
      cjmpR = &ctx->cjmpLR[dof];
      for (j=0; j<dof; j++) {
        PetscScalar jmpL,jmpR;
        jmpL = x[(i+0)*dof+j]-x[(i-1)*dof+j];
        jmpR = x[(i+1)*dof+j]-x[(i+0)*dof+j];
        for (k=0; k<dof; k++) {
          cjmpL[k] += ctx->Rinv[k+j*dof]*jmpL;
          cjmpR[k] += ctx->Rinv[k+j*dof]*jmpR;
        }
      }
      /* Apply limiter to the left and right characteristic jumps */
      info.m  = dof;
      info.hxs = hxs;
      info.hxf = hxf;
      (*ctx->limit2)(&info,cjmpL,cjmpR,ctx->sf,ctx->fs,i,ctx->cslope);
      for (j=0; j<dof; j++) {
        PetscScalar tmp = 0;
        for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof]*ctx->cslope[k];
          slope[i*dof+j] = tmp;
      }
    }
  }
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    uL = &ctx->uLR[0];
    uR = &ctx->uLR[dof];
    if (i < sf-lsbwidth) { /* slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hxs)); /* Max allowable value of 1/Delta t */
      if (i > xs) {
        for (j=0; j<dof; j++) f[(islow-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hxs;
        islow++;
      }
    }
    if (i == sf-lsbwidth) { /* interface between the slow region and the fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(islow-1)*dof+j] -= ctx->flux[j]/hxs;
      }
    }
    if (i == fs+rsbwidth) { /* slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hxs;
        islow++;
      }
    }
    if (i > fs+rsbwidth) { /* slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(islow-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
<<<<<<< HEAD
<<<<<<< HEAD
        for (j=0; j<dof; j++) f[(i-fs+sf)*dof+j] += ctx->flux[j]/hs;
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hs;
=======
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hxs;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
        islow++;
>>>>>>> Merged master and copied the fixed ex7 to ex4, runs in parallel now
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
<<<<<<< HEAD
<<<<<<< HEAD
  ierr = DMDARestoreArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
<<<<<<< HEAD
  ierr = MPI_Allreduce(&cfl_idt,&ctx->cfl_idt,1,MPIU_SCALAR,MPIU_MAX,PetscObjectComm((PetscObject)da));CHKERRMPI(ierr);
=======
  ierr = MPI_Allreduce(&cfl_idt,&ctx->cfl_idt,1,MPIU_SCALAR,MPIU_MAX,PetscObjectComm((PetscObject)da));CHKERRQ(ierr);
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  PetscFunctionReturn(0);
}

PetscErrorCode FVRHSFunctionslowbuffer_2WaySplit(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx          *ctx = (FVCtx*)vctx;
  PetscErrorCode ierr;
  PetscInt       i,j,k,Mx,dof,xs,xm,islow = 0,sf = ctx->sf,fs = ctx->fs,lsbwidth = ctx->lsbwidth,rsbwidth = ctx->rsbwidth;
  PetscReal      hxs,hxf;
  PetscScalar    *x,*f,*slope;
=======
=======
  ierr = DMDARestoreArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&cfl_idt,&ctx->cfl_idt,1,MPIU_SCALAR,MPIU_MAX,PetscObjectComm((PetscObject)da));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FVRHSFunctionslowbuffer_2WaySplit(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx          *ctx = (FVCtx*)vctx;
  PetscErrorCode ierr;
<<<<<<< HEAD
  PetscInt       i,j,Mx,dof,xs,xm,ifast = 0,sf = ctx->sf,fs = ctx->fs;
  PetscReal      hf,hs;
  PetscScalar    *x,*f,*r,*min,*alpha,*gamma;
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
  PetscInt       i,j,k,Mx,dof,xs,xm,islow = 0,sf = ctx->sf,fs = ctx->fs,lsbwidth = ctx->lsbwidth,rsbwidth = ctx->rsbwidth;
  PetscReal      hxs,hxf;
  PetscScalar    *x,*f,*slope;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  Vec            Xloc;
  DM             da;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  hxs  = (ctx->xmax-ctx->xmin)*3.0/8.0/ctx->sf;
  hxf  = (ctx->xmax-ctx->xmin)/4.0/(ctx->fs-ctx->sf);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
<<<<<<< HEAD
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = DMDAGetArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
=======
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);                          /* Xloc contains ghost points                                     */
  ierr = DMDAGetInfo(da,0,&Mx,0,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);   /* Mx is the number of center points                              */
  hs   = (ctx->xmax-ctx->xmin)/2.0*(ctx->hratio+1.0)/Mx;
  hf   = (ctx->xmax-ctx->xmin)/2.0*(1.0+1.0/ctx->hratio)/Mx;
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);       /* X is solution vector which does not contain ghost points       */
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = DMDAGetArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  ierr = PetscMalloc4(dof,&r,dof,&min,dof,&alpha,dof,&gamma);CHKERRQ(ierr);
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
  ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD); 
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD); 
>>>>>>> Minor edits
=======
  ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);
>>>>>>> add test to src/ts/tutorials/multirate/ex4.c
=======
>>>>>>> Small fixes to ex4

  if (ctx->bctype == FVBC_OUTFLOW) {
    for (i=xs-2; i<0; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[j];
    }
    for (i=Mx; i<xs+xm+2; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[(xs+xm-1)*dof+j];
    }
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  if (ctx->bctype == FVBC_INFLOW) {
    /* See LeVeque, R. (2002). Finite Volume Methods for Hyperbolic Problems. doi:10.1017/CBO9780511791253
    pages 137-138 for the scheme. */
    if (xs==0) { /* Left Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmin,ctx->ub);
      for (j=0; j<dof; j++) {
        if (ctx->physics2.bcinflowindex[j]==PETSC_TRUE) {
          for (i=-2; i<0; i++) x[i*dof+j] = 2.0*ctx->ub[j]-x[-(i+1)*dof+j];
        } else {
          for (i=-2; i<0; i++) x[i*dof+j] = x[j]; /* Outflow */
        }
      }
    }
    if (xs+xm==Mx) { /* Right Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmax,ctx->ub);
      for (j=0; j<dof; j++) {
        if (ctx->physics2.bcinflowindex[dof+j]==PETSC_TRUE) {
          for (i=Mx; i<Mx+2; i++) x[i*dof+j] = 2.0*ctx->ub[dof+j]-x[(2*Mx-(i+1))*dof+j];
        } else {
          for (i=Mx; i<Mx+2; i++) x[i*dof+j] = x[(Mx-1)*dof+j]; /* Outflow */
=======
  if (ctx->bctype == FVBC_INFLOW) {
    /* See LeVeque, R. (2002). Finite Volume Methods for Hyperbolic Problems. doi:10.1017/CBO9780511791253
    pages 137-138 for the scheme. */
    if(xs==0){ /* Left Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmin,ctx->ub);
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[j]==PETSC_TRUE){
          for(i=-2; i<0; i++) x[i*dof+j] = 2.0*ctx->ub[j]-x[-(i+1)*dof+j];
=======
  if (ctx->bctype == FVBC_INFLOW) {
    /* See LeVeque, R. (2002). Finite Volume Methods for Hyperbolic Problems. doi:10.1017/CBO9780511791253
    pages 137-138 for the scheme. */
    if(xs==0){ /* Left Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmin,ctx->ub);
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[j]==PETSC_TRUE){
<<<<<<< HEAD
          for(i=-2; i<0; i++) x[i*dof+j] = 2.0*ctx->ub[j]-x[-(i+1)*dof+j]; 
>>>>>>> Added Support for Inflow Boundary Conditions
=======
          for(i=-2; i<0; i++) x[i*dof+j] = 2.0*ctx->ub[j]-x[-(i+1)*dof+j];
>>>>>>> trivial edit: remove white spaces
        } else {
          for(i=-2; i<0; i++) x[i*dof+j] = x[j]; /* Outflow */
        }
      }
    }
    if(xs+xm==Mx){ /* Right Boundary */
<<<<<<< HEAD
<<<<<<< HEAD
      ctx->physics2.inflow(ctx,time,ctx->xmax,ctx->ub);
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[dof+j]==PETSC_TRUE){
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = 2.0*ctx->ub[dof+j]-x[(2*Mx-(i+1))*dof+j];
        } else {
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = x[(Mx-1)*dof+j]; /* Outflow */
>>>>>>> Added Support for Inflow Boundary Conditions
=======
      ctx->physics2.inflow(ctx,time,ctx->xmax,ctx->ub); 
=======
      ctx->physics2.inflow(ctx,time,ctx->xmax,ctx->ub);
>>>>>>> trivial edit: remove white spaces
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[dof+j]==PETSC_TRUE){
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = 2.0*ctx->ub[dof+j]-x[(2*Mx-(i+1))*dof+j];
        } else {
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = x[(Mx-1)*dof+j]; /* Outflow */
>>>>>>> Added Support for Inflow Boundary Conditions
        }
      }
    }
  }
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
>>>>>>> Added Support for Inflow Boundary Conditions
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
>>>>>>> Added Support for Inflow Boundary Conditions
  for (i=xs-1; i<xs+xm+1; i++) {
    struct _LimitInfo info;
    PetscScalar       *cjmpL,*cjmpR;
    if ((i > sf-lsbwidth-2 && i < sf+1) || (i > fs-2 && i < fs+rsbwidth+1)) {
      /* Determine the right eigenvectors R, where A = R \Lambda R^{-1} */
      ierr = (*ctx->physics2.characteristic2)(ctx->physics2.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds);CHKERRQ(ierr);
      /* Evaluate jumps across interfaces (i-1, i) and (i, i+1), put in characteristic basis */
      ierr  = PetscArrayzero(ctx->cjmpLR,2*dof);CHKERRQ(ierr);
      cjmpL = &ctx->cjmpLR[0];
      cjmpR = &ctx->cjmpLR[dof];
      for (j=0; j<dof; j++) {
        PetscScalar jmpL,jmpR;
        jmpL = x[(i+0)*dof+j]-x[(i-1)*dof+j];
        jmpR = x[(i+1)*dof+j]-x[(i+0)*dof+j];
        for (k=0; k<dof; k++) {
          cjmpL[k] += ctx->Rinv[k+j*dof]*jmpL;
          cjmpR[k] += ctx->Rinv[k+j*dof]*jmpR;
        }
      }
      /* Apply limiter to the left and right characteristic jumps */
      info.m  = dof;
      info.hxs = hxs;
      info.hxf = hxf;
      (*ctx->limit2)(&info,cjmpL,cjmpR,ctx->sf,ctx->fs,i,ctx->cslope);
      for (j=0; j<dof; j++) {
        PetscScalar tmp = 0;
        for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof]*ctx->cslope[k];
          slope[i*dof+j] = tmp;
      }
    }
  }
<<<<<<< HEAD

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    uL = &ctx->uLR[0];
    uR = &ctx->uLR[dof];
    if (i == sf-lsbwidth) {
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hxs;
        islow++;
      }
    }
    if (i > sf-lsbwidth && i < sf) {
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(islow-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hxs;
        islow++;
      }
    }
    if (i == sf) { /* interface between the slow region and the fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxf/2;
      }
      ierr = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(islow-1)*dof+j] -= ctx->flux[j]/hxs;
<<<<<<< HEAD
      }
    }
    if (i == fs) { /* interface between the fast region and the slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxf/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
=======
      }
    }
    if (i == fs) { /* interface between the fast region and the slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxf/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
      ierr = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hxs;
        islow++;
<<<<<<< HEAD
      }
    }
    if (i > fs && i < fs+rsbwidth) {
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(islow-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hxs;
        islow++;
      }
    }
    if (i == fs+rsbwidth) {
=======
      }
    }
    if (i > fs && i < fs+rsbwidth) {
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(islow-1)*dof+j] -= ctx->flux[j]/hxs;
<<<<<<< HEAD
=======
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hxs;
        islow++;
      }
    }
    if (i == fs+rsbwidth) {
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(islow-1)*dof+j] -= ctx->flux[j]/hxs;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    uL = &ctx->uLR[0];
    uR = &ctx->uLR[dof];
    if (i == sf-lsbwidth) {
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hxs;
        islow++;
      }
    }
    if (i > sf-lsbwidth && i < sf) {
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(islow-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hxs;
        islow++;
      }
    }
    if (i == sf) { /* interface between the slow region and the fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxf/2;
      }
      ierr = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(islow-1)*dof+j] -= ctx->flux[j]/hxs;
      }
    }
    if (i == fs) { /* interface between the fast region and the slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxf/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hxs;
        islow++;
      }
    }
    if (i > fs && i < fs+rsbwidth) {
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
<<<<<<< HEAD
      ierr =  (*ctx->physics.flux)(ctx->physics.user,u,ctx->flux,&maxspeed);CHKERRQ(ierr);
<<<<<<< HEAD
      if (i>xs) {
        for (j=0; j<dof; j++) f[(i-sf-1)*dof+j] -= ctx->flux[j]/hf;
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
<<<<<<< HEAD
  ierr = DMDARestoreArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------- Finite Volume Solver for fast  parts ----------------------------------- */
PetscErrorCode FVRHSFunctionfast_2WaySplit(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx          *ctx = (FVCtx*)vctx;
  PetscErrorCode ierr;
  PetscInt       i,j,k,Mx,dof,xs,xm,ifast = 0,sf = ctx->sf,fs = ctx->fs;
  PetscReal      hxs,hxf;
  PetscScalar    *x,*f,*slope;
  Vec            Xloc;
  DM             da;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  hxs  = (ctx->xmax-ctx->xmin)*3.0/8.0/ctx->sf;
  hxf  = (ctx->xmax-ctx->xmin)/4.0/(ctx->fs-ctx->sf);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = DMDAGetArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  if (ctx->bctype == FVBC_OUTFLOW) {
    for (i=xs-2; i<0; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[j];
    }
    for (i=Mx; i<xs+xm+2; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[(xs+xm-1)*dof+j];
    }
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  if (ctx->bctype == FVBC_INFLOW) {
    /* See LeVeque, R. (2002). Finite Volume Methods for Hyperbolic Problems. doi:10.1017/CBO9780511791253
    pages 137-138 for the scheme. */
    if (xs==0) { /* Left Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmin,ctx->ub);
      for (j=0; j<dof; j++) {
        if (ctx->physics2.bcinflowindex[j]==PETSC_TRUE) {
          for (i=-2; i<0; i++) x[i*dof+j] = 2.0*ctx->ub[j]-x[-(i+1)*dof+j];
        } else {
          for (i=-2; i<0; i++) x[i*dof+j] = x[j]; /* Outflow */
        }
      }
    }
    if (xs+xm==Mx) { /* Right Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmax,ctx->ub);
      for (j=0; j<dof; j++) {
        if (ctx->physics2.bcinflowindex[dof+j]==PETSC_TRUE) {
          for (i=Mx; i<Mx+2; i++) x[i*dof+j] = 2.0*ctx->ub[dof+j]-x[(2*Mx-(i+1))*dof+j];
        } else {
          for (i=Mx; i<Mx+2; i++) x[i*dof+j] = x[(Mx-1)*dof+j]; /* Outflow */
=======
  if (ctx->bctype == FVBC_INFLOW) {
    /* See LeVeque, R. (2002). Finite Volume Methods for Hyperbolic Problems. doi:10.1017/CBO9780511791253
    pages 137-138 for the scheme. */
    if(xs==0){ /* Left Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmin,ctx->ub);
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[j]==PETSC_TRUE){
          for(i=-2; i<0; i++) x[i*dof+j] = 2.0*ctx->ub[j]-x[-(i+1)*dof+j];
=======
  if (ctx->bctype == FVBC_INFLOW) {
    /* See LeVeque, R. (2002). Finite Volume Methods for Hyperbolic Problems. doi:10.1017/CBO9780511791253
    pages 137-138 for the scheme. */
    if(xs==0){ /* Left Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmin,ctx->ub);
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[j]==PETSC_TRUE){
<<<<<<< HEAD
          for(i=-2; i<0; i++) x[i*dof+j] = 2.0*ctx->ub[j]-x[-(i+1)*dof+j]; 
>>>>>>> Added Support for Inflow Boundary Conditions
=======
          for(i=-2; i<0; i++) x[i*dof+j] = 2.0*ctx->ub[j]-x[-(i+1)*dof+j];
>>>>>>> trivial edit: remove white spaces
        } else {
          for(i=-2; i<0; i++) x[i*dof+j] = x[j]; /* Outflow */
        }
      }
    }
    if(xs+xm==Mx){ /* Right Boundary */
<<<<<<< HEAD
<<<<<<< HEAD
      ctx->physics2.inflow(ctx,time,ctx->xmax,ctx->ub);
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[dof+j]==PETSC_TRUE){
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = 2.0*ctx->ub[dof+j]-x[(2*Mx-(i+1))*dof+j];
        } else {
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = x[(Mx-1)*dof+j]; /* Outflow */
>>>>>>> Added Support for Inflow Boundary Conditions
=======
      ctx->physics2.inflow(ctx,time,ctx->xmax,ctx->ub); 
=======
      ctx->physics2.inflow(ctx,time,ctx->xmax,ctx->ub);
>>>>>>> trivial edit: remove white spaces
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[dof+j]==PETSC_TRUE){
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = 2.0*ctx->ub[dof+j]-x[(2*Mx-(i+1))*dof+j];
        } else {
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = x[(Mx-1)*dof+j]; /* Outflow */
>>>>>>> Added Support for Inflow Boundary Conditions
        }
      }
    }
  }
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Added Support for Inflow Boundary Conditions
  for (i=xs-1; i<xs+xm+1; i++) {
    struct _LimitInfo info;
    PetscScalar       *cjmpL,*cjmpR;
    if (i > sf-2 && i < fs+1) {
      ierr = (*ctx->physics2.characteristic2)(ctx->physics2.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds);CHKERRQ(ierr);
      ierr  = PetscArrayzero(ctx->cjmpLR,2*dof);CHKERRQ(ierr);
      cjmpL = &ctx->cjmpLR[0];
      cjmpR = &ctx->cjmpLR[dof];
      for (j=0; j<dof; j++) {
        PetscScalar jmpL,jmpR;
        jmpL = x[(i+0)*dof+j]-x[(i-1)*dof+j];
        jmpR = x[(i+1)*dof+j]-x[(i+0)*dof+j];
        for (k=0; k<dof; k++) {
          cjmpL[k] += ctx->Rinv[k+j*dof]*jmpL;
          cjmpR[k] += ctx->Rinv[k+j*dof]*jmpR;
        }
      }
=======
=======
>>>>>>> Added Support for Inflow Boundary Conditions
  for (i=xs-1; i<xs+xm+1; i++) {
    struct _LimitInfo info;
    PetscScalar       *cjmpL,*cjmpR;
    if (i > sf-2 && i < fs+1) {
      ierr = (*ctx->physics2.characteristic2)(ctx->physics2.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds);CHKERRQ(ierr);
      ierr  = PetscArrayzero(ctx->cjmpLR,2*dof);CHKERRQ(ierr);
      cjmpL = &ctx->cjmpLR[0];
      cjmpR = &ctx->cjmpLR[dof];
      for (j=0; j<dof; j++) {
        PetscScalar jmpL,jmpR;
        jmpL = x[(i+0)*dof+j]-x[(i-1)*dof+j];
        jmpR = x[(i+1)*dof+j]-x[(i+0)*dof+j];
        for (k=0; k<dof; k++) {
          cjmpL[k] += ctx->Rinv[k+j*dof]*jmpL;
          cjmpR[k] += ctx->Rinv[k+j*dof]*jmpR;
        }
      }
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
      /* Apply limiter to the left and right characteristic jumps */
      info.m  = dof;
      info.hxs = hxs;
      info.hxf = hxf;
      (*ctx->limit2)(&info,cjmpL,cjmpR,ctx->sf,ctx->fs,i,ctx->cslope);
      for (j=0; j<dof; j++) {
      PetscScalar tmp = 0;
      for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof]*ctx->cslope[k];
        slope[i*dof+j] = tmp;
      }
    }
  }

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    uL = &ctx->uLR[0];
    uR = &ctx->uLR[dof];
    if (i == sf) { /* interface between the slow region and the fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxf/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[ifast*dof+j] += ctx->flux[j]/hxf;
        ifast++;
      }
    }
    if (i > sf && i < fs) { /* fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxf/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxf/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(ifast-1)*dof+j] -= ctx->flux[j]/hxf;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[ifast*dof+j] += ctx->flux[j]/hxf;
        ifast++;
      }
    }
    if (i == fs) { /* interface between the fast region and the slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxf/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(ifast-1)*dof+j] -= ctx->flux[j]/hxf;
=======
      if (i > xs) {
        for (j=0; j<dof; j++) f[(ifast-1)*dof+j] -= ctx->flux[j]/hf;
>>>>>>> Merged master and copied the fixed ex7 to ex4, runs in parallel now
=======
      ierr = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(islow-1)*dof+j] -= ctx->flux[j]/hxs;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[islow*dof+j] += ctx->flux[j]/hxs;
        islow++;
      }
    }
    if (i == fs+rsbwidth) {
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(islow-1)*dof+j] -= ctx->flux[j]/hxs;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = DMDARestoreArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
<<<<<<< HEAD
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
=======
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------- Finite Volume Solver for fast  parts ----------------------------------- */
PetscErrorCode FVRHSFunctionfast_2WaySplit(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx          *ctx = (FVCtx*)vctx;
  PetscErrorCode ierr;
  PetscInt       i,j,k,Mx,dof,xs,xm,ifast = 0,sf = ctx->sf,fs = ctx->fs;
  PetscReal      hxs,hxf;
  PetscScalar    *x,*f,*slope;
  Vec            Xloc;
  DM             da;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  hxs  = (ctx->xmax-ctx->xmin)*3.0/8.0/ctx->sf;
  hxf  = (ctx->xmax-ctx->xmin)/4.0/(ctx->fs-ctx->sf);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = DMDAGetArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  if (ctx->bctype == FVBC_OUTFLOW) {
    for (i=xs-2; i<0; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[j];
    }
    for (i=Mx; i<xs+xm+2; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[(xs+xm-1)*dof+j];
    }
  }
  if (ctx->bctype == FVBC_INFLOW) {
    /* See LeVeque, R. (2002). Finite Volume Methods for Hyperbolic Problems. doi:10.1017/CBO9780511791253
    pages 137-138 for the scheme. */
    if(xs==0){ /* Left Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmin,ctx->ub);
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[j]){
          for(i=-2; i<0; i++) x[i*dof+j] = 2.0*ctx->ub[j]-x[-(i+1)*dof+j];
        } else {
          for(i=-2; i<0; i++) x[i*dof+j] = x[j]; /* Outflow */
        }
      }
    }
    if(xs+xm==Mx){ /* Right Boundary */
      ctx->physics2.inflow(ctx,time,ctx->xmax,ctx->ub);
      for(j=0; j<dof; j++) {
        if(ctx->physics2.bcinflowindex[dof+j]){
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = 2.0*ctx->ub[dof+j]-x[(2*Mx-(i+1))*dof+j];
        } else {
          for(i=Mx; i<Mx+2; i++) x[i*dof+j] = x[(Mx-1)*dof+j]; /* Outflow */
        }
      }
    }
  }
  for (i=xs-1; i<xs+xm+1; i++) {
    struct _LimitInfo info;
    PetscScalar       *cjmpL,*cjmpR;
    if (i > sf-2 && i < fs+1) {
      ierr = (*ctx->physics2.characteristic2)(ctx->physics2.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds);CHKERRQ(ierr);
      ierr  = PetscArrayzero(ctx->cjmpLR,2*dof);CHKERRQ(ierr);
      cjmpL = &ctx->cjmpLR[0];
      cjmpR = &ctx->cjmpLR[dof];
      for (j=0; j<dof; j++) {
        PetscScalar jmpL,jmpR;
        jmpL = x[(i+0)*dof+j]-x[(i-1)*dof+j];
        jmpR = x[(i+1)*dof+j]-x[(i+0)*dof+j];
        for (k=0; k<dof; k++) {
          cjmpL[k] += ctx->Rinv[k+j*dof]*jmpL;
          cjmpR[k] += ctx->Rinv[k+j*dof]*jmpR;
        }
      }
      /* Apply limiter to the left and right characteristic jumps */
      info.m  = dof;
      info.hxs = hxs;
      info.hxf = hxf;
      (*ctx->limit2)(&info,cjmpL,cjmpR,ctx->sf,ctx->fs,i,ctx->cslope);
      for (j=0; j<dof; j++) {
      PetscScalar tmp = 0;
      for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof]*ctx->cslope[k];
        slope[i*dof+j] = tmp;
      }
    }
  }

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    uL = &ctx->uLR[0];
    uR = &ctx->uLR[dof];
    if (i == sf) { /* interface between the slow region and the fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxs/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxf/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[ifast*dof+j] += ctx->flux[j]/hxf;
        ifast++;
      }
    }
    if (i > sf && i < fs) { /* fast region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxf/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxf/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(ifast-1)*dof+j] -= ctx->flux[j]/hxf;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[ifast*dof+j] += ctx->flux[j]/hxf;
        ifast++;
      }
    }
    if (i == fs) { /* interface between the fast region and the slow region */
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hxf/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hxs/2;
      }
      ierr    = (*ctx->physics2.riemann2)(ctx->physics2.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
      if (i > xs) {
        for (j=0; j<dof; j++) f[(ifast-1)*dof+j] -= ctx->flux[j]/hxf;
      }
    }
  }
<<<<<<< HEAD
  ierr = VecRestoreArrayRead(X,&ptr_X);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Y,&ptr_Y);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
  ierr = DMDAVecRestoreArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = DMDARestoreArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  PetscFunctionReturn(0);
}

int main(int argc,char *argv[])
{
<<<<<<< HEAD
<<<<<<< HEAD
  char              lname[256] = "mc",physname[256] = "advect",final_fname[256] = "solution.m";
  PetscFunctionList limiters   = 0,physics = 0;
=======
  char              physname[256] = "advect",final_fname[256] = "solution.m";
  PetscFunctionList physics = 0;
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
  char              lname[256] = "mc",physname[256] = "advect",final_fname[256] = "solution.m";
  PetscFunctionList limiters   = 0,physics = 0;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  MPI_Comm          comm;
  TS                ts;
  DM                da;
  Vec               X,X0,R;
  FVCtx             ctx;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  PetscInt          i,k,dof,xs,xm,Mx,draw = 0,count_slow,count_fast,islow = 0,ifast =0,islowbuffer = 0,*index_slow,*index_fast,*index_slowbuffer;
=======
  PetscInt          bs,i,k,dof,xs,xm,Mx,draw = 0,count_slow,count_fast,islow = 0,ifast =0,islowbuffer = 0,*index_slow,*index_fast,*index_slowbuffer;
>>>>>>> Small fixes to ex4
=======
  PetscInt          i,k,dof,xs,xm,Mx,draw = 0,count_slow,count_fast,islow = 0,ifast =0,islowbuffer = 0,*index_slow,*index_fast,*index_slowbuffer;
>>>>>>> Documentation adjustments to ex4
  PetscBool         view_final = PETSC_FALSE;
  PetscReal         ptime,maxtime;
<<<<<<< HEAD
=======
  PetscInt          i,k,dof,xs,xm,Mx,draw = 0,count_slow,count_fast,islow = 0,ifast = 0,*index_slow,*index_fast;
=======
  PetscInt          i,k,dof,xs,xm,Mx,draw = 0,count_slow,count_fast,islow = 0,ifast =0,islowbuffer = 0,*index_slow,*index_fast,*index_slowbuffer;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  PetscBool         view_final = PETSC_FALSE;
  PetscReal         ptime;
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
>>>>>>> Added example specifications
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscMemzero(&ctx,sizeof(ctx));CHKERRQ(ierr);

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  /* Register limiters to be available on the command line */
  ierr = PetscFunctionListAdd(&limiters,"upwind"              ,Limit2_Upwind);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"lax-wendroff"        ,Limit2_LaxWendroff);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"beam-warming"        ,Limit2_BeamWarming);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"fromm"               ,Limit2_Fromm);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"minmod"              ,Limit2_Minmod);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"superbee"            ,Limit2_Superbee);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"mc"                  ,Limit2_MC);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"koren3"              ,Limit2_Koren3);CHKERRQ(ierr);

<<<<<<< HEAD
  /* Register physical models to be available on the command line */
  ierr = PetscFunctionListAdd(&physics,"shallow"         ,PhysicsCreate_Shallow);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&physics,"advect"          ,PhysicsCreate_Advect);CHKERRQ(ierr);

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> trivial edit: remove white spaces
  ctx.comm    = comm;
  ctx.cfl     = 0.9;
  ctx.bctype  = FVBC_PERIODIC;
  ctx.xmin    = -1.0;
  ctx.xmax    = 1.0;
  ctx.initial = 1;
  ctx.hratio  = 2;
  maxtime     = 10.0;
<<<<<<< HEAD
<<<<<<< HEAD
  ctx.simulation = PETSC_FALSE;
=======
=======
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  /* Register physical models to be available on the command line */
  ierr = PetscFunctionListAdd(&physics,"shallow"         ,PhysicsCreate_Shallow);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&physics,"advect"          ,PhysicsCreate_Advect);CHKERRQ(ierr);

<<<<<<< HEAD
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
  ctx.comm = comm;
  ctx.cfl  = 0.9;
  ctx.bctype = FVBC_PERIODIC;
  ctx.xmin = -1.0;
  ctx.xmax = 1.0;
<<<<<<< HEAD
<<<<<<< HEAD
  ctx.initial = 1; 
  ctx.hratio = 2; 
<<<<<<< HEAD
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
  maxtime = 10.0;
>>>>>>> Added example specifications
=======
>>>>>>> trivial edit: remove white spaces
=======
  ctx.simulation = PETSC_FALSE; 
>>>>>>> Small fixes to ex4
  ierr = PetscOptionsBegin(comm,NULL,"Finite Volume solver options","");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-xmin","X min","",ctx.xmin,&ctx.xmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-xmax","X max","",ctx.xmax,&ctx.xmax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-limit","Name of flux imiter to use","",limiters,lname,lname,sizeof(lname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-physics","Name of physics model to use","",physics,physname,physname,sizeof(physname),NULL);CHKERRQ(ierr);
=======
  ierr = PetscOptionsBegin(comm,NULL,"Finite Volume solver options","");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-xmin","X min","",ctx.xmin,&ctx.xmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-xmax","X max","",ctx.xmax,&ctx.xmax,NULL);CHKERRQ(ierr);
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
  ctx.initial = 1; 
  ctx.hratio = 2; 
  maxtime = 10.0;
=======
  ctx.comm    = comm;
  ctx.cfl     = 0.9;
  ctx.bctype  = FVBC_PERIODIC;
  ctx.xmin    = -1.0;
  ctx.xmax    = 1.0;
  ctx.initial = 1;
  ctx.hratio  = 2;
  maxtime     = 10.0;
<<<<<<< HEAD
>>>>>>> trivial edit: remove white spaces
=======
  ctx.simulation = PETSC_FALSE; 
>>>>>>> Small fixes to ex4
  ierr = PetscOptionsBegin(comm,NULL,"Finite Volume solver options","");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-xmin","X min","",ctx.xmin,&ctx.xmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-xmax","X max","",ctx.xmax,&ctx.xmax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-limit","Name of flux imiter to use","",limiters,lname,lname,sizeof(lname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-physics","Name of physics model to use","",physics,physname,physname,sizeof(physname),NULL);CHKERRQ(ierr);
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  ierr = PetscOptionsInt("-draw","Draw solution vector, bitwise OR of (1=initial,2=final,4=final error)","",draw,&draw,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-view_final","Write final solution in ASCII MATLAB format to given file name","",final_fname,final_fname,sizeof(final_fname),&view_final);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-initial","Initial condition (depends on the physics)","",ctx.initial,&ctx.initial,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-exact","Compare errors with exact solution","",ctx.exact,&ctx.exact,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simulation","Compare errors with reference solution","",ctx.simulation,&ctx.simulation,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-cfl","CFL number to time step at","",ctx.cfl,&ctx.cfl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-bc_type","Boundary condition","",FVBCTypes,(PetscEnum)ctx.bctype,(PetscEnum*)&ctx.bctype,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-hratio","Spacing ratio","",ctx.hratio,&ctx.hratio,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  /* Choose the limiter from the list of registered limiters */
  ierr = PetscFunctionListFind(limiters,lname,&ctx.limit2);CHKERRQ(ierr);
  if (!ctx.limit2) SETERRQ1(PETSC_COMM_SELF,1,"Limiter '%s' not found",lname);

<<<<<<< HEAD
=======
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  /* Choose the physics from the list of registered models */
  {
    PetscErrorCode (*r)(FVCtx*);
    ierr = PetscFunctionListFind(physics,physname,&r);CHKERRQ(ierr);
    if (!r) SETERRQ1(PETSC_COMM_SELF,1,"Physics '%s' not found",physname);
    /* Create the physics, will set the number of fields and their names */
    ierr = (*r)(&ctx);CHKERRQ(ierr);
  }

  /* Create a DMDA to manage the parallel grid */
<<<<<<< HEAD
<<<<<<< HEAD
  ierr = DMDACreate1d(comm,DM_BOUNDARY_PERIODIC,50,ctx.physics2.dof,2,NULL,&da);CHKERRQ(ierr);
=======
  ierr = DMDACreate1d(comm,DM_BOUNDARY_PERIODIC,50,ctx.physics.dof,2,NULL,&da);CHKERRQ(ierr);
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
  ierr = DMDACreate1d(comm,DM_BOUNDARY_PERIODIC,50,ctx.physics2.dof,2,NULL,&da);CHKERRQ(ierr);
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  /* Inform the DMDA of the field names provided by the physics. */
  /* The names will be shown in the title bars when run with -ts_monitor_draw_solution */
<<<<<<< HEAD
<<<<<<< HEAD
  for (i=0; i<ctx.physics2.dof; i++) {
    ierr = DMDASetFieldName(da,i,ctx.physics2.fieldname[i]);CHKERRQ(ierr);
  }
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
=======
  for (i=0; i<ctx.physics.dof; i++) {
    ierr = DMDASetFieldName(da,i,ctx.physics.fieldname[i]);CHKERRQ(ierr);
  }
  ierr = DMDAGetInfo(da,0,&Mx,0,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
  for (i=0; i<ctx.physics2.dof; i++) {
    ierr = DMDASetFieldName(da,i,ctx.physics2.fieldname[i]);CHKERRQ(ierr);
  }
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  /* Set coordinates of cell centers */
  ierr = DMDASetUniformCoordinates(da,ctx.xmin+0.5*(ctx.xmax-ctx.xmin)/Mx,ctx.xmax+0.5*(ctx.xmax-ctx.xmin)/Mx,0,0,0,0);CHKERRQ(ierr);

  /* Allocate work space for the Finite Volume solver (so it doesn't have to be reallocated on each function evaluation) */
<<<<<<< HEAD
<<<<<<< HEAD
  ierr = PetscMalloc4(dof*dof,&ctx.R,dof*dof,&ctx.Rinv,2*dof,&ctx.cjmpLR,1*dof,&ctx.cslope);CHKERRQ(ierr);
  ierr = PetscMalloc3(2*dof,&ctx.uLR,dof,&ctx.flux,dof,&ctx.speeds);CHKERRQ(ierr);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  ierr = PetscMalloc1(2*dof,&ctx.ub);CHKERRQ(ierr);
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
  ierr = PetscMalloc1(2*dof,&ctx.ub);CHKERRQ(ierr); 
>>>>>>> Added Support for Inflow Boundary Conditions
=======
  ierr = PetscMalloc1(2*dof,&ctx.ub);CHKERRQ(ierr);
>>>>>>> trivial edit: remove white spaces
=======
  ierr = PetscMalloc3(dof,&ctx.u,dof,&ctx.flux,dof,&ctx.speeds);CHKERRQ(ierr);
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
  ierr = PetscMalloc4(dof*dof,&ctx.R,dof*dof,&ctx.Rinv,2*dof,&ctx.cjmpLR,1*dof,&ctx.cslope);CHKERRQ(ierr);
  ierr = PetscMalloc3(2*dof,&ctx.uLR,dof,&ctx.flux,dof,&ctx.speeds);CHKERRQ(ierr);
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
  ierr = PetscMalloc1(2*dof,&ctx.ub);CHKERRQ(ierr); 
>>>>>>> Added Support for Inflow Boundary Conditions
=======
  ierr = PetscMalloc1(2*dof,&ctx.ub);CHKERRQ(ierr);
>>>>>>> trivial edit: remove white spaces

  /* Create a vector to store the solution and to save the initial state */
  ierr = DMCreateGlobalVector(da,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&X0);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&R);CHKERRQ(ierr);

<<<<<<< HEAD
<<<<<<< HEAD
  /* create index for slow parts and fast parts,
     count_slow + count_fast = Mx, counts_slow*hs = 0.5, counts_fast*hf = 0.5 */
  count_slow = Mx/(1.0+ctx.hratio/3.0);
<<<<<<< HEAD
<<<<<<< HEAD
  if (count_slow%2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Please adjust grid size Mx (-da_grid_x) and hratio (-hratio) so that Mx/(1+hratio/3) is even");
=======
  if (count_slow%2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Please adjust grid size Mx (-da_grid_x) and hratio (-hratio) so that Mx/(1+hartio/3) is even");
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
  if (count_slow%2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Please adjust grid size Mx (-da_grid_x) and hratio (-hratio) so that Mx/(1+hratio/3) is even");
>>>>>>> Small fixes to ex4
  count_fast = Mx-count_slow;
  ctx.sf = count_slow/2;
  ctx.fs = ctx.sf+count_fast;
  ierr = PetscMalloc1(xm*dof,&index_slow);CHKERRQ(ierr);
  ierr = PetscMalloc1(xm*dof,&index_fast);CHKERRQ(ierr);
  ierr = PetscMalloc1(8*dof,&index_slowbuffer);CHKERRQ(ierr);
<<<<<<< HEAD
<<<<<<< HEAD
  ctx.lsbwidth = 4;
  ctx.rsbwidth = 4;

<<<<<<< HEAD
=======
  if (((AdvectCtx*)ctx.physics2.user)->a > 0) {
    ctx.lsbwidth = 4;
    ctx.rsbwidth = 4;
  } else {
    ctx.lsbwidth = 4;
    ctx.rsbwidth = 4;
  }
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
  ctx.lsbwidth = 4;
  ctx.rsbwidth = 4;
<<<<<<< HEAD
  
>>>>>>> Added Shallow water equations to the top menu. Detailed description of options still required.
=======

<<<<<<< HEAD
>>>>>>> trivial edit: remove white spaces
=======
  ierr = VecGetBlockSize(X,&bs);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"BlockSize %D\n",bs);CHKERRQ(ierr);
>>>>>>> Small fixes to ex4
=======
>>>>>>> Documentation adjustments to ex4
  for (i=xs; i<xs+xm; i++) {
    if (i < ctx.sf-ctx.lsbwidth || i > ctx.fs+ctx.rsbwidth-1)
      for (k=0; k<dof; k++) index_slow[islow++] = i*dof+k;
    else if ((i >= ctx.sf-ctx.lsbwidth && i < ctx.sf) || (i > ctx.fs-1 && i <= ctx.fs+ctx.rsbwidth-1))
      for (k=0; k<dof; k++) index_slowbuffer[islowbuffer++] = i*dof+k;
=======
  /* create index for slow parts and fast parts*/
  count_slow = Mx/(1+ctx.hratio);
  if (count_slow%2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Please adjust grid size Mx (-da_grid_x) and hratio (-hratio) so that Mx/(1+hartio) is even");
=======
  /* create index for slow parts and fast parts,
     count_slow + count_fast = Mx, counts_slow*hs = 0.5, counts_fast*hf = 0.5 */
  count_slow = Mx/(1.0+ctx.hratio/3.0);
<<<<<<< HEAD
  if (count_slow%2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Please adjust grid size Mx (-da_grid_x) and hratio (-hratio) so that Mx/(1+hartio/3) is even");
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
  if (count_slow%2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Please adjust grid size Mx (-da_grid_x) and hratio (-hratio) so that Mx/(1+hratio/3) is even");
>>>>>>> Small fixes to ex4
  count_fast = Mx-count_slow;
  ctx.sf = count_slow/2;
  ctx.fs = ctx.sf+count_fast;
  ierr = PetscMalloc1(xm*dof,&index_slow);CHKERRQ(ierr);
  ierr = PetscMalloc1(xm*dof,&index_fast);CHKERRQ(ierr);
  ierr = PetscMalloc1(8*dof,&index_slowbuffer);CHKERRQ(ierr);
  ctx.lsbwidth = 4;
  ctx.rsbwidth = 4;

  for (i=xs; i<xs+xm; i++) {
    if (i < ctx.sf-ctx.lsbwidth || i > ctx.fs+ctx.rsbwidth-1)
      for (k=0; k<dof; k++) index_slow[islow++] = i*dof+k;
<<<<<<< HEAD
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
    else if ((i >= ctx.sf-ctx.lsbwidth && i < ctx.sf) || (i > ctx.fs-1 && i <= ctx.fs+ctx.rsbwidth-1))
      for (k=0; k<dof; k++) index_slowbuffer[islowbuffer++] = i*dof+k;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
    else
      for (k=0; k<dof; k++) index_fast[ifast++] = i*dof+k;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,islow,index_slow,PETSC_COPY_VALUES,&ctx.iss);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,ifast,index_fast,PETSC_COPY_VALUES,&ctx.isf);CHKERRQ(ierr);
<<<<<<< HEAD
<<<<<<< HEAD
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,islowbuffer,index_slowbuffer,PETSC_COPY_VALUES,&ctx.issb);CHKERRQ(ierr);
=======
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,islowbuffer,index_slowbuffer,PETSC_COPY_VALUES,&ctx.issb);CHKERRQ(ierr);
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.

  /* Create a time-stepping object */
  ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
<<<<<<< HEAD
<<<<<<< HEAD
  ierr = TSSetRHSFunction(ts,R,FVRHSFunction_2WaySplit,&ctx);CHKERRQ(ierr);
  ierr = TSRHSSplitSetIS(ts,"slow",ctx.iss);CHKERRQ(ierr);
  ierr = TSRHSSplitSetIS(ts,"slowbuffer",ctx.issb);CHKERRQ(ierr);
  ierr = TSRHSSplitSetIS(ts,"fast",ctx.isf);CHKERRQ(ierr);
  ierr = TSRHSSplitSetRHSFunction(ts,"slow",NULL,FVRHSFunctionslow_2WaySplit,&ctx);CHKERRQ(ierr);
  ierr = TSRHSSplitSetRHSFunction(ts,"fast",NULL,FVRHSFunctionfast_2WaySplit,&ctx);CHKERRQ(ierr);
  ierr = TSRHSSplitSetRHSFunction(ts,"slowbuffer",NULL,FVRHSFunctionslowbuffer_2WaySplit,&ctx);CHKERRQ(ierr);

  ierr = TSSetType(ts,TSMPRK);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,maxtime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  /* Compute initial conditions and starting time step */
  ierr = FVSample_2WaySplit(&ctx,da,0,X0);CHKERRQ(ierr);
  ierr = FVRHSFunction_2WaySplit(ts,0,X0,X,(void*)&ctx);CHKERRQ(ierr); /* Initial function evaluation, only used to determine max speed */
=======
  ierr = TSSetRHSFunction(ts,R,FVRHSFunction,&ctx);CHKERRQ(ierr);
=======
  ierr = TSSetRHSFunction(ts,R,FVRHSFunction_2WaySplit,&ctx);CHKERRQ(ierr);
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  ierr = TSRHSSplitSetIS(ts,"slow",ctx.iss);CHKERRQ(ierr);
  ierr = TSRHSSplitSetIS(ts,"slowbuffer",ctx.issb);CHKERRQ(ierr);
  ierr = TSRHSSplitSetIS(ts,"fast",ctx.isf);CHKERRQ(ierr);
  ierr = TSRHSSplitSetRHSFunction(ts,"slow",NULL,FVRHSFunctionslow_2WaySplit,&ctx);CHKERRQ(ierr);
  ierr = TSRHSSplitSetRHSFunction(ts,"fast",NULL,FVRHSFunctionfast_2WaySplit,&ctx);CHKERRQ(ierr);
  ierr = TSRHSSplitSetRHSFunction(ts,"slowbuffer",NULL,FVRHSFunctionslowbuffer_2WaySplit,&ctx);CHKERRQ(ierr);

  ierr = TSSetType(ts,TSMPRK);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,maxtime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  /* Compute initial conditions and starting time step */
<<<<<<< HEAD
  ierr = FVSample(&ctx,da,0,X0);CHKERRQ(ierr);
  ierr = FVRHSFunction(ts,0,X0,X,(void*)&ctx);CHKERRQ(ierr); /* Initial function evaluation, only used to determine max speed */
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
  ierr = FVSample_2WaySplit(&ctx,da,0,X0);CHKERRQ(ierr);
  ierr = FVRHSFunction_2WaySplit(ts,0,X0,X,(void*)&ctx);CHKERRQ(ierr); /* Initial function evaluation, only used to determine max speed */
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  ierr = VecCopy(X0,X);CHKERRQ(ierr);                        /* The function value was not used so we set X=X0 again */
  ierr = TSSetTimeStep(ts,ctx.cfl/ctx.cfl_idt);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr); /* Take runtime options */
  ierr = SolutionStatsView(da,X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  {
    PetscInt          steps;
    PetscScalar       mass_initial,mass_final,mass_difference,mass_differenceg;
    const PetscScalar *ptr_X,*ptr_X0;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    const PetscReal   hs = (ctx.xmax-ctx.xmin)*3.0/4.0/count_slow;
    const PetscReal   hf = (ctx.xmax-ctx.xmin)/4.0/count_fast;

=======
    const PetscReal   hs  = (ctx.xmax-ctx.xmin)/2.0*(ctx.hratio+1.0)/Mx;
    const PetscReal   hf  = (ctx.xmax-ctx.xmin)/2.0*(1.0+1.0/ctx.hratio)/Mx;
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
    const PetscReal   hs  = (ctx.xmax-ctx.xmin)/2.0/count_slow;
    const PetscReal   hf  = (ctx.xmax-ctx.xmin)/2.0/count_fast;
>>>>>>> Merged master and copied the fixed ex7 to ex4, runs in parallel now
=======
    const PetscReal   hs = (ctx.xmax-ctx.xmin)*3.0/4.0/count_slow;
    const PetscReal   hf = (ctx.xmax-ctx.xmin)/4.0/count_fast;

>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
    ierr = TSSolve(ts,X);CHKERRQ(ierr);
    ierr = TSGetSolveTime(ts,&ptime);CHKERRQ(ierr);
    ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
    /* calculate the total mass at initial time and final time */
    mass_initial = 0.0;
    mass_final   = 0.0;
    ierr = DMDAVecGetArrayRead(da,X0,(void*)&ptr_X0);CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(da,X,(void*)&ptr_X);CHKERRQ(ierr);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    for (i=xs;i<xs+xm;i++) {
      if (i < ctx.sf || i > ctx.fs-1) {
        for (k=0; k<dof; k++) {
          mass_initial = mass_initial + hs*ptr_X0[i*dof+k];
          mass_final = mass_final + hs*ptr_X[i*dof+k];
        }
      } else {
        for (k=0; k<dof; k++) {
          mass_initial = mass_initial + hf*ptr_X0[i*dof+k];
          mass_final = mass_final + hf*ptr_X[i*dof+k];
        }
=======
    for(i=0; i<Mx; i++) {
      if(i < count_slow/2 || i > count_slow/2+count_fast-1){
        mass_initial = mass_initial+hs*ptr_X0[i];
        mass_final = mass_final+hs*ptr_X[i];
      } else {
        mass_initial = mass_initial+hf*ptr_X0[i];
        mass_final = mass_final+hf*ptr_X[i];
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
    for(i=xs; i<xs+xm; i++) {
      if(i < ctx.sf || i > ctx.fs-1) {
=======
    for (i=xs;i<xs+xm;i++) {
      if (i < ctx.sf || i > ctx.fs-1) {
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
        for (k=0; k<dof; k++) {
          mass_initial = mass_initial + hs*ptr_X0[i*dof+k];
          mass_final = mass_final + hs*ptr_X[i*dof+k];
        }
      } else {
        for (k=0; k<dof; k++) {
          mass_initial = mass_initial + hf*ptr_X0[i*dof+k];
          mass_final = mass_final + hf*ptr_X[i*dof+k];
        }
>>>>>>> Merged master and copied the fixed ex7 to ex4, runs in parallel now
      }
    }
    ierr = DMDAVecRestoreArrayRead(da,X0,(void*)&ptr_X0);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(da,X,(void*)&ptr_X);CHKERRQ(ierr);
<<<<<<< HEAD
<<<<<<< HEAD
    mass_difference = mass_final - mass_initial;
<<<<<<< HEAD
    ierr = MPI_Allreduce(&mass_difference,&mass_differenceg,1,MPIU_SCALAR,MPIU_SUM,comm);CHKERRMPI(ierr);
=======
    ierr = MPI_Allreduce(&mass_difference,&mass_differenceg,1,MPIU_SCALAR,MPIU_SUM,comm);CHKERRQ(ierr);
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
    ierr = PetscPrintf(comm,"Mass difference %g\n",(double)mass_differenceg);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Final time %g, steps %D\n",(double)ptime,steps);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Maximum allowable stepsize according to CFL %g\n",(double)(1.0/ctx.cfl_idt));CHKERRQ(ierr);
    if (ctx.exact) {
      PetscReal nrm1=0;
      ierr = SolutionErrorNorms_2WaySplit(&ctx,da,ptime,X,&nrm1);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"Error ||x-x_e||_1 %g\n",(double)nrm1);CHKERRQ(ierr);
    }
    if (ctx.simulation) {
      PetscReal    nrm1=0;
      PetscViewer  fd;
      char         filename[PETSC_MAX_PATH_LEN] = "binaryoutput";
      Vec          XR;
      PetscBool    flg;
      const PetscScalar  *ptr_XR;
=======
    mass_difference = mass_final-mass_initial;
=======
    mass_difference = mass_final - mass_initial;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
    ierr = MPI_Allreduce(&mass_difference,&mass_differenceg,1,MPIU_SCALAR,MPIU_SUM,comm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Mass difference %g\n",(double)mass_differenceg);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Final time %g, steps %D\n",(double)ptime,steps);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Maximum allowable stepsize according to CFL %g\n",(double)(1.0/ctx.cfl_idt));CHKERRQ(ierr);
    if (ctx.exact) {
      PetscReal nrm1=0;
      ierr = SolutionErrorNorms_2WaySplit(&ctx,da,ptime,X,&nrm1);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"Error ||x-x_e||_1 %g\n",(double)nrm1);CHKERRQ(ierr);
    }
    if (ctx.simulation) {
<<<<<<< HEAD
      PetscReal         nrm1 = 0;
      PetscViewer       fd;
      char              filename[PETSC_MAX_PATH_LEN] = "binaryoutput";
      Vec               XR;
      PetscBool         flg;
      const PetscScalar *ptr_XR;
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
      PetscReal    nrm1=0;
      PetscViewer  fd;
      char         filename[PETSC_MAX_PATH_LEN] = "binaryoutput";
      Vec          XR;
      PetscBool    flg;
      const PetscScalar  *ptr_XR;
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
      ierr = PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&fd);CHKERRQ(ierr);
      ierr = VecDuplicate(X0,&XR);CHKERRQ(ierr);
      ierr = VecLoad(XR,fd);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
      ierr = VecGetArrayRead(X,&ptr_X);CHKERRQ(ierr);
      ierr = VecGetArrayRead(XR,&ptr_XR);CHKERRQ(ierr);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
      for (i=xs;i<xs+xm;i++) {
        if (i < ctx.sf || i > ctx.fs-1)
=======
      for(i=xs;i<xs+xm;i++) {
        if(i < ctx.sf || i > ctx.fs-1)
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
          for (k=0; k<dof; k++) nrm1 = nrm1 + hs*PetscAbs(ptr_X[i*dof+k]-ptr_XR[i*dof+k]);
        else
          for (k=0; k<dof; k++) nrm1 = nrm1 + hf*PetscAbs(ptr_X[i*dof+k]-ptr_XR[i*dof+k]);
=======
      for(i=0; i<Mx; i++) {
        if(i < count_slow/2 || i > count_slow/2+count_fast-1) nrm1 = nrm1 + hs*PetscAbs(ptr_X[i]-ptr_XR[i]);
        else nrm1 = nrm1 + hf*PetscAbs(ptr_X[i]-ptr_XR[i]);
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
      for(i=xs;i<xs+xm;i++) {
        if(i < ctx.sf || i > ctx.fs-1)
          for (k=0; k<dof; k++) nrm1 = nrm1 + hs*PetscAbs(ptr_X[i*dof+k]-ptr_XR[i*dof+k]);
        else
          for (k=0; k<dof; k++) nrm1 = nrm1 + hf*PetscAbs(ptr_X[i*dof+k]-ptr_XR[i*dof+k]);
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
      }
      ierr = VecRestoreArrayRead(X,&ptr_X);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(XR,&ptr_XR);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"Error ||x-x_e||_1 %g\n",(double)nrm1);CHKERRQ(ierr);
      ierr = VecDestroy(&XR);CHKERRQ(ierr);
    }
  }

  ierr = SolutionStatsView(da,X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
<<<<<<< HEAD
<<<<<<< HEAD
  if (draw & 0x1) {ierr = VecView(X0,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  if (draw & 0x2) {ierr = VecView(X,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  if (draw & 0x4) {
    Vec Y;
    ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);
    ierr = FVSample_2WaySplit(&ctx,da,ptime,Y);CHKERRQ(ierr);
=======
  if (draw & 0x1) { ierr = VecView(X0,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr); }
  if (draw & 0x2) { ierr = VecView(X,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr); }
  if (draw & 0x4) {
    Vec Y;
    ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);
    ierr = FVSample(&ctx,da,ptime,Y);CHKERRQ(ierr);
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
  if (draw & 0x1) {ierr = VecView(X0,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  if (draw & 0x2) {ierr = VecView(X,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  if (draw & 0x4) {
    Vec Y;
    ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);
    ierr = FVSample_2WaySplit(&ctx,da,ptime,Y);CHKERRQ(ierr);
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
    ierr = VecAYPX(Y,-1,X);CHKERRQ(ierr);
    ierr = VecView(Y,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    ierr = VecDestroy(&Y);CHKERRQ(ierr);
  }

  if (view_final) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,final_fname,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* Clean up */
<<<<<<< HEAD
<<<<<<< HEAD
  ierr = (*ctx.physics2.destroy)(ctx.physics2.user);CHKERRQ(ierr);
  for (i=0; i<ctx.physics2.dof; i++) {ierr = PetscFree(ctx.physics2.fieldname[i]);CHKERRQ(ierr);}
<<<<<<< HEAD
<<<<<<< HEAD
  ierr = PetscFree(ctx.physics2.bcinflowindex);CHKERRQ(ierr);
  ierr = PetscFree(ctx.ub);
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
  ierr = PetscFree(ctx.physics2.bcinflowindex);CHKERRQ(ierr);
  ierr = PetscFree(ctx.ub);
>>>>>>> Added Support for Inflow Boundary Conditions
  ierr = PetscFree4(ctx.R,ctx.Rinv,ctx.cjmpLR,ctx.cslope);CHKERRQ(ierr);
  ierr = PetscFree3(ctx.uLR,ctx.flux,ctx.speeds);CHKERRQ(ierr);
=======
  ierr = (*ctx.physics.destroy)(ctx.physics.user);CHKERRQ(ierr);
  for (i=0; i<ctx.physics.dof; i++) {ierr = PetscFree(ctx.physics.fieldname[i]);CHKERRQ(ierr);}
  ierr = PetscFree3(ctx.u,ctx.flux,ctx.speeds);CHKERRQ(ierr);
  ierr = ISDestroy(&ctx.iss);CHKERRQ(ierr);
  ierr = ISDestroy(&ctx.isf);CHKERRQ(ierr);
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
  ierr = (*ctx.physics2.destroy)(ctx.physics2.user);CHKERRQ(ierr);
  for (i=0; i<ctx.physics2.dof; i++) {ierr = PetscFree(ctx.physics2.fieldname[i]);CHKERRQ(ierr);}
  ierr = PetscFree(ctx.physics2.bcinflowindex);CHKERRQ(ierr);
  ierr = PetscFree(ctx.ub);
  ierr = PetscFree4(ctx.R,ctx.Rinv,ctx.cjmpLR,ctx.cslope);CHKERRQ(ierr);
  ierr = PetscFree3(ctx.uLR,ctx.flux,ctx.speeds);CHKERRQ(ierr);
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&X0);CHKERRQ(ierr);
  ierr = VecDestroy(&R);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
<<<<<<< HEAD
<<<<<<< HEAD
  ierr = ISDestroy(&ctx.iss);CHKERRQ(ierr);
  ierr = ISDestroy(&ctx.isf);CHKERRQ(ierr);
  ierr = ISDestroy(&ctx.issb);CHKERRQ(ierr);
  ierr = PetscFree(index_slow);CHKERRQ(ierr);
  ierr = PetscFree(index_fast);CHKERRQ(ierr);
  ierr = PetscFree(index_slowbuffer);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&limiters);CHKERRQ(ierr);
=======
  ierr = PetscFree(index_slow);CHKERRQ(ierr);
  ierr = PetscFree(index_fast);CHKERRQ(ierr);
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======
  ierr = ISDestroy(&ctx.iss);CHKERRQ(ierr);
  ierr = ISDestroy(&ctx.isf);CHKERRQ(ierr);
  ierr = ISDestroy(&ctx.issb);CHKERRQ(ierr);
  ierr = PetscFree(index_slow);CHKERRQ(ierr);
  ierr = PetscFree(index_fast);CHKERRQ(ierr);
  ierr = PetscFree(index_slowbuffer);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&limiters);CHKERRQ(ierr);
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
  ierr = PetscFunctionListDestroy(&physics);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    build:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
      requires: !complex !single
=======
      requires: !complex c99
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
      requires: !complex
>>>>>>> add test to src/ts/tutorials/multirate/ex4.c
=======
      requires: !complex 
>>>>>>> Added adaptive time stepping to fvnet
=======
      requires: !complex
>>>>>>> add missing output/ex4_1.out; small edits
      depends: finitevolume1d.c

    test:
      suffix: 1
      args: -da_grid_x 60 -initial 7 -xmin -1 -xmax 1 -hratio 2 -limit mc -ts_dt 0.025 -ts_max_steps 24 -ts_type mprk -ts_mprk_type 2a22
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
      output_file: output/ex4_1.out

    test:
      suffix: 2
      args: -da_grid_x 60 -initial 7 -xmin -1 -xmax 1 -hratio 2 -limit mc -ts_dt 0.025 -ts_max_steps 24 -ts_type mprk -ts_mprk_type 2a22 -ts_use_splitrhsfunction 1
      output_file: output/ex4_1.out

    test:
      suffix: 3
      args: -da_grid_x 40 -initial 1 -hratio 2 -limit mc -ts_dt 0.1 -ts_max_steps 24 -ts_max_time 7.0 -ts_type mprk -ts_mprk_type 2a22 -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
      output_file: output/ex4_3.out

    test:
      suffix: 4
      nsize: 2
      args: -da_grid_x 40 -initial 1 -hratio 2 -limit mc -ts_dt 0.1 -ts_max_steps 24 -ts_max_time 7.0 -ts_type mprk -ts_mprk_type 2a22 -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 1
      output_file: output/ex4_3.out

    test:
      suffix: 5
      nsize: 4
      args: args: -da_grid_x 40 -initial 1 -hratio 2 -limit mc -ts_dt 0.1 -ts_max_steps 24 -ts_max_time 7.0 -ts_type mprk -ts_mprk_type 2a22 -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 1
      output_file: output/ex4_3.out
=======
=======
      output_file: output/ex6_1.out
>>>>>>> add test to src/ts/tutorials/multirate/ex4.c
=======
      output_file: output/ex4_1.out
>>>>>>> Small fixes to ex4

    test:
      suffix: 2
      args: -da_grid_x 60 -initial 7 -xmin -1 -xmax 1 -hratio 2 -limit mc -ts_dt 0.025 -ts_max_steps 24 -ts_type mprk -ts_mprk_type 2a22 -ts_use_splitrhsfunction 1
      output_file: output/ex4_1.out

<<<<<<< HEAD
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
    test:
      suffix: 3
      args: -da_grid_x 40 -initial 1 -hratio 2 -limit mc -ts_dt 0.1 -ts_max_steps 24 -ts_max_time 7.0 -ts_type mprk -ts_mprk_type 2a22 -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
      output_file: output/ex4_3.out

    test:
      suffix: 4
      nsize: 2
      args: -da_grid_x 40 -initial 1 -hratio 2 -limit mc -ts_dt 0.1 -ts_max_steps 24 -ts_max_time 7.0 -ts_type mprk -ts_mprk_type 2a22 -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 1
      output_file: output/ex4_3.out

<<<<<<< HEAD
>>>>>>> add test to src/ts/tutorials/multirate/ex4.c
=======
    test: 
      suffix: 5
      nsize: 4
      args: args: -da_grid_x 40 -initial 1 -hratio 2 -limit mc -ts_dt 0.1 -ts_max_steps 24 -ts_max_time 7.0 -ts_type mprk -ts_mprk_type 2a22 -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 1
      output_file: output/ex4_3.out
>>>>>>> Added -np 4 test for ex4
=======
      requires: !complex c99
=======
      requires: !complex
>>>>>>> add test to src/ts/tutorials/multirate/ex4.c
=======
      requires: !complex 
>>>>>>> Added adaptive time stepping to fvnet
=======
      requires: !complex
>>>>>>> add missing output/ex4_1.out; small edits
      depends: finitevolume1d.c

    test:
      suffix: 1
      args: -da_grid_x 60 -initial 7 -xmin -1 -xmax 1 -hratio 2 -limit mc -ts_dt 0.025 -ts_max_steps 24 -ts_type mprk -ts_mprk_type 2a22
      output_file: output/ex4_1.out

    test:
      suffix: 2
      args: -da_grid_x 60 -initial 7 -xmin -1 -xmax 1 -hratio 2 -limit mc -ts_dt 0.025 -ts_max_steps 24 -ts_type mprk -ts_mprk_type 2a22 -ts_use_splitrhsfunction 1
      output_file: output/ex4_1.out

<<<<<<< HEAD
<<<<<<< HEAD
    test:
      suffix: 3
      args: -da_grid_x 60 -initial 7 -xmin -1 -xmax 1 -hratio 2 -ts_dt 0.025 -ts_max_steps 24 -ts_type mprk -ts_mprk_type 2a22 -ts_use_splitrhsfunction 0

    test:
      suffix: 4
      args: -da_grid_x 60 -initial 7 -xmin -1 -xmax 1 -hratio 2 -ts_dt 0.025 -ts_max_steps 24 -ts_type mprk -ts_mprk_type 2a22 -ts_use_splitrhsfunction 1
      output_file: output/ex7_3.out
<<<<<<< HEAD
>>>>>>> add src/ts/tutorials/multirate/ex4.c for a shallow water model
=======

    test:
      suffix: 5
      nsize: 2
      args: -da_grid_x 60 -initial 7 -xmin -1 -xmax 1 -hratio 2 -ts_dt 0.025 -ts_max_steps 24 -ts_type mprk -ts_mprk_type 2a22 -ts_use_splitrhsfunction 1
      output_file: output/ex7_3.out
>>>>>>> Merged master and copied the fixed ex7 to ex4, runs in parallel now
=======
>>>>>>> Modified .gitignore to include .vscode, added default hratio to ex6 (allows for an actual default run). Changed ex4 to be based on ex6.c (same problem with added slow buffer for better performance and slightly different mesh scaling, but both are of the form __slow__|___fast___|__slow__ ). Added initial support for shallow water equations (sourceless) to example 4. Still requires testing to verify  correctness.
=======
    test:
      suffix: 3
      args: -da_grid_x 40 -initial 1 -hratio 2 -limit mc -ts_dt 0.1 -ts_max_steps 24 -ts_max_time 7.0 -ts_type mprk -ts_mprk_type 2a22 -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 0
      output_file: output/ex4_3.out

    test:
      suffix: 4
      nsize: 2
      args: -da_grid_x 40 -initial 1 -hratio 2 -limit mc -ts_dt 0.1 -ts_max_steps 24 -ts_max_time 7.0 -ts_type mprk -ts_mprk_type 2a22 -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 1
      output_file: output/ex4_3.out

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> add test to src/ts/tutorials/multirate/ex4.c
=======
    test: 
=======
    test:
>>>>>>> rm white spaces
      suffix: 5
      nsize: 4
      args: args: -da_grid_x 40 -initial 1 -hratio 2 -limit mc -ts_dt 0.1 -ts_max_steps 24 -ts_max_time 7.0 -ts_type mprk -ts_mprk_type 2a22 -physics shallow -bc_type outflow -xmin 0 -xmax 50 -ts_use_splitrhsfunction 1
      output_file: output/ex4_3.out
>>>>>>> Added -np 4 test for ex4
TEST*/
