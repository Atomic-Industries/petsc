/* 
    Currently just a dumping ground for physics functions needed for the various tests. Namely flux functions 
    eigenvalues, characteristic decompositions, initial condition specifications, exact solutions, 
    network riemann solvers (to be removed as class is built for them specifically)
*/

#include "physics.h"

PETSC_STATIC_INLINE PetscReal MaxAbs(PetscReal a,PetscReal b) { return (PetscAbs(a) > PetscAbs(b)) ? a : b; }

/* --------------------------------- Shallow Water ----------------------------------- */
typedef struct {
  PetscReal gravity;
} ShallowCtx;

PETSC_STATIC_INLINE PetscErrorCode ShallowFlux(void *ctx,const PetscReal *u,PetscReal *f)
{
  ShallowCtx *phys = (ShallowCtx*)ctx;
  f[0] = u[1];
  f[1] = PetscSqr(u[1])/u[0] + 0.5*phys->gravity*PetscSqr(u[0]);
  PetscFunctionReturn(0);
}
PETSC_STATIC_INLINE void ShallowFluxVoid(void *ctx,const PetscReal *u,PetscReal *f)
{
  ShallowCtx *phys = (ShallowCtx*)ctx;
  f[0] = u[1];
  f[1] = PetscSqr(u[1])/u[0] + 0.5*phys->gravity*PetscSqr(u[0]);
}

PETSC_STATIC_INLINE void ShallowFlux2(ShallowCtx *phys,const PetscScalar *u,PetscScalar *f)
{
  f[0] = u[1]*u[0];
  f[1] = PetscSqr(u[1])*u[0] + 0.5*phys->gravity*PetscSqr(u[0]);
}

PETSC_STATIC_INLINE void ShallowEig(void *ctx,const PetscReal *u,PetscReal *eig)
{
    ShallowCtx *phys = (ShallowCtx*)ctx;
    eig[0] = u[1]/u[0] - PetscSqrtReal(phys->gravity*u[0]); /*left wave*/
    eig[1] = u[1]/u[0] + PetscSqrtReal(phys->gravity*u[0]); /*right wave*/
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

  /*simple positivity preserving limiter*/
  if (L.h < 0) L.h = 0;
  if (R.h < 0) R.h = 0;

  ShallowFlux2(phys,(PetscScalar*)&L,fL);
  ShallowFlux2(phys,(PetscScalar*)&R,fR);

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

  if (u[0] < tol) { /*Use conservative variables*/
    X[0*2+0]  = 1;
    X[0*2+1]  = 0;
    X[1*2+0]  = 0;
    X[1*2+1]  = 1;
    speeds[0] = - c;
    speeds[1] =   c;
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

static PetscErrorCode PhysicsCharacteristic_Shallow_Mat(void *vctx,const PetscScalar *u,Mat eigmat)
{
  ShallowCtx     *phys = (ShallowCtx*)vctx;
  PetscReal      c;
  PetscErrorCode ierr;
  PetscInt       m = 2,n = 2,i; 
  PetscReal      X[m][n];
  PetscInt       idxm[m],idxn[n]; 
  

  PetscFunctionBeginUser;
  c         = PetscSqrtScalar(u[0]*phys->gravity);

  for (i=0; i<m; i++) idxm[i] = i; 
  for (i=0; i<n; i++) idxn[i] = i; 
  /* Analytical formulation for the eigen basis of the Df for at u */
  X[0][0]  = 1;
  X[1][0]  = u[1]/u[0] - c;
  X[0][1]  = 1;
  X[1][1]  = u[1]/u[0] + c;
  ierr = MatSetValues(eigmat,m,idxm,n,idxn,(PetscReal *)X,INSERT_VALUES);CHKERRQ(ierr);
  MatAssemblyBegin(eigmat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(eigmat,MAT_FINAL_ASSEMBLY);
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsFluxDer_Shallow(void *vctx,const PetscReal *u,Mat jacobian)
{
  ShallowCtx     *phys = (ShallowCtx*)vctx;
  PetscErrorCode ierr;
  PetscInt       m = 2,n = 2,i; 
  PetscReal      X[m][n];
  PetscInt       idxm[m],idxn[n]; 
  

  PetscFunctionBeginUser;
  for (i=0; i<m; i++) idxm[i] = i; 
  for (i=0; i<n; i++) idxn[i] = i; 
  /* Analytical formulation for Df at u */
  X[0][0]  = 0.;
  X[1][0]  = - PetscSqr(u[1])/PetscSqr(u[0]) + phys->gravity*u[0];
  X[0][1]  = 1.;
  X[1][1]  = 2.*u[1]/u[0];
  ierr = MatSetValues(jacobian,m,idxm,n,idxn,(PetscReal *)X,INSERT_VALUES);CHKERRQ(ierr);
  MatAssemblyBegin(jacobian,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(jacobian,MAT_FINAL_ASSEMBLY);
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsRoeAvg_Shallow(void *ctx,const PetscReal *uL,const PetscReal *uR,PetscReal *uavg) 
{
  PetscFunctionBeginUser;
  uavg[0] = (uL[0]+uR[0])/2.0; 
  uavg[1] = (PetscSqrtReal(uL[0])*uL[1]+PetscSqrtReal(uR[0])*uR[1])/(PetscSqrtReal(uL[0])+PetscSqrtReal(uR[0]));
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsSample_ShallowNetwork(void *vctx,PetscInt initial,PetscReal t,PetscReal x,PetscReal *u,PetscInt edgeid)
{
  PetscFunctionBeginUser;
  if (t > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exact solutions not implemented for t > 0");
  switch (initial) {
    case 0:
      if (edgeid == 0) {
        u[0] = 50.0;
        u[1] = 0.0;
      }  else {
        u[0] = 1;
        u[1] = 0.0;
      }
      break;
    case 1: /* Initial 1-3 are from Jingmei's and Bennedito's paper */
      if (edgeid == 0) {
        u[0] = 0.5;
        u[1] = 0.1;
      } else if (edgeid == 1) {
        u[0] = 0.5;
        u[1] = 0.0;
      } else  {
        u[0] = 1;
        u[1] = 0.0;
      }
      break;
    case 2:
      if (edgeid == 0) {
        u[0] = 1.0+PetscExpReal(-20.0*(x+1.0)*(x+1.0));
        u[1] = u[0]/2.0;
      } else if (edgeid == 1) {
        u[0] = 1.0;
        u[1] = 0.0;
      } else  {
        u[0] = 0.5;
        u[1] = 0.0;
      }
      break;
    case 3:
      if (edgeid == 0) {
        u[0] = ((x>=0 && x<=0.2) || (x>=0.4 && x<=0.6) || (x>=0.8 && x<=1.0)) ? 1.5 : 1.0 ;
        u[1] = u[0]/5.0;
      } else if (edgeid == 1) {
        u[0] = 1.0;
        u[1] = 0.0;
      } else {
        u[0] = 0.5;
        u[1] = 0.0;
      }
      break;
    case 4: /* Sunny's Test Case*/
      if (edgeid == 0) { /* Not sure what the correct IC is here*/
          u[0] = ((x>=7 && x<=9)) ? 2.0-PetscSqr(x-8): 1.0; 
          u[1] = 0.0;
      } else {
          u[0] = 1.0; 
          u[1] = 0.0;
       }
      break;
    case 5: /* Roundabout Pulse */
      u[0] = !(edgeid%2) ? 2 : 1; 
      u[1] = 0;
      break;
 /* The following problems are based on geoemtrically 1d Networks, no notion of edgeid is considered */
    case 6:
      u[0] = (x < 10) ?   1 : 0.1; 
      u[1] = (x < 10) ? 2.5 : 0;
      break;
    case 7:
      u[0] = (x < 25) ?  1 : 1;
      u[1] = (x < 25) ? -5 : 5;
      break;
    case 8:
      u[0] = (x < 20) ?  1 : 0;
      u[1] = (x < 20) ?  0 : 0;
      break;
    case 9:
      u[0] = (x < 30) ? 0: 1;
      u[1] = (x < 30) ? 0 : 0;
      break;
    case 10:
      u[0] = (x < 25) ?  0.1 : 0.1;
      u[1] = (x < 25) ? -0.3 : 0.3;
      break;
    case 11:
      u[0] = 1+0.5*PetscSinReal(2*PETSC_PI*x);
      u[1] = 1*u[0];
      break;
    case 12:
      u[0] = 1.0;
      u[1] = 1.0;
      break;
    case 13:
      u[0] = (x < -2) ? 2 : 1; /* Standard Dam Break Problem */
      u[1] = (x < -2) ? 0 : 0;
      break;
    case 14:
      u[0] = (x < 25) ? 2 : 1; /* Standard Dam Break Problem */
      u[1] = (x < 25) ? 0 : 0;
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}

/*2 edge vertex flux for edge 1 pointing in and edge 2 pointing out */
static PetscErrorCode PhysicsVertexFlux_2Edge_InOut(const void* _dgnet,const PetscScalar *uV,const PetscBool *dir,PetscScalar *flux,PetscScalar *maxspeed,const void* _junct)
{
  PetscErrorCode  ierr;
  const DGNetwork fvnet = (DGNetwork)_dgnet;
  PetscInt        i,dof = fvnet->physics.dof;

  PetscFunctionBeginUser;
  /* First edge interpreted as uL, 2nd as uR. Use the user inputted Riemann function. */
  ierr = fvnet->physics.riemann(fvnet->physics.user,dof,uV,uV+dof,flux,maxspeed);CHKERRQ(ierr);
  /* Copy the flux */
  for (i = 0; i<dof; i++) {
    flux[i+dof] = flux[i];
  }
  PetscFunctionReturn(0);
}

/*2 edge vertex flux for edge 1 pointing out and edge 2 pointing in  */
static PetscErrorCode PhysicsVertexFlux_2Edge_OutIn(const void* _dgnet,const PetscScalar *uV,const PetscBool *dir,PetscScalar *flux,PetscScalar *maxspeed,const void* _junct)
{
  PetscErrorCode  ierr;
  const DGNetwork fvnet = (DGNetwork)_dgnet;
  PetscInt        i,dof = fvnet->physics.dof;

  PetscFunctionBeginUser;
  /* First edge interpreted as uR, 2nd as uL. Use the user inputted Riemann function. */
  ierr = fvnet->physics.riemann(fvnet->physics.user,dof,uV+dof,uV,flux,maxspeed);CHKERRQ(ierr);
  /* Copy the flux */
  for (i = 0; i<dof; i++) {
    flux[i+dof] = flux[i];
  }
  PetscFunctionReturn(0);
}

static PetscReal ShallowRiemannExact_Left(const PetscScalar hl, const PetscScalar vl, PetscScalar h)
{
  const PetscScalar g = 9.81;
  return h<hl ? vl-2.0*(PetscSqrtScalar(g*h)-PetscSqrtScalar(g*hl)) : vl- (h-hl)*PetscSqrtScalar(g*(h+hl)/(2.0*h*hl));
}

static PetscReal ShallowRiemannExact_Right(const PetscScalar hr, const PetscScalar vr, PetscScalar h)
{
  const PetscScalar g = 9.81;
  return h<hr ? vr+2.0*(PetscSqrtScalar(g*h)-PetscSqrtScalar(g*hr)) : vr+(h-hr)*PetscSqrtScalar(g*(h+hr)/(2.0*h*hr));
}

static PetscReal ShallowRiemannEig_Right(const PetscScalar hr, const PetscScalar vr)
{
  const PetscScalar g = 9.81;
  return vr + PetscSqrtScalar(g*hr);
}

static PetscReal ShallowRiemannEig_Left(const PetscScalar hl, const PetscScalar vl)
{
  const PetscScalar g = 9.81;
  return vl - PetscSqrtScalar(g*hl);
}

typedef struct {
  Junction          junct;
  const PetscScalar *uV;
} Shallow_Couple_InputWrapper;

static PetscErrorCode RiemannInvariant_Couple_Shallow(SNES snes,Vec x,Vec f, void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i,n,dof=2;
  Shallow_Couple_InputWrapper *wrapper= (Shallow_Couple_InputWrapper*) ctx;
  const Junction  junct = wrapper->junct;
  const PetscScalar *ustar,*uV = wrapper->uV;
  PetscScalar *F;

  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&ustar);CHKERRQ(ierr);
  ierr = VecGetArray(f,&F);CHKERRQ(ierr);

  F[n-2] = (junct->dir[junct->numedges-1] == EDGEIN) ?  ustar[n-1] : -ustar[n-1];

  for (i=0; i<junct->numedges-1; i++)
  {
    F[dof*i] = ustar[dof*i] - ustar[dof*(i+1)];
    F[n-2] += (junct->dir[i] == EDGEIN) ? ustar[dof*i+1] : -ustar[dof*i+1];
    /* output for this edges vstar eqn */
    if (junct->dir[i] == EDGEIN){
      F[dof*i+1] = ustar[dof*i+1] - ShallowRiemannExact_Left(uV[dof*i],uV[dof*i+1]/uV[dof*i],ustar[dof*i]);
    } else { /* junct->dir[i] == EDGEOUT */
      F[dof*i+1] = ustar[dof*i+1] - ShallowRiemannExact_Right(uV[dof*i],uV[dof*i+1]/uV[dof*i],ustar[dof*i]);
    }
  }

   if (junct->dir[junct->numedges-1] == EDGEIN){
      F[n-1] = ustar[n-1] - ShallowRiemannExact_Left(uV[n-2],uV[n-1]/uV[n-2],ustar[n-2]);
    } else { /* junct->dir[i] == EDGEOUT */
      F[n-1] = ustar[n-1] - ShallowRiemannExact_Right(uV[n-2],uV[n-1]/uV[n-2],ustar[n-2]);
    }

  ierr = VecRestoreArrayRead(x,&ustar);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsVertexFlux_Shallow_Full(const void* _dgnet,const PetscScalar *uV,const PetscBool *dir,PetscScalar *flux,PetscScalar *maxspeed,const void* _junct)
{
  PetscErrorCode  ierr;
  const DGNetwork fvnet = (DGNetwork)_dgnet;
  const Junction  junct = (Junction) _junct;
  Shallow_Couple_InputWrapper wrapper = {junct,uV};
  PetscInt        i,n,dof = fvnet->physics.dof;
  PetscScalar     *x;


  PetscFunctionBeginUser;

  ierr = SNESSetFunction(fvnet->snes,junct->rcouple,RiemannInvariant_Couple_Shallow,&wrapper);
  /* Set initial condition as the reconstructed h,v values*/
  ierr = VecGetArray(junct->xcouple,&x);CHKERRQ(ierr);
  ierr = VecGetSize(junct->xcouple,&n);CHKERRQ(ierr);
  for (i=0;i<junct->numedges;i++) {
    x[i*dof]   = uV[i*dof];
    x[i*dof+1] = uV[i*dof+1]/uV[i*dof];
  }
  ierr = VecRestoreArray(junct->xcouple,&x);CHKERRQ(ierr);
  ierr = SNESSolve(fvnet->snes,NULL,junct->xcouple);CHKERRQ(ierr);
  ierr = VecGetArray(junct->xcouple,&x);CHKERRQ(ierr);
  /* Compute the Flux from the computed star values */
  for (i=0;i<junct->numedges;i++) {
    flux[i*dof] = x[i*dof];
  }
  /* Compute the Flux from the computed star values */
  for (i=0;i<junct->numedges;i++) {
    flux[i*dof+1] = x[i*dof+1]*x[i*dof];
  }
  ierr = VecRestoreArray(junct->xcouple,&x);CHKERRQ(ierr);
  *maxspeed = 0.0; /* Ignore the computation of the maxspeed */
  PetscFunctionReturn(0);
}
static PetscErrorCode PhysicsVertexFlux_Shallow_Full_Linear(const void* _dgnet,const PetscScalar *uV,const PetscBool *dir,PetscScalar *flux,PetscScalar *maxspeed,const void* _junct)
{
  PetscErrorCode  ierr;
  const DGNetwork fvnet = (DGNetwork)_dgnet;
  const Junction  junct = (Junction) _junct;
  PetscInt        i,dof = fvnet->physics.dof;
  PetscScalar     *x,*r,eig,h,v,hv;
  PetscBool       nonzeroinitial; 

  PetscFunctionBeginUser;

  ierr = VecGetArray(junct->rcouple,&r);CHKERRQ(ierr);
  /* Build the system matrix and rhs vector */
  for (i=1;i<junct->numedges;i++) {
    h  = uV[i*dof];
    hv = uV[i*dof+1];
    v  = hv/h; 
    eig = junct->dir[i] == EDGEIN ? 
      ShallowRiemannEig_Left(h,v) : ShallowRiemannEig_Right(h,v);
    ierr = MatSetValue(junct->mat,i+1,i+1,-1 ,INSERT_VALUES);CHKERRQ(ierr); /* hv* term */
    ierr = MatSetValue(junct->mat,i+1,0,eig,INSERT_VALUES);CHKERRQ(ierr);      /* h* term */
    ierr = MatSetValue(junct->mat,1,i+1,junct->dir[i] == EDGEIN ? 1:-1,INSERT_VALUES);CHKERRQ(ierr);  
    r[i+1] = eig*h-hv; /* riemann invariant at the boundary */
  }
  /* Form the matrix in this reordered form to have nonzeros along the diagonal */
  h  = uV[0];
  hv = uV[1];
  v  = hv/h; 
  eig = junct->dir[0] == EDGEIN ? 
    ShallowRiemannEig_Left(h,v) : ShallowRiemannEig_Right(h,v);
  ierr = MatSetValue(junct->mat,0,1,-1 ,INSERT_VALUES);CHKERRQ(ierr); /* hv* term */
  ierr = MatSetValue(junct->mat,0,0,eig,INSERT_VALUES);CHKERRQ(ierr);      /* h* term */
  ierr = MatSetValue(junct->mat,1,1,junct->dir[0] == EDGEIN ? 1:-1,INSERT_VALUES);CHKERRQ(ierr);  
  r[0] = eig*h-hv; /* riemann invariant at the boundary */
  r[1] = 0.0;   
  ierr = VecRestoreArray(junct->rcouple,&r);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(junct->mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(junct->mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = KSPGetInitialGuessNonzero(fvnet->ksp,&nonzeroinitial);CHKERRQ(ierr);
  if(nonzeroinitial) {
    /* Set initial guess as the reconstructed h,v values */
    ierr = VecGetArray(junct->xcouple,&x);CHKERRQ(ierr);
    for (i=0;i<junct->numedges;i++) {
      x[i+1] = uV[i*dof+1];
    }
    x[0] = uV[0];
    ierr = VecRestoreArray(junct->xcouple,&x);CHKERRQ(ierr);
  }

  ierr = KSPSetOperators(fvnet->ksp,junct->mat,junct->mat);CHKERRQ(ierr);
  ierr = KSPSolve(fvnet->ksp,junct->rcouple,junct->xcouple);CHKERRQ(ierr);
  ierr = VecGetArray(junct->xcouple,&x);CHKERRQ(ierr);
  /* Compute the Flux from the computed star values */
  for (i=0;i<junct->numedges;i++) {
    flux[i*dof+1] = x[i+1];
    flux[i*dof]   = x[0];
  }
  ierr = VecRestoreArray(junct->xcouple,&x);CHKERRQ(ierr);
  *maxspeed = 0.0; /* Ignore the computation of the maxspeed */
  PetscFunctionReturn(0);
}
static PetscErrorCode PhysicsVertexFlux_Outflow_Simple(const void* _dgnet,const PetscScalar *uV,const PetscBool *dir,PetscScalar *flux,PetscScalar *maxspeed,const void* _junct) {
  PetscErrorCode  ierr;
  const DGNetwork dgnet = (DGNetwork)_dgnet;
  PetscInt        dof = dgnet->physics.dof;

  PetscFunctionBeginUser;
  *maxspeed = 0.0; 
  ierr = dgnet->physics.riemann(dgnet->physics.user,dof,uV,uV,flux,maxspeed);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode PhysicsAssignVertexFlux_Shallow(const void* _dgnet, Junction junct)
{
  PetscErrorCode  ierr;
  const DGNetwork fvnet = (DGNetwork)_dgnet;
  PetscInt        dof = 2;

  PetscFunctionBeginUser;
      if (junct->numedges == 2) {
        if (junct->dir[0] == EDGEIN) {
          if (junct->dir[1] == EDGEIN) {
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a valid directed graph for the current discretization method");
          } else { /* dir[1] == EDGEOUT */
            junct->couplingflux = PhysicsVertexFlux_2Edge_InOut;
          }
        } else { /* dir[0] == EDGEOUT */
          if (junct->dir[1] == EDGEIN) {
            junct->couplingflux = PhysicsVertexFlux_2Edge_OutIn;
          } else { /* dir[1] == EDGEOUT */
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a valid directed graph for the current discretization method");
          }
        }
      } else if (junct->numedges == 1) {
         if (junct->dir[0] == EDGEIN) {
            junct->couplingflux =   junct->couplingflux = PhysicsVertexFlux_Outflow_Simple;
         } else {
            junct->couplingflux =   junct->couplingflux = PhysicsVertexFlux_Outflow_Simple;
         }
      } else {
        if(!fvnet->linearcoupling) {
          ierr = VecCreateSeq(MPI_COMM_SELF,junct->numedges*dof,&junct->rcouple);CHKERRQ(ierr);
          ierr = VecDuplicate(junct->rcouple,&junct->xcouple);CHKERRQ(ierr);
          junct->couplingflux = PhysicsVertexFlux_Shallow_Full;
        } else {
          ierr = VecCreateSeq(MPI_COMM_SELF,junct->numedges+1,&junct->rcouple);CHKERRQ(ierr);
          ierr = VecDuplicate(junct->rcouple,&junct->xcouple);CHKERRQ(ierr);
          ierr = MatCreate(MPI_COMM_SELF,&junct->mat);CHKERRQ(ierr);
          ierr = MatSetSizes(junct->mat,PETSC_DECIDE,PETSC_DECIDE,junct->numedges+1,junct->numedges+1);CHKERRQ(ierr);
          ierr = MatSetFromOptions(junct->mat);CHKERRQ(ierr); 
          ierr = MatSetUp(junct->mat);CHKERRQ(ierr); /* Could use a specific create seq mat here for improved performance */
          junct->couplingflux = PhysicsVertexFlux_Shallow_Full_Linear;
        }
      }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsDestroyVertexFlux_Shallow(const void* _fvnet, Junction junct)
{
  PetscErrorCode  ierr;
  const DGNetwork fvnet = (DGNetwork)_fvnet;

  PetscFunctionBeginUser;
      if (junct->numedges == 2) {
        // Nothing to Destroy 
      } else {
        if(!fvnet->linearcoupling) {
          ierr = VecDestroy(&junct->rcouple);CHKERRQ(ierr);
          ierr = VecDestroy(&junct->xcouple);CHKERRQ(ierr);
        } else {
          ierr = VecDestroy(&junct->rcouple);CHKERRQ(ierr);
          ierr = VecDestroy(&junct->xcouple);CHKERRQ(ierr);
          ierr = MatDestroy(&junct->mat);CHKERRQ(ierr);
        }
      }
  PetscFunctionReturn(0);
}

PetscErrorCode PhysicsCreate_Shallow(DGNetwork fvnet)
{
  PetscErrorCode    ierr;
  ShallowCtx        *user;
  PetscFunctionList rlist = 0;
  char              rname[256] = "rusanov";
  PetscFunctionBeginUser;
  ierr = PetscNew(&user);CHKERRQ(ierr);
  fvnet->physics.samplenetwork   = PhysicsSample_ShallowNetwork;
  fvnet->physics.destroy         = PhysicsDestroy_SimpleFree_Net;
  fvnet->physics.riemann         = PhysicsRiemann_Shallow_Rusanov;
  fvnet->physics.characteristic  = PhysicsCharacteristic_Shallow;
  fvnet->physics.vfluxassign     = PhysicsAssignVertexFlux_Shallow;
  fvnet->physics.vfluxdestroy    = PhysicsDestroyVertexFlux_Shallow;
  fvnet->physics.flux            = ShallowFlux;
  fvnet->physics.user            = user;
  fvnet->physics.dof             = 2;
  fvnet->physics.flux2           = ShallowFluxVoid; 
  fvnet->physics.fluxeig         = ShallowEig;
  fvnet->physics.roeavg          = PhysicsRoeAvg_Shallow; 
  fvnet->physics.eigbasis        = PhysicsCharacteristic_Shallow_Mat;
  fvnet->physics.fluxder         = PhysicsFluxDer_Shallow;
  ierr = PetscStrallocpy("height",&fvnet->physics.fieldname[0]);CHKERRQ(ierr);
  ierr = PetscStrallocpy("momentum",&fvnet->physics.fieldname[1]);CHKERRQ(ierr);
  user->gravity = 9.81;

  ierr = PetscMalloc2(2,&fvnet->physics.lowbound,2,&fvnet->physics.upbound);CHKERRQ(ierr);
  fvnet->physics.lowbound[0] = 0;   fvnet->physics.lowbound[1] = -100; 
  fvnet->physics.upbound[0] = 100;   fvnet->physics.upbound[1] = 100; 

  ierr = RiemannListAdd_Net(&rlist,"rusanov",PhysicsRiemann_Shallow_Rusanov);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(fvnet->comm,fvnet->prefix,"Options for Shallow","");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-physics_shallow_riemann","Riemann solver","",rlist,rname,rname,sizeof(rname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = RiemannListFind_Net(rlist,rname,&fvnet->physics.riemann);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&rlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------- Traffic ----------------------------------- */

typedef struct {
  PetscReal a;
} TrafficCtx;

PETSC_STATIC_INLINE PetscScalar TrafficFlux(PetscScalar a,PetscScalar u) { return a*u*(1-u); }
PETSC_STATIC_INLINE PetscScalar TrafficChar(PetscScalar a,PetscScalar u) { return a*(1-2*u); }
PETSC_STATIC_INLINE PetscErrorCode TrafficFlux2(void *ctx,const PetscReal *u,PetscReal *f) {
  TrafficCtx *phys = (TrafficCtx*)ctx;
  f[0] = phys->a * u[0]*(1. - u[0]);
  PetscFunctionReturn(0);
}
PETSC_STATIC_INLINE void TrafficFluxVoid(void *ctx,const PetscReal *u,PetscReal *f) {
  TrafficCtx *phys = (TrafficCtx*)ctx;
  f[0] = phys->a * u[0]*(1. - u[0]);
}
static PetscErrorCode PhysicsRiemann_Traffic_Exact(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  PetscReal a = ((TrafficCtx*)vctx)->a;

  PetscFunctionBeginUser;
  if (uL[0] < uR[0]) {
    flux[0] = PetscMin(TrafficFlux(a,uL[0]),TrafficFlux(a,uR[0]));
  } else {
    flux[0] = (uR[0] < 0.5 && 0.5 < uL[0]) ? TrafficFlux(a,0.5) : PetscMax(TrafficFlux(a,uL[0]),TrafficFlux(a,uR[0]));
  }
  *maxspeed = a*MaxAbs(1-2*uL[0],1-2*uR[0]);
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsRiemann_Traffic_Roe(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  PetscReal a = ((TrafficCtx*)vctx)->a;
  PetscReal speed;

  PetscFunctionBeginUser;
  speed = a*(1 - (uL[0] + uR[0]));
  flux[0] = 0.5*(TrafficFlux(a,uL[0]) + TrafficFlux(a,uR[0])) - 0.5*PetscAbs(speed)*(uR[0]-uL[0]);
  *maxspeed = speed;
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsRiemann_Traffic_Rusanov(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  PetscReal a = ((TrafficCtx*)vctx)->a;
  PetscReal speed;

  PetscFunctionBeginUser;
  speed     = a*PetscMax(PetscAbs(1-2*uL[0]),PetscAbs(1-2*uR[0]));
  flux[0]   = 0.5*(TrafficFlux(a,uL[0]) + TrafficFlux(a,uR[0])) - 0.5*speed*(uR[0]-uL[0]);
  *maxspeed = speed;
  PetscFunctionReturn(0);
}

static void TrafficEig(void *ctx,const PetscReal *u,PetscScalar *eig) 
{
    PetscReal a = ((TrafficCtx*)ctx)->a;

    eig[0]      = TrafficChar(a,u[0]);
}

static PetscErrorCode PhysicsAssignVertexFlux_Traffic(const void* _fvnet, Junction junct)
{  
  PetscFunctionBeginUser;
      if (junct->numedges == 2) {
        if (junct->dir[0] == EDGEIN) {
          if (junct->dir[1] == EDGEIN) {
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a valid directed graph for the current discretization method");
          } else { /* dir[1] == EDGEOUT */
            junct->couplingflux = PhysicsVertexFlux_2Edge_InOut;
          }
        } else { /* dir[0] == EDGEOUT */
          if (junct->dir[1] == EDGEIN) {
            junct->couplingflux = PhysicsVertexFlux_2Edge_OutIn;
          } else { /* dir[1] == EDGEOUT */
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a valid directed graph for the current discretization method");
          }
        }
      } else {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"General coupling conditions are not yet implemented for traffic models");
      }
  PetscFunctionReturn(0);
}
typedef struct {
  PetscReal a,x,t;
} MethodCharCtx; 
/* TODO Generalize to arbitrary initial value */
static  PetscErrorCode TrafficCase1Char(SNES snes,Vec X,Vec f, void *ctx) {
  PetscReal      x,t,rhs,a;
  const PetscScalar *s; 
  PetscErrorCode ierr; 
  
  PetscFunctionBeginUser;
  x = ((MethodCharCtx*)ctx)->x;
  t = ((MethodCharCtx*)ctx)->t;
  a = ((MethodCharCtx*)ctx)->a;

  ierr = VecGetArrayRead(X,&s);CHKERRQ(ierr);
  rhs  = TrafficChar(a,PetscSinReal(PETSC_PI*(s[0]/5.0))+2)*t +s[0]-x; 
  ierr = VecSetValue(f,0,rhs,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* TODO Generalize to arbitrary initial value */
static  PetscErrorCode TrafficCase1Char_J(SNES snes,Vec X,Mat Amat,Mat Pmat, void *ctx) {
  PetscReal      x,t,rhs,a;
  const PetscScalar *s; 
  PetscErrorCode ierr; 
  
  PetscFunctionBeginUser;
  x = ((MethodCharCtx*)ctx)->x;
  t = ((MethodCharCtx*)ctx)->t;
  a = ((MethodCharCtx*)ctx)->a;

  ierr = VecGetArrayRead(X,&s);CHKERRQ(ierr);
  rhs = 1.0- t*a*2.0*PETSC_PI/5.0*PetscCosReal(PETSC_PI*(s[0]/5.0)); 
  ierr = MatSetValue(Pmat,0,0,rhs,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&s);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Amat != Pmat) {
    ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
static PetscErrorCode PhysicsSample_TrafficNetwork(void *vctx,PetscInt initial,PetscReal t,PetscReal x,PetscReal *u,PetscInt edgeid)
{
  SNES           snes; 
  Mat            J;
  Vec            X,R;
  PetscErrorCode ierr; 
  PetscReal      *s;
  MethodCharCtx  ctx;  

  PetscFunctionBeginUser;
  if (t<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"t must be >= 0 ");
  switch (initial) {
    case 0:
        if (t > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exact solution for case 0 not implemented for t > 0");
      if(edgeid == 0) {
        u[0] = (x < -2) ? 2 : 1; /* Traffic Break problem ?*/
      } else {
        u[0] = 1; 
      }
      break;
    case 1:
      if (t==0.0) {
        if(edgeid == 0) {
          u[0] = PetscSinReal(PETSC_PI*(x/5.0))+2;
        } else if(edgeid == 1) {
          u[0] = PetscSinReal(PETSC_PI*(-x/5.0))+2;
        } else {
          u[0] = 0; 
        }
      } else {
          /* Method of characteristics to solve for exact solution */
          ctx.t =t; ctx.a = 0.5;
          ctx.x = !edgeid ? x : -x; 
          ierr = VecCreate(PETSC_COMM_SELF,&X);CHKERRQ(ierr);
          ierr = VecSetSizes(X,PETSC_DECIDE,1);CHKERRQ(ierr);
          ierr = VecSetFromOptions(X);CHKERRQ(ierr);
          ierr = VecDuplicate(X,&R);CHKERRQ(ierr);
          ierr = MatCreate(PETSC_COMM_SELF,&J);CHKERRQ(ierr);
          ierr = MatSetSizes(J,1,1,1,1);CHKERRQ(ierr);
          ierr = MatSetFromOptions(J);CHKERRQ(ierr);
          ierr = MatSetUp(J);CHKERRQ(ierr);
          ierr = SNESCreate(PETSC_COMM_SELF,&snes);CHKERRQ(ierr);
          ierr = SNESSetFunction(snes,R,TrafficCase1Char,&ctx);
          ierr = SNESSetJacobian(snes,J,J,TrafficCase1Char_J,&ctx);
          ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
          ierr = VecSet(X,x);CHKERRQ(ierr);
          ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
          ierr = PetscMalloc1(1,&s);CHKERRQ(ierr);
          ierr = VecGetArray(X,&s);CHKERRQ(ierr);
      
          u[0] = PetscSinReal(PETSC_PI*(s[0]/5.0))+2;
         
          ierr = VecRestoreArray(X,&s);CHKERRQ(ierr);
          ierr = VecDestroy(&X);CHKERRQ(ierr);
          ierr = VecDestroy(&R);CHKERRQ(ierr);
          ierr = MatDestroy(&J);CHKERRQ(ierr);
          ierr = SNESDestroy(&snes);CHKERRQ(ierr);
          ierr = PetscFree(s); 
      } 
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}
static PetscErrorCode PhysicsDestroyVertexFlux(const void* _fvnet, Junction junct)
{
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
          ierr = VecDestroy(&junct->rcouple);CHKERRQ(ierr);
          ierr = VecDestroy(&junct->xcouple);CHKERRQ(ierr);
          ierr = MatDestroy(&junct->mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PhysicsCreate_Traffic(DGNetwork fvnet)
{
  PetscErrorCode    ierr;
  TrafficCtx        *user;
  RiemannFunction   r;
  PetscFunctionList rlist      = 0;
  char              rname[256] = "exact";

  PetscFunctionBeginUser;
  ierr = PetscNew(&user);CHKERRQ(ierr);
  fvnet->physics.samplenetwork  = PhysicsSample_TrafficNetwork;
  fvnet->physics.characteristic = PhysicsCharacteristic_Conservative;
  fvnet->physics.destroy        = PhysicsDestroy_SimpleFree_Net;
  fvnet->physics.user           = user;
  fvnet->physics.dof            = 1;
  fvnet->physics.destroy        = PhysicsDestroy_SimpleFree_Net;
  fvnet->physics.vfluxassign    = PhysicsAssignVertexFlux_Traffic;
  fvnet->physics.vfluxdestroy   = PhysicsDestroyVertexFlux;
  fvnet->physics.user           = user;
  fvnet->physics.flux           = TrafficFlux2;
  fvnet->physics.flux2          = TrafficFluxVoid;
  fvnet->physics.fluxeig        = TrafficEig;     

  ierr = PetscStrallocpy("density",&fvnet->physics.fieldname[0]);CHKERRQ(ierr);
  user->a = 0.5;
  ierr = RiemannListAdd_Net(&rlist,"rusanov",PhysicsRiemann_Traffic_Rusanov);CHKERRQ(ierr);
  ierr = RiemannListAdd_Net(&rlist,"exact",  PhysicsRiemann_Traffic_Exact);CHKERRQ(ierr);
  ierr = RiemannListAdd_Net(&rlist,"roe",    PhysicsRiemann_Traffic_Roe);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(fvnet->comm,fvnet->prefix,"Options for Traffic","");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-physics_traffic_a","Flux = a*u*(1-u)","",user->a,&user->a,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-physics_traffic_riemann","Riemann solver","",rlist,rname,rname,sizeof(rname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = RiemannListFind_Net(rlist,rname,&r);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&rlist);CHKERRQ(ierr);

  fvnet->physics.riemann = r;
  PetscFunctionReturn(0);
}