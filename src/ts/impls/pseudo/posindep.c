#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: posindep.c,v 1.25 1998/03/20 22:51:32 bsmith Exp bsmith $";
#endif
/*
       Code for Timestepping with implicit backwards Euler.
*/
#include <math.h>
#include "src/ts/tsimpl.h"                /*I   "ts.h"   I*/
#include "pinclude/pviewer.h"


typedef struct {
  Vec  update;      /* work vector where new solution is formed */
  Vec  func;        /* work vector where F(t[i],u[i]) is stored */
  Vec  rhs;         /* work vector for RHS; vec_sol/dt */

  /* information used for Pseudo-timestepping */

  int    (*dt)(TS,double*,void*);              /* compute next timestep, and related context */
  void   *dtctx;              
  int    (*verify)(TS,Vec,void*,double*,int*); /* verify previous timestep and related context */
  void   *verifyctx;     

  double initial_fnorm,fnorm;                  /* original and current norm of F(u) */
  double fnorm_previous;

  double dt_increment;                  /* scaling that dt is incremented each time-step */
  int    increment_dt_from_initial_dt;  
} TS_Pseudo;

/* ------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "TSPseudoComputeTimeStep"
/*@
    TSPseudoComputeTimeStep - Computes the next timestep for a currently running
    pseudo-timestepping process.

    Input Parameter:
.   ts - timestep context

    Output Parameter:
.   dt - newly computed timestep


    Notes:
    The routine to be called here to compute the timestep should be
    set by calling TSPseudoSetTimeStep().

.keywords: timestep, pseudo, compute

.seealso: TSPseudoDefaultTimeStep(), TSPseudoSetTimeStep()
@*/
int TSPseudoComputeTimeStep(TS ts,double *dt)
{
  TS_Pseudo *pseudo = (TS_Pseudo*) ts->data;
  int       ierr;

  PetscFunctionBegin;
  PLogEventBegin(TS_PseudoComputeTimeStep,ts,0,0,0);
  ierr = (*pseudo->dt)(ts,dt,pseudo->dtctx); CHKERRQ(ierr);
  PLogEventEnd(TS_PseudoComputeTimeStep,ts,0,0,0);
  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "TSPseudoDefaultVerifyTimeStep"
/*@C
   TSPseudoDefaultVerifyTimeStep - Default code to verify the quality of the last timestep.

   Input Parameters:
.  ts - the timestep context
.  dtctx - unused timestep context
.  update - latest solution vector

   Output Parameters:
.  newdt - the timestep to use for the next step
.  flag - flag indicating whether the last time step was acceptable

   Note:
   This routine always returns a flag of 1, indicating an acceptable 
   timestep.

.keywords: timestep, pseudo, default, verify 

.seealso: TSPseudoSetVerifyTimeStep(), TSPseudoVerifyTimeStep()
@*/
int TSPseudoDefaultVerifyTimeStep(TS ts,Vec update,void *dtctx,double *newdt,int *flag)
{
  PetscFunctionBegin;
  *flag = 1;
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "TSPseudoVerifyTimeStep"
/*@
    TSPseudoVerifyTimeStep - Verifies whether the last timestep was acceptable.

    Input Parameters:
.   ts - timestep context
.   update - latest solution vector

    Output Parameters:
.   dt - newly computed timestep (if it had to shrink)
.   flag - indicates if current timestep was ok

    Notes:
    The routine to be called here to compute the timestep should be
    set by calling TSPseudoSetVerifyTimeStep().

.keywords: timestep, pseudo, verify 

.seealso: TSPseudoSetVerifyTimeStep(), TSPseudoDefaultVerifyTimeStep()
@*/
int TSPseudoVerifyTimeStep(TS ts,Vec update,double *dt,int *flag)
{
  TS_Pseudo *pseudo = (TS_Pseudo*) ts->data;
  int       ierr;

  PetscFunctionBegin;
  if (!pseudo->verify) {*flag = 1; PetscFunctionReturn(0);}

  ierr = (*pseudo->verify)(ts,update,pseudo->verifyctx,dt,flag ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "TSStep_Pseudo"
static int TSStep_Pseudo(TS ts,int *steps,double *time)
{
  Vec       sol = ts->vec_sol;
  int       ierr,i,max_steps = ts->max_steps,its,ok,lits;
  TS_Pseudo *pseudo = (TS_Pseudo*) ts->data;
  double    current_time_step;
  
  PetscFunctionBegin;
  *steps = -ts->steps;

  ierr = VecCopy(sol,pseudo->update); CHKERRQ(ierr);
  for ( i=0; i<max_steps && ts->ptime < ts->max_time; i++ ) {
    ierr = TSPseudoComputeTimeStep(ts,&ts->time_step); CHKERRQ(ierr);
    current_time_step = ts->time_step;
    while (1) {
      ts->ptime  += current_time_step;
      ierr = SNESSolve(ts->snes,pseudo->update,&its); CHKERRQ(ierr);
      ierr = SNESGetNumberLinearIterations(ts->snes,&lits); CHKERRQ(ierr);
      ts->nonlinear_its += PetscAbsInt(its); ts->linear_its += lits;
      ierr = TSPseudoVerifyTimeStep(ts,pseudo->update,&ts->time_step,&ok); CHKERRQ(ierr);
      if (ok) break;
      ts->ptime        -= current_time_step;
      current_time_step = ts->time_step;
    }
    ierr = VecCopy(pseudo->update,sol); CHKERRQ(ierr);
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *time  = ts->ptime;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "TSDestroy_Pseudo"
static int TSDestroy_Pseudo(TS ts )
{
  TS_Pseudo *pseudo = (TS_Pseudo*) ts->data;
  int       ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(pseudo->update); CHKERRQ(ierr);
  if (pseudo->func) {ierr = VecDestroy(pseudo->func);CHKERRQ(ierr);}
  if (pseudo->rhs)  {ierr = VecDestroy(pseudo->rhs);CHKERRQ(ierr);}
  if (ts->Ashell)   {ierr = MatDestroy(ts->A); CHKERRQ(ierr);}
  PetscFree(pseudo);
  PetscFunctionReturn(0);
}


/*------------------------------------------------------------*/
/*
    This matrix shell multiply where user provided Shell matrix
*/

#undef __FUNC__  
#define __FUNC__ "TSPseudoMatMult"
int TSPseudoMatMult(Mat mat,Vec x,Vec y)
{
  TS     ts;
  Scalar mdt,mone = -1.0;
  int    ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ts);CHKERRQ(ierr);
  mdt = 1.0/ts->time_step;

  /* apply user provided function */
  ierr = MatMult(ts->Ashell,x,y); CHKERRQ(ierr);
  /* shift and scale by 1/dt - F */
  ierr = VecAXPBY(&mdt,&mone,x,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
    This defines the nonlinear equation that is to be solved with SNES

              (U^{n+1} - U^{n})/dt - F(U^{n+1})
*/
#undef __FUNC__  
#define __FUNC__ "TSPseudoFunction"
int TSPseudoFunction(SNES snes,Vec x,Vec y,void *ctx)
{
  TS     ts = (TS) ctx;
  Scalar mdt = 1.0/ts->time_step,*unp1,*un,*Funp1;
  int    ierr,i,n;

  PetscFunctionBegin;
  /* apply user provided function */
  ierr = TSComputeRHSFunction(ts,ts->ptime,x,y); CHKERRQ(ierr);
  /* compute (u^{n+1) - u^{n})/dt - F(u^{n+1}) */
  ierr = VecGetArray(ts->vec_sol,&un); CHKERRQ(ierr);
  ierr = VecGetArray(x,&unp1); CHKERRQ(ierr);
  ierr = VecGetArray(y,&Funp1); CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&n); CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    Funp1[i] = mdt*(unp1[i] - un[i]) - Funp1[i];
  }
  ierr = VecRestoreArray(ts->vec_sol,&un);
  ierr = VecRestoreArray(x,&unp1);
  ierr = VecRestoreArray(y,&Funp1);

  PetscFunctionReturn(0);
}

/*
   This constructs the Jacobian needed for SNES 

             J = I/dt - J_{F}   where J_{F} is the given Jacobian of F.
*/
#undef __FUNC__  
#define __FUNC__ "TSPseudoJacobian"
int TSPseudoJacobian(SNES snes,Vec x,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  TS      ts = (TS) ctx;
  int     ierr;
  Scalar  mone = -1.0, mdt = 1.0/ts->time_step;
  MatType mtype;

  PetscFunctionBegin;
  /* construct users Jacobian */
  if (ts->rhsjacobian) {
    ierr = (*ts->rhsjacobian)(ts,ts->ptime,x,AA,BB,str,ts->jacP);CHKERRQ(ierr);
  }

  /* shift and scale Jacobian, if not a shell matrix */
  ierr = MatGetType(*AA,&mtype,PETSC_NULL);
  if (mtype != MATSHELL) {
    ierr = MatScale(&mone,*AA); CHKERRQ(ierr);
    ierr = MatShift(&mdt,*AA); CHKERRQ(ierr);
  }
  ierr = MatGetType(*BB,&mtype,PETSC_NULL);
  if (*BB != *AA && *str != SAME_PRECONDITIONER && mtype != MATSHELL) {
    ierr = MatScale(&mone,*BB); CHKERRQ(ierr);
    ierr = MatShift(&mdt,*BB); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "TSSetUp_Pseudo"
static int TSSetUp_Pseudo(TS ts)
{
  TS_Pseudo *pseudo = (TS_Pseudo*) ts->data;
  int       ierr, M, m;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&pseudo->update); CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&pseudo->func); CHKERRQ(ierr);  
  ierr = SNESSetFunction(ts->snes,pseudo->func,TSPseudoFunction,ts);CHKERRQ(ierr);
  if (ts->Ashell) { /* construct new shell matrix */
    ierr = VecGetSize(ts->vec_sol,&M); CHKERRQ(ierr);
    ierr = VecGetLocalSize(ts->vec_sol,&m); CHKERRQ(ierr);
    ierr = MatCreateShell(ts->comm,m,M,M,M,ts,&ts->A); CHKERRQ(ierr);
    ierr = MatShellSetOperation(ts->A,MATOP_MULT,(void*)TSPseudoMatMult);CHKERRQ(ierr);
  }
  ierr = SNESSetJacobian(ts->snes,ts->A,ts->B,TSPseudoJacobian,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "TSPseudoDefaultMonitor"
int TSPseudoDefaultMonitor(TS ts, int step, double time,Vec v, void *ctx)
{
  TS_Pseudo *pseudo = (TS_Pseudo*) ts->data;

  PetscFunctionBegin;
  (*PetscHelpPrintf)(ts->comm,"TS %d dt %g time %g fnorm %g\n",step,ts->time_step,time,pseudo->fnorm);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSSetFromOptions_Pseudo"
static int TSSetFromOptions_Pseudo(TS ts)
{
  int    ierr,flg;
  double inc;

  PetscFunctionBegin;
  ierr = SNESSetFromOptions(ts->snes); CHKERRQ(ierr);

  ierr = OptionsHasName(ts->prefix,"-ts_monitor",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = TSSetMonitor(ts,TSPseudoDefaultMonitor,0); CHKERRQ(ierr);
  }
  ierr = OptionsGetDouble(ts->prefix,"-ts_pseudo_increment",&inc,&flg);  CHKERRQ(ierr);
  if (flg) {
    ierr = TSPseudoSetTimeStepIncrement(ts,inc);  CHKERRQ(ierr);
  }
  ierr = OptionsHasName(ts->prefix,"-ts_pseudo_increment_dt_from_initial_dt",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = TSPseudoIncrementDtFromInitialDt(ts); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSPrintHelp_Pseudo"
static int TSPrintHelp_Pseudo(TS ts,char *p)
{
  PetscFunctionBegin;
  (*PetscHelpPrintf)(ts->comm," Options for TS Pseudo timestepper:\n");
  (*PetscHelpPrintf)(ts->comm," %sts_pseudo_increment <value> : default 1.1\n",p);
  (*PetscHelpPrintf)(ts->comm," %sts_pseudo_increment_dt_from_initial_dt : use initial_dt *\n",p); 
  (*PetscHelpPrintf)(ts->comm,"     initial fnorm/current fnorm to determine new timestep\n");
  (*PetscHelpPrintf)(ts->comm,"     default is current_dt * previous fnorm/current fnorm\n"); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSView_Pseudo"
static int TSView_Pseudo(TS ts,Viewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "TSPseudoSetVerifyTimeStep"
/*@
   TSPseudoSetVerifyTimeStep - Sets a user-defined routine to verify the quality of the 
   last timestep.

   Input Parameters:
.  ts - timestep context
.  dt - user-defined function to verify timestep
.  ctx - [optional] user-defined context for private data
         for the timestep verification routine (may be PETSC_NULL)

   Calling sequence of func:
.  func (TS ts,Vec update,void *ctx,double *newdt,int *flag);

.  update - latest solution vector
.  ctx - [optional] timestep context
.  newdt - the timestep to use for the next step
.  flag - flag indicating whether the last time step was acceptable

   Notes:
   The routine set here will be called by TSPseudoVerifyTimeStep()
   during the timestepping process.

.keywords: timestep, pseudo, set, verify 

.seealso: TSPseudoDefaultVerifyTimeStep(), TSPseudoVerifyTimeStep()
@*/
int TSPseudoSetVerifyTimeStep(TS ts,int (*dt)(TS,Vec,void*,double*,int*),void* ctx)
{
  int ierr, (*f)(TS,int (*)(TS,Vec,void*,double *,int *),void *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);

  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPseudoSetVerifyTimeStep",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,dt,ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSPseudoSetTimeStepIncrement"
/*@
    TSPseudoSetTimeStepIncrement - Sets the scaling increment applied to 
    dt when using the TSPseudoDefaultTimeStep() routine.

    Input Parameters:
.   ts - the timestep context
.   inc - the scaling factor >= 1.0

    Options Database Key:
$    -ts_pseudo_increment <increment>

.keywords: timestep, pseudo, set, increment

.seealso: TSPseudoSetTimeStep(), TSPseudoDefaultTimeStep()
@*/
int TSPseudoSetTimeStepIncrement(TS ts,double inc)
{
  int ierr, (*f)(TS,double);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);

  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPseudoSetTimeStepIncremen",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,inc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSPseudoIncrementDtFromInitialDt"
/*@
    TSPseudoIncrementDtFromInitialDt - Indicates that a new timestep
    is computed via the formula
$         dt = initial_dt*initial_fnorm/current_fnorm 
      rather than the default update,
$         dt = current_dt*previous_fnorm/current_fnorm.

    Input Parameter:
.   ts - the timestep context

    Options Database Key:
$    -ts_pseudo_increment_dt_from_initial_dt

.keywords: timestep, pseudo, set, increment

.seealso: TSPseudoSetTimeStep(), TSPseudoDefaultTimeStep()
@*/
int TSPseudoIncrementDtFromInitialDt(TS ts)
{
  int ierr, (*f)(TS);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);

  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPseudoIncrementDtFromInitialDt",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "TSPseudoSetTimeStep"
/*@
   TSPseudoSetTimeStep - Sets the user-defined routine to be
   called at each pseudo-timestep to update the timestep.

   Input Parameters:
.  ts - timestep context
.  dt - function to compute timestep
.  ctx - [optional] user-defined context for private data
         required by the function (may be PETSC_NULL)

   Calling sequence of func:
.  func (TS ts,double *newdt,void *ctx);

.  newdt - the newly computed timestep
.  ctx - [optional] timestep context

   Notes:
   The routine set here will be called by TSPseudoComputeTimeStep()
   during the timestepping process.

.keywords: timestep, pseudo, set

.seealso: TSPseudoDefaultTimeStep(), TSPseudoComputeTimeStep()
@*/
int TSPseudoSetTimeStep(TS ts,int (*dt)(TS,double*,void*),void* ctx)
{
  int ierr, (*f)(TS,int (*)(TS,double *,void *),void *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);

  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPseudoSetTimeStep",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,dt,ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------- */

#undef __FUNC__  
#define __FUNC__ "TSPseudoSetVerifyTimeStep_Pseudo"
int TSPseudoSetVerifyTimeStep_Pseudo(TS ts,int (*dt)(TS,Vec,void*,double*,int*),void* ctx)
{
  TS_Pseudo *pseudo;

  PetscFunctionBegin;
  pseudo              = (TS_Pseudo*) ts->data;
  pseudo->verify      = dt;
  pseudo->verifyctx   = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSPseudoSetTimeStepIncrement_Pseudo"
int TSPseudoSetTimeStepIncrement_Pseudo(TS ts,double inc)
{
  TS_Pseudo *pseudo;

  PetscFunctionBegin;
  pseudo               = (TS_Pseudo*) ts->data;
  pseudo->dt_increment = inc;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSPseudoIncrementDtFromInitialDt_Pseudo"
int TSPseudoIncrementDtFromInitialDt_Pseudo(TS ts)
{
  TS_Pseudo *pseudo;

  PetscFunctionBegin;
  pseudo                               = (TS_Pseudo*) ts->data;
  pseudo->increment_dt_from_initial_dt = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSPseudoSetTimeStep_Pseudo"
int TSPseudoSetTimeStep_Pseudo(TS ts,int (*dt)(TS,double*,void*),void* ctx)
{
  TS_Pseudo *pseudo;

  PetscFunctionBegin;
  pseudo          = (TS_Pseudo*) ts->data;
  pseudo->dt      = dt;
  pseudo->dtctx   = ctx;
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------- */

#undef __FUNC__  
#define __FUNC__ "TSCreate_Pseudo"
int TSCreate_Pseudo(TS ts )
{
  TS_Pseudo *pseudo;
  int       ierr;
  MatType   mtype;

  PetscFunctionBegin;
  ts->destroy         = TSDestroy_Pseudo;
  ts->printhelp       = TSPrintHelp_Pseudo;
  ts->view            = TSView_Pseudo;

  if (ts->problem_type == TS_LINEAR) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Only for nonlinear problems");
  }
  if (!ts->A) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Must set Jacobian");
  }
  ierr = MatGetType(ts->A,&mtype,PETSC_NULL);
  if (mtype == MATSHELL) {
    ts->Ashell = ts->A;
  }
  ts->setup           = TSSetUp_Pseudo;  
  ts->step            = TSStep_Pseudo;
  ts->setfromoptions  = TSSetFromOptions_Pseudo;

  /* create the required nonlinear solver context */
  ierr = SNESCreate(ts->comm,SNES_NONLINEAR_EQUATIONS,&ts->snes);CHKERRQ(ierr);

  pseudo   = PetscNew(TS_Pseudo); CHKPTRQ(pseudo);
  PLogObjectMemory(ts,sizeof(TS_Pseudo));

  PetscMemzero(pseudo,sizeof(TS_Pseudo));
  ts->data = (void *) pseudo;

  pseudo->dt_increment                 = 1.1;
  pseudo->increment_dt_from_initial_dt = 0;
  pseudo->dt                           = TSPseudoDefaultTimeStep;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSPseudoSetVerifyTimeStep",
                    "TSPseudoSetVerifyTimeStep_Pseudo",
                    (void*)TSPseudoSetVerifyTimeStep_Pseudo);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSPseudoSetTimeStepIncrement",
                    "TSPseudoSetTimeStepIncrement_Pseudo",
                    (void*)TSPseudoSetTimeStepIncrement_Pseudo);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSPseudoIncrementDtFromInitialDt",
                    "TSPseudoIncrementDtFromInitialDt_Pseudo",
                    (void*)TSPseudoIncrementDtFromInitialDt_Pseudo);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSPseudoSetTimeStep","TSPseudoSetTimeStep_Pseudo",
                    (void*)TSPseudoSetTimeStep_Pseudo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TSPseudoDefaultTimeStep"
/*@C
   TSPseudoDefaultTimeStep - Default code to compute pseudo-timestepping.
   Use with TSPseudoSetTimeStep().

   Input Parameters:
.  ts - the timestep context
.  dtctx - unused timestep context

   Output Parameter:
.  newdt - the timestep to use for the next step

.keywords: timestep, pseudo, default

.seealso: TSPseudoSetTimeStep(), TSPseudoComputeTimeStep()
@*/
int TSPseudoDefaultTimeStep(TS ts,double* newdt,void* dtctx)
{
  TS_Pseudo *pseudo = (TS_Pseudo*) ts->data;
  double    inc = pseudo->dt_increment,fnorm_previous = pseudo->fnorm_previous;
  int       ierr;

  PetscFunctionBegin;
  ierr = TSComputeRHSFunction(ts,ts->ptime,ts->vec_sol,pseudo->func);CHKERRQ(ierr);  
  ierr = VecNorm(pseudo->func,NORM_2,&pseudo->fnorm); CHKERRQ(ierr); 
  if (pseudo->initial_fnorm == 0.0) {
    /* first time through so compute initial function norm */
    pseudo->initial_fnorm = pseudo->fnorm;
    fnorm_previous        = pseudo->fnorm;
  }
  if (pseudo->fnorm == 0.0) {
    *newdt = 1.e12*inc*ts->time_step; 
  }
  else if (pseudo->increment_dt_from_initial_dt) {
    *newdt = inc*ts->initial_time_step*pseudo->initial_fnorm/pseudo->fnorm;
  } else {
    *newdt = inc*ts->time_step*fnorm_previous/pseudo->fnorm;
  }
  pseudo->fnorm_previous = pseudo->fnorm;
  PetscFunctionReturn(0);
}

