/*
   p_t = -x_t*p_x -y_t*p_y + f(t)*p_yy
   xmin < x < xmax, ymin < y < ymax;
   x_t = (y - ws)  y_t = (ws/2H)*(Pm - Pmax*sin(x))

   Boundary conditions: -bc_type 0 => Zero dirichlet boundary
                        -bc_type 1 => Steady state boundary condition
   Steady state boundary condition found by setting p_t = 0
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

typedef struct {
  PetscInt  runs,steps,N,i;
  PetscReal dt;
  Vec       *all_x,*all_dx;
  PetscReal *all_t;
  PetscBool fd_der;
} Data;


/*
   User-defined data structures and routines
*/
typedef struct {
  PetscScalar scale_u_x;
  PetscScalar scale_u_y;
  PetscScalar scale_y_u_x;
  PetscScalar scale_sin_x_u_y;
  PetscScalar scale_nexp_t_u_yy;

  PetscScalar ws;   /* Synchronous speed */
  PetscScalar H;    /* Inertia constant */
  PetscScalar D;    /* Damping constant */
  PetscScalar Pmax; /* Maximum power output of generator */
  PetscScalar PM_min; /* Mean mechanical power input */
  PetscScalar lambda; /* correlation time */
  PetscScalar q;      /* noise strength */
  PetscScalar mux;    /* Initial average angle */
  PetscScalar sigmax; /* Standard deviation of initial angle */
  PetscScalar muy;    /* Average speed */
  PetscScalar sigmay; /* standard deviation of initial speed */
  PetscScalar rho;    /* Cross-correlation coefficient */
  PetscScalar t0;     /* Initial time */
  PetscScalar dt;     /* Timestep size */
  PetscScalar tmax;   /* Final time */
  PetscScalar xmin;   /* left boundary of angle */
  PetscScalar xmax;   /* right boundary of angle */
  PetscScalar ymin;   /* bottom boundary of speed */
  PetscScalar ymax;   /* top boundary of speed */
  PetscScalar dx;     /* x step size */
  PetscScalar dy;     /* y step size */
  PetscInt    steps;  /* Number of timesteps */
  PetscInt    bc; /* Boundary conditions */
  PetscScalar disper_coe; /* Dispersion coefficient */
  Data        data;
  DM          da;
} AppCtx;

PetscErrorCode Parameter_settings(AppCtx*);
PetscErrorCode ini_bou(Vec,AppCtx*);
PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*);
PetscErrorCode IJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
PetscErrorCode PostStep(TS);
PetscErrorCode DataComputeDerivative_FD(Data*);
PetscErrorCode adv1(PetscScalar**,PetscScalar,PetscInt,PetscInt,PetscInt,PetscScalar*,AppCtx*);
PetscErrorCode adv2(PetscScalar**,PetscScalar,PetscInt,PetscInt,PetscInt,PetscScalar*,AppCtx*);
PetscErrorCode diffuse(PetscScalar**,PetscInt,PetscInt,PetscReal,PetscScalar*,AppCtx*);
PetscErrorCode BoundaryConditions(PetscScalar **,DMDACoor2d **,PetscInt,PetscInt,PetscInt, PetscInt,PetscScalar **,AppCtx *);

PetscErrorCode GetData(PetscInt* N_p, Vec** all_x_p, Vec** all_dx_p, PetscReal** all_t_p, DM* dm_p)
{
  PetscErrorCode ierr;
  Vec            x;  /* Solution vector */
  TS             ts;   /* Time-stepping context */
  AppCtx         user; /* Application context */
  Mat            J;
  PetscViewer    viewer;

  /* Get physics and time parameters */
  ierr = Parameter_settings(&user);CHKERRQ(ierr);

  /* Create a 2D DA with dof = 1 */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,4,4,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&user.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.da);CHKERRQ(ierr);
  ierr = DMSetUp(user.da);CHKERRQ(ierr);
  /* Set x and y coordinates */
  ierr = DMDASetUniformCoordinates(user.da,user.xmin,user.xmax,user.ymin,user.ymax,0.0,1.0);CHKERRQ(ierr);

  /* Get global vector x from DM  */
  ierr = DMCreateGlobalVector(user.da,&x);CHKERRQ(ierr);

  ierr = ini_bou(x,&user);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"ini_x",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(x,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /* Create Vecs to hold recorded data.*/
  user.data.runs = 1;
  user.data.steps = user.steps;
  user.data.dt = user.dt;
  user.data.N = user.data.runs * user.data.steps;
  user.data.i = 0;
  ierr = VecDuplicateVecs(x, user.data.N, &user.data.all_x);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(x, user.data.N, &user.data.all_dx);CHKERRQ(ierr);
  ierr = PetscMalloc1(user.data.N, &user.data.all_t);
  user.data.fd_der = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-fd_der",&user.data.fd_der,NULL);CHKERRQ(ierr);

  /* Get Jacobian matrix structure from the da */
  ierr = DMSetMatType(user.da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(user.da,&J);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,IJacobian,&user);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts,&user);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user.t0 + (user.steps-1) * user.dt);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTime(ts,user.t0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,user.dt);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSetPostStep(ts,PostStep);CHKERRQ(ierr);
  ierr = TSSetSolution(ts, x);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  /* Record initial values and then run. */
  ierr = PostStep(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,NULL);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"fin_x",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(x,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /* Get derivative data. */
  if (user.data.fd_der) {
    ierr = DataComputeDerivative_FD(&user.data);CHKERRQ(ierr);
  }

  /* Expected values:
    fx = kx * (y - user.ws);
    fy = ky * (user.PM_min - user.Pmax*PetscSinScalar(x));

    du/dx = ky2*(1.0-exp(-t/user.lambda)) * p_yy
           - fx*p_x
           - fy*p_y;
    du/dx =
             ky2*p_yy - ky2*exp(-t/user.lambda)*p_yy
           - kx*y*p_x + kx*user.ws*p_x
           - ky*user.PM_min*p_y + ky*user.Pmax*PetscSinScalar(x)*p_y

    (1.0-exp(-t/l)) = 1 - (1 + -t/l + (-t/l)^2/2 + (-t/l)^3/6 + (-t/l)^4/24 + ...)
                         = t/l - t^2/(2*l^2) + t^3/(6*l^3) - t^4/(24*l^4) + ...
  */
  PetscScalar ky2 = PetscPowScalar((user.lambda*user.ws)/(2*user.H), 2) * user.q;
  PetscScalar kx = 1;
  PetscScalar ky = user.ws/(2*user.H);
  PetscScalar scaled_kyy = user.scale_nexp_t_u_yy * ky2;
  printf("Expected:\n");
  printf("lambda: % g\n", user.lambda);
  printf("                        u_x: % g\n", user.scale_u_x         * kx*user.ws);
  printf("                        u_y: % g\n", user.scale_u_y         * -ky*user.PM_min);
  printf("                    y * u_x: % g\n", user.scale_y_u_x       * -kx);
  printf("               sin(x) * u_y: % g\n", user.scale_sin_x_u_y   * ky*user.Pmax);
  printf("(1 - exp(-t/lambda)) * u_yy: % g\n", scaled_kyy);
  printf("\n");

  printf("Maclaurin series expansion for t:\n");
  printf("             t * u_yy[j][i]: % g\n",   scaled_kyy / user.lambda);
  printf("           t^2 * u_yy[j][i]: % g\n", - scaled_kyy / (2 * PetscPowScalarInt(user.lambda, 2)));
  printf("           t^3 * u_yy[j][i]: % g\n",   scaled_kyy / (6 * PetscPowScalarInt(user.lambda, 3)));
  printf("           t^4 * u_yy[j][i]: % g\n", - scaled_kyy / (24 * PetscPowScalarInt(user.lambda, 4)));



  /* Write output parameters. */
  *N_p = user.data.N;
  *all_x_p = user.data.all_x;
  *all_dx_p = user.data.all_dx;
  *all_t_p = user.data.all_t;
  *dm_p = user.da;

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *user=(AppCtx*)ctx;
  DM             cda;
  DMDACoor2d     **coors;
  PetscScalar    **p,**f;
  PetscInt       i,j;
  PetscInt       xs,ys,xm,ym,M,N;
  Vec            localX,gc;
  PetscScalar    p_adv1,p_adv2,p_diff;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(user->da,NULL,&M,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(user->da,&cda);CHKERRQ(ierr);
  ierr = DMDAGetCorners(cda,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);

  ierr = DMGetLocalVector(user->da,&localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(user->da,&gc);CHKERRQ(ierr);

  ierr = DMDAVecGetArrayRead(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(user->da,localX,&p);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,F,&f);CHKERRQ(ierr);

  user->disper_coe = PetscPowScalar((user->lambda*user->ws)/(2*user->H),2)*user->q*(1.0-PetscExpScalar(-t/user->lambda));
  for (i=xs; i < xs+xm; i++) {
    for (j=ys; j < ys+ym; j++) {
      if (i == 0 || j == 0 || i == M-1 || j == N-1) {
        ierr = BoundaryConditions(p,coors,i,j,M,N,f,user);CHKERRQ(ierr);
      } else {
        ierr = adv1(p,coors[j][i].y,i,j,M,&p_adv1,user);CHKERRQ(ierr);
        ierr = adv2(p,coors[j][i].x,i,j,N,&p_adv2,user);CHKERRQ(ierr);
        ierr = diffuse(p,i,j,t,&p_diff,user);CHKERRQ(ierr);
        f[j][i] = -p_adv1 - p_adv2 + p_diff;
      }
    }
  }
  ierr = DMDAVecRestoreArrayRead(user->da,localX,&p);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->da,&localX);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(cda,gc,&coors);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode DataComputeDerivative_FD(Data* data)
{
  PetscErrorCode ierr;
  PetscInt       i,t,r;

  /* Get derivate data using fourth-order central difference. */
  PetscFunctionBegin;
  i = 0;
  for (r = 0; r < data->runs; r++) {
    for (t = 0; t < data->steps; t++) {
      ierr = VecSet(data->all_dx[i], 0);CHKERRQ(ierr);
      if (t >= 2 && t < data->steps - 2) {
        ierr = VecAXPY(data->all_dx[i], -1.0, data->all_x[i+2]);CHKERRQ(ierr);
        ierr = VecAXPY(data->all_dx[i],  8.0, data->all_x[i+1]);CHKERRQ(ierr);
        ierr = VecAXPY(data->all_dx[i], -8.0, data->all_x[i-1]);CHKERRQ(ierr);
        ierr = VecAXPY(data->all_dx[i],  1.0, data->all_x[i-2]);CHKERRQ(ierr);
        ierr = VecScale(data->all_dx[i], 1.0/(12.0*data->dt));CHKERRQ(ierr);
      }
      i++;
    }
    /* Set boundary values to 0. */
    ierr = VecSet(data->all_x[i-1], 0);CHKERRQ(ierr);
    ierr = VecSet(data->all_x[i-2], 0);CHKERRQ(ierr);
    ierr = VecSet(data->all_x[i-data->steps], 0);CHKERRQ(ierr);
    ierr = VecSet(data->all_x[i-data->steps+1], 0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PostStep(TS ts)
{
  PetscErrorCode ierr;
  Vec            X;
  AppCtx         *user;
  // PetscScalar    sum;
  // PetscReal      t;
  Data           *data;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts,&user);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
  // ierr = VecSum(X,&sum);CHKERRQ(ierr);
  // ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  // ierr = PetscPrintf(PETSC_COMM_WORLD,"sum(p)*dw*dtheta at t = %3.2f = %3.6f\n",(double)t,(double)sum*user->dx*user->dy);CHKERRQ(ierr);
  data = &user->data;
  if (data->i == data->N) {
    // SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Cannot record more than %d vectors.",data->N);
    printf("Cannot record more than %d vectors. Currently at %d.\n", data->N, data->i);
    data->i++;
    return 0;
  }
  ierr = VecCopy(X, data->all_x[data->i]);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &data->all_t[data->i]);CHKERRQ(ierr);
  if (!data->fd_der) {
    PetscReal     t;
    ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
    ierr = RHSFunction(ts, t, X, data->all_dx[data->i], user);CHKERRQ(ierr);
  }
  data->i++;
  PetscFunctionReturn(0);
}

PetscErrorCode ini_bou(Vec X,AppCtx* user)
{
  PetscErrorCode ierr;
  DM             cda;
  DMDACoor2d     **coors;
  PetscScalar    **p;
  Vec            gc;
  PetscInt       i,j;
  PetscInt       xs,ys,xm,ym,M,N;
  PetscScalar    xi,yi;
  PetscScalar    sigmax=user->sigmax,sigmay=user->sigmay;
  PetscScalar    rho   =user->rho;
  PetscScalar    mux   =user->mux,muy=user->muy;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = DMDAGetInfo(user->da,NULL,&M,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  user->dx = (user->xmax - user->xmin)/(M-1); user->dy = (user->ymax - user->ymin)/(N-1);
  ierr = DMGetCoordinateDM(user->da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(user->da,&gc);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,X,&p);CHKERRQ(ierr);
  ierr = DMDAGetCorners(cda,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (i=xs; i < xs+xm; i++) {
    for (j=ys; j < ys+ym; j++) {
      xi = coors[j][i].x; yi = coors[j][i].y;
      if (i == 0 || j == 0 || i == M-1 || j == N-1) p[j][i] = 0.0;
      else p[j][i] = (0.5/(PETSC_PI*sigmax*sigmay*PetscSqrtScalar(1.0-rho*rho)))*PetscExpScalar(-0.5/(1-rho*rho)*(PetscPowScalar((xi-mux)/sigmax,2) + PetscPowScalar((yi-muy)/sigmay,2) - 2*rho*(xi-mux)*(yi-muy)/(sigmax*sigmay)));
    }
  }
  /*  p[N/2+N%2][M/2+M%2] = 1/(user->dx*user->dy); */

  ierr = DMDAVecRestoreArray(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da,X,&p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* First advection term */
PetscErrorCode adv1(PetscScalar **p,PetscScalar y,PetscInt i,PetscInt j,PetscInt M,PetscScalar *p1,AppCtx *user)
{
  PetscScalar f;
  /*  PetscScalar v1,v2,v3,v4,v5,s1,s2,s3; */
  PetscFunctionBegin;
  /*  if (i > 2 && i < M-2) {
    v1 = (y-user->ws)*(p[j][i-2] - p[j][i-3])/user->dx;
    v2 = (y-user->ws)*(p[j][i-1] - p[j][i-2])/user->dx;
    v3 = (y-user->ws)*(p[j][i] - p[j][i-1])/user->dx;
    v4 = (y-user->ws)*(p[j][i+1] - p[j][i])/user->dx;
    v5 = (y-user->ws)*(p[j][i+1] - p[j][i+2])/user->dx;

    s1 = v1/3.0 - (7.0/6.0)*v2 + (11.0/6.0)*v3;
    s2 =-v2/6.0 + (5.0/6.0)*v3 + (1.0/3.0)*v4;
    s3 = v3/3.0 + (5.0/6.0)*v4 - (1.0/6.0)*v5;

    *p1 = 0.1*s1 + 0.6*s2 + 0.3*s3;
    } else *p1 = 0.0; */
  f   =  (user->scale_y_u_x * y - user->scale_u_x * user->ws);
  *p1 = f*(p[j][i+1] - p[j][i-1])/(2*user->dx);
  PetscFunctionReturn(0);
}

/* Second advection term */
PetscErrorCode adv2(PetscScalar **p,PetscScalar x,PetscInt i,PetscInt j,PetscInt N,PetscScalar *p2,AppCtx *user)
{
  PetscScalar f;
  /*  PetscScalar v1,v2,v3,v4,v5,s1,s2,s3; */
  PetscFunctionBegin;
  /*  if (j > 2 && j < N-2) {
    v1 = (user->ws/(2*user->H))*(user->PM_min - user->Pmax*sin(x))*(p[j-2][i] - p[j-3][i])/user->dy;
    v2 = (user->ws/(2*user->H))*(user->PM_min - user->Pmax*sin(x))*(p[j-1][i] - p[j-2][i])/user->dy;
    v3 = (user->ws/(2*user->H))*(user->PM_min - user->Pmax*sin(x))*(p[j][i] - p[j-1][i])/user->dy;
    v4 = (user->ws/(2*user->H))*(user->PM_min - user->Pmax*sin(x))*(p[j+1][i] - p[j][i])/user->dy;
    v5 = (user->ws/(2*user->H))*(user->PM_min - user->Pmax*sin(x))*(p[j+2][i] - p[j+1][i])/user->dy;

    s1 = v1/3.0 - (7.0/6.0)*v2 + (11.0/6.0)*v3;
    s2 =-v2/6.0 + (5.0/6.0)*v3 + (1.0/3.0)*v4;
    s3 = v3/3.0 + (5.0/6.0)*v4 - (1.0/6.0)*v5;

    *p2 = 0.1*s1 + 0.6*s2 + 0.3*s3;
    } else *p2 = 0.0; */
  f   = (user->ws/(2*user->H))*(user->scale_u_y * user->PM_min - user->scale_sin_x_u_y* user->Pmax*PetscSinScalar(x));
  *p2 = f*(p[j+1][i] - p[j-1][i])/(2*user->dy);
  PetscFunctionReturn(0);
}

/* Diffusion term */
PetscErrorCode diffuse(PetscScalar **p,PetscInt i,PetscInt j,PetscReal t,PetscScalar *p_diff,AppCtx * user)
{
  PetscFunctionBeginUser;
  *p_diff = user->scale_nexp_t_u_yy * user->disper_coe*((p[j-1][i] - 2*p[j][i] + p[j+1][i])/(user->dy*user->dy));
  PetscFunctionReturn(0);
}

PetscErrorCode BoundaryConditions(PetscScalar **p,DMDACoor2d **coors,PetscInt i,PetscInt j,PetscInt M, PetscInt N,PetscScalar **f,AppCtx *user)
{
  PetscScalar fwc,fthetac;
  PetscScalar w=coors[j][i].y,theta=coors[j][i].x;

  PetscFunctionBeginUser;
  if (user->bc == 0) { /* Natural boundary condition */
    f[j][i] = p[j][i];
  } else { /* Steady state boundary condition */
    fthetac = user->ws/(2*user->H)*(user->PM_min - user->Pmax*PetscSinScalar(theta));
    fwc = (w*w/2.0 - user->ws*w);
    if (i == 0 && j == 0) { /* left bottom corner */
      f[j][i] = fwc*(p[j][i+1] - p[j][i])/user->dx + fthetac*p[j][i] - user->disper_coe*(p[j+1][i] - p[j][i])/user->dy;
    } else if (i == 0 && j == N-1) { /* right bottom corner */
      f[j][i] = fwc*(p[j][i+1] - p[j][i])/user->dx + fthetac*p[j][i] - user->disper_coe*(p[j][i] - p[j-1][i])/user->dy;
    } else if (i == M-1 && j == 0) { /* left top corner */
      f[j][i] = fwc*(p[j][i] - p[j][i-1])/user->dx + fthetac*p[j][i] - user->disper_coe*(p[j+1][i] - p[j][i])/user->dy;
    } else if (i == M-1 && j == N-1) { /* right top corner */
      f[j][i] = fwc*(p[j][i] - p[j][i-1])/user->dx + fthetac*p[j][i] - user->disper_coe*(p[j][i] - p[j-1][i])/user->dy;
    } else if (i == 0) { /* Bottom edge */
      f[j][i] = fwc*(p[j][i+1] - p[j][i])/(user->dx) + fthetac*p[j][i] - user->disper_coe*(p[j+1][i] - p[j-1][i])/(2*user->dy);
    } else if (i == M-1) { /* Top edge */
      f[j][i] = fwc*(p[j][i] - p[j][i-1])/(user->dx) + fthetac*p[j][i] - user->disper_coe*(p[j+1][i] - p[j-1][i])/(2*user->dy);
    } else if (j == 0) { /* Left edge */
      f[j][i] = fwc*(p[j][i+1] - p[j][i-1])/(2*user->dx) + fthetac*p[j][i] - user->disper_coe*(p[j+1][i] - p[j][i])/(user->dy);
    } else if (j == N-1) { /* Right edge */
      f[j][i] = fwc*(p[j][i+1] - p[j][i-1])/(2*user->dx) + fthetac*p[j][i] - user->disper_coe*(p[j][i] - p[j-1][i])/(user->dy);
    }
  }    
  PetscFunctionReturn(0);
}

PetscErrorCode IFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *user=(AppCtx*)ctx;
  DM             cda;
  DMDACoor2d     **coors;
  PetscScalar    **p,**f,**pdot;
  PetscInt       i,j;
  PetscInt       xs,ys,xm,ym,M,N;
  Vec            localX,gc,localXdot;
  PetscScalar    p_adv1,p_adv2,p_diff;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(user->da,NULL,&M,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(user->da,&cda);CHKERRQ(ierr);
  ierr = DMDAGetCorners(cda,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);

  ierr = DMGetLocalVector(user->da,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->da,&localXdot);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user->da,Xdot,INSERT_VALUES,localXdot);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da,Xdot,INSERT_VALUES,localXdot);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(user->da,&gc);CHKERRQ(ierr);

  ierr = DMDAVecGetArrayRead(cda,gc,&coors);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(user->da,localX,&p);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(user->da,localXdot,&pdot);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,F,&f);CHKERRQ(ierr);

  user->disper_coe = PetscPowScalar((user->lambda*user->ws)/(2*user->H),2)*user->q*(1.0-PetscExpScalar(-t/user->lambda));
  for (i=xs; i < xs+xm; i++) {
    for (j=ys; j < ys+ym; j++) {
      if (i == 0 || j == 0 || i == M-1 || j == N-1) {
        ierr = BoundaryConditions(p,coors,i,j,M,N,f,user);CHKERRQ(ierr);
      } else {
        ierr = adv1(p,coors[j][i].y,i,j,M,&p_adv1,user);CHKERRQ(ierr);
        ierr = adv2(p,coors[j][i].x,i,j,N,&p_adv2,user);CHKERRQ(ierr);
        ierr = diffuse(p,i,j,t,&p_diff,user);CHKERRQ(ierr);
        f[j][i] = -p_adv1 - p_adv2 + p_diff - pdot[j][i];
      }
    }
  }
  ierr = DMDAVecRestoreArrayRead(user->da,localX,&p);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(user->da,localXdot,&pdot);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->da,&localX);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->da,&localXdot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(cda,gc,&coors);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat J,Mat Jpre,void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *user=(AppCtx*)ctx;
  DM             cda;
  DMDACoor2d     **coors;
  PetscInt       i,j;
  PetscInt       xs,ys,xm,ym,M,N;
  Vec            gc;
  PetscScalar    val[5],xi,yi;
  MatStencil     row,col[5];
  PetscScalar    c1,c3,c5;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(user->da,NULL,&M,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(user->da,&cda);CHKERRQ(ierr);
  ierr = DMDAGetCorners(cda,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(user->da,&gc);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cda,gc,&coors);CHKERRQ(ierr);
  for (i=xs; i < xs+xm; i++) {
    for (j=ys; j < ys+ym; j++) {
      PetscInt nc = 0;
      xi = coors[j][i].x; yi = coors[j][i].y;
      row.i = i; row.j = j;
      if (i == 0 || j == 0 || i == M-1 || j == N-1) {
        if (user->bc == 0) {
          col[nc].i = i; col[nc].j = j; val[nc++] = 1.0;
        } else {
          PetscScalar fthetac,fwc;
          fthetac = user->ws/(2*user->H)*(user->PM_min - user->Pmax*PetscSinScalar(xi));
          fwc     = (yi*yi/2.0 - user->ws*yi);
          if (i==0 && j==0) {
            col[nc].i = i+1; col[nc].j = j;   val[nc++] = fwc/user->dx;
            col[nc].i = i;   col[nc].j = j+1; val[nc++] = -user->disper_coe/user->dy;
            col[nc].i = i;   col[nc].j = j;   val[nc++] = -fwc/user->dx + fthetac + user->disper_coe/user->dy;
          } else if (i==0 && j == N-1) {
            col[nc].i = i+1; col[nc].j = j;   val[nc++] = fwc/user->dx;
            col[nc].i = i;   col[nc].j = j-1; val[nc++] = user->disper_coe/user->dy;
            col[nc].i = i;   col[nc].j = j;   val[nc++] = -fwc/user->dx + fthetac - user->disper_coe/user->dy;
          } else if (i== M-1 && j == 0) {
            col[nc].i = i-1; col[nc].j = j;   val[nc++] = -fwc/user->dx;
            col[nc].i = i;   col[nc].j = j+1; val[nc++] = -user->disper_coe/user->dy;
            col[nc].i = i;   col[nc].j = j;   val[nc++] =  fwc/user->dx + fthetac + user->disper_coe/user->dy;
          } else if (i == M-1 && j == N-1) {
            col[nc].i = i-1; col[nc].j = j;   val[nc++] = -fwc/user->dx;
            col[nc].i = i;   col[nc].j = j-1; val[nc++] =  user->disper_coe/user->dy;
            col[nc].i = i;   col[nc].j = j;   val[nc++] =  fwc/user->dx + fthetac - user->disper_coe/user->dy;
          } else if (i==0) {
            col[nc].i = i+1; col[nc].j = j;   val[nc++] = fwc/user->dx;
            col[nc].i = i;   col[nc].j = j+1; val[nc++] = -user->disper_coe/(2*user->dy);
            col[nc].i = i;   col[nc].j = j-1; val[nc++] =  user->disper_coe/(2*user->dy);
            col[nc].i = i;   col[nc].j = j;   val[nc++] = -fwc/user->dx + fthetac;
          } else if (i == M-1) {
            col[nc].i = i-1; col[nc].j = j;   val[nc++] = -fwc/user->dx;
            col[nc].i = i;   col[nc].j = j+1; val[nc++] = -user->disper_coe/(2*user->dy);
            col[nc].i = i;   col[nc].j = j-1; val[nc++] =  user->disper_coe/(2*user->dy);
            col[nc].i = i;   col[nc].j = j;   val[nc++] = fwc/user->dx + fthetac;
          } else if (j==0) {
            col[nc].i = i+1; col[nc].j = j;   val[nc++] = fwc/(2*user->dx);
            col[nc].i = i-1; col[nc].j = j;   val[nc++] = -fwc/(2*user->dx);
            col[nc].i = i;   col[nc].j = j+1; val[nc++] = -user->disper_coe/user->dy;
            col[nc].i = i;   col[nc].j = j;   val[nc++] = user->disper_coe/user->dy + fthetac;
          } else if (j == N-1) {
            col[nc].i = i+1; col[nc].j = j;   val[nc++] = fwc/(2*user->dx);
            col[nc].i = i-1; col[nc].j = j;   val[nc++] = -fwc/(2*user->dx);
            col[nc].i = i;   col[nc].j = j-1; val[nc++] = user->disper_coe/user->dy;
            col[nc].i = i;   col[nc].j = j;   val[nc++] = -user->disper_coe/user->dy + fthetac;
          }
        }
      } else {
        c1        = (yi-user->ws)/(2*user->dx);
        c3        = (user->ws/(2.0*user->H))*(user->PM_min - user->Pmax*PetscSinScalar(xi))/(2*user->dy);
        c5        = (PetscPowScalar((user->lambda*user->ws)/(2*user->H),2)*user->q*(1.0-PetscExpScalar(-t/user->lambda)))/(user->dy*user->dy);
        col[nc].i = i-1; col[nc].j = j;   val[nc++] = c1;
        col[nc].i = i+1; col[nc].j = j;   val[nc++] = -c1;
        col[nc].i = i;   col[nc].j = j-1; val[nc++] = c3 + c5;
        col[nc].i = i;   col[nc].j = j+1; val[nc++] = -c3 + c5;
        col[nc].i = i;   col[nc].j = j;   val[nc++] = -2*c5 -a;
      }
      ierr = MatSetValuesStencil(Jpre,1,&row,nc,col,val,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = DMDAVecRestoreArrayRead(cda,gc,&coors);CHKERRQ(ierr);

  ierr =  MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != Jpre) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Parameter_settings(AppCtx *user)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBeginUser;

  /* Set default parameters */
  user->ws     = 1.0;
  user->H      = 5.0;  user->Pmax   = 2.1;
  user->PM_min = 1.0;  user->lambda = 0.1;
  user->q      = 1.0;  user->mux    = PetscAsinScalar(user->PM_min/user->Pmax);
  user->sigmax = 0.1;
  user->sigmay = 0.1;  user->rho  = 0.0;
  user->t0     = 0.0;  user->dt   = .005; user->steps = 800; 
  user->xmin   = -1.0; user->xmax = 10.0;
  user->ymin   = -1.0; user->ymax = 10.0;
  user->bc     = 0;

  ierr = PetscOptionsGetScalar(NULL,NULL,"-ws",&user->ws,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-Inertia",&user->H,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-Pmax",&user->Pmax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-PM_min",&user->PM_min,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-lambda",&user->lambda,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-q",&user->q,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-mux",&user->mux,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-sigmax",&user->sigmax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-muy",&user->muy,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-sigmay",&user->sigmay,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-rho",&user->rho,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-t0",&user->t0,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-dt",&user->dt,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-xmin",&user->xmin,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-xmax",&user->xmax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-ymin",&user->ymin,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-ymax",&user->ymax,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-steps",&user->steps,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-bc_type",&user->bc,&flg);CHKERRQ(ierr);


  user->scale_u_x         = 1;
  user->scale_u_y         = 1;
  user->scale_y_u_x       = 1;
  user->scale_sin_x_u_y   = 1;
  user->scale_nexp_t_u_yy = 1;
  ierr = PetscOptionsGetScalar(NULL,NULL,"-scale_u_x",&user->scale_u_x,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-scale_u_y",&user->scale_u_y,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-scale_y_u_x",&user->scale_y_u_x,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-scale_sin_x_u_y",&user->scale_sin_x_u_y,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-scale_nexp_t_u_yy",&user->scale_nexp_t_u_yy,&flg);CHKERRQ(ierr);

  user->muy = user->ws;
  PetscFunctionReturn(0);
}