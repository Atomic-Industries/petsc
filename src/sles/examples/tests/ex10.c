
static char help[] = 
"This example calculates the stiffness matrix for a brick in three\n\
dimensions using 20 node serendipity elements and the equations of linear\n\
elasticity. This demonstrates use of MatGetSubMatrix() and the block\n\
diagonal data structure.\n\n";

#include "sles.h"
#include "petsc.h"
#include <stdio.h>

/* This code is not intended as an efficient implementation, it is only
   here to produce an interesting sparse matrix quickly.  As originally
   written, this program was too complex for Microsoft C Version 7 to
   handle.  It has been re-arranged to simplify the code. */

#define MAX(a,b)  ((a) > (b) ? (a) : (b))
#define ABS(a)    ((a) < 0.0 ? -(a) : (a))

int main(int argc,char **args)
{
  Mat     mat;
  int     ierr, i, its, m = 3, rdim, cdim, rstart, rend, mytid, numtids;
  Scalar  norm, v, one = 1.0, neg1 = -1.0;
  Vec     u, x, b;
  SLES    sles;
  KSP     ksp;

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,help);
  OptionsGetInt(0,"-m",&m);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);

  /* Form matrix */
  ierr = GetElasticityMatrix(m,&mat); CHKERRA(ierr);

  /* Generate vectors */
  ierr = MatGetSize(mat,&rdim,&cdim); CHKERRA(ierr);
  ierr = MatGetOwnershipRange(mat,&rstart,&rend); CHKERRA(ierr);
  ierr = VecCreate(MPI_COMM_WORLD,rdim,&u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b); CHKERRA(ierr);
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  for (i=rstart; i<rend; i++) {
    v = (Scalar)(i-rstart + 100*mytid); 
    ierr = VecSetValues(u,1,&i,&v,INSERTVALUES); CHKERRA(ierr);
  } 
  ierr = VecAssemblyBegin(u); CHKERRA(ierr);
  ierr = VecAssemblyEnd(u); CHKERRA(ierr);
  
  /* Compute right-hand-side */
  ierr = MatMult(mat,u,b); CHKERRA(ierr);
  
  /* Solve linear system */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,mat,mat,MAT_SAME_NONZERO_PATTERN);
          CHKERRA(ierr);
  ierr = SLESGetKSP(sles,&ksp); CHKERRA(ierr);

  { Scalar val;
    KSPSetInitialGuessNonzero(ksp);
    for (i=0; i<rdim; i++) {
      val = (Scalar)i;
      ierr = VecSetValues(x,1,&i,&val,INSERTVALUES); CHKERRA(ierr);
    }
    ierr = VecAssemblyBegin(x); CHKERRA(ierr);
    ierr = VecAssemblyEnd(x); CHKERRA(ierr);
  }

  ierr = KSPGMRESSetRestart(ksp,2*m); CHKERRA(ierr);
  ierr = KSPSetRelativeTolerance(ksp,1.e-12); CHKERRA(ierr);
  ierr = KSPSetMethod(ksp,KSPCG); CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
 
  /* Check error */
  ierr = VecAXPY(&neg1,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,&norm); CHKERRA(ierr);
  MPIU_printf(MPI_COMM_WORLD,"Norm of error %g, Number of iterations %d\n",norm,its);

  /* Free work space */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(mat); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
/* -------------------------------------------------------------------- */
/* 
  GetElasticityMatrix - Forms 3D linear elasticity matrix.
 */
int GetElasticityMatrix(int m,Mat *newmat)
{
  int     i,j,k,i1,i2,j1,j2,k1,k2,h1,h2,l1,l2,shiftx,shifty,shiftz,nzalloc;
  int     ict, nz, base, r1, r2, N, *rows, *rowkeep, nstart, ierr, mem;
  IS      iskeep;
  double  **K;
  Mat     mat, submat, *newmat1;

  m /= 2;   /* This is done just to be consistent with the old example */
  N = 3*(2*m+1)*(2*m+1)*(2*m+1);
  printf("m = %d, N=%d\n", m, N );
  ierr = MatCreateSequentialAIJ(MPI_COMM_SELF,N,N,80,0,&mat); CHKERRQ(ierr); 

  /* Form stiffness for element */
  K = (double **) PETSCMALLOC(81*sizeof(double *)); CHKPTRQ(K);
  for ( i=0; i<81; i++ ) {
    K[i] = (double *) PETSCMALLOC(81*sizeof(double)); CHKPTRQ(K[i]);
  }
  ierr = Elastic20Stiff(K); CHKERRQ(ierr);

  /* Loop over elements and add contribution to stiffness */
  shiftx = 3; shifty = 3*(2*m+1); shiftz = 3*(2*m+1)*(2*m+1);
  for ( k=0; k<m; k++ ) {
    for ( j=0; j<m; j++ ) {
      for ( i=0; i<m; i++ ) {
	h1 = 0; 
        base = 2*k*shiftz + 2*j*shifty + 2*i*shiftx;
	for ( k1=0; k1<3; k1++ ) {
	  for ( j1=0; j1<3; j1++ ) {
	    for ( i1=0; i1<3; i1++ ) {
	      h2 = 0;
	      r1 = base + i1*shiftx + j1*shifty + k1*shiftz;
	      for ( k2=0; k2<3; k2++ ) {
	        for ( j2=0; j2<3; j2++ ) {
	          for ( i2=0; i2<3; i2++ ) {
	            r2 = base + i2*shiftx + j2*shifty + k2*shiftz;
		    ierr = AddElement( mat, r1, r2, K, h1, h2 ); CHKERRA(ierr);
		    h2 += 3;
	          }
                }
              }
	      h1 += 3;
	    }
          }
        }
      }
    }
  }

  for ( i=0; i<81; i++ ) {
    PETSCFREE(K[i]);
  }
  PETSCFREE(K);

  ierr = MatAssemblyBegin(mat,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(mat,FINAL_ASSEMBLY); CHKERRA(ierr);

  /* Exclude any superfluous rows and columns */
  nstart = 3*(2*m+1)*(2*m+1);
  ict = 0;
  rowkeep = (int *) PETSCMALLOC((N-nstart)*sizeof(int)); CHKPTRQ(rowkeep);
  for (i=nstart; i<N; i++) {
    ierr = MatGetRow(mat,i,&nz,0,0); CHKERRA(ierr);
    if (nz) rowkeep[ict++] = i;
    ierr = MatRestoreRow(mat,i,&nz,0,0); CHKERRA(ierr);
  }
  ierr = ISCreateSequential(MPI_COMM_SELF,ict,rowkeep,&iskeep); CHKERRA(ierr);
  ierr = MatGetSubMatrix(mat,iskeep,iskeep,&submat); CHKERRA(ierr);
  PETSCFREE(rowkeep);
  ierr = MatDestroy(mat); CHKERRA(ierr);

  /* Convert storage formats -- just to demonstrate conversion to various
     formats (in particular, block diagonal storage) */
  { MatType type = MATBDIAG;
  if (OptionsHasName(0,"-mat_row")) type = MATROW; 
  if (OptionsHasName(0,"-mat_aij")) type = MATSAME;
  if (OptionsHasName(0,"-mat_dense")) type = MATDENSE;
  if (OptionsHasName(0,"-mat_mpiaij")) type = MATMPIAIJ;
  if (OptionsHasName(0,"-mat_mpirow")) type = MATMPIROW;
  if (OptionsHasName(0,"-mat_mpirowbs")) type = MATMPIROW_BS;
  if (OptionsHasName(0,"-mat_mpibdiag")) type = MATMPIBDIAG;
  ierr = MatConvert(submat,type,newmat); CHKERRQ(ierr);
  ierr = MatDestroy(submat); CHKERRQ(ierr);
  }

  /* Display matrix information and nonzero structure */
  MatGetInfo(*newmat,MAT_LOCAL,&nz,&nzalloc,&mem); CHKERRA(ierr);
  printf("matrix nonzeros = %d, allocated nonzeros = %d, memory = %d bytes\n",
          nz,nzalloc,mem);
  return 0;
}
/* -------------------------------------------------------------------- */
int AddElement(Mat mat,int r1,int r2,Scalar **K,int h1,int h2)
{
  Scalar val;
  int    l1, l2, row, col, ierr;

  for ( l1=0; l1<3; l1++ ) {
    for ( l2=0; l2<3; l2++ ) {
      if (ABS(K[h1+l1][h2+l2]) != 0.0) {
        row = r1+l1; col = r2+l2; val = K[h1+l1][h2+l2]; 
	ierr = MatSetValues(mat,1,&row,1,&col,&val,ADDVALUES); CHKERRQ(ierr);
        row = r2+l2; col = r1+l1;
	ierr = MatSetValues(mat,1,&row,1,&col,&val,ADDVALUES); CHKERRQ(ierr);
      }
    }
  }
  return 0;
}
/* -------------------------------------------------------------------- */
double	N[20][64];	   /* Interpolation function. */
double	part_N[3][20][64]; /* Partials of interpolation function. */
double	rst[3][64];	   /* Location of integration pts in (r,s,t) */
double	weight[64];	   /* Gaussian quadrature weights. */
double	xyz[20][3];	   /* (x,y,z) coordinates of nodes  */
double	E, nu;		   /* Physcial constants. */
int	n_int, N_int;	   /* N_int = n_int^3, number of int. pts. */
/* Ordering of the vertices, (r,s,t) coordinates, of the canonical cell. */
double	r2[20] = {-1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 1.0,
                 -1.0, 1.0, -1.0, 1.0, 
                 -1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 1.0};
double	s2[20] = {-1.0, -1.0,  -1.0, 0.0, 0.0, 1.0,  1.0,  1.0,
                 -1.0, -1.0, 1.0, 1.0,
                 -1.0, -1.0,  -1.0, 0.0, 0.0, 1.0,  1.0,  1.0};
double	t2[20] =  {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                 0.0, 0.0, 0.0, 0.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
int     rmap[20] = {0,1,2,3,5,6,7,8,9,11,15,17,18,19,20,21,23,24,25,26};
/* -------------------------------------------------------------------- */
/* 
  Elastic20Stiff - Forms 20 node elastic stiffness for element.
 */
int Elastic20Stiff(Ke)
double **Ke;
{
  double                 K[60][60],x,y,z,dx,dy,dz,m,v;
  int                    i,j,k,l,I,J;

  paulsetup20();

  x = -1.0;  y = -1.0; z = -1.0; dx = 2.0; dy = 2.0; dz = 2.0;
  xyz[0][0] = x;          xyz[0][1] = y;          xyz[0][2] = z;
  xyz[1][0] = x + dx;     xyz[1][1] = y;          xyz[1][2] = z;
  xyz[2][0] = x + 2.*dx;  xyz[2][1] = y;          xyz[2][2] = z;
  xyz[3][0] = x;          xyz[3][1] = y + dy;     xyz[3][2] = z;
  xyz[4][0] = x + 2.*dx;  xyz[4][1] = y + dy;     xyz[4][2] = z;
  xyz[5][0] = x;          xyz[5][1] = y + 2.*dy;  xyz[5][2] = z;
  xyz[6][0] = x + dx;     xyz[6][1] = y + 2.*dy;  xyz[6][2] = z;
  xyz[7][0] = x + 2.*dx;  xyz[7][1] = y + 2.*dy;  xyz[7][2] = z;
  xyz[8][0] = x;          xyz[8][1] = y;          xyz[8][2] = z + dz;
  xyz[9][0] = x + 2.*dx;  xyz[9][1] = y;          xyz[9][2] = z + dz;
  xyz[10][0] = x;         xyz[10][1] = y + 2.*dy; xyz[10][2] = z + dz;
  xyz[11][0] = x + 2.*dx; xyz[11][1] = y + 2.*dy; xyz[11][2] = z + dz;
  xyz[12][0] = x;         xyz[12][1] = y;         xyz[12][2] = z + 2.*dz;
  xyz[13][0] = x + dx;    xyz[13][1] = y;         xyz[13][2] = z + 2.*dz;
  xyz[14][0] = x + 2.*dx; xyz[14][1] = y;         xyz[14][2] = z + 2.*dz;
  xyz[15][0] = x;         xyz[15][1] = y + dy;    xyz[15][2] = z + 2.*dz;
  xyz[16][0] = x + 2.*dx; xyz[16][1] = y + dy;    xyz[16][2] = z + 2.*dz;
  xyz[17][0] = x;         xyz[17][1] = y + 2.*dy; xyz[17][2] = z + 2.*dz;
  xyz[18][0] = x + dx;    xyz[18][1] = y + 2.*dy; xyz[18][2] = z + 2.*dz;
  xyz[19][0] = x + 2.*dx; xyz[19][1] = y + 2.*dy; xyz[19][2] = z + 2.*dz;
  paulintegrate20(K);

  /* copy the stiffness from K into format used by Ke */
  for ( i=0; i<81; i++ ) {
    for ( j=0; j<81; j++ ) {
          Ke[i][j] = 0.0;
    }
  }
  I = 0;
  m = 0.0;
  for ( i=0; i<20; i++ ) {
    J = 0;
    for ( j=0; j<20; j++ ) {
      for ( k=0; k<3; k++ ) {
        for ( l=0; l<3; l++ ) {
          Ke[3*rmap[i]+k][3*rmap[j]+l] = v = K[I+k][J+l];
          m = MAX(m,ABS(v));
        }
      }
      J += 3;
    }
    I += 3;
  }
  /* zero out the extremely small values */
  m = (1.e-8)*m;
  for ( i=0; i<81; i++ ) {
    for ( j=0; j<81; j++ ) {
      if (ABS(Ke[i][j]) < m)  Ke[i][j] = 0.0;
    }
  }  
  /* force the matrix to be exactly symmetric */
  for ( i=0; i<81; i++ ) {
    for ( j=0; j<i; j++ ) {
      Ke[i][j] = ( Ke[i][j] + Ke[j][i] )/2.0;
    }
  } 
  return 0;
}
/* -------------------------------------------------------------------- */
/* 
  paulsetup20 - Sets up data structure for forming local elastic stiffness.
 */
int paulsetup20()
{
  int     i, j, k, cnt;
  double  x[4],w[4];
  double  c;

  n_int = 3;
  nu = 0.3;
  E = 1.0;

  /* Assign integration points and weights for
       Gaussian quadrature formulae. */
  if(n_int == 2)  {
		x[0] = (-0.577350269189626);
		x[1] = (0.577350269189626);
		w[0] = 1.0000000;
		w[1] = 1.0000000;
  }
  else if(n_int == 3) {
		x[0] = (-0.774596669241483);
		x[1] = 0.0000000;
		x[2] = 0.774596669241483;
		w[0] = 0.555555555555555;
		w[1] = 0.888888888888888;
		w[2] = 0.555555555555555;
  }
  else if(n_int == 4) {
		x[0] = (-0.861136311594053);
		x[1] = (-0.339981043584856);
		x[2] = 0.339981043584856;
		x[3] = 0.861136311594053;
		w[0] = 0.347854845137454;
		w[1] = 0.652145154862546;
		w[2] = 0.652145154862546;
		w[3] = 0.347854845137454;
  }
  else {
    SETERRQ(1,"Unknown value for n_int"); return -1;
  }

  /* rst[][i] contains the location of the i-th integration point
      in the canonical (r,s,t) coordinate system.  weight[i] contains
      the Gaussian weighting factor. */

  cnt = 0;
  for (i=0; i<n_int;i++) {
    for (j=0; j<n_int;j++) {
      for (k=0; k<n_int;k++) {
        rst[0][cnt]=x[i];
        rst[1][cnt]=x[j];
        rst[2][cnt]=x[k];
        weight[cnt] = w[i]*w[j]*w[k];
        ++cnt;
      }
    }
  }
  N_int = cnt;

  /* N[][j] is the interpolation vector, N[][j] .* xyz[] */
  /* yields the (x,y,z)  locations of the integration point. */
  /*  part_N[][][j] is the partials of the N function */
  /*  w.r.t. (r,s,t). */

  c = 1.0/8.0;
  for (j=0; j<N_int; j++) {
    for (i=0; i<20; i++) {
      if ( i==0 || i==2 || i==5 || i==7 || i==12 || i==14 || i== 17 || i==19 ){ 
        N[i][j] = c*(1.0 + r2[i]*rst[0][j])*
                (1.0 + s2[i]*rst[1][j])*(1.0 + t2[i]*rst[2][j])*
                (-2.0 + r2[i]*rst[0][j] + s2[i]*rst[1][j] + t2[i]*rst[2][j]);
        part_N[0][i][j] = c*r2[i]*(1 + s2[i]*rst[1][j])*(1 + t2[i]*rst[2][j])*
                 (-1.0 + 2.0*r2[i]*rst[0][j] + s2[i]*rst[1][j] + 
                 t2[i]*rst[2][j]);
        part_N[1][i][j] = c*s2[i]*(1 + r2[i]*rst[0][j])*(1 + t2[i]*rst[2][j])*
                 (-1.0 + r2[i]*rst[0][j] + 2.0*s2[i]*rst[1][j] + 
                 t2[i]*rst[2][j]);
        part_N[2][i][j] = c*t2[i]*(1 + r2[i]*rst[0][j])*(1 + s2[i]*rst[1][j])*
                 (-1.0 + r2[i]*rst[0][j] + s2[i]*rst[1][j] + 
                 2.0*t2[i]*rst[2][j]);
      }
      else if ( i==1 || i==6 || i==13 || i==18 ) {
        N[i][j] = .25*(1.0 - rst[0][j]*rst[0][j])*
                (1.0 + s2[i]*rst[1][j])*(1.0 + t2[i]*rst[2][j]);
        part_N[0][i][j] = -.5*rst[0][j]*(1 + s2[i]*rst[1][j])*
                         (1 + t2[i]*rst[2][j]);
        part_N[1][i][j] = .25*s2[i]*(1 + t2[i]*rst[2][j])*
                          (1.0 - rst[0][j]*rst[0][j]);
        part_N[2][i][j] = .25*t2[i]*(1.0 - rst[0][j]*rst[0][j])*
                          (1 + s2[i]*rst[1][j]);
      }
      else if ( i==3 || i==4 || i==15 || i==16 ) {
        N[i][j] = .25*(1.0 - rst[1][j]*rst[1][j])*
                (1.0 + r2[i]*rst[0][j])*(1.0 + t2[i]*rst[2][j]);
        part_N[0][i][j] = .25*r2[i]*(1 + t2[i]*rst[2][j])*
                           (1.0 - rst[1][j]*rst[1][j]);
        part_N[1][i][j] = -.5*rst[1][j]*(1 + r2[i]*rst[0][j])*
                           (1 + t2[i]*rst[2][j]);
        part_N[2][i][j] = .25*t2[i]*(1.0 - rst[1][j]*rst[1][j])*
                          (1 + r2[i]*rst[0][j]);
      }
      else if ( i==8 || i==9 || i==10 || i==11 ) {
        N[i][j] = .25*(1.0 - rst[2][j]*rst[2][j])*
                (1.0 + r2[i]*rst[0][j])*(1.0 + s2[i]*rst[1][j]);
        part_N[0][i][j] = .25*r2[i]*(1 + s2[i]*rst[1][j])*
                          (1.0 - rst[2][j]*rst[2][j]);
        part_N[1][i][j] = .25*s2[i]*(1.0 - rst[2][j]*rst[2][j])*
                          (1 + r2[i]*rst[0][j]);
        part_N[2][i][j] = -.5*rst[2][j]*(1 + r2[i]*rst[0][j])*
                           (1 + s2[i]*rst[1][j]);
      }
    }
  }
  return 0;
}
/* -------------------------------------------------------------------- */
/* 
   paulintegrate20 - Does actual numerical integration on 20 node element.
 */
int paulintegrate20(K)
double K[60][60];
{
  double  det_jac, jac[3][3], inv_jac[3][3], v[3];
  double  B[6][60], B_temp[6][60], C[6][6];
  double  temp;
  int     i, j, k, step;

  /* Zero out K, since we will accumulate the result here */
  for (i=0; i<60; i++) {
    for (j=0; j<60; j++) {
      K[i][j] = 0.0;
    }
  }

  /* Loop over integration points ... */
  for (step=0; step<N_int; step++) {

    /* Compute the Jacobian, its determinant, and inverse. */
    for (i=0; i<3; i++) {
      for (j=0; j<3; j++) {
        jac[i][j] = 0;
        for (k=0; k<20; k++) {
          jac[i][j] += part_N[i][k][step]*xyz[k][j];
        }
      }
    }
    det_jac = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])
              + jac[0][1]*(jac[1][2]*jac[2][0]-jac[1][0]*jac[2][2])
              + jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
    inv_jac[0][0] = (jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])/det_jac;
    inv_jac[0][1] = (jac[0][2]*jac[2][1]-jac[0][1]*jac[2][2])/det_jac;
    inv_jac[0][2] = (jac[0][1]*jac[1][2]-jac[1][1]*jac[0][2])/det_jac;
    inv_jac[1][0] = (jac[1][2]*jac[2][0]-jac[1][0]*jac[2][2])/det_jac;
    inv_jac[1][1] = (jac[0][0]*jac[2][2]-jac[2][0]*jac[0][2])/det_jac;
    inv_jac[1][2] = (jac[0][2]*jac[1][0]-jac[0][0]*jac[1][2])/det_jac;
    inv_jac[2][0] = (jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0])/det_jac;
    inv_jac[2][1] = (jac[0][1]*jac[2][0]-jac[0][0]*jac[2][1])/det_jac;
    inv_jac[2][2] = (jac[0][0]*jac[1][1]-jac[1][0]*jac[0][1])/det_jac;

    /* Compute the B matrix. */
    for (i=0; i<3; i++) {
      for (j=0; j<20; j++) {
        B_temp[i][j] = 0.0;
        for (k=0; k<3; k++) {
          B_temp[i][j] += inv_jac[i][k]*part_N[k][j][step];
        }
      }
    }
    for (i=0; i<6; i++) {
      for (j=0; j<60; j++) {
        B[i][j] = 0.0;
      }
    }

    /* Put values in correct places in B. */
    for (k=0; k<20; k++) {
      B[0][3*k]   = B_temp[0][k];
      B[1][3*k+1] = B_temp[1][k];
      B[2][3*k+2] = B_temp[2][k];
      B[3][3*k]   = B_temp[1][k];
      B[3][3*k+1] = B_temp[0][k];
      B[4][3*k+1] = B_temp[2][k];
      B[4][3*k+2] = B_temp[1][k];
      B[5][3*k]   = B_temp[2][k];
      B[5][3*k+2] = B_temp[0][k];
    }
  
    /* Construct the C matrix, uses the constants "nu" and "E". */
    for (i=0; i<6; i++) {
      for (j=0; j<6; j++) {
        C[i][j] = 0.0;
      }
    }
    temp = (1.0 + nu)*(1.0 - 2.0*nu);
    temp = E/temp;
    C[0][0] = temp*(1.0 - nu);
    C[1][1] = C[0][0];
    C[2][2] = C[0][0];
    C[3][3] = temp*(0.5 - nu);
    C[4][4] = C[3][3];
    C[5][5] = C[3][3];
    C[0][1] = temp*nu;
    C[0][2] = C[0][1];
    C[1][0] = C[0][1];
    C[1][2] = C[0][1];
    C[2][0] = C[0][1];
    C[2][1] = C[0][1];
  
    for (i=0; i<6; i++) {
      for (j=0; j<60; j++) {
        B_temp[i][j] = 0.0;
        for (k=0; k<6; k++) {
          B_temp[i][j] += C[i][k]*B[k][j];
        }
        B_temp[i][j] *= det_jac;
      }
    }

    /* Accumulate B'*C*B*det(J)*weight, as a function of (r,s,t), in K. */
    for (i=0; i<60; i++) {
      for (j=0; j<60; j++) {
        temp = 0.0;
        for (k=0; k<6; k++) {
          temp += B[k][i]*B_temp[k][j];
        }
        K[i][j] += temp*weight[step];
      }
    }
  }  /* end of loop over integration points */
  return 0;
}

