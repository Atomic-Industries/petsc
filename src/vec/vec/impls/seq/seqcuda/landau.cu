/*
   Implements the Landau kernal
*/
#include <petscconf.h>
#include <petsc/private/dmpleximpl.h>   /*I   "petscdmplex.h"   I*/
#include <petsc/private/vecimpl.h>      /* put CUDA stuff in veccuda */
#include <../src/mat/impls/aij/seq/aij.h>  /* put CUDA SeqAIJ */
#include <omp.h>

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)
// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaDeviceSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

texture<int, 2, cudaReadModeElementType> tex;

__global__ void kernel (int m, int n)
{
    int val;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            val = tex2D (tex, col+0.5f, row+0.5f);
            printf ("%3d  ", val);
        }
        printf ("\n");
    }
}

PetscErrorCode FPCUDATest ()
{
  PetscInt m = 4; // height = #rows
  PetscInt n = 3; // width  = #columns
  size_t pitch, tex_ofs;
  PetscInt arr[4][3]= {{10, 11, 12},
		       {20, 21, 22},
		       {30, 31, 32},
		       {40, 41, 42}};
  PetscInt *arr_d = 0;
  PetscFunctionBegin;
  CUDA_SAFE_CALL(cudaMallocPitch((void**)&arr_d,&pitch,n*sizeof(*arr_d),m));
  CUDA_SAFE_CALL(cudaMemcpy2D(arr_d, pitch, arr, n*sizeof(arr[0][0]),
			      n*sizeof(arr[0][0]),m,cudaMemcpyHostToDevice));
  tex.normalized = false;
  CUDA_SAFE_CALL (cudaBindTexture2D (&tex_ofs, &tex, arr_d, &tex.channelDesc,
				     n, m, pitch));
  if (tex_ofs !=0) {
    printf ("tex_ofs = %zu\n", tex_ofs);
    return EXIT_FAILURE;
  }
  printf ("reading texture:\n");
  kernel<<<1,1>>>(m, n);
  CHECK_LAUNCH_ERROR();
  CUDA_SAFE_CALL (cudaDeviceSynchronize());
  PetscFunctionReturn(EXIT_SUCCESS);
}

#if defined(__INTEL_COMPILER)
#if defined(PETSC_USE_REAL_SINGLE)
#define MYINVSQRT(q) invsqrtf(q)
#define MYSQRT(q) sqrtf(q)
#else
#define MYINVSQRT(q) invsqrt(q)
#define MYSQRT(q) sqrt(q)
#endif
#else
#define MYINVSQRT(q) (1./PetscSqrtReal(q))
#define MYSQRT(q) PetscSqrtReal(q)
#endif

#if defined(PETSC_USE_REAL_SINGLE)
#define MYSMALL 1.e-12F
#define MYVERYSMALL 1.e-30F
#define MYLOG logf
            #else
#define MYSMALL 1.e-12
#define MYVERYSMALL 1.e-300
#define MYLOG log
#endif

#if defined(PETSC_USE_REAL_SINGLE)
__constant__ PetscReal P2[] = {
  1.53552577301013293365E-4F,
  2.50888492163602060990E-3F,
  8.68786816565889628429E-3F,
  1.07350949056076193403E-2F,
  7.77395492516787092951E-3F,
  7.58395289413514708519E-3F,
  1.15688436810574127319E-2F,
  2.18317996015557253103E-2F,
  5.68051945617860553470E-2F,
  4.43147180560990850618E-1F,
  1.00000000000000000299E0F
};
__constant__ PetscReal Q2[] = {
  3.27954898576485872656E-5F,
  1.00962792679356715133E-3F,
  6.50609489976927491433E-3F,
  1.68862163993311317300E-2F,
  2.61769742454493659583E-2F,
  3.34833904888224918614E-2F,
  4.27180926518931511717E-2F,
  5.85936634471101055642E-2F,
  9.37499997197644278445E-2F,
  2.49999999999888314361E-1F
};
#else
__constant__ PetscReal P2[] = {
  1.53552577301013293365E-4,
  2.50888492163602060990E-3,
  8.68786816565889628429E-3,
  1.07350949056076193403E-2,
  7.77395492516787092951E-3,
  7.58395289413514708519E-3,
  1.15688436810574127319E-2,
  2.18317996015557253103E-2,
  5.68051945617860553470E-2,
  4.43147180560990850618E-1,
  1.00000000000000000299E0
};
__constant__ PetscReal Q2[] = {
  3.27954898576485872656E-5,
  1.00962792679356715133E-3,
  6.50609489976927491433E-3,
  1.68862163993311317300E-2,
  2.61769742454493659583E-2,
  3.34833904888224918614E-2,
  4.27180926518931511717E-2,
  5.85936634471101055642E-2,
  9.37499997197644278445E-2,
  2.49999999999888314361E-1
};
#endif
#if defined(PETSC_USE_REAL_SINGLE)
__constant__ PetscReal P1[] =
{
 1.37982864606273237150E-4F,
 2.28025724005875567385E-3F,
 7.97404013220415179367E-3F,
 9.85821379021226008714E-3F,
 6.87489687449949877925E-3F,
 6.18901033637687613229E-3F,
 8.79078273952743772254E-3F,
 1.49380448916805252718E-2F,
 3.08851465246711995998E-2F,
 9.65735902811690126535E-2F,
 1.38629436111989062502E0F
};
__constant__ PetscReal Q1[] =
{
 2.94078955048598507511E-5F,
 9.14184723865917226571E-4F,
 5.94058303753167793257E-3F,
 1.54850516649762399335E-2F,
 2.39089602715924892727E-2F,
 3.01204715227604046988E-2F,
 3.73774314173823228969E-2F,
 4.88280347570998239232E-2F,
 7.03124996963957469739E-2F,
 1.24999999999870820058E-1F,
 4.99999999999999999821E-1F
};
#else
__constant__ PetscReal P1[] =
{
 1.37982864606273237150E-4,
 2.28025724005875567385E-3,
 7.97404013220415179367E-3,
 9.85821379021226008714E-3,
 6.87489687449949877925E-3,
 6.18901033637687613229E-3,
 8.79078273952743772254E-3,
 1.49380448916805252718E-2,
 3.08851465246711995998E-2,
 9.65735902811690126535E-2,
 1.38629436111989062502E0
};
__constant__ PetscReal Q1[] =
{
 2.94078955048598507511E-5,
 9.14184723865917226571E-4,
 5.94058303753167793257E-3,
 1.54850516649762399335E-2,
 2.39089602715924892727E-2,
 3.01204715227604046988E-2,
 3.73774314173823228969E-2,
 4.88280347570998239232E-2,
 7.03124996963957469739E-2,
 1.24999999999870820058E-1,
 4.99999999999999999821E-1
};
#endif

/* elliptic functions
 */
__device__ PetscReal polevl_10( PetscReal x, PetscReal coef[] )
{
  PetscReal ans;
  int       i;
  ans = coef[0];
  for (i=1; i<11; i++) ans = ans * x + coef[i];
  return( ans );
}
__device__ PetscReal polevl_9( PetscReal x, PetscReal coef[] )
{
  PetscReal ans;
  int       i;
  ans = coef[0];
  for (i=1; i<10; i++) ans = ans * x + coef[i];
  return( ans );
}
/*
 *	Complete elliptic integral of the second kind
 */
__device__ void ellipticE(PetscReal x,PetscReal *ret)
{
  x = 1 - x; /* where m = 1 - m1 */
  *ret = polevl_10(x,P2) - MYLOG(x) * (x * polevl_9(x,Q2));
}
/*
 *	Complete elliptic integral of the first kind
 */
__device__ void ellipticK(PetscReal x,PetscReal *ret)
{
  x = 1 - x; /* where m = 1 - m1 */
  *ret = polevl_10(x,P1) - MYLOG(x) * polevl_10(x,Q1);
}

/* integration point functions */
/* Evaluates the tensor U=(I-(x-y)(x-y)/(x-y)^2)/|x-y| at point x,y */
/* if x==y we will return zero. This is not the correct result */
/* since the tensor diverges for x==y but when integrated */
/* the divergent part is antisymmetric and vanishes. This is not  */
/* trivial, but can be proven. */
__device__ void LandauTensor3D(const PetscReal x1[], const PetscReal xp, const PetscReal yp, const PetscReal zp, PetscReal U[][3], PetscReal mask)
{
  PetscReal dx[3],inorm3,inorm,inorm2,norm2,x2[] = {xp,yp,zp};
  PetscInt  d;
  for (d = 0, norm2 = MYVERYSMALL; d < 3; ++d) {
    dx[d] = x2[d] - x1[d];
    norm2 += dx[d] * dx[d];
  }
  inorm2 = mask/norm2;
  inorm = MYSQRT(inorm2);
  inorm3 = inorm2*inorm;
  for (d = 0; d < 3; ++d) U[d][d] = inorm - inorm3 * dx[d] * dx[d];
  U[1][0] = U[0][1] = -inorm3 * dx[0] * dx[1];
  U[1][2] = U[2][1] = -inorm3 * dx[2] * dx[1];
  U[2][0] = U[0][2] = -inorm3 * dx[0] * dx[2];
}

__device__ void LandauTensor2D(const PetscReal x[], const PetscReal rp, const PetscReal zp, PetscReal Ud[][2], PetscReal Uk[][2], const PetscReal mask)
{
  PetscReal l,s,r=x[0],z=x[1],i1func,i2func,i3func,ks,es,pi4pow,sqrt_1s,r2,rp2,r2prp2,zmzp,zmzp2,tt;
  //PetscReal mask /* = !!(r!=rp || z!=zp) */;
  /* !!(zmzp2 > 1.e-12 || (r-rp) >  1.e-12 || (r-rp) < -1.e-12); */
  r2=PetscSqr(r);
  zmzp=z-zp;
  rp2=PetscSqr(rp);
  zmzp2=PetscSqr(zmzp);
  r2prp2=r2+rp2;
  l = r2 + rp2 + zmzp2;
  /* if      ( zmzp2 >  MYSMALL) mask = 1; */
  /* else if ( (tt=(r-rp)) >  MYSMALL) mask = 1; */
  /* else if (  tt         < -MYSMALL) mask = 1; */
  /* else mask = 0; */
  s = mask*2*r*rp/l; /* mask for vectorization */
  tt = 1./(1+s);
  pi4pow = 4*M_PI*MYINVSQRT(PetscSqr(l)*l);
  sqrt_1s = MYSQRT(1.+s);
  /* sp.ellipe(2.*s/(1.+s)) */
  ellipticE(2*s*tt,&es); /* 44 flops * 2 + 75 = 163 flops including 2 logs, 1 sqrt, 1 pow, 21 mult */
  /* sp.ellipk(2.*s/(1.+s)) */
  ellipticK(2*s*tt,&ks); /* 44 flops + 75 in rest, 21 mult */
  /* mask is needed here just for single precision */
  i2func = 2./((1-s)*sqrt_1s) * es;
  i1func = 4./(PetscSqr(s)*sqrt_1s + MYVERYSMALL) * mask * ( ks - (1.+s) * es);
  i3func = 2./((1-s)*(s)*sqrt_1s + MYVERYSMALL) * (es - (1-s) * ks);
  Ud[0][0]=                    pi4pow*(rp2*i1func+PetscSqr(zmzp)*i2func);
  Ud[0][1]=Ud[1][0]=Uk[0][1]= -pi4pow*(zmzp)*(r*i2func-rp*i3func);
  Uk[1][1]=Ud[1][1]=           pi4pow*((r2prp2)*i2func-2*r*rp*i3func)*mask;
  Uk[0][0]=                    pi4pow*(zmzp2*i3func+r*rp*i1func);
  Uk[1][0]=                   -pi4pow*(zmzp)*(r*i3func-rp*i2func); /* 48 mults + 21 + 21 = 90 mults and divs */
}

typedef struct {
  PetscReal    *v; // x[3]
  //PetscReal    *f[FP_MAX_SPECIES];
  //PetscReal    *df[3][FP_MAX_SPECIES];
  //PetscInt     nip; /* number of integration points */
  //PetscInt     ns; /* number of species or fields */
  //PetscInt     dim;
} FPLandPointDataFlat;
static PetscErrorCode FPLandPointDataCreateDevice(FPLandPointDataFlat *ld, const FPLandPointData * const src)
{
  PetscErrorCode ierr;
  int            s,idx,d;
  const PetscInt pntsz = src->dim*src->ns + src->ns + src->dim, totsz = src->nip*sizeof(PetscReal)*pntsz;
  PetscReal      *newv,*pp;
  PetscFunctionBeginUser;
  ierr = PetscMalloc(totsz,&newv);CHKERRQ(ierr);
  pp = newv;
  for (idx=0;idx<src->nip;idx++) {
    for (d=0;d<src->dim;d++) *pp++ = src->x[d][idx];
    for (s=0;s<src->ns;s++)  *pp++ = src->f[s][idx];
    for (s=0;s<src->ns;s++) {
      for (d=0;d<src->dim;d++) *pp++ = src->df[d][s][idx];
    }
  }
  CUDA_SAFE_CALL(cudaMalloc((void **)&ld->v, totsz));
  CUDA_SAFE_CALL(cudaMemcpy(ld->v, newv, totsz, cudaMemcpyHostToDevice));
  ierr = PetscFree(newv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode FPLandPointDataDestroyDevice(FPLandPointDataFlat *ld)
{
  PetscFunctionBeginUser;
  CUDA_SAFE_CALL(cudaFree(ld->v));
  PetscFunctionReturn(0);
}

#if !defined(FP_DIM)
#define FP_DIM 2
#endif
// #define FP_USE_SHARED_GPU_MEM
//
// The GPU Landau kernel
//
__global__
void land_kernel(const PetscInt nip, const PetscInt dim, const PetscInt totDim, const PetscInt Nf, const PetscInt Nb, const PetscInt nSubBlocks_dummy,
		 const PetscReal vj[], const PetscReal Jj[], const PetscReal invJj[],
		 const PetscReal *a_nu_m0_ma, const PetscReal invMass[FP_MAX_SPECIES], const PetscReal Eq_m[FP_MAX_SPECIES],
		 const PetscReal * const a_TabBD, const PetscReal quadWeights[], const PetscInt foffsets[],
		 const FPLandPointDataFlat IPDataGlobal, const PetscReal wiGlobal[],
#if !defined(FP_USE_SHARED_GPU_MEM)
		 PetscReal *g2arr, PetscReal *g3arr,
#endif
		 PetscBool quarter3DDomain, PetscScalar elemMats_out[])
{
  const PetscInt  Nq = blockDim.x, myelem = blockIdx.x;
#if defined(FP_USE_SHARED_GPU_MEM)
  extern __shared__ PetscReal g2_g3_qi[]; // Nq * { [NSubBlocks][Nf][dim] ; [NSubBlocks][Nf][dim][dim] }
  PetscReal       (*g2)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM]         = (PetscReal (*)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM])         &g2_g3_qi[0];
  PetscReal       (*g3)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM][FP_DIM] = (PetscReal (*)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM][FP_DIM]) &g2_g3_qi[FP_MAX_SUB_THREAD_BLOCKS*FP_MAX_NQ*FP_MAX_SPECIES*FP_DIM];
#else
  PetscReal       (*g2)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM]         = (PetscReal (*)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM])         &g2arr[myelem*FP_MAX_SUB_THREAD_BLOCKS*FP_MAX_NQ*FP_MAX_SPECIES*FP_DIM       ];
  PetscReal       (*g3)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM][FP_DIM] = (PetscReal (*)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM][FP_DIM]) &g3arr[myelem*FP_MAX_SUB_THREAD_BLOCKS*FP_MAX_NQ*FP_MAX_SPECIES*FP_DIM*FP_DIM];
#endif
  const PetscInt  mythread = threadIdx.x + blockDim.x*threadIdx.y, myqi = threadIdx.x, mySubBlk = threadIdx.y, nSubBlocks = blockDim.y;
  const PetscInt  jpidx = myqi + myelem * Nq;
  const PetscInt  pntsz = dim*Nf + Nf + dim; // x[dim], f[Ns], df[dim*Nf]
  const PetscInt  subblocksz = nip/nSubBlocks + !!(nip%nSubBlocks), ip_start = mySubBlk*subblocksz, ip_end = (mySubBlk+1)*subblocksz > nip ? nip : (mySubBlk+1)*subblocksz; /* this could be wrong with very few global IPs */
  PetscReal       nu_m0_ma[FP_MAX_SPECIES][FP_MAX_SPECIES];
  const PetscReal *iTab,*TabBD[FP_MAX_SPECIES][2],*pvj = &vj[jpidx*dim];
  const PetscReal wj = wiGlobal[jpidx];
  PetscReal       gg2[FP_MAX_SPECIES][FP_DIM],gg3[FP_MAX_SPECIES][FP_DIM][FP_DIM];
  PetscInt        d,f,d2,dp,d3,fieldB,ipidx,fieldA;
  for (iTab = a_TabBD, fieldA = 0 ; fieldA < Nf ; fieldA++, iTab += Nq*Nb*(1+dim)) { // get pointers for convenience
    for (fieldB = 0; fieldB < Nf; ++fieldB) {
      nu_m0_ma[fieldA][fieldB] = a_nu_m0_ma[fieldA*FP_MAX_SPECIES + fieldB];
    }
    TabBD[fieldA][0] = iTab;
    TabBD[fieldA][1] = &iTab[Nq*Nb];
  }
  // create g2 & g3
  for (d=0;d<dim;d++) { // clear accumulation data D & K
    for (f=0;f<Nf;f++) {
      gg2[f][d] = 0;
      for (d2=0;d2<dim;d2++) gg3[f][d][d2] = 0;
    }
  }
  const PetscReal * __restrict__ data = IPDataGlobal.v + ip_start*pntsz;
  for (ipidx = ip_start; ipidx < ip_end; ++ipidx, data += pntsz) {
    const PetscReal wi = wiGlobal[ipidx];
    const PetscReal * __restrict__ f  = &data[dim];
    const PetscReal * __restrict__ df = &data[dim + Nf];
    if (dim==2) {
      PetscReal       Ud[2][2], Uk[2][2];
      LandauTensor2D(pvj, data[0], data[1], Ud, Uk, (ipidx==jpidx) ? 0. : 1.);
      for (fieldA = 0; fieldA < Nf; ++fieldA) {
       for (fieldB = 0; fieldB < Nf; ++fieldB) {
         for (d2 = 0; d2 < 2; ++d2) {
           for (d3 = 0; d3 < 2; ++d3) {
             /* K = U * grad(f): g2=e: i,A */
             gg2[fieldA][d2] += nu_m0_ma[fieldA][fieldB] * invMass[fieldB] * Uk[d2][d3] * df[d3 + fieldB*dim] * wi;
             /* D = -U * (I \kron (fx)): g3=f: i,j,A */
             gg3[fieldA][d2][d3] -= nu_m0_ma[fieldA][fieldB] * invMass[fieldA] * Ud[d2][d3] * f[fieldB] * wi;
           }
         }
       }
      }
    } else {
      PetscReal U[3][3], R[2][2] = {{-1,1},{1,-1}};
      if (!quarter3DDomain) {
      LandauTensor3D(pvj,data[0], data[1], data[2],U, (ipidx==jpidx) ? 0. : 1.);
      for (fieldA = 0; fieldA < Nf; ++fieldA) {
	for (fieldB = 0; fieldB < Nf; ++fieldB) {
	  for (d2 = 0; d2 < 3; ++d2) {
	    for (d3 = 0; d3 < 3; ++d3) {
	      /* K = U * grad(f): g2 = e: i,A */
	      gg2[fieldA][d2] += nu_m0_ma[fieldA][fieldB] * invMass[fieldB] * U[d2][d3] * df[d3 + fieldB*3] * wi;
	      /* D = -U * (I \kron (fx)): g3 = f: i,j,A */
	      gg3[fieldA][d2][d3] -= nu_m0_ma[fieldA][fieldB] * invMass[fieldA] * U[d2][d3] * f[fieldB] * wi;
	    }
	  }
	}
      }
      } else {
	PetscReal lxx[] = {data[0], data[1]};
	PetscReal ldf[3*FP_MAX_SPECIES];
	for (fieldB = 0; fieldB < Nf; ++fieldB) for (d3 = 0; d3 < 3; ++d3) ldf[d3 + fieldB*3] = df[d3 + fieldB*3] * wi * invMass[fieldB];
	for (dp=0;dp<4;dp++) {
	  LandauTensor3D(pvj, lxx[0], lxx[1], data[2], U, (ipidx==jpidx) ? 0. : 1.);
	  for (fieldA = 0; fieldA < Nf; ++fieldA) {
	    for (fieldB = 0; fieldB < Nf; ++fieldB) {
	      for (d2 = 0; d2 < 3; ++d2) {
		for (d3 = 0; d3 < 3; ++d3) {
		  /* K = U * grad(f): g2 = e: i,A */
		  gg2[fieldA][d2] += nu_m0_ma[fieldA][fieldB] * U[d2][d3] * ldf[d3 + fieldB*3];
		  /* D = -U * (I \kron (fx)): g3 = f: i,j,A */
		  gg3[fieldA][d2][d3] -= nu_m0_ma[fieldA][fieldB] * invMass[fieldA] * U[d2][d3] * f[fieldB] * wi;
		}
	      }
	    }
	  }
	  for (d3 = 0; d3 < 2; ++d3) {
	    lxx[d3] *= R[d3][dp%2];
	    for (fieldB = 0; fieldB < Nf; ++fieldB) {
	      ldf[d3 + fieldB*3] *= R[d3][dp%2];
	    }
	  }
	}
      }
    }
  } /* IPs */
  /* Jacobian transform - g2 */
  for (fieldA = 0; fieldA < Nf; ++fieldA) {
    if (mySubBlk==0) gg2[fieldA][dim-1] += Eq_m[fieldA]; /* add electric field term once per IP */
    for (d = 0; d < dim; ++d) {
      (*g2)[myqi][mySubBlk][fieldA][d] = 0.0;
      for (d2 = 0; d2 < dim; ++d2) {
	(*g2)[myqi][mySubBlk][fieldA][d] += invJj[jpidx * dim * dim + d*dim+d2]*gg2[fieldA][d2];
      }
      (*g2)[myqi][mySubBlk][fieldA][d] *= wj;
    }
  }
  /* g3 */
  for (fieldA = 0; fieldA < Nf; ++fieldA) {
    for (d = 0; d < dim; ++d) {
      for (dp = 0; dp < dim; ++dp) {
	(*g3)[myqi][mySubBlk][fieldA][d][dp] = 0.0;
	for (d2 = 0; d2 < dim; ++d2) {
	  for (d3 = 0; d3 < dim; ++d3) {
	    (*g3)[myqi][mySubBlk][fieldA][d][dp] += invJj[jpidx * dim * dim + d*dim + d2]*gg3[fieldA][d2][d3]*invJj[jpidx * dim * dim + dp*dim + d3];
	  }
	}
	(*g3)[myqi][mySubBlk][fieldA][d][dp] *= wj;
      }
    }
  }
  // Synchronize (ensure all the data is available) and sum g2 & g3
  __syncthreads();
  if (mySubBlk==0) { /* on one thread, sum up */
    for (fieldA = 0; fieldA < Nf; ++fieldA) {
      for (d = 0; d < dim; ++d) {
	for (ipidx = 1; ipidx < nSubBlocks; ++ipidx) {
	  (*g2)[myqi][0][fieldA][d] += (*g2)[myqi][ipidx][fieldA][d];
	  for (dp = 0; dp < dim; ++dp) {
	    (*g3)[myqi][0][fieldA][d][dp] += (*g3)[myqi][ipidx][fieldA][d][dp];
	  }
	}
      }
    }
  }
  // Synchronize (ensure all the data is available) and sum IP matrices
  __syncthreads();
  if (mythread==0) { // on one thread, sum up
    PetscScalar *elemMat  = &elemMats_out[myelem*totDim*totDim]; /* my output */
    int qj,ii;
    for (ii=0;ii<totDim*totDim;ii++) elemMat[ii] = 0;
    /* assemble - on the diagonal (I,I) */
    for (fieldA = 0; fieldA < Nf; ++fieldA) {
      PetscInt        f,g;
      for (f = 0; f < Nb; ++f) {
	const PetscInt i    = foffsets[fieldA] + f; /* Element matrix row */
	for (g = 0; g < Nb; ++g) {
	  const PetscInt j    = foffsets[fieldA] + g; /* Element matrix column */
	  const PetscInt fOff = i*totDim + j;
	  for (qj=0;qj<Nq;qj++) {
	    const PetscReal *B = TabBD[fieldA][0], *D = TabBD[fieldA][1], *BJq = &B[qj*Nb], *DIq = &D[qj*Nb*dim], *DJq = &D[qj*Nb*dim];
	    for (d = 0; d < dim; ++d) {
	      elemMat[fOff] += DIq[f*dim+d]*(*g2)[qj][0][fieldA][d]*BJq[g];
	      for (d2 = 0; d2 < dim; ++d2) {
		elemMat[fOff] += DIq[f*dim + d]*(*g3)[qj][0][fieldA][d][d2]*DJq[g*dim + d2];
	      }
	    }
	  }
	}
      }
    } // qj
  }
}
/* < v, u > */
static void g0_1(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.;
}

struct _ISColoring_ctx {
  ISColoring coloring;
};
typedef struct _ISColoring_ctx * ISColoring_ctx;

static PetscErrorCode destroy_coloring (void *obj)
{
  PetscErrorCode    ierr;
  ISColoring_ctx    coloring_ctx = (ISColoring_ctx)obj;
  ISColoring        isc = coloring_ctx->coloring;
  PetscFunctionBegin;
  ierr = ISColoringDestroy(&isc);CHKERRQ(ierr);
  ierr = PetscFree(coloring_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
__global__ void assemble_kernel(const PetscInt nidx_arr[], PetscInt *idx_arr[], PetscScalar *new_el_mats[], const ISColoringValue colors[], Mat_SeqAIJ mats[]);
PetscErrorCode FPLandauCUDAJacobian( DM plex, PetscQuadrature quad, const PetscInt foffsets[], const PetscReal nu_m0_ma[FP_MAX_SPECIES][FP_MAX_SPECIES],
				     const PetscReal invMass[FP_MAX_SPECIES], const PetscReal Eq_m[FP_MAX_SPECIES],const FPLandPointData * const IPDataGlobal,
				     const PetscReal wiGlobal[], const PetscInt num_sub_blocks, const PetscLogEvent events[], PetscBool quarter3DDomain, Mat JacP)
{
  PetscErrorCode    ierr;
  PetscInt          ii,ej,fieldA,*Nbf,Nb,nqdimGC,nqdim2GC,cStart,cEnd,Nf,dim,numGCells,Nq,totDim,nip,szf=sizeof(PetscReal);
  PetscInt          *d_foffsets;
  PetscReal         *vj,*Jj,*invJj,*vj_a,*Jj_a,*invJj_a;
  const PetscReal   *quadWeights;
  PetscReal         *d_quadWeights,*d_TabBD,*iTab;
  PetscReal         *d_vj,*d_Jj,*d_invJj,*d_wiGlobal,*d_nu_m0_ma,*d_invMass,*d_Eq_m;
  PetscScalar       *elemMats,*d_elemMats;
  PetscLogDouble    flops;
  PetscTabulation   *Tf;
  PetscDS           prob;
  PetscSection      section, globalSection;
  FPLandPointDataFlat IPDataGlobalDevice;
  PetscContainer    container = NULL;
  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(events[3],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = DMGetDimension(plex, &dim);CHKERRQ(ierr);
  if (dim!=FP_DIM) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "FP_DIM != dim");
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  numGCells = cEnd - cStart;
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, &quadWeights);CHKERRQ(ierr);
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_quadWeights,Nq*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(d_quadWeights, quadWeights, Nq*szf, cudaMemcpyHostToDevice));
  nip  = numGCells*Nq; /* length of inner global interation */
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nbf);CHKERRQ(ierr); Nb = Nbf[0];
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  if (Nb!=Nq)SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nb!=Nq over integration or simplices?");
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &Tf);CHKERRQ(ierr);
  ierr = DMGetLocalSection(plex, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(plex, &globalSection);CHKERRQ(ierr);
  // create data
  ierr = FPLandPointDataCreateDevice(&IPDataGlobalDevice, IPDataGlobal);CHKERRQ(ierr); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_nu_m0_ma, Nf*FP_MAX_SPECIES*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_invMass,                 Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Eq_m,                    Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_foffsets,                Nf*sizeof(PetscInt))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(d_nu_m0_ma, nu_m0_ma, Nf*FP_MAX_SPECIES*szf,              cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_invMass,  invMass,                 Nf*szf,              cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_Eq_m,     Eq_m,                    Nf*szf,              cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_foffsets, foffsets,                Nf*sizeof(PetscInt), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_TabBD,                   Nf*Nq*Nb*(1+dim)*szf)); // kernel input
  for (ii=0,iTab=d_TabBD;ii<Nf;ii++,iTab += Nq*Nb*(1+dim)) {
    CUDA_SAFE_CALL(cudaMemcpy( iTab,        Tf[ii]->T[0], Nq*Nb*szf,     cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(&iTab[Nq*Nb], Tf[ii]->T[1], Nq*Nb*dim*szf, cudaMemcpyHostToDevice));
  }
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_wiGlobal,           Nq*numGCells*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_wiGlobal, wiGlobal, Nq*numGCells*szf,   cudaMemcpyHostToDevice));
  // collect geometry
  flops = (PetscLogDouble)numGCells*(PetscLogDouble)Nq*(PetscLogDouble)(5.*dim*dim*Nf*Nf + 165.);
  nqdim2GC = Nq*dim*dim*numGCells;
  nqdimGC  = Nq*dim*numGCells;
  ierr = PetscMalloc3(nqdimGC,&vj_a,nqdim2GC,&Jj_a,nqdim2GC,&invJj_a);CHKERRQ(ierr);
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_vj,    nqdimGC*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Jj,    nqdim2GC*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_invJj, nqdim2GC*szf)); // kernel input
  for (ej = cStart, vj = vj_a, Jj = Jj_a, invJj = invJj_a;
       ej < cEnd;
       ++ej, vj += Nq*dim, Jj += Nq*dim*dim, invJj += Nq*dim*dim) {
    PetscReal  detJ[FP_MAX_NQ];
    ierr = DMPlexComputeCellGeometryFEM(plex, cStart+ej, quad, vj, Jj, invJj, detJ);CHKERRQ(ierr);
  }
  CUDA_SAFE_CALL(cudaMemcpy(d_vj,       vj_a, nqdimGC*szf,        cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_Jj,       Jj_a, nqdim2GC*szf,       cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_invJj, invJj_a, nqdim2GC*szf,       cudaMemcpyHostToDevice));
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[3],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(flops*nip);CHKERRQ(ierr);
#endif
  {
    PetscReal  *d_g2g3;
    dim3 dimBlock(Nq,num_sub_blocks);
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_elemMats, totDim*totDim*numGCells*sizeof(PetscScalar))); // kernel output
    ii = FP_MAX_NQ*FP_MAX_SPECIES*FP_DIM*(1+FP_DIM)*FP_MAX_SUB_THREAD_BLOCKS;
#if defined(FP_USE_SHARED_GPU_MEM)
    /* PetscPrintf(PETSC_COMM_SELF,"Call land_kernel with %D words shared memory\n",ii); */
    land_kernel<<<numGCells,dimBlock,ii*szf>>>(nip,dim,totDim,Nf,Nb,num_sub_blocks,d_vj,d_Jj,d_invJj,d_nu_m0_ma,d_invMass,d_Eq_m,
					       d_TabBD, d_quadWeights, d_foffsets, IPDataGlobalDevice, d_wiGlobal, quarter3DDomain, d_elemMats);
    CHECK_LAUNCH_ERROR();
#else
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_g2g3, ii*szf*numGCells)); // kernel input
    PetscReal  *g2 = &d_g2g3[0];
    PetscReal  *g3 = &d_g2g3[FP_MAX_SUB_THREAD_BLOCKS*FP_MAX_NQ*FP_MAX_SPECIES*FP_DIM*numGCells];
    land_kernel<<<numGCells,dimBlock>>>(nip,dim,totDim,Nf,Nb,num_sub_blocks,d_vj,d_Jj,d_invJj,d_nu_m0_ma,d_invMass,d_Eq_m,
					d_TabBD, d_quadWeights, d_foffsets, IPDataGlobalDevice, d_wiGlobal, g2, g3, quarter3DDomain, d_elemMats);
    CHECK_LAUNCH_ERROR();
    CUDA_SAFE_CALL (cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaFree(d_g2g3));
  }
#endif
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[4],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(events[5],0,0,0,0);CHKERRQ(ierr);
#endif
  // delete device data
  ierr = FPLandPointDataDestroyDevice(&IPDataGlobalDevice);CHKERRQ(ierr);
  CUDA_SAFE_CALL(cudaFree(d_vj));
  CUDA_SAFE_CALL(cudaFree(d_Jj));
  CUDA_SAFE_CALL(cudaFree(d_invJj));
  CUDA_SAFE_CALL(cudaFree(d_quadWeights));
  CUDA_SAFE_CALL(cudaFree(d_wiGlobal));
  CUDA_SAFE_CALL(cudaFree(d_nu_m0_ma));
  CUDA_SAFE_CALL(cudaFree(d_invMass));
  CUDA_SAFE_CALL(cudaFree(d_Eq_m));
  CUDA_SAFE_CALL(cudaFree(d_foffsets));
  CUDA_SAFE_CALL(cudaFree(d_TabBD));
  ierr = PetscFree3(vj_a,Jj_a,invJj_a);CHKERRQ(ierr);
  ierr = PetscMalloc1(totDim*totDim*numGCells,&elemMats);CHKERRQ(ierr);
  CUDA_SAFE_CALL(cudaMemcpy(elemMats, d_elemMats, totDim*totDim*numGCells*sizeof(PetscScalar), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_elemMats));
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[5],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(events[6],0,0,0,0);CHKERRQ(ierr);
#endif
  /* coloring */
  ierr = PetscObjectQuery((PetscObject)JacP,"coloring",(PetscObject*)&container);CHKERRQ(ierr);
  if (!container) {
    PetscSection   csection;
    DM             colordm;
    Vec            color_vec, eidx_vec;
    PetscInt       i,nc;
    PetscInt       numComp[1];
    PetscInt       numDof[3];
    PetscFE        fe;
    PetscDS        prob;
    Mat            mat;
    ISColoring     iscoloring = NULL;
    ISColoring_ctx coloring_ctx = NULL;
    /* Create a scalar field u, a vector field v, and a surface vector field w */
    numComp[0] = 1;
    ierr = DMClone(plex, &colordm);CHKERRQ(ierr);
    /* we do not need the right degree for coloring so color_ prefix work */
    ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) plex), dim, 1, PETSC_FALSE, "color_", PETSC_DECIDE, &fe);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fe, "color");CHKERRQ(ierr);
    ierr = DMSetField(colordm, 0, NULL, (PetscObject)fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
    for (i = 0; i < (dim+1); ++i) numDof[i] = 0;
    numDof[dim]   = 1;
    /* Create a PetscSection with this data layout */
    ierr = DMPlexCreateSection(colordm, NULL, numComp, numDof, 0, 0, NULL, NULL, NULL, &csection);CHKERRQ(ierr);
    /* Name the Field variables */
    ierr = PetscSectionSetFieldName(csection, 0, "color");CHKERRQ(ierr);
    /* Tell the DM to use this data layout */
    ierr = DMSetLocalSection(colordm, csection);CHKERRQ(ierr);
    ierr = DMCreateDS(colordm);CHKERRQ(ierr);
    ierr = DMGetDS(colordm, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, g0_1, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = DMViewFromOptions(colordm,NULL,"-color_dm_view");CHKERRQ(ierr);
    ierr = DMSetAdjacency(colordm, 0, PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMCreateMatrix(colordm, &mat);CHKERRQ(ierr);
    /* Create a Mat and Vec with this layout and view it */
    ierr = DMGetGlobalVector(colordm, &color_vec);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(colordm, &eidx_vec);CHKERRQ(ierr);
    ierr = DMPlexSNESComputeJacobianFEM(colordm, color_vec, mat, mat, NULL);CHKERRQ(ierr);
    ierr = MatViewFromOptions(mat,NULL,"-color_mat_view");CHKERRQ(ierr);
    {
      MatColoring     mc;
      IS             *is;
      PetscInt        csize,colour,j,k;
      const PetscInt *indices;
      ierr = MatColoringCreate(mat,&mc);CHKERRQ(ierr);
      ierr = MatColoringSetDistance(mc,1);CHKERRQ(ierr);
      ierr = MatColoringSetType(mc,MATCOLORINGJP);CHKERRQ(ierr);
      ierr = MatColoringSetFromOptions(mc);CHKERRQ(ierr);
      ierr = MatColoringApply(mc,&iscoloring);CHKERRQ(ierr);
      ierr = MatColoringDestroy(&mc);CHKERRQ(ierr);
      /* view */
      ierr = ISColoringViewFromOptions(iscoloring,NULL,"-coloring_is_view");CHKERRQ(ierr);
      ierr = ISColoringGetIS(iscoloring,PETSC_USE_POINTER,&nc,&is);CHKERRQ(ierr);
      for (colour=0; colour<nc; colour++) {
	ierr = ISGetLocalSize(is[colour],&csize);CHKERRQ(ierr);
	ierr = ISGetIndices(is[colour],&indices);CHKERRQ(ierr);
	for (j=0; j<csize; j++) {
	  PetscScalar v = (PetscScalar)colour;
	  k = indices[j];
	  ierr = VecSetValues(color_vec,1,&k,&v,INSERT_VALUES);
	  v = (PetscScalar)k;
	  ierr = VecSetValues(eidx_vec,1,&k,&v,INSERT_VALUES);
	}
	ierr = ISRestoreIndices(is[colour],&indices);CHKERRQ(ierr);
      }
      ierr = ISColoringRestoreIS(iscoloring,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
    }
    /* view coloring */
    if (1) {
      PetscViewer    viewer;
      ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERVTK);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "color.vtk");CHKERRQ(ierr);
      ierr = VecView(color_vec, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERVTK);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "eidx.vtk");CHKERRQ(ierr);
      ierr = VecView(eidx_vec, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    /* Cleanup */
    ierr = MatDestroy(&mat);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(colordm, &color_vec);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(colordm, &eidx_vec);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&csection);CHKERRQ(ierr);
    ierr = DMDestroy(&colordm);CHKERRQ(ierr);
    /* stash coloring */
    ierr = PetscContainerCreate(PETSC_COMM_SELF, &container);CHKERRQ(ierr);
    ierr = PetscNew(&coloring_ctx);CHKERRQ(ierr);
    coloring_ctx->coloring = iscoloring;
    ierr = PetscContainerSetPointer(container,(void*)coloring_ctx);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container, destroy_coloring);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)JacP,"coloring",(PetscObject)container);CHKERRQ(ierr);
#if defined(PETSC_HAVE_OPENMP)
    if (1) {
      int thread_id,num_threads;
      char name[MPI_MAX_PROCESSOR_NAME];
      int resultlength;
      MPI_Get_processor_name(name, &resultlength);
#pragma omp parallel default(shared) private(thread_id)
      {
	thread_id = omp_get_thread_num();
	num_threads = omp_get_num_threads();
	PetscPrintf(PETSC_COMM_SELF, "Made coloring with %D colors. Node %s, OMP_threadID %d of %d\n", nc, name, thread_id, num_threads);
      }
    }
#endif
  }
  if (0) {
    PetscScalar *elMat;
    for (ej = cStart, elMat = elemMats ; ej < cEnd; ++ej, elMat += totDim*totDim) {
      ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, ej, elMat, ADD_VALUES);CHKERRQ(ierr);
    }
  } else if (0) { /* OMP assembly */
    /* assemble with coloring */
    IS             *is;
    PetscInt        nc,colour,j;
    const PetscInt *clr_idxs;
    ISColoring_ctx  coloring_ctx = NULL;
    ISColoring      iscoloring;
    ierr = PetscContainerGetPointer(container,(void**)&coloring_ctx);CHKERRQ(ierr);
    iscoloring = coloring_ctx->coloring;
    ierr = ISColoringGetIS(iscoloring,PETSC_USE_POINTER,&nc,&is);CHKERRQ(ierr);
    for (colour=0; colour<nc; colour++) {
      PetscInt    *idx_arr[64];
      PetscScalar *new_el_mats[64];
      PetscInt     idx_size[64],csize;
      ierr = ISGetLocalSize(is[colour],&csize);CHKERRQ(ierr);
      if (csize>64) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "too many elements in color. %D > %D",csize,64);
      ierr = ISGetIndices(is[colour],&clr_idxs);CHKERRQ(ierr);
      /* get indices and mats */
      for (j=0; j<csize; j++) {
	PetscInt cell = cStart + clr_idxs[j];
	PetscInt numindices,*indices;
	PetscScalar *elMat = &elemMats[clr_idxs[j]*totDim*totDim];
	PetscScalar *valuesOrig = elMat;
	ierr = DMPlexGetClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
	idx_size[j] = numindices;
	ierr = PetscMalloc2(numindices,&idx_arr[j],numindices*numindices,&new_el_mats[j]);CHKERRQ(ierr);
	ierr = PetscMemcpy(idx_arr[j],indices,numindices*sizeof(PetscInt));CHKERRQ(ierr);
	ierr = PetscMemcpy(new_el_mats[j],elMat,numindices*numindices*sizeof(PetscScalar));CHKERRQ(ierr);
	ierr = DMPlexRestoreClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);	
	if (elMat != valuesOrig) {ierr = DMRestoreWorkArray(plex, numindices*numindices, MPIU_SCALAR, &elMat);}
      }
      ierr = ISRestoreIndices(is[colour],&clr_idxs);CHKERRQ(ierr);
      /* assemble matrix */
#pragma omp parallel for shared(JacP,idx_size,idx_arr,new_el_mats) private(j) schedule(static)
      for (j=0; j<csize; j++) {
	PetscInt numindices = idx_size[j], *indices = idx_arr[j];
	PetscScalar *elMat = new_el_mats[j];
	MatSetValues(JacP,numindices,indices,numindices,indices,elMat,ADD_VALUES);
      }
      /* free */
      for (j=0; j<csize; j++) {
	ierr = PetscFree2(idx_arr[j],new_el_mats[j]);CHKERRQ(ierr);
      }
    }
    ierr = ISColoringRestoreIS(iscoloring,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
  } else {  /* gpu assembly */
#define FP_MAX_COLORS 16
#define FP_MAX_ELEMS 256
    PetscInt               nelems=cEnd-cStart,nc,ej,j;
    const ISColoringValue *colors;
    ISColoringValue       *d_colors,colour;
    Mat_SeqAIJ             h_mats[FP_MAX_COLORS], *jaca = (Mat_SeqAIJ *)JacP->data, *d_mats;
    PetscInt              *h_idx_arr[FP_MAX_ELEMS], h_nidx_arr[FP_MAX_ELEMS], *d_nidx_arr, **d_idx_arr;
    const PetscInt         n = JacP->rmap->n, nnz = jaca->i[n];
    PetscScalar           *h_new_el_mats[FP_MAX_ELEMS], *val_buf, **d_new_el_mats;
    ISColoring_ctx         coloring_ctx = NULL;
    ISColoring             iscoloring;
    IS                     *is;
    const PetscInt         *clr_idxs;
    ierr = PetscContainerGetPointer(container,(void**)&coloring_ctx);CHKERRQ(ierr);
    iscoloring = coloring_ctx->coloring;
    /* get colors */
    ierr = ISColoringGetColors(iscoloring, &j, &nc, &colors);CHKERRQ(ierr);
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors,         nelems*sizeof(ISColoringValue))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          d_colors, colors, nelems*sizeof(ISColoringValue), cudaMemcpyHostToDevice));
    ierr = ISColoringGetIS(iscoloring,PETSC_USE_POINTER,&nc,&is);CHKERRQ(ierr);
    if (nelems>FP_MAX_ELEMS) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "too many elements. %D > %D",nelems,FP_MAX_ELEMS);
    if (nc>FP_MAX_COLORS) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "too many colors. %D > %D",nc,FP_MAX_COLORS);
    /* get indices and element matrices */
    for (colour=0; colour<nc; colour++) {
      PetscInt     csize;
      ierr = ISGetLocalSize(is[colour],&csize);CHKERRQ(ierr);
      ierr = ISGetIndices(is[colour],&clr_idxs);CHKERRQ(ierr);
      /* get indices and mats */
      for (j=0; j<csize; j++) {
	const PetscInt eidx = clr_idxs[j], cell = cStart + eidx;
	PetscInt numindices,*indices;
	PetscScalar *elMat = &elemMats[eidx*totDim*totDim];
	PetscScalar *valuesOrig = elMat;
	if(colors[eidx] != colour) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "color != colour");
	ierr = DMPlexGetClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
	h_nidx_arr[eidx] = numindices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&h_idx_arr[eidx],          numindices*sizeof(PetscInt))); // kernel input
	CUDA_SAFE_CALL(cudaMemcpy(          h_idx_arr[eidx], indices, numindices*sizeof(PetscInt), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void **)&h_new_el_mats[eidx],        numindices*numindices*sizeof(PetscScalar))); // kernel input
	CUDA_SAFE_CALL(cudaMemcpy(          h_new_el_mats[eidx], elMat, numindices*numindices*sizeof(PetscScalar), cudaMemcpyHostToDevice));
	ierr = DMPlexRestoreClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
	if (elMat != valuesOrig) {ierr = DMRestoreWorkArray(plex, numindices*numindices, MPIU_SCALAR, &elMat);CHKERRQ(ierr);}
      }
      ierr = ISRestoreIndices(is[colour],&clr_idxs);CHKERRQ(ierr);
    }
    ierr = ISColoringRestoreIS(iscoloring,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_nidx_arr,             nelems*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          d_nidx_arr, h_nidx_arr, nelems*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc(&d_idx_arr,            nelems*sizeof(PetscInt*))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy( d_idx_arr, h_idx_arr, nelems*sizeof(PetscInt*), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc(&d_new_el_mats,                nelems*sizeof(PetscScalar*))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy( d_new_el_mats, h_new_el_mats, nelems*sizeof(PetscScalar*), cudaMemcpyHostToDevice));
    /* make matrix buffers */
    for (colour=0; colour<nc; colour++) {
      Mat_SeqAIJ *a = &h_mats[colour];
      /* create on GPU and copy to GPU */
      CUDA_SAFE_CALL(cudaMalloc((void **)&a->i,          (n+1)*sizeof(PetscInt))); // kernel input
      CUDA_SAFE_CALL(cudaMemcpy(          a->i, jaca->i, (n+1)*sizeof(PetscInt), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMalloc((void **)&a->ilen,             (n)*sizeof(PetscInt))); // kernel input
      CUDA_SAFE_CALL(cudaMemcpy(          a->ilen, jaca->ilen, (n)*sizeof(PetscInt), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMalloc((void **)&a->j,          (nnz)*sizeof(PetscInt))); // kernel input
      CUDA_SAFE_CALL(cudaMemcpy(          a->j, jaca->j, (nnz)*sizeof(PetscInt), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMalloc((void **)&a->a,          (nnz)*sizeof(PetscScalar))); // kernel output
      CUDA_SAFE_CALL(cudaMemset(          a->a, 0,       (nnz)*sizeof(PetscScalar)));
    }
    CUDA_SAFE_CALL(cudaMalloc(&d_mats,         nc*sizeof(Mat_SeqAIJ))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy( d_mats, h_mats, nc*sizeof(Mat_SeqAIJ), cudaMemcpyHostToDevice));
    /* do it */
    assemble_kernel<<<nelems,1>>>(d_nidx_arr, d_idx_arr, d_new_el_mats, d_colors, d_mats);
    CHECK_LAUNCH_ERROR();
    /* cleanup */
    CUDA_SAFE_CALL(cudaFree(d_colors));
    CUDA_SAFE_CALL(cudaFree(d_nidx_arr));
    for (ej = cStart ; ej < cEnd; ++ej) {
      CUDA_SAFE_CALL(cudaFree(h_idx_arr[ej]));
      CUDA_SAFE_CALL(cudaFree(h_new_el_mats[ej]));
    }
    CUDA_SAFE_CALL(cudaFree(d_idx_arr));
    CUDA_SAFE_CALL(cudaFree(d_new_el_mats));
    /* copy & add Mat data back to CPU to JacP */
    ierr = PetscMalloc1(nnz,&val_buf);CHKERRQ(ierr);
    ierr = PetscMemzero(jaca->a,nnz*sizeof(PetscScalar));CHKERRQ(ierr);
    for (colour=0; colour<nc; colour++) {
      Mat_SeqAIJ *a = &h_mats[colour];
      CUDA_SAFE_CALL(cudaMemcpy(val_buf, a->a, (nnz)*sizeof(PetscScalar), cudaMemcpyDeviceToHost));
      for(ii=0;ii<nnz;ii++) jaca->a[ii] += val_buf[ii];
      //PetscKernelAXPY(jaca->a,1.0,val_buf,nnz);
      /* destroy mat */
      CUDA_SAFE_CALL(cudaFree(a->i));
      CUDA_SAFE_CALL(cudaFree(a->ilen));
      CUDA_SAFE_CALL(cudaFree(a->j));
      CUDA_SAFE_CALL(cudaFree(a->a));
    }
    CUDA_SAFE_CALL(cudaFree(d_mats));
    ierr = PetscFree(val_buf);CHKERRQ(ierr);
  }
  ierr = PetscFree(elemMats);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[6],0,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

__global__
void assemble_kernel(const PetscInt nidx_arr[], PetscInt *idx_arr[], PetscScalar *new_el_mats[], const ISColoringValue colors[], Mat_SeqAIJ mats[])
{
  const PetscInt myelem = blockIdx.x;
  Mat_SeqAIJ a = mats[colors[myelem]]; /* copy to GPU */
  const PetscScalar *v = new_el_mats[myelem];
  const PetscInt *in = idx_arr[myelem], *im = idx_arr[myelem], n = nidx_arr[myelem], m = nidx_arr[myelem];  
  /* mat set values */
  PetscInt       *rp,k,low,high,t,row,nrow,i,col,l;
  PetscInt       *ai = a.i,*ailen = a.ilen;
  PetscInt       *aj = a.j,lastcol = -1;
  MatScalar      *ap=NULL,value=0.0,*aa = a.a;
  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) continue;
    rp   = aj + ai[row];
    ap = aa + ai[row];
    nrow = ailen[row];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) {
	printf("\t\tin[l] < 0 ?????\n");
	continue;
      }
      col = in[l];
      value = v[l + k*n];
      if (col <= lastcol) low = 0;
      else high = nrow;
      lastcol = col;
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else low = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
	  if (ap[i] != 0.0) printf("error element %d, color %d, (%d %d) = %g --> %g\n", myelem, colors[myelem], row, col, ap[i], value);
	  ap[i] = value;
	  low = i + 1;
          goto noinsert;
        }
      }
      printf("\t\t\t ERROR in assemble_kernel\n");
    noinsert:;
    }
  }
}
