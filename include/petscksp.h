/* $Id: ksp.h,v 1.48 1997/01/12 20:34:31 curfman Exp bsmith $ */
/*
   Defines the interface functions for the Krylov subspace accelerators.
*/
#ifndef __KSP_PACKAGE
#define __KSP_PACKAGE
#include "petsc.h"
#include "vec.h"
#include "mat.h"
#include "pc.h"

#define KSP_COOKIE  PETSC_COOKIE+8

typedef struct _KSP*     KSP;

typedef enum { KSPRICHARDSON, KSPCHEBYCHEV, KSPCG, KSPGMRES, KSPTCQMR, KSPBCGS, 
               KSPCGS, KSPTFQMR, KSPCR, KSPLSQR, KSPPREONLY, KSPQCG, KSPNEW} KSPType;

extern int KSPCreate(MPI_Comm,KSP *);
extern int KSPSetType(KSP,KSPType);
extern int KSPSetUp(KSP);
extern int KSPSolve(KSP,int *);
extern int KSPDestroy(KSP);

extern int KSPRegisterAll();
extern int KSPRegisterDestroy();
extern int KSPRegister(KSPType,KSPType*,char *,int (*)(KSP));
extern int KSPRegisterAllCalled;

extern int KSPGetType(KSP, KSPType *,char **);
extern int KSPSetPreconditionerSide(KSP,PCSide);
extern int KSPGetPreconditionerSide(KSP,PCSide*);
extern int KSPGetTolerances(KSP,double*,double*,double*,int*);
extern int KSPSetTolerances(KSP,double,double,double,int);
extern int KSPSetComputeResidual(KSP,PetscTruth);
extern int KSPSetUsePreconditionedResidual(KSP);
extern int KSPSetInitialGuessNonzero(KSP);
extern int KSPSetComputeEigenvalues(KSP);
extern int KSPSetComputeSingularValues(KSP);
extern int KSPSetRhs(KSP,Vec);
extern int KSPGetRhs(KSP,Vec *);
extern int KSPSetSolution(KSP,Vec);
extern int KSPGetSolution(KSP,Vec *);

extern int KSPSetPC(KSP,PC);
extern int KSPGetPC(KSP,PC*);

extern int KSPSetMonitor(KSP,int (*)(KSP,int,double, void*), void *);
extern int KSPGetMonitorContext(KSP,void **);
extern int KSPSetResidualHistory(KSP, double *,int);

extern int KSPSetConvergenceTest(KSP,int (*)(KSP,int,double, void*), void *);
extern int KSPGetConvergenceContext(KSP,void **);

extern int KSPBuildSolution(KSP, Vec,Vec *);
extern int KSPBuildResidual(KSP, Vec, Vec,Vec *);

extern int KSPRichardsonSetScale(KSP , double);
extern int KSPChebychevSetEigenvalues(KSP , double, double);
extern int KSPComputeExtremeSingularValues(KSP, double*,double*);
extern int KSPComputeEigenvalues(KSP,int,double*,double*);
extern int KSPComputeEigenvaluesExplicitly(KSP,int,double*,double*);

extern int KSPGMRESSetRestart(KSP, int);
extern int KSPGMRESSetPreAllocateVectors(KSP);
extern int KSPGMRESSetOrthogonalization(KSP,int (*)(KSP,int));
extern int KSPGMRESUnmodifiedGramSchmidtOrthogonalization(KSP,int);
extern int KSPGMRESModifiedGramSchmidtOrthogonalization(KSP,int);
extern int KSPGMRESIROrthogonalization(KSP,int);

extern int KSPSetFromOptions(KSP);
extern int KSPAddOptionsChecker(int (*)(KSP));

extern int KSPSingularValueMonitor(KSP,int,double, void * );
extern int KSPDefaultMonitor(KSP,int,double, void *);
extern int KSPTrueMonitor(KSP,int,double, void *);
extern int KSPDefaultSMonitor(KSP,int,double, void *);

extern int KSPDefaultConverged(KSP,int,double, void *);
extern int KSPCGDefaultConverged(KSP,int,double, void *);

extern int KSPResidual(KSP,Vec,Vec,Vec,Vec,Vec,Vec);
extern int KSPUnwindPreconditioner(KSP,Vec,Vec);
extern int KSPDefaultBuildSolution(KSP,Vec,Vec*);
extern int KSPDefaultBuildResidual(KSP,Vec,Vec,Vec *);

extern int KSPPrintHelp(KSP);

extern int KSPSetOptionsPrefix(KSP,char*);
extern int KSPAppendOptionsPrefix(KSP,char*);
extern int KSPGetOptionsPrefix(KSP,char**);

extern int KSPView(KSP,Viewer);

extern int KSPComputeExplicitOperator(KSP,Mat *);

typedef enum {KSP_CG_SYMMETRIC=1, KSP_CG_HERMITIAN=2} KSPCGType;
extern int KSPCGSetType(KSP,KSPCGType);

#if defined(__DRAW_PACKAGE)
extern int KSPLGMonitorCreate(char*,char*,int,int,int,int,DrawLG*);
extern int KSPLGMonitor(KSP,int,double,void*);
extern int KSPLGMonitorDestroy(DrawLG);
extern int KSPLGTrueMonitorCreate(MPI_Comm,char*,char*,int,int,int,int,DrawLG*);
extern int KSPLGTrueMonitor(KSP,int,double,void*);
extern int KSPLGTrueMonitorDestroy(DrawLG);
#endif 

extern int    PCPreSolve(PC,KSP);
extern int    PCPostSolve(PC,KSP);

#endif


