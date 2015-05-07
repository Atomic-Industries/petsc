/*
      Preconditioner module.
*/
#if !defined(__PETSCPC_H)
#define __PETSCPC_H
#include <petscmat.h>
#include <petscpctypes.h>

PETSC_EXTERN PetscErrorCode PCInitializePackage(void);

/*
    PCList contains the list of preconditioners currently registered
   These are added with PCRegister()
*/
PETSC_EXTERN PetscFunctionList PCList;

/* Logging support */
PETSC_EXTERN PetscClassId PC_CLASSID;

PETSC_EXTERN PetscErrorCode PCCreate(MPI_Comm,PC*);
PETSC_EXTERN PetscErrorCode PCSetType(PC,PCType);
PETSC_EXTERN PetscErrorCode PCGetType(PC,PCType*);
PETSC_EXTERN PetscErrorCode PCSetUp(PC);
PETSC_EXTERN PetscErrorCode PCSetUpOnBlocks(PC);
PETSC_EXTERN PetscErrorCode PCApply(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCApplySymmetricLeft(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCApplySymmetricRight(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCApplyBAorAB(PC,PCSide,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCApplyTranspose(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCApplyTransposeExists(PC,PetscBool *);
PETSC_EXTERN PetscErrorCode PCApplyBAorABTranspose(PC,PCSide,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCSetReusePreconditioner(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCGetReusePreconditioner(PC,PetscBool*);

#define PC_FILE_CLASSID 1211222

PETSC_EXTERN PetscErrorCode PCApplyRichardson(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt,PetscBool ,PetscInt*,PCRichardsonConvergedReason*);
PETSC_EXTERN PetscErrorCode PCApplyRichardsonExists(PC,PetscBool *);
PETSC_EXTERN PetscErrorCode PCSetInitialGuessNonzero(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCGetInitialGuessNonzero(PC,PetscBool*);
PETSC_EXTERN PetscErrorCode PCSetUseAmat(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCGetUseAmat(PC,PetscBool*);


PETSC_EXTERN PetscErrorCode PCRegister(const char[],PetscErrorCode(*)(PC));

PETSC_EXTERN PetscErrorCode PCReset(PC);
PETSC_EXTERN PetscErrorCode PCDestroy(PC*);
PETSC_EXTERN PetscErrorCode PCSetFromOptions(PC);

PETSC_EXTERN PetscErrorCode PCFactorGetMatrix(PC,Mat*);
PETSC_EXTERN PetscErrorCode PCSetModifySubMatrices(PC,PetscErrorCode(*)(PC,PetscInt,const IS[],const IS[],Mat[],void*),void*);
PETSC_EXTERN PetscErrorCode PCModifySubMatrices(PC,PetscInt,const IS[],const IS[],Mat[],void*);

PETSC_EXTERN PetscErrorCode PCSetOperators(PC,Mat,Mat);
PETSC_EXTERN PetscErrorCode PCGetOperators(PC,Mat*,Mat*);
PETSC_EXTERN PetscErrorCode PCGetOperatorsSet(PC,PetscBool *,PetscBool *);

PETSC_EXTERN PetscErrorCode PCView(PC,PetscViewer);
PETSC_EXTERN PetscErrorCode PCLoad(PC,PetscViewer);
PETSC_STATIC_INLINE PetscErrorCode PCViewFromOptions(PC A,const char prefix[],const char name[]) {return PetscObjectViewFromOptions((PetscObject)A,prefix,name);}

PETSC_EXTERN PetscErrorCode PCSetOptionsPrefix(PC,const char[]);
PETSC_EXTERN PetscErrorCode PCAppendOptionsPrefix(PC,const char[]);
PETSC_EXTERN PetscErrorCode PCGetOptionsPrefix(PC,const char*[]);

PETSC_EXTERN PetscErrorCode PCComputeExplicitOperator(PC,Mat*);

/*
      These are used to provide extra scaling of preconditioned
   operator for time-stepping schemes like in SUNDIALS
*/
PETSC_EXTERN PetscErrorCode PCGetDiagonalScale(PC,PetscBool *);
PETSC_EXTERN PetscErrorCode PCDiagonalScaleLeft(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCDiagonalScaleRight(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCSetDiagonalScale(PC,Vec);

/* ------------- options specific to particular preconditioners --------- */

PETSC_EXTERN PetscErrorCode PCJacobiSetType(PC,PCJacobiType);
PETSC_EXTERN PetscErrorCode PCJacobiGetType(PC,PCJacobiType*);
PETSC_EXTERN PetscErrorCode PCJacobiSetUseAbs(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCJacobiGetUseAbs(PC,PetscBool*);
PETSC_EXTERN PetscErrorCode PCSORSetSymmetric(PC,MatSORType);
PETSC_EXTERN PetscErrorCode PCSORGetSymmetric(PC,MatSORType*);
PETSC_EXTERN PetscErrorCode PCSORSetOmega(PC,PetscReal);
PETSC_EXTERN PetscErrorCode PCSORGetOmega(PC,PetscReal*);
PETSC_EXTERN PetscErrorCode PCSORSetIterations(PC,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode PCSORGetIterations(PC,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode PCEisenstatSetOmega(PC,PetscReal);
PETSC_EXTERN PetscErrorCode PCEisenstatGetOmega(PC,PetscReal*);
PETSC_EXTERN PetscErrorCode PCEisenstatSetNoDiagonalScaling(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCEisenstatGetNoDiagonalScaling(PC,PetscBool*);

PETSC_EXTERN PetscErrorCode PCBJacobiSetTotalBlocks(PC,PetscInt,const PetscInt[]);
PETSC_EXTERN PetscErrorCode PCBJacobiGetTotalBlocks(PC,PetscInt*,const PetscInt*[]);
PETSC_EXTERN PetscErrorCode PCBJacobiSetLocalBlocks(PC,PetscInt,const PetscInt[]);
PETSC_EXTERN PetscErrorCode PCBJacobiGetLocalBlocks(PC,PetscInt*,const PetscInt*[]);

PETSC_EXTERN PetscErrorCode PCShellSetApply(PC,PetscErrorCode (*)(PC,Vec,Vec));
PETSC_EXTERN PetscErrorCode PCShellSetApplyBA(PC,PetscErrorCode (*)(PC,PCSide,Vec,Vec,Vec));
PETSC_EXTERN PetscErrorCode PCShellSetApplyTranspose(PC,PetscErrorCode (*)(PC,Vec,Vec));
PETSC_EXTERN PetscErrorCode PCShellSetSetUp(PC,PetscErrorCode (*)(PC));
PETSC_EXTERN PetscErrorCode PCShellSetApplyRichardson(PC,PetscErrorCode (*)(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,PetscInt,PetscBool ,PetscInt*,PCRichardsonConvergedReason*));
PETSC_EXTERN PetscErrorCode PCShellSetView(PC,PetscErrorCode (*)(PC,PetscViewer));
PETSC_EXTERN PetscErrorCode PCShellSetDestroy(PC,PetscErrorCode (*)(PC));
PETSC_EXTERN PetscErrorCode PCShellSetContext(PC,void*);
PETSC_EXTERN PetscErrorCode PCShellGetContext(PC,void**);
PETSC_EXTERN PetscErrorCode PCShellSetName(PC,const char[]);
PETSC_EXTERN PetscErrorCode PCShellGetName(PC,const char*[]);

PETSC_EXTERN PetscErrorCode PCFactorSetZeroPivot(PC,PetscReal);

PETSC_EXTERN PetscErrorCode PCFactorSetShiftType(PC,MatFactorShiftType);
PETSC_EXTERN PetscErrorCode PCFactorSetShiftAmount(PC,PetscReal);

PETSC_EXTERN PetscErrorCode PCFactorSetMatSolverPackage(PC,const MatSolverPackage);
PETSC_EXTERN PetscErrorCode PCFactorGetMatSolverPackage(PC,const MatSolverPackage*);
PETSC_EXTERN PetscErrorCode PCFactorSetUpMatSolverPackage(PC);

PETSC_EXTERN PetscErrorCode PCFactorSetFill(PC,PetscReal);
PETSC_EXTERN PetscErrorCode PCFactorSetColumnPivot(PC,PetscReal);
PETSC_EXTERN PetscErrorCode PCFactorReorderForNonzeroDiagonal(PC,PetscReal);

PETSC_EXTERN PetscErrorCode PCFactorSetMatOrderingType(PC,MatOrderingType);
PETSC_EXTERN PetscErrorCode PCFactorSetReuseOrdering(PC,PetscBool );
PETSC_EXTERN PetscErrorCode PCFactorSetReuseFill(PC,PetscBool );
PETSC_EXTERN PetscErrorCode PCFactorSetUseInPlace(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCFactorGetUseInPlace(PC,PetscBool*);
PETSC_EXTERN PetscErrorCode PCFactorSetAllowDiagonalFill(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCFactorGetAllowDiagonalFill(PC,PetscBool*);
PETSC_EXTERN PetscErrorCode PCFactorSetPivotInBlocks(PC,PetscBool);

PETSC_EXTERN PetscErrorCode PCFactorSetLevels(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCFactorGetLevels(PC,PetscInt*);
PETSC_EXTERN PetscErrorCode PCFactorSetDropTolerance(PC,PetscReal,PetscReal,PetscInt);

PETSC_EXTERN PetscErrorCode PCASMSetLocalSubdomains(PC,PetscInt,IS[],IS[]);
PETSC_EXTERN PetscErrorCode PCASMSetTotalSubdomains(PC,PetscInt,IS[],IS[]);
PETSC_EXTERN PetscErrorCode PCASMSetOverlap(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCASMSetDMSubdomains(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCASMGetDMSubdomains(PC,PetscBool*);
PETSC_EXTERN PetscErrorCode PCASMSetSortIndices(PC,PetscBool);

PETSC_EXTERN PetscErrorCode PCASMSetType(PC,PCASMType);
PETSC_EXTERN PetscErrorCode PCASMGetType(PC,PCASMType*);
PETSC_EXTERN PetscErrorCode PCASMSetLocalType(PC,PCCompositeType);
PETSC_EXTERN PetscErrorCode PCASMGetLocalType(PC,PCCompositeType*);
PETSC_EXTERN PetscErrorCode PCASMCreateSubdomains(Mat,PetscInt,IS*[]);
PETSC_EXTERN PetscErrorCode PCASMDestroySubdomains(PetscInt,IS[],IS[]);
PETSC_EXTERN PetscErrorCode PCASMCreateSubdomains2D(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,IS**,IS**);
PETSC_EXTERN PetscErrorCode PCASMGetLocalSubdomains(PC,PetscInt*,IS*[],IS*[]);
PETSC_EXTERN PetscErrorCode PCASMGetLocalSubmatrices(PC,PetscInt*,Mat*[]);

PETSC_EXTERN PetscErrorCode PCGASMSetSubdomains(PC,PetscInt,IS[],IS[]);
PETSC_EXTERN PetscErrorCode PCGASMSetTotalSubdomains(PC,PetscInt,PetscBool);
PETSC_EXTERN PetscErrorCode PCGASMSetOverlap(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCGASMSetDMSubdomains(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCGASMGetDMSubdomains(PC,PetscBool*);
PETSC_EXTERN PetscErrorCode PCGASMSetSortIndices(PC,PetscBool );

PETSC_EXTERN PetscErrorCode PCGASMSetType(PC,PCGASMType);
PETSC_EXTERN PetscErrorCode PCGASMCreateLocalSubdomains(Mat,PetscInt,PetscInt,IS*[],IS*[]);
PETSC_EXTERN PetscErrorCode PCGASMDestroySubdomains(PetscInt,IS[],IS[]);
PETSC_EXTERN PetscErrorCode PCGASMCreateSubdomains2D(PC,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,IS**,IS**);
PETSC_EXTERN PetscErrorCode PCGASMGetSubdomains(PC,PetscInt*,IS*[],IS*[]);
PETSC_EXTERN PetscErrorCode PCGASMGetSubmatrices(PC,PetscInt*,Mat*[]);

PETSC_EXTERN PetscErrorCode PCCompositeSetType(PC,PCCompositeType);
PETSC_EXTERN PetscErrorCode PCCompositeGetType(PC,PCCompositeType*);
PETSC_EXTERN PetscErrorCode PCCompositeAddPC(PC,PCType);
PETSC_EXTERN PetscErrorCode PCCompositeGetPC(PC,PetscInt,PC *);
PETSC_EXTERN PetscErrorCode PCCompositeSpecialSetAlpha(PC,PetscScalar);

PETSC_EXTERN PetscErrorCode PCRedundantSetNumber(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCRedundantSetScatter(PC,VecScatter,VecScatter);
PETSC_EXTERN PetscErrorCode PCRedundantGetOperators(PC,Mat*,Mat*);

PETSC_EXTERN PetscErrorCode PCSPAISetEpsilon(PC,double);
PETSC_EXTERN PetscErrorCode PCSPAISetNBSteps(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCSPAISetMax(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCSPAISetMaxNew(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCSPAISetBlockSize(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCSPAISetCacheSize(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCSPAISetVerbose(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCSPAISetSp(PC,PetscInt);

PETSC_EXTERN PetscErrorCode PCHYPRESetType(PC,const char[]);
PETSC_EXTERN PetscErrorCode PCHYPREGetType(PC,const char*[]);
PETSC_EXTERN PetscErrorCode PCHYPRESetDiscreteGradient(PC,Mat);
PETSC_EXTERN PetscErrorCode PCHYPRESetDiscreteCurl(PC,Mat);
PETSC_EXTERN PetscErrorCode PCHYPRESetEdgeConstantVectors(PC,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCHYPRESetAlphaPoissonMatrix(PC,Mat);
PETSC_EXTERN PetscErrorCode PCHYPRESetBetaPoissonMatrix(PC,Mat);
PETSC_EXTERN PetscErrorCode PCBJacobiGetLocalBlocks(PC,PetscInt*,const PetscInt*[]);
PETSC_EXTERN PetscErrorCode PCBJacobiGetTotalBlocks(PC,PetscInt*,const PetscInt*[]);

PETSC_EXTERN PetscErrorCode PCFieldSplitSetFields(PC,const char[],PetscInt,const PetscInt*,const PetscInt*);
PETSC_EXTERN PetscErrorCode PCFieldSplitSetType(PC,PCCompositeType);
PETSC_EXTERN PetscErrorCode PCFieldSplitGetType(PC,PCCompositeType*);
PETSC_EXTERN PetscErrorCode PCFieldSplitSetBlockSize(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCFieldSplitSetIS(PC,const char[],IS);
PETSC_EXTERN PetscErrorCode PCFieldSplitGetIS(PC,const char[],IS*);
PETSC_EXTERN PetscErrorCode PCFieldSplitSetDMSplits(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCFieldSplitGetDMSplits(PC,PetscBool*);
PETSC_EXTERN PetscErrorCode PCFieldSplitSetDiagUseAmat(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCFieldSplitGetDiagUseAmat(PC,PetscBool*);
PETSC_EXTERN PetscErrorCode PCFieldSplitSetOffDiagUseAmat(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCFieldSplitGetOffDiagUseAmat(PC,PetscBool*);

PETSC_EXTERN PETSC_DEPRECATED("Use PCFieldSplitSetSchurPre") PetscErrorCode PCFieldSplitSchurPrecondition(PC,PCFieldSplitSchurPreType,Mat);
PETSC_EXTERN PetscErrorCode PCFieldSplitSetSchurPre(PC,PCFieldSplitSchurPreType,Mat);
PETSC_EXTERN PetscErrorCode PCFieldSplitGetSchurPre(PC,PCFieldSplitSchurPreType*,Mat*);
PETSC_EXTERN PetscErrorCode PCFieldSplitSetSchurFactType(PC,PCFieldSplitSchurFactType);
PETSC_EXTERN PetscErrorCode PCFieldSplitGetSchurBlocks(PC,Mat*,Mat*,Mat*,Mat*);
PETSC_EXTERN PetscErrorCode PCFieldSplitSchurGetS(PC,Mat *S);
PETSC_EXTERN PetscErrorCode PCFieldSplitSchurRestoreS(PC,Mat *S);

PETSC_EXTERN PetscErrorCode PCGalerkinSetRestriction(PC,Mat);
PETSC_EXTERN PetscErrorCode PCGalerkinSetInterpolation(PC,Mat);

PETSC_EXTERN PetscErrorCode PCSetCoordinates(PC,PetscInt,PetscInt,PetscReal*);

PETSC_EXTERN PetscErrorCode PCPythonSetType(PC,const char[]);

PETSC_EXTERN PetscErrorCode PCSetDM(PC,DM);
PETSC_EXTERN PetscErrorCode PCGetDM(PC,DM*);

PETSC_EXTERN PetscErrorCode PCSetApplicationContext(PC,void*);
PETSC_EXTERN PetscErrorCode PCGetApplicationContext(PC,void*);

PETSC_EXTERN PetscErrorCode PCBiCGStabCUSPSetTolerance(PC,PetscReal);
PETSC_EXTERN PetscErrorCode PCBiCGStabCUSPSetIterations(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCBiCGStabCUSPSetUseVerboseMonitor(PC,PetscBool);

PETSC_EXTERN PetscErrorCode PCAINVCUSPSetDropTolerance(PC,PetscReal);
PETSC_EXTERN PetscErrorCode PCAINVCUSPUseScaling(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCAINVCUSPSetNonzeros(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCAINVCUSPSetLinParameter(PC,PetscInt);

PETSC_EXTERN PetscErrorCode PCPARMSSetGlobal(PC,PCPARMSGlobalType);
PETSC_EXTERN PetscErrorCode PCPARMSSetLocal(PC,PCPARMSLocalType);
PETSC_EXTERN PetscErrorCode PCPARMSSetSolveTolerances(PC,PetscReal,PetscInt);
PETSC_EXTERN PetscErrorCode PCPARMSSetSolveRestart(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCPARMSSetNonsymPerm(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCPARMSSetFill(PC,PetscInt,PetscInt,PetscInt);

PETSC_EXTERN PetscErrorCode PCGAMGSetType( PC,PCGAMGType);
PETSC_EXTERN PetscErrorCode PCGAMGGetType( PC,PCGAMGType*);
PETSC_EXTERN PetscErrorCode PCGAMGSetProcEqLim(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCGAMGSetRepartitioning(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCGAMGSetUseASMAggs(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCGAMGSetSolverType(PC,char[],PetscInt);
PETSC_EXTERN PetscErrorCode PCGAMGSetThreshold(PC,PetscReal);
PETSC_EXTERN PetscErrorCode PCGAMGSetCoarseEqLim(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCGAMGSetNlevels(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCGAMGSetNSmooths(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCGAMGSetSymGraph(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCGAMGSetSquareGraph(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCGAMGSetReuseInterpolation(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCGAMGFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PCGAMGInitializePackage(void);
PETSC_EXTERN PetscErrorCode PCGAMGRegister(PCGAMGType,PetscErrorCode (*)(PC));

PETSC_EXTERN PetscErrorCode PCGAMGClassicalSetType(PC,PCGAMGClassicalType);
PETSC_EXTERN PetscErrorCode PCGAMGClassicalGetType(PC,PCGAMGClassicalType*);

PETSC_EXTERN PetscErrorCode PCBDDCSetChangeOfBasisMat(PC,Mat);
PETSC_EXTERN PetscErrorCode PCBDDCSetPrimalVerticesLocalIS(PC,IS);
PETSC_EXTERN PetscErrorCode PCBDDCSetCoarseningRatio(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCBDDCSetLevels(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCBDDCSetNullSpace(PC,MatNullSpace);
PETSC_EXTERN PetscErrorCode PCBDDCSetDirichletBoundaries(PC,IS);
PETSC_EXTERN PetscErrorCode PCBDDCSetDirichletBoundariesLocal(PC,IS);
PETSC_EXTERN PetscErrorCode PCBDDCGetDirichletBoundaries(PC,IS*);
PETSC_EXTERN PetscErrorCode PCBDDCGetDirichletBoundariesLocal(PC,IS*);
PETSC_EXTERN PetscErrorCode PCBDDCSetNeumannBoundaries(PC,IS);
PETSC_EXTERN PetscErrorCode PCBDDCSetNeumannBoundariesLocal(PC,IS);
PETSC_EXTERN PetscErrorCode PCBDDCGetNeumannBoundaries(PC,IS*);
PETSC_EXTERN PetscErrorCode PCBDDCGetNeumannBoundariesLocal(PC,IS*);
PETSC_EXTERN PetscErrorCode PCBDDCSetDofsSplitting(PC,PetscInt,IS[]);
PETSC_EXTERN PetscErrorCode PCBDDCSetDofsSplittingLocal(PC,PetscInt,IS[]);
PETSC_EXTERN PetscErrorCode PCBDDCSetLocalAdjacencyGraph(PC,PetscInt,const PetscInt[],const PetscInt[],PetscCopyMode);
PETSC_EXTERN PetscErrorCode PCBDDCCreateFETIDPOperators(PC,Mat*,PC*);
PETSC_EXTERN PetscErrorCode PCBDDCMatFETIDPGetRHS(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCBDDCMatFETIDPGetSolution(Mat,Vec,Vec);

PETSC_EXTERN PetscErrorCode PCISSetUseStiffnessScaling(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCISSetSubdomainScalingFactor(PC,PetscScalar);
PETSC_EXTERN PetscErrorCode PCISSetSubdomainDiagonalScaling(PC,Vec);

PETSC_EXTERN PetscErrorCode PCMGSetType(PC,PCMGType);
PETSC_EXTERN PetscErrorCode PCMGGetType(PC,PCMGType*);
PETSC_EXTERN PetscErrorCode PCMGSetLevels(PC,PetscInt,MPI_Comm*);
PETSC_EXTERN PetscErrorCode PCMGGetLevels(PC,PetscInt*);

PETSC_EXTERN PetscErrorCode PCMGSetNumberSmoothUp(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCMGSetNumberSmoothDown(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCMGSetCycleType(PC,PCMGCycleType);
PETSC_EXTERN PetscErrorCode PCMGSetCycleTypeOnLevel(PC,PetscInt,PCMGCycleType);
PETSC_EXTERN PetscErrorCode PCMGSetCyclesOnLevel(PC,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode PCMGMultiplicativeSetCycles(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCMGSetGalerkin(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCMGGetGalerkin(PC,PetscBool*);

PETSC_EXTERN PetscErrorCode PCMGSetRhs(PC,PetscInt,Vec);
PETSC_EXTERN PetscErrorCode PCMGSetX(PC,PetscInt,Vec);
PETSC_EXTERN PetscErrorCode PCMGSetR(PC,PetscInt,Vec);

PETSC_EXTERN PetscErrorCode PCMGSetRestriction(PC,PetscInt,Mat);
PETSC_EXTERN PetscErrorCode PCMGGetRestriction(PC,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode PCMGSetInterpolation(PC,PetscInt,Mat);
PETSC_EXTERN PetscErrorCode PCMGGetInterpolation(PC,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode PCMGSetRScale(PC,PetscInt,Vec);
PETSC_EXTERN PetscErrorCode PCMGGetRScale(PC,PetscInt,Vec*);
PETSC_EXTERN PetscErrorCode PCMGSetResidual(PC,PetscInt,PetscErrorCode (*)(Mat,Vec,Vec,Vec),Mat);
PETSC_EXTERN PetscErrorCode PCMGResidualDefault(Mat,Vec,Vec,Vec);

#endif /* __PETSCPC_H */
