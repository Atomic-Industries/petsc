#define TAOLINESEARCH_DLL
#include "private/taolinesearch_impl.h"

EXTERN_C_BEGIN
extern PetscErrorCode TaoLineSearchCreate_Unit(TaoLineSearch);
extern PetscErrorCode TaoLineSearchCreate_MT(TaoLineSearch);
extern PetscErrorCode TaoLineSearchCreate_GPCG(TaoLineSearch);
extern PetscErrorCode TaoLineSearchCreate_Armijo(TaoLineSearch);
EXTERN_C_END


extern PetscBool TaoLineSearchInitialized;

//PetscClassId TAOLINESEARCH_CLASSID=0;

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
#define TaoLineSearchRegisterDynamic(a,b,c,d) TaoLineSearchRegister(a,b,c,0)
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_taolinesearch"
PetscErrorCode PetscDLLibraryRegister_taolinesearch(const char path[]) 
{
    PetscErrorCode info;
    info = PetscInitializeNoArguments(); if (info) return 1;
    PetscFunctionBegin;
    info = TaoLineSearchInitializePackage(path); CHKERRQ(info);
    PetscFunctionReturn(0);
}
EXTERN_C_END
#else
#define TaoLineSearchRegisterDynamic(a,b,c,d) TaoLineSearchRegister(a,b,c,d)
#endif

    



#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchInitializePackage"
/*@C
  TaoLineSearchInitializePackage - This function registers the line-search
  algorithms in TAO.
  When using static libraries, this function is called from the
  first entry to TaoCreate(); when using shared libraries, it is called
  from PetscDLLibraryRegister()

  Input parameter:
. path - The dynamic library path or PETSC_NULL

  Level: developer

.seealso: TaoLineSearchCreate()
@*/
PetscErrorCode TaoLineSearchInitializePackage(const char path[])
{
    PetscErrorCode info;

    PetscFunctionBegin;
    if (TaoLineSearchInitialized) PetscFunctionReturn(0);
    TaoLineSearchInitialized=PETSC_TRUE;

    info = PetscClassIdRegister("TaoLineSearch",&TAOLINESEARCH_CLASSID); CHKERRQ(info);


    info = TaoLineSearchRegisterDynamic("unit",path,"TaoLineSearchCreate_Unit",TaoLineSearchCreate_Unit); CHKERRQ(info);
    info = TaoLineSearchRegisterDynamic("more-thuente",path,"TaoLineSearchCreate_MT",TaoLineSearchCreate_MT); CHKERRQ(info);
    info = TaoLineSearchRegisterDynamic("gpcg",path,"TaoLineSearchCreate_GPCG",TaoLineSearchCreate_GPCG); CHKERRQ(info);
    info = TaoLineSearchRegisterDynamic("armijo",path,"TaoLineSearchCreate_Armijo",TaoLineSearchCreate_Armijo); CHKERRQ(info);

    info = PetscLogEventRegister(  "TaoLineSearchApply",TAOLINESEARCH_CLASSID,&TaoLineSearch_ApplyEvent); CHKERRQ(info);
    info = PetscLogEventRegister("TaoLineSearchComputeObjective[Gradient]",TAOLINESEARCH_CLASSID,&TaoLineSearch_EvalEvent); CHKERRQ(info);
    PetscFunctionReturn(0);
}



    

