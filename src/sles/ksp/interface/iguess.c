#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: iguess.c,v 1.20 1997/07/09 20:50:16 balay Exp bsmith $";
#endif

#include "src/ksp/kspimpl.h"  /*I "ksp.h" I*/
/* 
  This code inplements Paul Fischer's initial guess code for situations where
  a linear system is solved repeatedly 
 */

typedef struct {
    int      curl,     /* Current number of basis vectors */
             maxl;     /* Maximum number of basis vectors */
    Scalar   *alpha;   /* */
    Vec      *xtilde,  /* Saved x vectors */
             *btilde;  /* Saved b vectors */
} KSPIGUESS;

#undef __FUNC__  
#define __FUNC__ "KSPGuessCreate" 
int KSPGuessCreate(KSP itctx,int  maxl,void **ITG )
{
  KSPIGUESS *itg;

  *ITG = 0;
  PetscValidHeaderSpecific(itctx,KSP_COOKIE);
  itg  = (KSPIGUESS* ) PetscMalloc(sizeof(KSPIGUESS)); CHKPTRQ(itg);
  itg->curl = 0;
  itg->maxl = maxl;
  itg->alpha = (Scalar *)PetscMalloc( maxl * sizeof(Scalar) );  CHKPTRQ(itg->alpha);
  PLogObjectMemory(itctx,sizeof(KSPIGUESS) + maxl*sizeof(Scalar));
  VecDuplicateVecs(itctx->vec_rhs,maxl,&itg->xtilde);
  PLogObjectParents(itctx,maxl,itg->xtilde);
  VecDuplicateVecs(itctx->vec_rhs,maxl,&itg->btilde);
  PLogObjectParents(itctx,maxl,itg->btilde);
  *ITG = (void *)itg;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPGuessDestroy" 
int KSPGuessDestroy( KSP itctx, KSPIGUESS *itg )
{
  PetscValidHeaderSpecific(itctx,KSP_COOKIE);
  PetscFree( itg->alpha );
  VecDestroyVecs( itg->btilde, itg->maxl );
  VecDestroyVecs( itg->xtilde, itg->maxl );
  PetscFree( itg );
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPGuessFormB"
int KSPGuessFormB( KSP itctx, KSPIGUESS *itg, Vec b )
{
  int    i;
  Scalar tmp;

  PetscValidHeaderSpecific(itctx,KSP_COOKIE);
  for (i=1; i<=itg->curl; i++) {
    VecDot(itg->btilde[i-1],b,&(itg->alpha[i-1]));
    tmp = -itg->alpha[i-1];
    VecAXPY(&tmp,itg->btilde[i-1],b);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPGuessFormX"
int KSPGuessFormX( KSP itctx, KSPIGUESS *itg, Vec x )
{
  int i;
  PetscValidHeaderSpecific(itctx,KSP_COOKIE);
  VecCopy(x,itg->xtilde[itg->curl]);
  for (i=1; i<=itg->curl; i++) {
    VecAXPY(&itg->alpha[i-1],itg->xtilde[i-1],x);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPGuessUpdate"
int  KSPGuessUpdate( KSP itctx, Vec x, KSPIGUESS *itg )
{
  double       normax, norm;
  Scalar       tmp;
  MatStructure pflag;
  int          curl = itg->curl, i;
  Mat          Amat, Pmat;

  PetscValidHeaderSpecific(itctx,KSP_COOKIE);
  PCGetOperators(itctx->B,&Amat,&Pmat,&pflag);
  if (curl == itg->maxl) {
    MatMult(Amat,x,itg->btilde[0] );
    VecNorm(itg->btilde[0],NORM_2,&normax);
    tmp = 1.0/normax; VecScale(&tmp,itg->btilde[0]);
    /* VCOPY(itctx->vc,x,itg->xtilde[0]); */
    VecScale(&tmp,itg->xtilde[0]);
  }
  else {
    MatMult( Amat, itg->xtilde[curl], itg->btilde[curl] );
    for (i=1; i<=curl; i++) 
      VecDot(itg->btilde[curl],itg->btilde[i-1],itg->alpha+i-1);
    for (i=1; i<=curl; i++) {
      tmp = -itg->alpha[i-1];
      VecAXPY(&tmp,itg->btilde[i-1],itg->btilde[curl]);
      VecAXPY(&itg->alpha[i-1],itg->xtilde[i-1],itg->xtilde[curl]);
    }
    VecNorm(itg->btilde[curl],NORM_2,&norm);
    tmp = 1.0/norm; VecScale(&tmp,itg->btilde[curl]);
    VecNorm(itg->xtilde[curl],NORM_2,&norm);
    VecScale(&tmp,itg->xtilde[curl]);
    itg->curl++;
  }
  return 0;
}
