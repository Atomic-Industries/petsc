#if !defined(__TAODEF_H)
#define __TAODEF_H

#include "petsc/finclude/petscts.h"

#define TaoType character*(80)
#define TaoLineSearchType character*(80)

#define Tao PetscFortranAddr
#define TaoLineSearch PetscFortranAddr
#define TaoConvergedReason PetscEnum
#define TaoADMMUpdateType PetscEnum
#define TaoADMMRegularizerType PetscEnum

#define TAOLMVM     'lmvm'
#define TAONLS      'nls'
#define TAONTR      'ntr'
#define TAONTL      'ntl'
#define TAOCG       'cg'
#define TAOTRON     'tron'
#define TAOOWLQN    'owlqn'
#define TAOBMRM     'bmrm'
#define TAOBLMVM    'blmvm'
#define TAOBQNLS    'bqnls'
#define TAOBNCG     'bncg'
#define TAOBNLS     'bnls'
#define TAOBNTR     'bntr'
#define TAOBNTL     'bntl'
#define TAOBQNKLS   'bqnkls'
#define TAOBQNKTR   'bqnktr'
#define TAOBQNKTL   'bqnktl'
#define TAOBQPIP    'bqpip'
#define TAOGPCG     'gpcg'
#define TAONM       'nm'
#define TAOPOUNDERS 'pounders'
#define TAOBRGN     'brgn'
#define TAOLCL      'lcl'
#define TAOSSILS    'ssils'
#define TAOSSFLS    'ssfls'
#define TAOASILS    'asils'
#define TAOASFLS    'asfls'
#define TAOIPM      'ipm'
#define TAOADMM     "admm"
#define TAOFDTEST   "test"

#endif
