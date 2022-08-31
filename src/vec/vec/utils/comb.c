
/*
      Split phase global vector reductions with support for combining the
   communication portion of several operations. Using MPI-1.1 support only

      The idea for this and much of the initial code is contributed by
   Victor Eijkhout.

       Usage:
             VecDotBegin(Vec,Vec,PetscScalar *);
             VecNormBegin(Vec,NormType,PetscReal *);
             ....
             VecDotEnd(Vec,Vec,PetscScalar *);
             VecNormEnd(Vec,NormType,PetscReal *);

       Limitations:
         - The order of the xxxEnd() functions MUST be in the same order
           as the xxxBegin(). There is extensive error checking to try to
           insure that the user calls the routines in the correct order
*/

#include <petsc/private/vecimpl.h> /*I   "petscvec.h"    I*/
#include <petsc/private/deviceimpl.h>

static PetscErrorCode MPIPetsc_Iallreduce(void *sendbuf, void *recvbuf, PetscMPIInt count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request) {
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPI_NONBLOCKING_COLLECTIVES)
  PetscCallMPI(MPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request));
#else
  PetscCall(MPIU_Allreduce(sendbuf, recvbuf, count, datatype, op, comm));
  *request = MPI_REQUEST_NULL;
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSplitReductionApply(PetscSplitReduction *);

/*
   PetscSplitReductionCreate - Creates a data structure to contain the queued information.
*/
static PetscErrorCode PetscSplitReductionCreate(MPI_Comm comm, PetscSplitReduction **sr) {
  PetscFunctionBegin;
  PetscCall(PetscNew(sr));
  (*sr)->numopsbegin = 0;
  (*sr)->numopsend   = 0;
  (*sr)->state       = STATE_BEGIN;
#define MAXOPS 32
  (*sr)->maxops = MAXOPS;
  PetscCall(PetscMalloc6(MAXOPS, &(*sr)->lvalues, MAXOPS, &(*sr)->gvalues, MAXOPS, &(*sr)->invecs, MAXOPS, &(*sr)->reducetype, MAXOPS, &(*sr)->lvalues_mix, MAXOPS, &(*sr)->gvalues_mix));
#undef MAXOPS
  (*sr)->comm    = comm;
  (*sr)->request = MPI_REQUEST_NULL;
  (*sr)->mix     = PETSC_FALSE;
  (*sr)->async   = PETSC_FALSE;
#if defined(PETSC_HAVE_MPI_NONBLOCKING_COLLECTIVES)
  (*sr)->async = PETSC_TRUE; /* Enable by default */
#endif
  /* always check for option; so that tests that run on systems without support don't warn about unhandled options */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-splitreduction_async", &(*sr)->async, NULL));
  PetscFunctionReturn(0);
}

/*
       This function is the MPI reduction operation used when there is
   a combination of sums and max in the reduction. The call below to
   MPI_Op_create() converts the function PetscSplitReduction_Local() to the
   MPI operator PetscSplitReduction_Op.
*/
MPI_Op PetscSplitReduction_Op = 0;

PETSC_EXTERN void MPIAPI PetscSplitReduction_Local(void *in, void *out, PetscMPIInt *cnt, MPI_Datatype *datatype) {
  struct PetscScalarInt {
    PetscScalar v;
    PetscInt    i;
  };
  struct PetscScalarInt *xin  = (struct PetscScalarInt *)in;
  struct PetscScalarInt *xout = (struct PetscScalarInt *)out;
  PetscInt               i, count = (PetscInt)*cnt;

  PetscFunctionBegin;
  if (*datatype != MPIU_SCALAR_INT) {
    (*PetscErrorPrintf)("Can only handle MPIU_SCALAR_INT data types");
    PETSCABORT(MPI_COMM_SELF, PETSC_ERR_ARG_WRONG);
  }
  for (i = 0; i < count; i++) {
    if (xin[i].i == PETSC_SR_REDUCE_SUM) xout[i].v += xin[i].v;
    else if (xin[i].i == PETSC_SR_REDUCE_MAX) xout[i].v = PetscMax(PetscRealPart(xout[i].v), PetscRealPart(xin[i].v));
    else if (xin[i].i == PETSC_SR_REDUCE_MIN) xout[i].v = PetscMin(PetscRealPart(xout[i].v), PetscRealPart(xin[i].v));
    else {
      (*PetscErrorPrintf)("Reduction type input is not PETSC_SR_REDUCE_SUM, PETSC_SR_REDUCE_MAX, or PETSC_SR_REDUCE_MIN");
      PETSCABORT(MPI_COMM_SELF, PETSC_ERR_ARG_WRONG);
    }
  }
  PetscFunctionReturnVoid();
}

/*@
   PetscCommSplitReductionBegin - Begin an asynchronous split-mode reduction

   Collective but not synchronizing

   Input Parameter:
   comm - communicator on which split reduction has been queued

   Level: advanced

   Note:
   Calling this function is optional when using split-mode reduction. On supporting hardware, calling this after all
   VecXxxBegin() allows the reduction to make asynchronous progress before the result is needed (in VecXxxEnd()).

.seealso: `VecNormBegin()`, `VecNormEnd()`, `VecDotBegin()`, `VecDotEnd()`, `VecTDotBegin()`, `VecTDotEnd()`, `VecMDotBegin()`, `VecMDotEnd()`, `VecMTDotBegin()`, `VecMTDotEnd()`
@*/
PetscErrorCode PetscCommSplitReductionBegin(MPI_Comm comm) {
  PetscSplitReduction *sr;

  PetscFunctionBegin;
  PetscCall(PetscSplitReductionGet(comm, &sr));
  PetscCheck(sr->numopsend <= 0, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Cannot call this after VecxxxEnd() has been called");
  if (sr->async) { /* Bad reuse, setup code copied from PetscSplitReductionApply(). */
    PetscInt     i, numops = sr->numopsbegin, *reducetype = sr->reducetype;
    PetscScalar *lvalues = sr->lvalues, *gvalues = sr->gvalues;
    PetscInt     sum_flg = 0, max_flg = 0, min_flg = 0;
    MPI_Comm     comm = sr->comm;
    PetscMPIInt  size, cmul = sizeof(PetscScalar) / sizeof(PetscReal);

    PetscCall(PetscLogEventBegin(VEC_ReduceBegin, 0, 0, 0, 0));
    PetscCallMPI(MPI_Comm_size(sr->comm, &size));
    if (size == 1) {
      PetscCall(PetscArraycpy(gvalues, lvalues, numops));
    } else {
      /* determine if all reductions are sum, max, or min */
      for (i = 0; i < numops; i++) {
        if (reducetype[i] == PETSC_SR_REDUCE_MAX) max_flg = 1;
        else if (reducetype[i] == PETSC_SR_REDUCE_SUM) sum_flg = 1;
        else if (reducetype[i] == PETSC_SR_REDUCE_MIN) min_flg = 1;
        else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in PetscSplitReduction() data structure, probably memory corruption");
      }
      PetscCheck(sum_flg + max_flg + min_flg <= 1 || !sr->mix, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in PetscSplitReduction() data structure, probably memory corruption");
      if (sum_flg + max_flg + min_flg > 1) {
        sr->mix = PETSC_TRUE;
        for (i = 0; i < numops; i++) {
          sr->lvalues_mix[i].v = lvalues[i];
          sr->lvalues_mix[i].i = reducetype[i];
        }
        PetscCall(MPIPetsc_Iallreduce(sr->lvalues_mix, sr->gvalues_mix, numops, MPIU_SCALAR_INT, PetscSplitReduction_Op, comm, &sr->request));
      } else if (max_flg) { /* Compute max of real and imag parts separately, presumably only the real part is used */
        PetscCall(MPIPetsc_Iallreduce((PetscReal *)lvalues, (PetscReal *)gvalues, cmul * numops, MPIU_REAL, MPIU_MAX, comm, &sr->request));
      } else if (min_flg) {
        PetscCall(MPIPetsc_Iallreduce((PetscReal *)lvalues, (PetscReal *)gvalues, cmul * numops, MPIU_REAL, MPIU_MIN, comm, &sr->request));
      } else {
        PetscCall(MPIPetsc_Iallreduce(lvalues, gvalues, numops, MPIU_SCALAR, MPIU_SUM, comm, &sr->request));
      }
    }
    sr->state     = STATE_PENDING;
    sr->numopsend = 0;
    PetscCall(PetscLogEventEnd(VEC_ReduceBegin, 0, 0, 0, 0));
  } else {
    PetscCall(PetscSplitReductionApply(sr));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSplitReductionEnd(PetscSplitReduction *sr) {
  PetscFunctionBegin;
  switch (sr->state) {
  case STATE_BEGIN: /* We are doing synchronous communication and this is the first call to VecXxxEnd() so do the communication */ PetscCall(PetscSplitReductionApply(sr)); break;
  case STATE_PENDING:
    /* We are doing asynchronous-mode communication and this is the first VecXxxEnd() so wait for comm to complete */
    PetscCall(PetscLogEventBegin(VEC_ReduceEnd, 0, 0, 0, 0));
    if (sr->request != MPI_REQUEST_NULL) PetscCallMPI(MPI_Wait(&sr->request, MPI_STATUS_IGNORE));
    sr->state = STATE_END;
    if (sr->mix) {
      PetscInt i;
      for (i = 0; i < sr->numopsbegin; i++) sr->gvalues[i] = sr->gvalues_mix[i].v;
      sr->mix = PETSC_FALSE;
    }
    PetscCall(PetscLogEventEnd(VEC_ReduceEnd, 0, 0, 0, 0));
    break;
  default: break; /* everything is already done */
  }
  PetscFunctionReturn(0);
}

/*
   PetscSplitReductionApply - Actually do the communication required for a split phase reduction
*/
static PetscErrorCode PetscSplitReductionApply(PetscSplitReduction *sr) {
  PetscInt     i, numops = sr->numopsbegin, *reducetype = sr->reducetype;
  PetscScalar *lvalues = sr->lvalues, *gvalues = sr->gvalues;
  PetscInt     sum_flg = 0, max_flg = 0, min_flg = 0;
  MPI_Comm     comm = sr->comm;
  PetscMPIInt  size, cmul = sizeof(PetscScalar) / sizeof(PetscReal);

  PetscFunctionBegin;
  PetscCheck(sr->numopsend <= 0, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Cannot call this after VecxxxEnd() has been called");
  PetscCall(PetscLogEventBegin(VEC_ReduceCommunication, 0, 0, 0, 0));
  PetscCallMPI(MPI_Comm_size(sr->comm, &size));
  if (size == 1) {
    PetscCall(PetscArraycpy(gvalues, lvalues, numops));
  } else {
    /* determine if all reductions are sum, max, or min */
    for (i = 0; i < numops; i++) {
      if (reducetype[i] == PETSC_SR_REDUCE_MAX) max_flg = 1;
      else if (reducetype[i] == PETSC_SR_REDUCE_SUM) sum_flg = 1;
      else if (reducetype[i] == PETSC_SR_REDUCE_MIN) min_flg = 1;
      else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in PetscSplitReduction() data structure, probably memory corruption");
    }
    if (sum_flg + max_flg + min_flg > 1) {
      PetscCheck(!sr->mix, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in PetscSplitReduction() data structure, probably memory corruption");
      for (i = 0; i < numops; i++) {
        sr->lvalues_mix[i].v = lvalues[i];
        sr->lvalues_mix[i].i = reducetype[i];
      }
      PetscCall(MPIU_Allreduce(sr->lvalues_mix, sr->gvalues_mix, numops, MPIU_SCALAR_INT, PetscSplitReduction_Op, comm));
      for (i = 0; i < numops; i++) sr->gvalues[i] = sr->gvalues_mix[i].v;
    } else if (max_flg) { /* Compute max of real and imag parts separately, presumably only the real part is used */
      PetscCall(MPIU_Allreduce((PetscReal *)lvalues, (PetscReal *)gvalues, cmul * numops, MPIU_REAL, MPIU_MAX, comm));
    } else if (min_flg) {
      PetscCall(MPIU_Allreduce((PetscReal *)lvalues, (PetscReal *)gvalues, cmul * numops, MPIU_REAL, MPIU_MIN, comm));
    } else {
      PetscCall(MPIU_Allreduce(lvalues, gvalues, numops, MPIU_SCALAR, MPIU_SUM, comm));
    }
  }
  sr->state     = STATE_END;
  sr->numopsend = 0;
  PetscCall(PetscLogEventEnd(VEC_ReduceCommunication, 0, 0, 0, 0));
  PetscFunctionReturn(0);
}

/*
   PetscSplitReductionExtend - Double the amount of space (slots) allocated for a split reduction object.
*/
PetscErrorCode PetscSplitReductionExtend(PetscSplitReduction *sr) {
  struct PetscScalarInt {
    PetscScalar v;
    PetscInt    i;
  };
  PetscInt               maxops = sr->maxops, *reducetype = sr->reducetype;
  PetscScalar           *lvalues = sr->lvalues, *gvalues = sr->gvalues;
  struct PetscScalarInt *lvalues_mix = (struct PetscScalarInt *)sr->lvalues_mix;
  struct PetscScalarInt *gvalues_mix = (struct PetscScalarInt *)sr->gvalues_mix;
  void                 **invecs      = sr->invecs;

  PetscFunctionBegin;
  sr->maxops = 2 * maxops;
  PetscCall(PetscMalloc6(2 * maxops, &sr->lvalues, 2 * maxops, &sr->gvalues, 2 * maxops, &sr->reducetype, 2 * maxops, &sr->invecs, 2 * maxops, &sr->lvalues_mix, 2 * maxops, &sr->gvalues_mix));
  PetscCall(PetscArraycpy(sr->lvalues, lvalues, maxops));
  PetscCall(PetscArraycpy(sr->gvalues, gvalues, maxops));
  PetscCall(PetscArraycpy(sr->reducetype, reducetype, maxops));
  PetscCall(PetscArraycpy(sr->invecs, invecs, maxops));
  PetscCall(PetscArraycpy(sr->lvalues_mix, lvalues_mix, maxops));
  PetscCall(PetscArraycpy(sr->gvalues_mix, gvalues_mix, maxops));
  PetscCall(PetscFree6(lvalues, gvalues, reducetype, invecs, lvalues_mix, gvalues_mix));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSplitReductionDestroy(PetscSplitReduction *sr) {
  PetscFunctionBegin;
  PetscCall(PetscFree6(sr->lvalues, sr->gvalues, sr->reducetype, sr->invecs, sr->lvalues_mix, sr->gvalues_mix));
  PetscCall(PetscFree(sr));
  PetscFunctionReturn(0);
}

PetscMPIInt Petsc_Reduction_keyval = MPI_KEYVAL_INVALID;

/*
   Private routine to delete internal storage when a communicator is freed.
  This is called by MPI, not by users.

  The binding for the first argument changed from MPI 1.0 to 1.1; in 1.0
  it was MPI_Comm *comm.
*/
PETSC_EXTERN int MPIAPI Petsc_DelReduction(MPI_Comm comm, int keyval, void *attr_val, void *extra_state) {
  PetscFunctionBegin;
  PetscCallMPI(PetscInfo(0, "Deleting reduction data in an MPI_Comm %ld\n", (long)comm));
  PetscCallMPI(PetscSplitReductionDestroy((PetscSplitReduction *)attr_val));
  PetscFunctionReturn(0);
}

/*
     PetscSplitReductionGet - Gets the split reduction object from a
        PETSc vector, creates if it does not exit.

*/
PetscErrorCode PetscSplitReductionGet(MPI_Comm comm, PetscSplitReduction **sr) {
  PetscMPIInt flag;

  PetscFunctionBegin;
  if (Petsc_Reduction_keyval == MPI_KEYVAL_INVALID) {
    /*
       The calling sequence of the 2nd argument to this function changed
       between MPI Standard 1.0 and the revisions 1.1 Here we match the
       new standard, if you are using an MPI implementation that uses
       the older version you will get a warning message about the next line;
       it is only a warning message and should do no harm.
    */
    PetscCallMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, Petsc_DelReduction, &Petsc_Reduction_keyval, NULL));
  }
  PetscCallMPI(MPI_Comm_get_attr(comm, Petsc_Reduction_keyval, (void **)sr, &flag));
  if (!flag) { /* doesn't exist yet so create it and put it in */
    PetscCall(PetscSplitReductionCreate(comm, sr));
    PetscCallMPI(MPI_Comm_set_attr(comm, Petsc_Reduction_keyval, *sr));
    PetscCall(PetscInfo(0, "Putting reduction data in an MPI_Comm %ld\n", (long)comm));
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------------------*/

static PetscErrorCode VecXDotBeginAsync(Vec x, Vec y, PetscManagedScalar PETSC_UNUSED result, PetscDeviceContext dctx, PetscErrorCode (*const op_local)(Vec, Vec, PetscManagedScalar, PetscDeviceContext)) {
  PetscSplitReduction *sr;
  PetscManagedScalar   tmp;
  MPI_Comm             comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  PetscCheckSameTypeAndComm(x, 1, y, 2);
  VecCheckSameSize(x, 1, y, 2);
  PetscCall(PetscObjectGetComm((PetscObject)x, &comm));
  PetscCall(PetscSplitReductionGet(comm, &sr));
  PetscCheck(sr->state == STATE_BEGIN, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Called before all VecxxxEnd() called");
  PetscCheck(op_local, PETSC_COMM_SELF, PETSC_ERR_SUP, "Vector does not support local dots");
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

  if (sr->numopsbegin >= sr->maxops) PetscCall(PetscSplitReductionExtend(sr));
  sr->reducetype[sr->numopsbegin] = PETSC_SR_REDUCE_SUM;
  sr->invecs[sr->numopsbegin]     = (void *)x;

  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  PetscCall(PetscManageHostScalar(dctx, sr->lvalues + sr->numopsbegin++, 1, &tmp));
  PetscCall(PetscLogEventBegin(VEC_ReduceArithmetic, 0, 0, 0, 0));
  PetscCall((*op_local)(x, y, tmp, dctx));
  PetscCall(PetscLogEventEnd(VEC_ReduceArithmetic, 0, 0, 0, 0));
  PetscCall(PetscManagedHostScalarDestroy(dctx, &tmp));
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotBeginAsync(Vec x, Vec y, PetscManagedScalar result, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  // check before we dereference for ops
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCall(VecXDotBeginAsync(x, y, NULL, dctx, x->ops->dot_local));
  PetscFunctionReturn(0);
}

/*@
   VecDotBegin - Starts a split phase dot product computation.

   Input Parameters:
+   x - the first vector
.   y - the second vector
-   result - where the result will go (can be NULL)

   Level: advanced

   Notes:
   Each call to VecDotBegin() should be paired with a call to VecDotEnd().

seealso: VecDotEnd(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(),
         VecTDotBegin(), VecTDotEnd(), PetscCommSplitReductionBegin()
@*/
PetscErrorCode VecDotBegin(Vec x, Vec y, PetscScalar *result) {
  PetscFunctionBegin;
  if (result) PetscValidScalarPointer(result, 3);
  PetscCall(VecDotBeginAsync(x, y, NULL, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotEndAsync(Vec x, Vec y, PetscManagedScalar result, PetscDeviceContext dctx) {
  PetscSplitReduction *sr;
  MPI_Comm             comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  PetscCheckSameTypeAndComm(x, 1, y, 2);
  VecCheckSameSize(x, 1, y, 2);
  PetscCall(PetscObjectGetComm((PetscObject)x, &comm));
  PetscCall(PetscSplitReductionGet(comm, &sr));
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  // sync for MPI
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(PetscSplitReductionEnd(sr));

  PetscCheck(sr->numopsend < sr->numopsbegin, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Called VecxxxEnd() more times then VecxxxBegin()");
  PetscCheck(!x || (void *)x == sr->invecs[sr->numopsend], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Called VecxxxEnd() in a different order or with a different vector than VecxxxBegin()");
  PetscCheck(sr->reducetype[sr->numopsend] == PETSC_SR_REDUCE_SUM, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Called VecDotEnd() on a reduction started with VecNormBegin()");

  PetscCall(PetscManagedScalarSetValues(dctx, result, PETSC_MEMTYPE_HOST, sr->gvalues + sr->numopsend++, 1));
  /*
     We are finished getting all the results so reset to no outstanding requests
  */
  if (sr->numopsend == sr->numopsbegin) {
    sr->state       = STATE_BEGIN;
    sr->numopsend   = 0;
    sr->numopsbegin = 0;
    sr->mix         = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*@
   VecDotEnd - Ends a split phase dot product computation.

   Input Parameters:
+  x - the first vector (can be NULL)
.  y - the second vector (can be NULL)
-  result - where the result will go

   Level: advanced

   Notes:
   Each call to VecDotBegin() should be paired with a call to VecDotEnd().

.seealso: `VecDotBegin()`, `VecNormBegin()`, `VecNormEnd()`, `VecNorm()`, `VecDot()`, `VecMDot()`,
          `VecTDotBegin()`, `VecTDotEnd()`, `PetscCommSplitReductionBegin()`

@*/
PetscErrorCode VecDotEnd(Vec x, Vec y, PetscScalar *result) {
  PetscManagedScalar scal;

  PetscFunctionBegin;
  PetscValidScalarPointer(result, 3);
  PetscCall(PetscManageHostScalar(NULL, result, 1, &scal));
  PetscCall(VecDotEndAsync(x, y, scal, NULL));
  PetscCall(PetscManagedHostScalarDestroy(NULL, &scal));
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDotBeginAsync(Vec x, Vec y, PetscManagedScalar result, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  // check before we dereference for ops
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCall(VecXDotBeginAsync(x, y, NULL, dctx, x->ops->tdot_local));
  PetscFunctionReturn(0);
}

/*@
   VecTDotBegin - Starts a split phase transpose dot product computation.

   Input Parameters:
+  x - the first vector
.  y - the second vector
-  result - where the result will go (can be NULL)

   Level: advanced

   Notes:
   Each call to VecTDotBegin() should be paired with a call to VecTDotEnd().

.seealso: `VecTDotEnd()`, `VecNormBegin()`, `VecNormEnd()`, `VecNorm()`, `VecDot()`, `VecMDot()`,
          `VecDotBegin()`, `VecDotEnd()`, `PetscCommSplitReductionBegin()`

@*/
PetscErrorCode VecTDotBegin(Vec x, Vec y, PetscScalar *result) {
  PetscFunctionBegin;
  if (result) PetscValidScalarPointer(result, 3);
  PetscCall(VecTDotBeginAsync(x, y, NULL, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDotEndAsync(Vec x, Vec y, PetscManagedScalar result, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  /* TDotEnd() is the same as DotEnd() so reuse the code */
  PetscCall(VecDotEndAsync(x, y, result, dctx));
  PetscFunctionReturn(0);
}

/*@
   VecTDotEnd - Ends a split phase transpose dot product computation.

   Input Parameters:
+  x - the first vector (can be NULL)
.  y - the second vector (can be NULL)
-  result - where the result will go

   Level: advanced

   Notes:
   Each call to VecTDotBegin() should be paired with a call to VecTDotEnd().

seealso: VecTDotBegin(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(),
         VecDotBegin(), VecDotEnd()
@*/
PetscErrorCode VecTDotEnd(Vec x, Vec y, PetscScalar *result) {
  PetscFunctionBegin;
  /* TDotEnd() is the same as DotEnd() so reuse the code */
  PetscCall(VecDotEnd(x, y, result));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------*/

PetscErrorCode VecNormBeginAsync(Vec x, NormType ntype, PetscManagedReal result, PetscDeviceContext dctx) {
  PetscSplitReduction *sr;
  PetscReal           *resptr;
  MPI_Comm             comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

  PetscCall(PetscObjectGetComm((PetscObject)x, &comm));
  PetscCall(PetscSplitReductionGet(comm, &sr));
  PetscCheck(sr->state == STATE_BEGIN, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Called before all VecxxxEnd() called");
  PetscCheck(x->ops->norm_local, PETSC_COMM_SELF, PETSC_ERR_SUP, "Vector does not support local norms");

  if (sr->numopsbegin >= sr->maxops || (sr->numopsbegin == sr->maxops - 1 && ntype == NORM_1_AND_2)) { PetscCall(PetscSplitReductionExtend(sr)); }
  sr->invecs[sr->numopsbegin] = (void *)x;

  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_ReduceArithmetic, 0, 0, 0, 0));
  PetscCall((*x->ops->norm_local)(x, ntype, result, dctx));
  PetscCall(PetscLogEventEnd(VEC_ReduceArithmetic, 0, 0, 0, 0));
  PetscCall(VecLockReadPop(x));

  // implicit sync, can likely do this better without a sync necessary
  PetscCall(PetscManagedRealGetValues(dctx, result, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &resptr));

  sr->reducetype[sr->numopsbegin] = ntype == NORM_MAX ? PETSC_SR_REDUCE_MAX : PETSC_SR_REDUCE_SUM;
  sr->lvalues[sr->numopsbegin++]  = ntype == NORM_2 ? PetscSqr(resptr[0]) : resptr[0];
  if (ntype == NORM_1_AND_2) {
    sr->reducetype[sr->numopsbegin] = PETSC_SR_REDUCE_SUM;
    sr->lvalues[sr->numopsbegin++]  = resptr[1] * resptr[1];
  }
  PetscFunctionReturn(0);
}

/*@
   VecNormBegin - Starts a split phase norm computation.

   Input Parameters:
+  x - the first vector
.  ntype - norm type, one of NORM_1, NORM_2, NORM_MAX, NORM_1_AND_2
-  result - where the result will go (can be NULL)

   Level: advanced

   Notes:
   Each call to VecNormBegin() should be paired with a call to VecNormEnd().

.seealso: `VecNormEnd()`, `VecNorm()`, `VecDot()`, `VecMDot()`, `VecDotBegin()`, `VecDotEnd()`, `PetscCommSplitReductionBegin()`

@*/
PetscErrorCode VecNormBegin(Vec x, NormType ntype, PetscReal *result) {
  PetscManagedReal   tmp;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscValidRealPointer(result, 3);
  PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  PetscCall(PetscManageHostReal(dctx, result, 1 + (ntype == NORM_1_AND_2), &tmp));
  PetscCall(VecNormBeginAsync(x, ntype, tmp, dctx));
  PetscCall(PetscManagedHostRealDestroy(dctx, &tmp));
  PetscFunctionReturn(0);
}

PetscErrorCode VecNormEndAsync(Vec x, NormType ntype, PetscManagedReal result, PetscDeviceContext dctx) {
  PetscSplitReduction *sr;
  PetscReal            tmp[2] = {0.0, 0.0};
  MPI_Comm             comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

  PetscCall(PetscObjectGetComm((PetscObject)x, &comm));
  PetscCall(PetscSplitReductionGet(comm, &sr));
  // sync for MPI
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(PetscSplitReductionEnd(sr));

  PetscCheck(sr->numopsend < sr->numopsbegin, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Called VecxxxEnd() more times then VecxxxBegin()");
  PetscCheck((void *)x == sr->invecs[sr->numopsend], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Called VecxxxEnd() in a different order or with a different vector than VecxxxBegin()");
  PetscCheck(sr->reducetype[sr->numopsend] == PETSC_SR_REDUCE_MAX || ntype != NORM_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Called VecNormEnd(,NORM_MAX,) on a reduction started with VecDotBegin() or NORM_1 or NORM_2");

  tmp[0] = PetscRealPart(sr->gvalues[sr->numopsend++]);
  if (ntype == NORM_2) tmp[0] = PetscSqrtReal(tmp[0]);
  if (ntype == NORM_1_AND_2) {
    tmp[1] = PetscSqrtReal(PetscRealPart(sr->gvalues[sr->numopsend++]));
  } else {
    PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[ntype], tmp[0]));
  }
  PetscCall(PetscManagedRealSetValues(dctx, result, PETSC_MEMTYPE_HOST, tmp, 1 + (ntype == NORM_1_AND_2)));
  if (sr->numopsend == sr->numopsbegin) {
    sr->state       = STATE_BEGIN;
    sr->numopsend   = 0;
    sr->numopsbegin = 0;
  }
  PetscFunctionReturn(0);
}

/*@
   VecNormEnd - Ends a split phase norm computation.

   Input Parameters:
+  x - the first vector
.  ntype - norm type, one of NORM_1, NORM_2, NORM_MAX, NORM_1_AND_2
-  result - where the result will go

   Level: advanced

   Notes:
   Each call to VecNormBegin() should be paired with a call to VecNormEnd().

   The x vector is not allowed to be NULL, otherwise the vector would not have its correctly cached norm value

.seealso: `VecNormBegin()`, `VecNorm()`, `VecDot()`, `VecMDot()`, `VecDotBegin()`, `VecDotEnd()`, `PetscCommSplitReductionBegin()`

@*/
PetscErrorCode VecNormEnd(Vec x, NormType ntype, PetscReal *result) {
  PetscManagedReal scal;

  PetscFunctionBegin;
  PetscValidRealPointer(result, 3);
  PetscCall(PetscManageHostReal(NULL, result, 1 + (ntype == NORM_1_AND_2), &scal));
  PetscCall(VecNormEndAsync(x, ntype, scal, NULL));
  PetscCall(PetscManagedHostRealDestroy(NULL, &scal));
  PetscFunctionReturn(0);
}

/*
   Possibly add

     PetscReductionSumBegin/End()
     PetscReductionMaxBegin/End()
     PetscReductionMinBegin/End()
   or have more like MPI with a single function with flag for Op? Like first better
*/
static PetscErrorCode VecMXDotBeginAsync_Private(Vec x, PetscManagedInt nv, const Vec y[], PetscManagedScalar PETSC_UNUSED result, PetscDeviceContext dctx, PetscErrorCode (*const op_local)(Vec, PetscManagedInt, const Vec *, PetscManagedScalar, PetscDeviceContext)) {
  PetscSplitReduction *sr;
  PetscManagedScalar   scal;
  MPI_Comm             comm;
  PetscInt            *nvptr;
  PetscInt             nvval;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscValidPointer(y, 3);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(PetscObjectGetComm((PetscObject)x, &comm));
  PetscCall(PetscSplitReductionGet(comm, &sr));
  PetscCheck(sr->state == STATE_BEGIN, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Called before all VecxxxEnd() called");
  PetscCheck(op_local, PETSC_COMM_SELF, PETSC_ERR_SUP, "Vector does not support local mdots");

  // implicit sync
  PetscCall(PetscManagedIntGetValues(dctx, nv, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &nvptr));
  nvval = *nvptr;
  for (PetscInt i = 0; i < nvval; ++i) {
    // use this opportunity to check y
    PetscValidType(y[i], 3);
    PetscCheckSameTypeAndComm(x, 1, y[i], 3);
    VecCheckSameSize(x, 1, y[i], 3);
    PetscCall(VecLockReadPush(y[i]));

    if (sr->numopsbegin + i >= sr->maxops) PetscCall(PetscSplitReductionExtend(sr));
    sr->reducetype[sr->numopsbegin + i] = PETSC_SR_REDUCE_SUM;
    sr->invecs[sr->numopsbegin + i]     = (void *)x;
  }
  PetscCall(PetscManageHostScalar(dctx, sr->lvalues + sr->numopsbegin, nvval, &scal));
  sr->numopsbegin += nvval;
  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_ReduceArithmetic, 0, 0, 0, 0));
  PetscCall((*op_local)(x, nv, y, scal, dctx));
  PetscCall(PetscLogEventEnd(VEC_ReduceArithmetic, 0, 0, 0, 0));
  PetscCall(PetscManagedHostScalarDestroy(dctx, &scal));
  PetscCall(VecLockReadPop(x));
  for (PetscInt i = 0; i < nvval; ++i) PetscCall(VecLockReadPop(y[i]));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMXDotBegin_Private(Vec x, PetscInt nv, const Vec y[], PetscScalar PETSC_UNUSED result[], PetscErrorCode (*const VecMXDotBeginAsync_Func)(Vec, PetscManagedInt, const Vec[], PetscManagedScalar, PetscDeviceContext)) {
  PetscDeviceContext dctx;
  PetscManagedInt    nvtmp;

  PetscFunctionBegin;
  if (nv) PetscValidScalarPointer(result, 4);
  PetscValidFunction(VecMXDotBeginAsync_Func, 5);
  PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  PetscCall(PetscManageHostInt(dctx, &nv, 1, &nvtmp));
  PetscCall(VecMXDotBeginAsync_Func(x, nvtmp, y, NULL, dctx));
  PetscCall(PetscManagedIntDestroy(dctx, &nvtmp));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDotBeginAsync(Vec x, PetscManagedInt nv, const Vec y[], PetscManagedScalar result, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCall(VecMXDotBeginAsync_Private(x, nv, y, result, dctx, x->ops->mdot_local));
  PetscFunctionReturn(0);
}

/*@
   VecMDotBegin - Starts a split phase multiple dot product computation.

   Input Parameters:
+   x - the first vector
.   nv - number of vectors
.   y - array of vectors
-   result - where the result will go (can be NULL)

   Level: advanced

   Notes:
   Each call to VecMDotBegin() should be paired with a call to VecMDotEnd().

.seealso: `VecMDotEnd()`, `VecNormBegin()`, `VecNormEnd()`, `VecNorm()`, `VecDot()`, `VecMDot()`,
          `VecTDotBegin()`, `VecTDotEnd()`, `VecMTDotBegin()`, `VecMTDotEnd()`, `PetscCommSplitReductionBegin()`
@*/
PetscErrorCode VecMDotBegin(Vec x, PetscInt nv, const Vec y[], PetscScalar result[]) {
  PetscFunctionBegin;
  PetscCall(VecMXDotBegin_Private(x, nv, y, result, VecMDotBeginAsync));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMTDotBeginAsync(Vec x, PetscManagedInt nv, const Vec y[], PetscManagedScalar result, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCall(VecMXDotBeginAsync_Private(x, nv, y, result, dctx, x->ops->mtdot_local));
  PetscFunctionReturn(0);
}

/*@
   VecMTDotBegin - Starts a split phase transpose multiple dot product computation.

   Input Parameters:
+  x - the first vector
.  nv - number of vectors
.  y - array of  vectors
-  result - where the result will go (can be NULL)

   Level: advanced

   Notes:
   Each call to VecMTDotBegin() should be paired with a call to VecMTDotEnd().

.seealso: VecMTDotEnd(), VecNormBegin(), VecNormEnd(), VecNorm(), VecDot(), VecMDot(),
         VecDotBegin(), VecDotEnd(), VecMDotBegin(), VecMDotEnd(), PetscCommSplitReductionBegin()

@*/
PetscErrorCode VecMTDotBegin(Vec x, PetscInt nv, const Vec y[], PetscScalar result[]) {
  PetscFunctionBegin;
  PetscCall(VecMXDotBegin_Private(x, nv, y, result, VecMTDotBeginAsync));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMXDotEndAsync_Private(Vec x, PetscManagedInt nv, const Vec y[], PetscManagedScalar result, PetscDeviceContext dctx) {
  PetscSplitReduction *sr;
  PetscInt            *nvptr;
  MPI_Comm             comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (nv) PetscValidPointer(y, 3);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(PetscObjectGetComm((PetscObject)x, &comm));
  PetscCall(PetscSplitReductionGet(comm, &sr));
  // sync for MPI
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(PetscSplitReductionEnd(sr));

  PetscCheck(sr->numopsend < sr->numopsbegin, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Called VecxxxEnd() more times then VecxxxBegin()");
  PetscCheck(!x || (void *)x == sr->invecs[sr->numopsend], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Called VecxxxEnd() in a different order or with a different vector than VecxxxBegin()");
  PetscCheck(sr->reducetype[sr->numopsend] == PETSC_SR_REDUCE_SUM, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Called VecDotEnd() on a reduction started with VecNormBegin()");
  // implicit sync
  PetscCall(PetscManagedIntGetValues(dctx, nv, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &nvptr));
  PetscCall(PetscManagedScalarSetValues(dctx, result, PETSC_MEMTYPE_HOST, sr->gvalues + sr->numopsend, *nvptr));
  sr->numopsend += *nvptr;

  /*
     We are finished getting all the results so reset to no outstanding requests
  */
  if (sr->numopsend == sr->numopsbegin) {
    sr->state       = STATE_BEGIN;
    sr->numopsend   = 0;
    sr->numopsbegin = 0;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMXDotEnd_Private(Vec x, PetscInt nv, const Vec y[], PetscScalar result[], PetscErrorCode (*const VecMXDotEndAsync_Func)(Vec, PetscManagedInt, const Vec[], PetscManagedScalar, PetscDeviceContext)) {
  PetscManagedInt    nvtmp;
  PetscManagedScalar restmp;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  if (nv) PetscValidScalarPointer(result, 4);
  PetscValidFunction(VecMXDotEndAsync_Func, 5);
  PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  PetscCall(PetscManageHostInt(dctx, &nv, 1, &nvtmp));
  PetscCall(PetscManageHostScalar(dctx, result, nv, &restmp));
  PetscCall(VecMXDotEndAsync_Func(x, nvtmp, y, restmp, dctx));
  PetscCall(PetscManagedHostScalarDestroy(dctx, &restmp));
  PetscCall(PetscManagedIntDestroy(dctx, &nvtmp));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDotEndAsync(Vec x, PetscManagedInt nv, const Vec y[], PetscManagedScalar result, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(VecMXDotEndAsync_Private(x, nv, y, result, dctx));
  PetscFunctionReturn(0);
}

/*@
   VecMDotEnd - Ends a split phase multiple dot product computation.

   Input Parameters:
+   x - the first vector (can be NULL)
.   nv - number of vectors
-   y - array of vectors (can be NULL)

   Output Parameters:
.   result - where the result will go

   Level: advanced

   Notes:
   Each call to VecMDotBegin() should be paired with a call to VecMDotEnd().

.seealso: `VecMDotBegin()`, `VecNormBegin()`, `VecNormEnd()`, `VecNorm()`, `VecDot()`, `VecMDot()`,
          `VecTDotBegin()`, `VecTDotEnd()`, `VecMTDotBegin()`, `VecMTDotEnd()`, `PetscCommSplitReductionBegin()`

@*/
PetscErrorCode VecMDotEnd(Vec x, PetscInt nv, const Vec y[], PetscScalar result[]) {
  PetscFunctionBegin;
  PetscCall(VecMXDotEnd_Private(x, nv, y, result, VecMDotEndAsync));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMTDotEndAsync(Vec x, PetscManagedInt nv, const Vec y[], PetscManagedScalar result, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  /* MTDotEnd() is the same as MDotEnd() so reuse the code */
  PetscCall(VecMDotEndAsync(x, nv, y, result, dctx));
  PetscFunctionReturn(0);
}

/*@
   VecMTDotEnd - Ends a split phase transpose multiple dot product computation.

   Input Parameters:
+  x - the first vector (can be NULL)
.  nv - number of vectors
-  y - array of  vectors (can be NULL)

   Output Parameters:
.  result - where the result will go

   Level: advanced

   Notes:
   Each call to VecTDotBegin() should be paired with a call to VecTDotEnd().

.seealso: `VecMTDotBegin()`, `VecNormBegin()`, `VecNormEnd()`, `VecNorm()`, `VecDot()`, `VecMDot()`,
          `VecDotBegin()`, `VecDotEnd()`, `VecMDotBegin()`, `VecMDotEnd()`, `PetscCommSplitReductionBegin()`
@*/
PetscErrorCode VecMTDotEnd(Vec x, PetscInt nv, const Vec y[], PetscScalar result[]) {
  PetscFunctionBegin;
  PetscCall(VecMXDotEnd_Private(x, nv, y, result, VecMTDotEndAsync));
  PetscFunctionReturn(0);
}
