#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: bvec2.c,v 1.146 1999/01/25 19:23:11 bsmith Exp bsmith $";
#endif
/*
   Implements the sequential vectors.
*/

#include "src/vec/vecimpl.h"          /*I  "vec.h"   I*/
#include "src/vec/impls/dvecimpl.h" 
#include "pinclude/blaslapack.h"
#if defined(HAVE_AMS)
extern int ViewerAMSGetAMSComm(Viewer,AMS_Comm *);
#endif

#undef __FUNC__  
#define __FUNC__ "VecNorm_Seq"
int VecNorm_Seq(Vec xin,NormType type,double* z )
{
  Vec_Seq * x = (Vec_Seq *) xin->data;
  int     ierr,one = 1;

  PetscFunctionBegin;
  if (type == NORM_2) {
    /*
      This is because the Fortran BLAS 1 Norm is very slow! 
    */
#if defined(HAVE_SLOW_NRM2)
    {
      int i;
      Scalar sum=0.0;
      for ( i=0; i<x->n; i++) {
        sum += (x->array[i])*(PetscConj(x->array[i]));
      }
      *z = sqrt(PetscReal(sum));
    }
#else
    *z = BLnrm2_( &x->n, x->array, &one );
#endif
    PLogFlops(2*x->n-1);
  } else if (type == NORM_INFINITY) {
    register int    i, n = x->n;
    register double max = 0.0, tmp;
    Scalar          *xx = x->array;

    for (i=0; i<n; i++) {
      if ((tmp = PetscAbsScalar(*xx)) > max) max = tmp;
      /* check special case of tmp == NaN */
      if (tmp != tmp) {max = tmp; break;}
      xx++;
    }
    *z   = max;
  } else if (type == NORM_1) {
    *z = BLasum_( &x->n, x->array, &one );
    PLogFlops(x->n-1);
  } else if (type == NORM_1_AND_2) {
    ierr = VecNorm_Seq(xin,NORM_1,z);CHKERRQ(ierr);
    ierr = VecNorm_Seq(xin,NORM_2,z+1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecGetOwnershipRange_Seq"
int VecGetOwnershipRange_Seq(Vec xin, int *low,int *high )
{
  Vec_Seq *x = (Vec_Seq *) xin->data;

  PetscFunctionBegin;
  *low = 0; *high = x->n;
  PetscFunctionReturn(0);
}
#include "viewer.h"
#include "sys.h"

#undef __FUNC__  
#define __FUNC__ "VecView_Seq_File"
int VecView_Seq_File(Vec xin,Viewer viewer)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
  int      i, n = x->n,ierr,format;
  FILE     *fd;
  char     *outputname;

  PetscFunctionBegin;
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);

  ierr = ViewerGetFormat(viewer,&format);
  if (format == VIEWER_FORMAT_ASCII_MATLAB) {
    ierr = ViewerGetOutputname(viewer,&outputname); CHKERRQ(ierr);
    fprintf(fd,"%s = [\n",outputname);
    for (i=0; i<n; i++ ) {
#if defined(USE_PETSC_COMPLEX)
      if (PetscImaginary(x->array[i]) > 0.0) {
        fprintf(fd,"%18.16e + %18.16e i\n",PetscReal(x->array[i]),PetscImaginary(x->array[i]));
      } else if (PetscImaginary(x->array[i]) < 0.0) {
        fprintf(fd,"%18.16e - %18.16e i\n",PetscReal(x->array[i]),-PetscImaginary(x->array[i]));
      } else {
        fprintf(fd,"%18.16e\n",PetscReal(x->array[i]));
      }
#else
      fprintf(fd,"%18.16e\n",x->array[i]);
#endif
    }
    fprintf(fd,"];\n");
  } else if (format == VIEWER_FORMAT_ASCII_SYMMODU) {
    for (i=0; i<n; i++ ) {
#if defined(USE_PETSC_COMPLEX)
      fprintf(fd,"%18.16e %18.16e\n",PetscReal(x->array[i]),PetscImaginary(x->array[i]));
#else
      fprintf(fd,"%18.16e\n",x->array[i]);
#endif
    }
  } else {
    for (i=0; i<n; i++ ) {
      if (format == VIEWER_FORMAT_ASCII_INDEX) {
        fprintf(fd,"%d: ",i);
      }
#if defined(USE_PETSC_COMPLEX)
      if (PetscImaginary(x->array[i]) > 0.0) {
        fprintf(fd,"%g + %g i\n",PetscReal(x->array[i]),PetscImaginary(x->array[i]));
      } else if (PetscImaginary(x->array[i]) < 0.0) {
        fprintf(fd,"%g - %g i\n",PetscReal(x->array[i]),-PetscImaginary(x->array[i]));
      } else {
        fprintf(fd,"%g\n",PetscReal(x->array[i]));
      }
#else
      fprintf(fd,"%g\n",x->array[i]);
#endif
    }
  }
  fflush(fd);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecView_Seq_Draw_LG"
static int VecView_Seq_Draw_LG(Vec xin,Viewer v)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
  int      i, n = x->n,ierr;
  Draw     win;
  double   *xx;
  DrawLG   lg;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDrawLG(v,0,&lg); CHKERRQ(ierr);
  ierr = DrawLGGetDraw(lg,&win); CHKERRQ(ierr);
  ierr = DrawCheckResizedWindow(win);CHKERRQ(ierr);
  ierr = DrawLGReset(lg); CHKERRQ(ierr);
  xx = (double *) PetscMalloc( (n+1)*sizeof(double) ); CHKPTRQ(xx);
  for ( i=0; i<n; i++ ) {
    xx[i] = (double) i;
  }
#if !defined(USE_PETSC_COMPLEX)
  DrawLGAddPoints(lg,n,&xx,&x->array);
#else 
  {
    double *yy;
    yy = (double *) PetscMalloc( (n+1)*sizeof(double) ); CHKPTRQ(yy);    
    for ( i=0; i<n; i++ ) {
      yy[i] = PetscReal(x->array[i]);
    }
    DrawLGAddPoints(lg,n,&xx,&yy);
    PetscFree(yy);
  }
#endif
  PetscFree(xx);
  DrawLGDraw(lg);
  DrawSynchronizedFlush(win);
  DrawPause(win);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecView_Seq_Draw"
static int VecView_Seq_Draw(Vec xin,Viewer v)
{
  int        ierr;
  Draw       draw;
  PetscTruth isnull;
  int        format;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDraw(v,0,&draw); CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
  
  ierr = ViewerGetFormat(v,&format); CHKERRQ(ierr);
  /*
     Currently it only supports drawing to a line graph */
  if (format != VIEWER_FORMAT_DRAW_LG) {
    ViewerPushFormat(v,VIEWER_FORMAT_DRAW_LG,PETSC_NULL);
  } 
  ierr = VecView_Seq_Draw_LG(xin,v); CHKERRQ(ierr);
  if (format != VIEWER_FORMAT_DRAW_LG) {
    ViewerPopFormat(v);
  } 

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecView_Seq_Binary"
static int VecView_Seq_Binary(Vec xin,Viewer viewer)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
  int      ierr,fdes,n = x->n;

  PetscFunctionBegin;
  ierr  = ViewerBinaryGetDescriptor(viewer,&fdes); CHKERRQ(ierr);
  /* Write vector header */
  ierr = PetscBinaryWrite(fdes,&xin->cookie,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(fdes,&n,1,PETSC_INT,0); CHKERRQ(ierr);

  /* Write vector contents */
  ierr = PetscBinaryWrite(fdes,x->array,n,PETSC_SCALAR,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "VecView_Seq"
int VecView_Seq(Vec xin,Viewer viewer)
{
  Vec_Seq     *x = (Vec_Seq *)xin->data;
  ViewerType  vtype;
  int         ierr;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,DRAW_VIEWER)){ 
    ierr = VecView_Seq_Draw(xin,viewer);CHKERRQ(ierr);
  } else if (PetscTypeCompare(vtype,ASCII_VIEWER)){
    ierr = VecView_Seq_File(xin,viewer);CHKERRQ(ierr);
  } else if (PetscTypeCompare(vtype,SOCKET_VIEWER)) {
    ierr = ViewerSocketPutScalar_Private(viewer,x->n,1,x->array);CHKERRQ(ierr);
  } else if (PetscTypeCompare(vtype,BINARY_VIEWER)) {
    ierr = VecView_Seq_Binary(xin,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Viewer type not supported by PETSc object");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSetValues_Seq"
int VecSetValues_Seq(Vec xin, int ni, int *ix,Scalar* y,InsertMode m)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
  Scalar   *xx = x->array;
  int      i;

  PetscFunctionBegin;
  if (m == INSERT_VALUES) {
    for ( i=0; i<ni; i++ ) {
      if (ix[i] < 0) continue;
#if defined(USE_PETSC_BOPT_g)
      if (ix[i] >= x->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Out of range index value %d maximum %d",ix[i],x->n);
#endif
      xx[ix[i]] = y[i];
    }
  } else {
    for ( i=0; i<ni; i++ ) {
      if (ix[i] < 0) continue;
#if defined(USE_PETSC_BOPT_g)
      if (ix[i] >= x->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Out of range index value %d maximum %d",ix[i],x->n);
#endif
      xx[ix[i]] += y[i];
    }  
  }  
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSetValuesBlocked_Seq"
int VecSetValuesBlocked_Seq(Vec xin, int ni, int *ix,Scalar* y,InsertMode m)
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
  Scalar   *xx = x->array;
  int      i,bs = xin->bs,start,j;

  /*
       For optimization could treat bs = 2, 3, 4, 5 as special cases with loop unrolling
  */
  PetscFunctionBegin;
  if (m == INSERT_VALUES) {
    for ( i=0; i<ni; i++ ) {
      start = bs*ix[i];
      if (start < 0) continue;
#if defined(USE_PETSC_BOPT_g)
      if (start >= x->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Out of range index value %d maximum %d",start,x->n);
#endif
      for (j=0; j<bs; j++) {
        xx[start+j] = y[j];
      }
      y += bs;
    }
  } else {
    for ( i=0; i<ni; i++ ) {
      start = bs*ix[i];
      if (start < 0) continue;
#if defined(USE_PETSC_BOPT_g)
      if (start >= x->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Out of range index value %d maximum %d",start,x->n);
#endif
      for (j=0; j<bs; j++) {
        xx[start+j] += y[j];
      }
      y += bs;
    }  
  }  
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecDestroy_Seq"
int VecDestroy_Seq(Vec v)
{
  Vec_Seq *vs = (Vec_Seq*) v->data;

  PetscFunctionBegin;

#if defined(USE_PETSC_LOG)
  PLogObjectState((PetscObject)v,"Length=%d",((Vec_Seq *)v->data)->n);
#endif
  if (vs->array_allocated) PetscFree(vs->array_allocated);
  PetscFree(vs);

  /* if memory was published with AMS then destroy it */
#if defined(HAVE_AMS)
  if (v->amem >= 0) {
    int     ierr;
    ierr = AMS_Memory_destroy(v->amem);CHKERRQ(ierr);
  }
#endif

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecPublish_Seq"
static int VecPublish_Seq(PetscObject object)
{
#if defined(HAVE_AMS)
  Vec          v = (Vec) object;
  Vec_Seq      *s = (Vec_Seq *) v->data;
  int          ierr;
  int          (*f)(AMS_Memory,char *,Vec);
  
  PetscFunctionBegin;

  /* if it is already published then return */
  if (v->amem >=0 ) PetscFunctionReturn(0);

  ierr = PetscObjectPublishBaseBegin(object);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field((AMS_Memory)v->amem,"values",s->array,v->n,AMS_DOUBLE,AMS_READ,
                                AMS_DISTRIBUTED,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

  /* if the vector knows its "layout" let it set it*/
  ierr = PetscObjectQueryFunction((PetscObject)v,"AMSSetFieldBlock_C",(void**)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)((AMS_Memory)v->amem,"values",v);CHKERRQ(ierr);
  }
  ierr = PetscObjectPublishBaseEnd(object);CHKERRQ(ierr);

#else
  PetscFunctionBegin;
#endif

  PetscFunctionReturn(0);
}

static struct _VecOps DvOps = {VecDuplicate_Seq, 
            VecDuplicateVecs_Default,
            VecDestroyVecs_Default, 
            VecDot_Seq, 
            VecMDot_Seq,
            VecNorm_Seq,  
            VecTDot_Seq, 
            VecMTDot_Seq,
            VecScale_Seq, 
            VecCopy_Seq,
            VecSet_Seq, 
            VecSwap_Seq, 
            VecAXPY_Seq, 
            VecAXPBY_Seq,
            VecMAXPY_Seq, 
            VecAYPX_Seq,
            VecWAXPY_Seq, 
            VecPointwiseMult_Seq,
            VecPointwiseDivide_Seq,  
            VecSetValues_Seq,0,0,
            VecGetArray_Seq, 
            VecGetSize_Seq,
            VecGetSize_Seq,
            VecGetOwnershipRange_Seq,
            VecRestoreArray_Seq,
            VecMax_Seq,
            VecMin_Seq,
            VecSetRandom_Seq,0,
            VecSetValuesBlocked_Seq,
            VecDestroy_Seq,
            VecView_Seq,
            VecPlaceArray_Seq,
            VecGetMap_Seq};

/*
      This is called by VecCreate_Seq() (i.e. VecCreateSeq()) and VecCreateSeqWithArray()
*/
#undef __FUNC__  
#define __FUNC__ "VecCreate_Seq_Private"
static int VecCreate_Seq_Private(Vec v,Scalar *array)
{
  Vec_Seq *s;
  int     ierr;

  PetscFunctionBegin;
  PetscMemcpy(v->ops,&DvOps,sizeof(DvOps));
  s                  = (Vec_Seq *) PetscMalloc(sizeof(Vec_Seq)); CHKPTRQ(s);
  v->data            = (void *) s;
  v->bops->publish   = VecPublish_Seq;
  s->n               = PetscMax(v->n,v->N);
  v->n               = PetscMax(v->n,v->N);; 
  v->N               = PetscMax(v->n,v->N);; 
  s->array           = array;
  s->array_allocated = 0;
  if (!v->map) {
    ierr = MapCreateMPI(v->comm,v->n,v->N,&v->map); CHKERRQ(ierr);
  }
  v->type_name       = (char *) PetscMalloc((1+PetscStrlen(VEC_SEQ))*sizeof(char));CHKPTRQ(v->type_name);
  PetscStrcpy(v->type_name,VEC_SEQ);
  PetscPublishAll(v);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecCreateSeqWithArray"
/*@C
   VecCreateSeqWithArray - Creates a standard, sequential array-style vector,
   where the user provides the array space to store the vector values.

   Collective on MPI_Comm

   Input Parameter:
+  comm - the communicator, should be PETSC_COMM_SELF
.  n - the vector length 
-  array - memory where the vector elements are to be stored.

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is PETSC_NULL, then VecPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

.keywords: vector, sequential, create, BLAS

.seealso: VecCreateMPIWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), 
          VecCreateGhost(), VecCreateSeq(), VecPlaceArray()
@*/
int VecCreateSeqWithArray(MPI_Comm comm,int n,Scalar *array,Vec *V)
{
  int  ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,n,n,V);CHKERRQ(ierr);
  ierr = VecCreate_Seq_Private(*V,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "VecCreate_Seq"
int VecCreate_Seq(Vec V)
{
  Vec_Seq *s;
  Scalar  *array;
  int     ierr, n = PetscMax(V->n,V->N);

  PetscFunctionBegin;
  array              = (Scalar *) PetscMalloc((n+1)*sizeof(Scalar));CHKPTRQ(array);
  PetscMemzero(array,n*sizeof(Scalar));
  ierr               = VecCreate_Seq_Private(V,array);CHKERRQ(ierr);
  s                  = (Vec_Seq *) V->data;
  s->array_allocated = array;
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNC__  
#define __FUNC__ "VecGetMap_Seq"
int VecGetMap_Seq(Vec win,Map *m)
{
  PetscFunctionBegin;
  *m = win->map;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecDuplicate_Seq"
int VecDuplicate_Seq(Vec win,Vec *V)
{
  int     ierr;
  Vec_Seq *w = (Vec_Seq *)win->data;

  PetscFunctionBegin;
  ierr = VecCreateSeq(win->comm,w->n,V); CHKERRQ(ierr);
  if (win->mapping) {
    (*V)->mapping = win->mapping;
    PetscObjectReference((PetscObject)win->mapping);
  }
  if (win->bmapping) {
    (*V)->bmapping = win->bmapping;
    PetscObjectReference((PetscObject)win->bmapping);
  }
  (*V)->bs = win->bs;
  ierr = OListDuplicate(win->olist,&(*V)->olist);CHKERRQ(ierr);
  ierr = FListDuplicate(win->qlist,&(*V)->qlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

