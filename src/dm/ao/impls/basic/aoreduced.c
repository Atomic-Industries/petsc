
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aoreduced.c,v 1.6 1997/11/28 16:22:36 bsmith Exp bsmith $";
#endif

#include "src/ao/aoimpl.h"
#include "pinclude/pviewer.h"
#include "sys.h"
#include "src/inline/bitarray.h"

int AODataSegmentGetReduced_Basic(AOData ao,char *name,char *segname,int n,int *keys,IS *is)
{
  AODataSegment    *segment; 
  AODataKey        *key;
  int              ierr,dsize,i,bs,flag,*found,count,imin,imax,*out;
  char             *idata, *odata;
  BT               mask;

  PetscFunctionBegin;
  /* find the correct segment */
  ierr = AODataSegmentFind_Private(ao,name,segname,&flag,&key,&segment);CHKERRQ(ierr);
  if (flag != 1) SETERRQ(PETSC_ERR_ARG_WRONG,1,"Cannot locate segment");

  if (segment->datatype != PETSC_INT) SETERRQ(PETSC_ERR_ARG_WRONG,1,"Only for PETSC_INT data");

  /*
     Copy the found values into a contiguous location, keeping them in the 
     order of the requested keys
  */
  ierr  = PetscDataTypeGetSize(segment->datatype,&dsize); CHKERRQ(ierr);
  bs    = segment->bs;
  odata = (char *) PetscMalloc((n+1)*bs*dsize);CHKPTRQ(odata);
  idata = (char *) segment->data;
  for ( i=0; i<n; i++ ) {
    PetscMemcpy(odata + i*bs*dsize,idata + keys[i]*bs*dsize,bs*dsize);
  }

  found = (int *) odata;
  n     = n*bs;

  /*  Determine the max and min values */
  if (n) {
    imin = PETSC_MAX_INT;
    imax = 0;  
    for ( i=0; i<n; i++ ) {
      if (found[i] < 0) continue;
      imin = PetscMin(imin,found[i]);
      imax = PetscMax(imax,found[i]);
    }
  } else {
    imin = imax = 0;
  }
  ierr = BTCreate(imax-imin,mask); CHKERRQ(ierr);
  /* Put the values into the mask and count them */
  count = 0;
  for ( i=0; i<n; i++ ) {
    if (found[i] < 0) continue;
    if (!BTLookupSet(mask,found[i] - imin)) count++;
  }
  BTMemzero(imax-imin,mask);
  out = (int *) PetscMalloc((count+1)*sizeof(int));CHKPTRQ(out);
  count = 0;
  for ( i=0; i<n; i++ ) {
    if (found[i] < 0) continue;
    if (!BTLookupSet(mask,found[i] - imin)) {out[count++] = found[i];}
  }
  BTDestroy(mask);
  PetscFree(found);

  ierr = ISCreateGeneral(ao->comm,count,out,is);CHKERRQ(ierr);
  PetscFree(out);
  PetscFunctionReturn(0);
}






