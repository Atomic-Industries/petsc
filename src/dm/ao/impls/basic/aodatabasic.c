


#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aodatabasic.c,v 1.23 1998/04/03 23:18:51 bsmith Exp bsmith $";
#endif

/*
    The most basic AOData routines. These store the 
  entire database on each processor. These are very simple, not that
  we do not even use a private data structure for AOData and the 
  private datastructure for AODataSegment is just used as a simple array.

    These are made slightly complicated by having to be able to handle
  logical variables stored in bit arrays. Thus

  *  before mallocing to hold a bit array, we shrunk the array length by a factor
     of 8 using BTLength()

  *  we use PetscBitMemcpy() to allow us to copy at the individual bit level.
     (for regular datatypes this just does a regular memcpy().
*/

#include "src/ao/aoimpl.h"
#include "pinclude/pviewer.h"
#include "sys.h"

#undef __FUNC__  
#define __FUNC__ "AODataDestroy_Basic"
int AODataDestroy_Basic(AOData ao)
{
  int           ierr;
  AODataKey     *key = ao->keys,*nextkey;
  AODataSegment *seg,*nextseg;

  PetscFunctionBegin;
  while (key) {
    PetscFree(key->name);
    if (key->ltog) {
      ierr = ISLocalToGlobalMappingDestroy(key->ltog);CHKERRQ(ierr);
    }
    seg = key->segments;
    while (seg) {
      PetscFree(seg->data);
      PetscFree(seg->name);
      nextseg = seg->next;
      PetscFree(seg);
      seg     = nextseg;
    }
    PetscFree(key->rowners);
    nextkey = key->next;
    PetscFree(key);
    key     = nextkey;
  }
  
  PLogObjectDestroy(ao);
  PetscHeaderDestroy(ao);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataView_Basic_Binary"
int AODataView_Basic_Binary(AOData ao,Viewer viewer)
{
  int             ierr,N, fd;
  AODataSegment   *segment;
  AODataKey       *key = ao->keys;
  char            paddedname[256];

  PetscFunctionBegin;

  ierr  = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);

  /* write out number of keys */
  ierr = PetscBinaryWrite(fd,&ao->nkeys,1,PETSC_INT,0);CHKERRQ(ierr);

  while (key) {
    N   = key->N;
    /* 
       Write out name of key - use a fixed length for the name in the binary 
       file to make seeking easier
    */
    PetscMemzero(paddedname,256*sizeof(char));
    PetscStrncpy(paddedname,key->name,255);
    ierr = PetscBinaryWrite(fd,paddedname,256,PETSC_CHAR,0);CHKERRQ(ierr);
    /* write out the number of indices */
    ierr = PetscBinaryWrite(fd,&key->N,1,PETSC_INT,0);CHKERRQ(ierr);
    /* write out number of segments */
    ierr = PetscBinaryWrite(fd,&key->nsegments,1,PETSC_INT,0);CHKERRQ(ierr);
   
    /* loop over segments writing them out */
    segment = key->segments;
    while (segment) {
      /* 
         Write out name of segment - use a fixed length for the name in the binary 
         file to make seeking easier
      */
      PetscMemzero(paddedname,256*sizeof(char));
      PetscStrncpy(paddedname,segment->name,255);
      ierr = PetscBinaryWrite(fd,paddedname,256,PETSC_CHAR,0);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,&segment->bs,1,PETSC_INT,0);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,&segment->datatype,1,PETSC_INT,0);CHKERRQ(ierr);
      /* write out the data */
      ierr = PetscBinaryWrite(fd,segment->data,N*segment->bs,segment->datatype,0);CHKERRQ(ierr);
      segment = segment->next;
    }
    key = key->next;
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataView_Basic_ASCII"
int AODataView_Basic_ASCII(AOData ao,Viewer viewer)
{
  int             ierr,format,j,k,l,rank,size,nkeys,nsegs,i,N,bs,zero = 0;
  FILE            *fd;
  char            *dt,**keynames,**segnames,*stype,*segvalue;
  AODataSegment   *segment;
  AODataKey       *key = ao->keys;
  PetscDataType   dtype;

  PetscFunctionBegin;
  MPI_Comm_rank(ao->comm,&rank); if (rank) PetscFunctionReturn(0);  
  MPI_Comm_size(ao->comm,&size);

  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);
  if (format == VIEWER_FORMAT_ASCII_INFO) {
    ierr = AODataGetInfo(ao,&nkeys,&keynames);CHKERRQ(ierr);
    for ( i=0; i<nkeys; i++) {
      ierr = AODataKeyGetInfo(ao,keynames[i],&N,0,&nsegs,&segnames);CHKERRQ(ierr);
      printf("  %s: (%d)\n",keynames[i],N);
      for ( j=0; j<nsegs; j++ ) {
        ierr = AODataSegmentGetInfo(ao,keynames[i],segnames[j],&bs,&dtype);CHKERRQ(ierr);
        ierr = PetscDataTypeGetName(dtype,&stype);CHKERRQ(ierr);
        if (dtype == PETSC_CHAR) {
          ierr = AODataSegmentGet(ao,keynames[i],segnames[j],1,&zero,(void **)&segvalue);CHKERRQ(ierr);
          printf("      %s: (%d) %s -> %s\n",segnames[j],bs,stype,segvalue);
          ierr = AODataSegmentRestore(ao,keynames[i],segnames[j],1,&zero,(void **)&segvalue);CHKERRQ(ierr);
        } else {
          printf("      %s: (%d) %s\n",segnames[j],bs,stype);
        }
      }
    }
    PetscFree(keynames);
    PetscFunctionReturn(0);
  }

  while (key) {
    fprintf(fd,"AOData Key: %s Length %d Ownership: ",key->name,key->N);
    for (j=0; j<size+1; j++) {fprintf(fd,"%d ",key->rowners[j]);}
    fprintf(fd,"\n");

    segment = key->segments;
    while (segment) {      
      ierr = PetscDataTypeGetName(segment->datatype,&dt); CHKERRQ(ierr);    
      fprintf(fd,"  AOData Segment: %s Blocksize %d datatype %s\n",segment->name,segment->bs,dt);
      if (segment->datatype == PETSC_INT) {
        int *mdata = (int *) segment->data;
        for ( k=0; k<key->N; k++ ) {
          fprintf(fd," %d: ",k);
          for ( l=0; l<segment->bs; l++ ) {
            fprintf(fd,"   %d ",mdata[k*segment->bs + l]);
          }
          fprintf(fd,"\n");
        }
      } else if (segment->datatype == PETSC_DOUBLE) {
        double *mdata = (double *) segment->data;
        for ( k=0; k<key->N; k++ ) {
          fprintf(fd," %d: ",k);
          for ( l=0; l<segment->bs; l++ ) {
            fprintf(fd,"   %18.16e ",mdata[k*segment->bs + l]);
          }
          fprintf(fd,"\n");
        }
      } else if (segment->datatype == PETSC_SCALAR) {
        Scalar *mdata = (Scalar *) segment->data;
        for ( k=0; k<key->N; k++ ) {
          fprintf(fd," %d: ",k);
          for ( l=0; l<segment->bs; l++ ) {
#if !defined(USE_PETSC_COMPLEX)
            fprintf(fd,"   %18.16e ",mdata[k*segment->bs + l]);
#else
            Scalar x = mdata[k*segment->bs + l];
            if (imag(x) > 0.0) {
              fprintf(fd," %18.16e + %18.16e i \n",real(x),imag(x));
            } else if (imag(x) < 0.0) {
              fprintf(fd,"   %18.16e - %18.16e i \n",real(x),-imag(x));
            } else {
              fprintf(fd,"   %18.16e \n",real(x));
            }
#endif
          }
        }
        fprintf(fd,"\n");
      } else if (segment->datatype == PETSC_LOGICAL) {
        BT mdata = (BT) segment->data;
        for ( k=0; k<key->N; k++ ) {
          fprintf(fd," %d: ",k);
          for ( l=0; l<segment->bs; l++ ) {
            fprintf(fd,"   %d ",(int) BTLookup(mdata,k*segment->bs + l));
          }
          fprintf(fd,"\n");
        }
      } else if (segment->datatype == PETSC_CHAR) {
        char * mdata = (char *) segment->data;
        for ( k=0; k<key->N; k++ ) {
          fprintf(fd,"  %s ",mdata + k*segment->bs);
        }
        fprintf(fd,"\n");
      } else {
        SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Unknown PETSc data format");
      }
      segment = segment->next;
    }
    key = key->next;
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "AODataView_Basic"
int AODataView_Basic(AOData ao,Viewer viewer)
{
  int             rank,ierr;
  ViewerType      vtype;

  PetscFunctionBegin;
  MPI_Comm_rank(ao->comm,&rank); if (rank) PetscFunctionReturn(0);

  if (!viewer) {
    viewer = VIEWER_STDOUT_SELF; 
  }

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) { 
    ierr = AODataView_Basic_ASCII(ao,viewer); CHKERRQ(ierr);
  } else if (vtype == BINARY_FILE_VIEWER) {
    ierr = AODataView_Basic_Binary(ao,viewer); CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyRemove_Basic"
int AODataKeyRemove_Basic(AOData aodata,char *name)
{
  AODataSegment    *segment,*iseg;
  AODataKey        *key,*ikey;
  int              ierr,flag;

  PetscFunctionBegin;
  ierr = AODataKeyFind_Private(aodata,name,&flag,&key);CHKERRQ(ierr);
  if (flag != 1)  PetscFunctionReturn(0);
  aodata->nkeys--;

  segment = key->segments;
  while (segment) {
    iseg    = segment->next;
    PetscFree(segment->name); 
    PetscFree(segment->data);
    PetscFree(segment);
    segment = iseg;
  }
  ikey = aodata->keys;
  if (key == ikey) {
    aodata->keys = key->next;
    goto finishup;
  }
  while (ikey->next != key) {
    ikey = ikey->next;
  }
  ikey->next = key->next;

  finishup:

  PetscFree(key->name);
  PetscFree(key->rowners);
  PetscFree(key);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentRemove_Basic"
int AODataSegmentRemove_Basic(AOData aodata,char *name,char *segname)
{
  AODataSegment    *segment,*iseg;
  AODataKey        *key;
  int              ierr,flag;

  PetscFunctionBegin;
  ierr = AODataSegmentFind_Private(aodata,name,segname,&flag,&key,&iseg);CHKERRQ(ierr);
  if (flag != 1)  PetscFunctionReturn(0);
  key->nsegments--;

  segment = key->segments;
  if (segment == iseg) {
    key->segments = segment->next;
    goto finishup;
  }
  while (segment->next != iseg) {
    segment = segment->next;
  }
  segment->next = iseg->next;
  segment       = iseg;
  
  finishup:

  PetscFree(segment->name); 
  PetscFree(segment->data);
  PetscFree(segment);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "AODataSegmentAdd_Basic"
int AODataSegmentAdd_Basic(AOData aodata,char *name,char *segname,int bs,int n,int *keys,void *data,
                           PetscDataType dtype)
{
  AODataSegment    *segment,*iseg;
  AODataKey        *key;
  int              N,size,rank,ierr,*lens,i,*disp,*akeys,datasize,*fkeys,len,flag,Nb,j;
  MPI_Datatype     mtype;
  char             *adata;
  MPI_Comm         comm = aodata->comm;

  PetscFunctionBegin;
  ierr = AODataSegmentFind_Private(aodata,name,segname,&flag,&key,&iseg);CHKERRQ(ierr);
  if (flag == -1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"No key created");
  if (flag == 1)  SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Segment already exists");

  segment = PetscNew(AODataSegment);CHKPTRQ(segment);
  if (iseg) {
    iseg->next    = segment;
  } else {
    key->segments = segment;
  }
  segment->next     = 0;
  segment->bs       = bs;
  segment->datatype = dtype;

  ierr = PetscDataTypeGetSize(dtype,&datasize); CHKERRQ(ierr);

  /*
     If keys not given, assume each processor provides entire data 
  */
  if (!keys && n == key->N) {
    char *fdata;
    if (dtype == PETSC_LOGICAL) Nb = BTLength(key->N); else Nb = key->N;
    fdata = (char *) PetscMalloc((Nb*bs+1)*datasize); CHKPTRQ(fdata);
    PetscBitMemcpy(fdata,0,data,0,key->N*bs,dtype);
    segment->data = (void *) fdata;
  } else if (!keys) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Keys not given, but not all data given on each processor");
  } else {
    /* transmit all lengths to all processors */
    MPI_Comm_size(comm,&size);
    MPI_Comm_rank(comm,&rank);
    lens = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(lens);
    disp = lens + size;
    ierr = MPI_Allgather(&n,1,MPI_INT,lens,1,MPI_INT,comm);CHKERRQ(ierr);
    N =  0;
    for ( i=0; i<size; i++ ) {
      disp[i]  = N;
      N       += lens[i];
    }
    if (N != key->N) {
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Did not provide correct number of keys for keyname");
    }

    /*
      Allocate space for all keys and all data 
    */
    akeys = (int *) PetscMalloc((N+1)*sizeof(int)); CHKPTRQ(akeys);
    adata = (char *) PetscMalloc((N*bs+1)*datasize); CHKPTRQ(adata);

    ierr = MPI_Allgatherv(keys,n,MPI_INT,akeys,lens,disp,MPI_INT,comm);CHKERRQ(ierr);
    for ( i=0; i<size; i++ ) {
      disp[i] *= bs;
      lens[i] *= bs;
    }
   
    if (dtype != PETSC_LOGICAL) {
      char *fdata;

      ierr = PetscDataTypeToMPIDataType(dtype,&mtype);CHKERRQ(ierr);
      ierr = MPI_Allgatherv(data,n*bs,mtype,adata,lens,disp,mtype,comm);CHKERRQ(ierr);
      PetscFree(lens);

      /*
        Now we have all the keys and data we need to put it in order
      */
      fkeys = (int *) PetscMalloc((key->N+1)*sizeof(int)); CHKPTRQ(fkeys);
      PetscMemzero(fkeys, (key->N+1)*sizeof(int));
      fdata = (char *) PetscMalloc((key->N*bs+1)*datasize); CHKPTRQ(fdata);

      for ( i=0; i<N; i++ ) {
        if (fkeys[akeys[i]] != 0) {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Duplicate key");
        }
        if (fkeys[akeys[i]] >= N) {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Key out of range");
        }
        fkeys[akeys[i]] = 1;
        PetscBitMemcpy(fdata,akeys[i]*bs,adata,i*bs,bs,dtype);
      }
      for ( i=0; i<N; i++ ) {
        if (!fkeys[i]) {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Missing key");
        }
      }
      segment->data = (void *) fdata;
    } else {
      /*
            For logical input the length is given by the user in bits; we need to 
            convert to bytes to send with MPI
      */
      BT fdata,mvalues = (BT) data;
      char *values = (char *) PetscMalloc((n+1)*bs*sizeof(char));CHKPTRQ(values);
      for ( i=0; i<n; i++ ) {
        for ( j=0; j<bs; j++ ) {
          if (BTLookup(mvalues,i*bs+j)) values[i*bs+j] = 1; else values[i*bs+j] = 0;
        }
      }

      ierr = MPI_Allgatherv(values,n*bs,MPI_BYTE,adata,lens,disp,MPI_BYTE,comm);CHKERRQ(ierr);
      PetscFree(lens); PetscFree(values);

      /*
        Now we have all the keys and data we need to put it in order
      */
      fkeys = (int *) PetscMalloc((key->N+1)*sizeof(int)); CHKPTRQ(fkeys);
      PetscMemzero(fkeys, (key->N+1)*sizeof(int));
      BTCreate(N*bs,fdata);

      for ( i=0; i<N; i++ ) {
        if (fkeys[akeys[i]] != 0) {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Duplicate key");
        }
        if (fkeys[akeys[i]] >= N) {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Key out of range");
        }
        fkeys[akeys[i]] = 1;
        for ( j=0; j<bs; j++ ) {
          if (adata[i*bs+j]) BTSet(fdata,i*bs+j); 
        }
      }
      for ( i=0; i<N; i++ ) {
        if (!fkeys[i]) {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Missing key");
        }
      }
      segment->data = (void *) fdata;
    }
    PetscFree(akeys);
    PetscFree(adata);
    PetscFree(fkeys);
  }

  key->nsegments++;

  len           = PetscStrlen(segname);
  segment->name = (char *) PetscMalloc((len+1)*sizeof(char));CHKPTRQ(segment->name);
  PetscStrcpy(segment->name,segname);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentExtrema_Basic"
int AODataSegmentGetExtrema_Basic(AOData ao,char *name,char *segname,void *xmax,void *xmin)
{
  AODataSegment    *segment; 
  AODataKey        *key;
  int              ierr,i,bs,flag,n,j;
  
  PetscFunctionBegin;
  /* find the correct segment */
  ierr = AODataSegmentFind_Private(ao,name,segname,&flag,&key,&segment);CHKERRQ(ierr);
  if (flag != 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Cannot locate segment");

  n       = key->N;
  bs      = segment->bs;

  if (segment->datatype == PETSC_INT) {
    int *vmax = (int *) xmax, *vmin = (int *) xmin, *values = (int *) segment->data;
    for ( j=0; j<bs; j++ ) {
      vmax[j] = vmin[j] = values[j];
    }
    for ( i=1; i<n; i++ ) {
      for ( j=0; j<bs; j++ ) {
        vmax[j] = PetscMax(vmax[j],values[bs*i+j]);
        vmin[j] = PetscMin(vmin[j],values[bs*i+j]);
      }
    }
  } else if (segment->datatype == PETSC_DOUBLE) {
    double *vmax = (double *) xmax, *vmin = (double *) xmin, *values = (double *) segment->data;
    for ( j=0; j<bs; j++ ) {
      vmax[j] = vmin[j] = values[j];
    }
    for ( i=1; i<n; i++ ) {
      for ( j=0; j<bs; j++ ) {
        vmax[j] = PetscMax(vmax[j],values[bs*i+j]);
        vmin[j] = PetscMin(vmin[j],values[bs*i+j]);
      }
    }
  } else SETERRQ(PETSC_ERR_SUP,1,"Cannot find extrema for this data type");

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentGet_Basic"
int AODataSegmentGet_Basic(AOData ao,char *name,char *segname,int n,int *keys,void **data)
{
  AODataSegment    *segment; 
  AODataKey        *key;
  int              ierr,dsize,i,bs,flag,nb;
  char             *idata, *odata;
  
  PetscFunctionBegin;
  /* find the correct segment */
  ierr = AODataSegmentFind_Private(ao,name,segname,&flag,&key,&segment);CHKERRQ(ierr);
  if (flag != 1) SETERRQ(PETSC_ERR_ARG_WRONG,1,"Cannot locate segment");

  ierr  = PetscDataTypeGetSize(segment->datatype,&dsize); CHKERRQ(ierr);
  bs    = segment->bs;
  if (segment->datatype == PETSC_LOGICAL) nb = BTLength(n); else nb = n;
  odata = (char *) PetscMalloc((nb+1)*bs*dsize);CHKPTRQ(odata);
  idata = (char *) segment->data;
  for ( i=0; i<n; i++ ) {
    PetscBitMemcpy(odata,i*bs,idata,keys[i]*bs,bs,segment->datatype);
  }
  *data = (void *) odata;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentRestore_Basic"
int AODataSegmentRestore_Basic(AOData aodata,char *name,char *segname,int n,int *keys,void **data)
{
  PetscFunctionBegin;
  PetscFree(*data);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentGetLocal_Basic"
int AODataSegmentGetLocal_Basic(AOData ao,char *name,char *segname,int n,int *keys,void **data)
{
  int           ierr,flag,*globals,*locals,bs;
  PetscDataType dtype;
  AODataKey     *key;

  PetscFunctionBegin;
  ierr = AODataKeyFind_Private(ao,segname,&flag,&key);CHKERRQ(ierr);
  if (flag != 1) SETERRQ(PETSC_ERR_ARG_WRONG,1,"Segment does not have corresponding key");
  if (!key->ltog) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,1,"No local to global mapping set for key");
  ierr = AODataSegmentGetInfo(ao,name,segname,&bs,&dtype); CHKERRQ(ierr);
  if (dtype != PETSC_INT) SETERRQ(PETSC_ERR_ARG_WRONG,1,"Datatype of segment must be PETSC_INT");

  /* get the values in global indexing */
  ierr = AODataSegmentGet_Basic(ao,name,segname,n,keys,(void **)&globals);CHKERRQ(ierr);
  
  /* allocate space to store them in local indexing */
  locals = (int *) PetscMalloc((n+1)*bs*sizeof(int));CHKPTRQ(locals);

  ierr = ISGlobalToLocalMappingApply(key->ltog,IS_GTOLM_MASK,n*bs,globals,PETSC_NULL,locals);CHKERRQ(ierr);

  ierr = AODataSegmentRestore_Basic(ao,name,segname,n,keys,(void **)&globals);CHKERRQ(ierr);

  *data = (void *) locals;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentRestoreLocal_Basic"
int AODataSegmentRestoreLocal_Basic(AOData aodata,char *name,char *segname,int n,int *keys,void **data)
{
  PetscFunctionBegin;
  PetscFree(*data);
  PetscFunctionReturn(0);
}

extern int AOBasicGetIndices_Private(AO,int **,int **);

#undef __FUNC__  
#define __FUNC__ "AODataKeyRemap_Basic"
int AODataKeyRemap_Basic(AOData aodata, char *keyname,AO ao)
{
  int           ierr,*inew,k,*ii,nk,flag,dsize,bs,nkb;
  char          *data,*tmpdata;
  AODataKey     *key;
  AODataSegment *seg;

  PetscFunctionBegin;

  /* remap all the values in the segments that match the key */
  key = aodata->keys;
  while (key) {
    seg = key->segments;
    while (seg) {
      if (PetscStrcmp(seg->name,keyname)) {
        seg = seg->next;
        continue;
      }
      if (seg->datatype != PETSC_INT) {
        SETERRQ( PETSC_ERR_ARG_WRONG,1,"Segment name same as key but not integer type");
      }
      nk   = seg->bs*key->N;
      ii   = (int *) seg->data;
      ierr = AOPetscToApplication(ao,nk,ii); CHKERRQ(ierr);
      seg  = seg->next;
    }
    key = key->next;
  }
  
  ierr = AOBasicGetIndices_Private(ao,&inew,PETSC_NULL);CHKERRQ(ierr);
  /* reorder in the arrays all the values for the key */
  ierr = AODataKeyFind_Private(aodata,keyname,&flag,&key);CHKERRQ(ierr);
  if (flag != 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Could not find key");
  nk  = key->N;
  seg = key->segments;
  while (seg) {
    ierr    = PetscDataTypeGetSize(seg->datatype,&dsize);CHKERRQ(ierr);
    bs      = seg->bs;
    data    = (char *) seg->data;
    if (seg->datatype == PETSC_LOGICAL) nkb = BTLength(nk*bs); else nkb = nk*bs;
    tmpdata = (char *) PetscMalloc((nkb+1)*dsize);CHKPTRQ(tmpdata);

    for ( k=0; k<nk; k++ ) {
      PetscBitMemcpy(tmpdata,inew[k]*bs,data,k*bs,bs,seg->datatype);
    }
    PetscMemcpy(data,tmpdata,nkb*dsize);
    PetscFree(tmpdata);
    seg = seg->next;
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetAdjacency_Basic"
int AODataKeyGetAdjacency_Basic(AOData aodata, char *keyname,Mat *adj)
{
  int           ierr,cnt,i,j,*jj,*ii,nlocal,n,flag,*nb,bs,ls;
  AODataKey     *key;
  AODataSegment *seg;

  PetscFunctionBegin;
  ierr = AODataSegmentFind_Private(aodata,keyname,keyname,&flag,&key,&seg);CHKERRQ(ierr);
  if (flag != 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Cannot locate key with neighbor segment");

  /*
     Get the beginning of the neighbor list for this processor 
  */
  bs     = seg->bs;
  nb     = (int *) seg->data;
  nb    += bs*key->rstart;
  nlocal = key->rend - key->rstart;
  n      = bs*key->N;

  /*
      Assemble the adjacency graph: first we determine total number of entries
  */
  cnt = 0;
  for ( i=0; i<bs*nlocal; i++ ) {
    if (nb[i] >= 0) cnt++;
  }
  ii    = (int *) PetscMalloc((nlocal + 1)*sizeof(int)); CHKPTRQ(ii);
  jj    = (int *) PetscMalloc((cnt+1)*sizeof(int));CHKPTRQ(jj);
  ii[0] = 0;
  cnt   = 0;
  for ( i=0; i<nlocal; i++ ) {
    ls = 0;
    for ( j=0; j<bs; j++ ) {
      if (nb[bs*i+j] >= 0) {
        jj[cnt++] = nb[bs*i+j];
        ls++;
      }
    }
    /* now sort the column indices for this row */
    PetscSortInt(ls,jj+cnt-ls);
    ii[i+1] = cnt;
  }

  ierr = MatCreateMPIAdj(aodata->comm,nlocal,n,ii,jj,adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "AODataSegmentPartition_Basic"
int AODataSegmentPartition_Basic(AOData aodata,char *keyname,char *segname)
{
  int             ierr,flag,size,bs,i,j,*idx,nc,*isc;
  AO              ao;
  AODataKey       *key,*keyseg;
  AODataSegment   *segment;

  PetscFunctionBegin;

  ierr = AODataKeyFind_Private(aodata,segname,&flag,&keyseg);CHKERRQ(ierr);
  if (flag != 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Cannot locate segment as a key");
  isc     = (int *) PetscMalloc(keyseg->N*sizeof(int));CHKPTRQ(isc);
  PetscMemzero(isc,keyseg->N*sizeof(int));

  ierr = AODataSegmentFind_Private(aodata,keyname,segname,&flag,&key,&segment);CHKERRQ(ierr);
  if (flag != 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Cannot locate segment");
  MPI_Comm_size(aodata->comm,&size);

  bs                = segment->bs;

  idx = (int *) segment->data;
  nc  = 0;
  for ( i=0; i<size; i++ ) {
    for ( j=bs*key->rowners[i]; j<bs*key->rowners[i+1]; j++ ) {
      if (!isc[idx[j]]) {
        isc[idx[j]] = ++nc;
      }
    }
  }
  for ( i=0; i<keyseg->N; i++ ) {
    isc[i]--;
  }

  ierr = AOCreateBasic(aodata->comm,keyseg->nlocal,isc+keyseg->rstart,PETSC_NULL,&ao);CHKERRQ(ierr);
  PetscFree(isc);

  ierr = AODataKeyRemap(aodata,segname,ao);CHKERRQ(ierr);
  ierr = AODestroy(ao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetActive_Basic" 
int AODataKeyGetActive_Basic(AOData aodata,char *name,char *segname,int n,int *keys,int wl,IS *is)
{
  int           ierr,i,cnt,*fnd,flag,bs;
  AODataKey     *key;
  AODataSegment *segment;
  BT            bt;

  PetscFunctionBegin;
  ierr = AODataSegmentFind_Private(aodata,name, segname, &flag,&key,&segment);CHKERRQ(ierr);
  if (flag != 1) SETERRQ(1,1,"Cannot locate segment");

  bt = (BT) segment->data;
  bs = segment->bs;

  if (wl >= bs) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Bit field (wl) argument too large");

  /* count the active ones */
  cnt = 0;
  for ( i=0; i<n; i++ ) {
    if (BTLookup(bt,keys[i]*bs+wl)) {
      cnt++;
    }
  }

  fnd = (int *) PetscMalloc((cnt+1)*sizeof(int));CHKPTRQ(fnd);
  cnt = 0;
  for ( i=0; i<n; i++ ) {
    if (BTLookup(bt,keys[i]*bs+wl)) {
      fnd[cnt++] = keys[i];
    }
  }
  
  ierr = ISCreateGeneral(aodata->comm,cnt,fnd,is);CHKERRQ(ierr);
  PetscFree(fnd);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetActiveLocal_Basic" 
int AODataKeyGetActiveLocal_Basic(AOData aodata,char *name,char *segname,int n,int *keys,int wl,IS *is)
{
  int           ierr,i,cnt,*fnd,flag,bs,*locals;
  AODataKey     *key;
  AODataSegment *segment;
  BT            bt;

  PetscFunctionBegin;
  ierr = AODataSegmentFind_Private(aodata,name, segname, &flag,&key,&segment);CHKERRQ(ierr);
  if (flag != 1) SETERRQ(1,1,"Cannot locate segment");

  bt = (BT) segment->data;
  bs = segment->bs;

  if (wl >= bs) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Bit field (wl) argument too large");

  /* count the active ones */
  cnt = 0;
  for ( i=0; i<n; i++ ) {
    if (BTLookup(bt,keys[i]*bs+wl)) {
      cnt++;
    }
  }

  fnd = (int *) PetscMalloc((cnt+1)*sizeof(int));CHKPTRQ(fnd);
  cnt = 0;
  for ( i=0; i<n; i++ ) {
    if (BTLookup(bt,keys[i]*bs+wl)) {
      fnd[cnt++] = keys[i];
    }
  }
  
  locals = (int *) PetscMalloc( (n+1)*sizeof(int));CHKPTRQ(locals);
  ierr = ISGlobalToLocalMappingApply(key->ltog,IS_GTOLM_MASK,cnt,fnd,PETSC_NULL,locals);CHKERRQ(ierr);  
  PetscFree(fnd);
  ierr = ISCreateGeneral(aodata->comm,cnt,locals,is);CHKERRQ(ierr);
  PetscFree(locals);
  PetscFunctionReturn(0);
}

extern int AODataSegmentGetReduced_Basic(AOData,char *,char *,int,int*,IS *);

static struct _AODataOps myops = {AODataSegmentAdd_Basic,
                                  AODataSegmentGet_Basic,
                                  AODataSegmentRestore_Basic,
                                  AODataSegmentGetLocal_Basic,
                                  AODataSegmentRestoreLocal_Basic,
                                  AODataSegmentGetReduced_Basic,
                                  AODataSegmentGetExtrema_Basic,
                                  AODataKeyRemap_Basic,
                                  AODataKeyGetAdjacency_Basic,
                                  AODataKeyGetActive_Basic,
                                  AODataKeyGetActiveLocal_Basic,
                                  AODataSegmentPartition_Basic,
                                  AODataKeyRemove_Basic,
                                  AODataSegmentRemove_Basic};

#undef __FUNC__  
#define __FUNC__ "AODataCreateBasic" 
/*@C
   AODataCreateBasic - Creates a AO datastructure.

   Input Parameters:
.  comm  - MPI communicator that is to share AO
.  n - total number of keys that will be added

   Output Parameter:
.  aoout - the new database

   Options Database Key:
$   -ao_data_view : call AODataView() at the conclusion of AODataAdd()
$   -ao_data_view_info : call AODataView() at the conclusion of AODataAdd()

.keywords: AOData, create

.seealso: AODataAdd(), AODataDestroy()
@*/
int AODataCreateBasic(MPI_Comm comm,AOData *aoout)
{
  AOData        ao;

  PetscFunctionBegin;
  *aoout = 0;
  PetscHeaderCreate(ao, _p_AOData,struct _AODataOps,AODATA_COOKIE,AODATA_BASIC,comm,AODataDestroy,AODataView); 
  PLogObjectCreate(ao);
  PLogObjectMemory(ao,sizeof(struct _p_AOData));

  PetscMemcpy(ao->ops,&myops,sizeof(myops));
  ao->ops->destroy  = AODataDestroy_Basic;
  ao->ops->view     = AODataView_Basic;

  ao->nkeys        = 0;
  ao->keys         = 0;
  ao->datacomplete = 0;

  *aoout = ao; PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataLoadBasic" 
/*@C
   AODataLoadBasic - Loads a AO database from a file.

   Input Parameters:
.  viewer - the binary file containing the data

   Output Parameter:
.  aoout - the new database

   Options Database Key:
$   -ao_data_view : call AODataView() at the conclusion of AODataLoadBasic()
$   -ao_data_view_info : call AODataView() at the conclusion of AODataLoadBasic()

.keywords: AOData, create, load

.seealso: AODataAdd(), AODataDestroy(), AODataCreateBasic()
@*/
int AODataLoadBasic(Viewer viewer,AOData *aoout)
{
  AOData        ao;
  int           fd,nkeys,i,len,flg1,ierr,dsize,j,size,rank,Nb;
  char          paddedname[256];
  AODataSegment *seg = 0;
  AODataKey     *key = 0;
  MPI_Comm      comm;
  ViewerType    vtype;

  PetscFunctionBegin;
  *aoout = 0;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  != BINARY_FILE_VIEWER) {
    SETERRQ(PETSC_ERR_ARG_WRONG,1,"Viewer must be obtained from ViewerFileOpenBinary()");
  }

  ierr = PetscObjectGetComm((PetscObject)viewer,&comm); CHKERRQ(ierr);
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);

  /* read in number of segments */
  ierr = PetscBinaryRead(fd,&nkeys,1,PETSC_INT);CHKERRQ(ierr);

  PetscHeaderCreate(ao, _p_AOData,struct _AODataOps,AODATA_COOKIE,AODATA_BASIC,comm,AODataDestroy,AODataView); 
  PLogObjectCreate(ao);
  PLogObjectMemory(ao,sizeof(struct _p_AOData) + nkeys*sizeof(void *));

  PetscMemcpy(ao->ops,&myops,sizeof(myops));
  ao->ops->destroy  = AODataDestroy_Basic;
  ao->ops->view     = AODataView_Basic;

  ao->nkeys      = nkeys;
  
  for ( i=0; i<nkeys; i++ ) {
    if (i == 0) {
      key = ao->keys  = PetscNew(AODataKey);CHKPTRQ(ao->keys);
    } else {
      key->next       = PetscNew(AODataKey);CHKPTRQ(key);
      key             = key->next;
    }
    key->ltog = 0;
    key->next = 0;

    /* read in key name */
    ierr = PetscBinaryRead(fd,paddedname,256,PETSC_CHAR); CHKERRQ(ierr);
    len  = PetscStrlen(paddedname);
    key->name = (char *) PetscMalloc((len+1)*sizeof(char));CHKPTRQ(key->name);
    PetscStrcpy(key->name,paddedname);

    ierr = PetscBinaryRead(fd,&key->N,1,PETSC_INT); CHKERRQ(ierr);    

    /* determine Nlocal and rowners for key */
    key->nlocal  = key->N/size + ((key->N % size) > rank);
    key->rowners = (int *) PetscMalloc((size+1)*sizeof(int));CHKPTRQ(key->rowners);
    ierr = MPI_Allgather(&key->nlocal,1,MPI_INT,key->rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
    key->rowners[0] = 0;
    for (j=2; j<=size; j++ ) {
      key->rowners[j] += key->rowners[j-1];
    }
    key->rstart        = key->rowners[rank];
    key->rend          = key->rowners[rank+1];

    /* loop key's segments, reading them in */
    ierr = PetscBinaryRead(fd,&key->nsegments,1,PETSC_INT); CHKERRQ(ierr);    

    for ( j=0; j<key->nsegments; j++ ) {
      if (j == 0) {
        seg = key->segments = PetscNew(AODataSegment); CHKPTRQ(seg);
      } else {
        seg->next = PetscNew(AODataSegment); CHKPTRQ(seg->next);
        seg       = seg->next;
      }

      /* read in segment name */
      ierr = PetscBinaryRead(fd,paddedname,256,PETSC_CHAR); CHKERRQ(ierr);
      len  = PetscStrlen(paddedname);
      seg->name = (char *) PetscMalloc((len+1)*sizeof(char));CHKPTRQ(seg->name);
      PetscStrcpy(seg->name,paddedname);

      /* read in segment blocksize and datatype */
      ierr = PetscBinaryRead(fd,&seg->bs,1,PETSC_INT);CHKERRQ(ierr);
      ierr = PetscBinaryRead(fd,&seg->datatype,1,PETSC_INT);CHKERRQ(ierr);

      /* allocate the space for the data */
      ierr = PetscDataTypeGetSize(seg->datatype,&dsize); CHKERRQ(ierr);
      if (seg->datatype == PETSC_LOGICAL) Nb = BTLength(key->N*seg->bs); else Nb = key->N*seg->bs;
      seg->data = (void *) PetscMalloc(Nb*dsize);CHKPTRQ(seg->data);
      /* read in the data */
      ierr = PetscBinaryRead(fd,seg->data,key->N*seg->bs,seg->datatype);CHKERRQ(ierr);
      seg->next = 0;
    }
  }
  *aoout = ao; 

  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view",&flg1); CHKERRQ(ierr);
  if (flg1) {
    ierr = AODataView(ao,VIEWER_STDOUT_(comm)); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view_info",&flg1); CHKERRQ(ierr);
  if (flg1) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_(comm),VIEWER_FORMAT_ASCII_INFO,0);CHKERRQ(ierr);
    ierr = AODataView(ao,VIEWER_STDOUT_(comm)); CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}




