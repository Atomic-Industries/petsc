#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tagm.c,v 1.3 1998/06/29 21:01:23 balay Exp balay $";
#endif
/*
      Some PETSc utilites
*/
#include "petsc.h"        
#include "sys.h"             /*I    "sys.h"   I*/
#include <stdarg.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"

/* ---------------------------------------------------------------- */
/*
   A simple way to manage tags inside a private 
   communicator.  It uses the attribute to determine if a new communicator
   is needed.

   Notes on the implementation

   The tagvalues to use are stored in a two element array.  The first element
   is the first free tag value.  The second is used to indicate how
   many "copies" of the communicator there are used in destroying.
*/

static int Petsc_Tag_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ "Petsc_DelTag" 
/*
   Private routine to delete internal storage when a communicator is freed.
  This is called by MPI, not by users.

  The binding for the first argument changed from MPI 1.0 to 1.1; in 1.0
  it was MPI_Comm *comm.  
*/
int Petsc_DelTag(MPI_Comm comm,int keyval,void* attr_val,void* extra_state )
{
  PetscFunctionBegin;
  PetscFree( attr_val );
  PetscFunctionReturn(MPI_SUCCESS);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetNewTag" 
/*@
    PetscObjectGetNewTag - Gets a unique new tag from a PETSc object. All 
    processors that share the object MUST call this routine EXACTLY the same
    number of times.  This tag should only be used with the current object's
    communicator; do NOT use it with any other MPI communicator.

    Collective on PetscObject

    Input Parameter:
.   obj - the PETSc object; this must be cast with a (PetscObject), for example, 
         PetscObjectGetNewTag((PetscObject) mat,&tag);

    Output Parameter:
.   tag - the new tag

.keywords: object, get, new, tag

.seealso: PetscObjectRestoreNewTag()
@*/
int PetscObjectGetNewTag(PetscObject obj,int *tag)
{
  int ierr,*tagvalp=0,flag;

  PetscFunctionBegin;
  PetscValidHeader(obj);
  PetscValidIntPointer(tag);

  ierr = MPI_Attr_get(obj->comm,Petsc_Tag_keyval,(void**)&tagvalp,&flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Bad comm in PETSc object");

  if (*tagvalp < 1) SETERRQ(PETSC_ERR_PLIB,0,"Out of tags for object");
  *tag = tagvalp[0]--;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectRestoreNewTag" 
/*@
    PetscObjectRestoreNewTag - Restores a new tag from a PETSc object. All 
    processors that share the object MUST call this routine EXACTLY the same
    number of times. 

    Collective on PetscObject

    Input Parameter:
.   obj - the PETSc object; this must be cast with a (PetscObject), for example, 
          PetscObjectRestoreNewTag((PetscObject) mat,&tag);

    Output Parameter:
.   tag - the new tag

.keywords: object, restore, new, tag

.seealso: PetscObjectGetNewTag()
@*/
int PetscObjectRestoreNewTag(PetscObject obj,int *tag)
{
  int ierr,*tagvalp=0,flag;

  PetscFunctionBegin;
  PetscValidHeader(obj);
  PetscValidIntPointer(tag);

  ierr = MPI_Attr_get(obj->comm,Petsc_Tag_keyval,(void**)&tagvalp,&flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Bad comm in PETSc object");

  if (*tagvalp == *tag - 1) {
    tagvalp[0]++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscCommDup_Private" 
/*
  PetscCommDup_Private - Duplicates the communicator only if it is not already PETSc 
                         communicator.

  Input Parameters:
. comm_in - Input communicator

  Output Parameters:
. comm_out - Output communicator.  May be 'comm_in'.
. first_tag - First tag available

  Notes:
  This routine returns one tag number.
  Call Petsc_Comm_free() when finished with the communicator.
*/
int PetscCommDup_Private(MPI_Comm comm_in,MPI_Comm *comm_out,int* first_tag)
{
  int ierr = MPI_SUCCESS,*tagvalp, flag,*maxval;

  PetscFunctionBegin;
  if (Petsc_Tag_keyval == MPI_KEYVAL_INVALID) {
    /* 
       The calling sequence of the 2nd argument to this function changed
       between MPI Standard 1.0 and the revisions 1.1 Here we match the 
       new standard, if you are using an MPI implementation that uses 
       the older version you will get a warning message about the next line;
       it is only a warning message and should do no harm.
    */
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN, Petsc_DelTag,&Petsc_Tag_keyval,(void*)0);CHKERRQ(ierr);
  }

  ierr = MPI_Attr_get(comm_in,Petsc_Tag_keyval,(void**)&tagvalp,&flag);CHKERRQ(ierr);

  if (!flag) {
    /* This communicator is not yet known to this system, so we dup it and set its value */
    ierr       = MPI_Comm_dup( comm_in, comm_out );CHKERRQ(ierr);
    ierr       = MPI_Attr_get( MPI_COMM_WORLD, MPI_TAG_UB, (void**)&maxval, &flag );CHKERRQ(ierr);
    tagvalp    = (int *) PetscMalloc( 2*sizeof(int) ); CHKPTRQ(tagvalp);
    tagvalp[0] = *maxval;
    tagvalp[1] = 0;
    ierr       = MPI_Attr_put(*comm_out,Petsc_Tag_keyval, tagvalp);CHKERRQ(ierr);
  } else {
    *comm_out = comm_in;
  }

  if (*tagvalp < 1) SETERRQ(PETSC_ERR_PLIB,0,"Out of tags for object");
  *first_tag = tagvalp[0]--;
  tagvalp[1]++;
#if defined(USE_PETSC_BOPT_g)
  if (*comm_out == comm_in) {
    int size;
    MPI_Comm_size(*comm_out,&size);
    if (size > 1) {
      int tag1 = *first_tag, tag2;
      ierr = MPI_Allreduce(&tag1,&tag2,1,MPI_INT,MPI_BOR,*comm_out);CHKERRQ(ierr);
      if (tag2 != tag1) {
        SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Communicator was used on subset\n of processors.");
      }
    }
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscCommFree_Private" 
/*
  PetscCommFree_Private - Frees communicator.  Use in conjunction with PetscCommDup_Private().
*/
int PetscCommFree_Private(MPI_Comm *comm)
{
  int ierr,*tagvalp,flag;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(*comm,Petsc_Tag_keyval,(void**)&tagvalp,&flag);CHKERRQ(ierr);
  if (!flag) {
    SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Error freeing PETSc object, problem with corrupted memory");
  }
  tagvalp[1]--;
  if (!tagvalp[1]) {ierr = MPI_Comm_free(comm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}




