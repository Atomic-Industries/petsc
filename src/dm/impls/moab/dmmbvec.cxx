#include <petsc-private/dmmbimpl.h> /*I  "petscdm.h"   I*/
#include <petsc-private/vecimpl.h> /*I  "petscdm.h"   I*/

#include <petscdmmoab.h>
#include <MBTagConventions.hpp>

// declare for use later but before they're defined
static PetscErrorCode DMMoab_VecUserDestroy(void *user);
static PetscErrorCode DMMoab_VecDuplicate(Vec x,Vec *y);
static PetscErrorCode DMMoab_VecCreateTagName_Private(moab::ParallelComm *pcomm,char** tag_name);

#undef __FUNCT__
#define __FUNCT__ "DMMoab_CreateVector_Private"
PetscErrorCode DMMoab_CreateVector_Private(DM dm,moab::Tag tag,moab::Range* userrange,PetscBool is_global_vec,PetscBool destroy_tag,Vec *vec)
{
  PetscErrorCode         ierr;
  moab::ErrorCode        merr;
  PetscBool              is_newtag;
  moab::Range           *range;
  PetscInt               count,lnative_vec,gnative_vec;
  std::string ttname;
  PetscScalar *data_ptr;
  ISLocalToGlobalMapping ltogb;

  Vec_MOAB *vmoab;
  DM_Moab *dmmoab = (DM_Moab*)dm->data;
  moab::ParallelComm *pcomm = dmmoab->pcomm;
  moab::Interface *mbiface = dmmoab->mbiface;

  PetscFunctionBegin;
  if(!userrange) range = dmmoab->vowned;
  else range = userrange;
  if(!range) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Input range cannot be empty or call DMSetUp first.");

  /* If the tag data is in a single sequence, use PETSc native vector since tag_iterate isn't useful anymore */
  lnative_vec=(range->psize()-1);

//  lnative_vec=1; /* NOTE: Testing PETSc vector: will force to create native vector all the time */
  ierr = MPI_Allreduce(&lnative_vec, &gnative_vec, 1, MPI_INT, MPI_MAX, pcomm->comm());CHKERRQ(ierr);

  /* Create the MOAB internal data object */
  ierr = PetscNew(Vec_MOAB,&vmoab);CHKERRQ(ierr);
  vmoab->is_native_vec=(gnative_vec>0?PETSC_TRUE:PETSC_FALSE);

  if (!vmoab->is_native_vec) {
    merr = mbiface->tag_get_name(tag, ttname);
    if (!ttname.length() && merr !=moab::MB_SUCCESS) {
      /* get the new name for the anonymous MOABVec -> the tag_name will be destroyed along with Tag */
      char *tag_name = PETSC_NULL;
      ierr = DMMoab_VecCreateTagName_Private(pcomm,&tag_name);CHKERRQ(ierr);
      is_newtag = PETSC_TRUE;

      /* Create the default value for the tag (all zeros) */
      std::vector<PetscScalar> default_value(dmmoab->nfields, 0.0);

      /* Create the tag */
      merr = mbiface->tag_get_handle(tag_name,dmmoab->nfields,moab::MB_TYPE_DOUBLE,tag,
                                    moab::MB_TAG_DENSE|moab::MB_TAG_CREAT,default_value.data());MBERRNM(merr);
      ierr = PetscFree(tag_name);CHKERRQ(ierr);
    }
    else {
      /* Make sure the tag data is of type "double" */
      moab::DataType tag_type;
      merr = mbiface->tag_get_data_type(tag, tag_type);MBERRNM(merr);
      if(tag_type != moab::MB_TYPE_DOUBLE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Tag data type must be MB_TYPE_DOUBLE");
      is_newtag = destroy_tag;
    }

    vmoab->tag = tag;
    vmoab->new_tag = is_newtag;
  }
  vmoab->mbiface = mbiface;
  vmoab->pcomm = pcomm;
  vmoab->is_global_vec = is_global_vec;
  vmoab->tag_size=dmmoab->bs;
  
  if (vmoab->is_native_vec) {

    /* Create the PETSc Vector directly and attach our functions accordingly */
    if (!is_global_vec) {
      /* This is an MPI Vector with ghosted padding */
      ierr = VecCreateGhostBlock(pcomm->comm(),dmmoab->bs,dmmoab->nfields*dmmoab->nloc,
                                 dmmoab->nfields*dmmoab->n,dmmoab->nghost,&dmmoab->gsindices[dmmoab->nloc],vec);CHKERRQ(ierr);
    }
    else {
      /* This is an MPI/SEQ Vector */
      ierr = VecCreate(pcomm->comm(),vec);CHKERRQ(ierr);
      ierr = VecSetSizes(*vec,dmmoab->nfields*dmmoab->nloc,PETSC_DECIDE);CHKERRQ(ierr);
      ierr = VecSetBlockSize(*vec,dmmoab->bs);CHKERRQ(ierr);
      ierr = VecSetType(*vec, VECMPI);CHKERRQ(ierr);
    }
  }
  else {
    /* Call tag_iterate. This will cause MOAB to allocate memory for the
       tag data if it hasn't already happened */
    merr = mbiface->tag_iterate(tag,range->begin(),range->end(),count,(void*&)data_ptr);MBERRNM(merr);

    /* set the reference for vector range */
    vmoab->tag_range = new moab::Range(*range);
    merr = mbiface->tag_get_length(tag,dmmoab->nfields);MBERRNM(merr);

    /* Create the PETSc Vector
      Query MOAB mesh to check if there are any ghosted entities
        -> if we do, create a ghosted vector to map correctly to the same layout
        -> else, create a non-ghosted parallel vector */
    if (!is_global_vec) {
      /* This is an MPI Vector with ghosted padding */
      ierr = VecCreateGhostBlockWithArray(pcomm->comm(),dmmoab->bs,dmmoab->nfields*dmmoab->nloc,
                                dmmoab->nfields*dmmoab->n,dmmoab->nghost,&dmmoab->gsindices[dmmoab->nloc],data_ptr,vec);CHKERRQ(ierr);
    }
    else {
      /* This is an MPI Vector without ghosted padding */
      ierr = VecCreateMPIWithArray(pcomm->comm(),dmmoab->bs,dmmoab->nfields*range->size(),
                                PETSC_DECIDE,data_ptr,vec);CHKERRQ(ierr);
    }
  }
  ierr = VecSetFromOptions(*vec);CHKERRQ(ierr);

  PetscContainer moabdata;
  ierr = PetscContainerCreate(PETSC_COMM_WORLD,&moabdata);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(moabdata,vmoab);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(moabdata,DMMoab_VecUserDestroy);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*vec,"MOABData",(PetscObject)moabdata);CHKERRQ(ierr);
  (*vec)->ops->duplicate = DMMoab_VecDuplicate;
  ierr = PetscContainerDestroy(&moabdata);CHKERRQ(ierr);

  /* Vector created, manually set local to global mapping */
  ierr = VecSetLocalToGlobalMapping(*vec,dmmoab->ltog_map);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingBlock(dmmoab->ltog_map,dmmoab->bs,&ltogb);
  ierr = VecSetLocalToGlobalMappingBlock(*vec,ltogb);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&ltogb);CHKERRQ(ierr);

  /* set the DM reference to the vector */
  ierr = VecSetDM(*vec, dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabCreateVector"
/*@
  DMMoabCreateVector - Create a Vec from either an existing tag, or a specified tag size, and a range of entities

  Collective on MPI_Comm

  Input Parameter:
. dm              - The DMMoab object being set
. tag             - If non-zero, block size will be taken from the tag size
. range           - If non-empty, Vec corresponds to these entities, otherwise to the entities set on the DMMoab
. is_global_vec   - If true, this is a local representation of the Vec (including ghosts in parallel), otherwise a truly parallel one
. destroy_tag     - If true, MOAB tag is destroyed with Vec, otherwise it is left on MOAB

  Output Parameter:
. vec             - The created vector

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabCreateVector(DM dm,moab::Tag tag,moab::Range* range,PetscBool is_global_vec,PetscBool destroy_tag,Vec *vec)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if(!tag && (!range || range->empty())) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Both tag and range cannot be null.");

  ierr = DMMoab_CreateVector_Private(dm,tag,range,is_global_vec,destroy_tag,vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_Moab"
PetscErrorCode DMCreateGlobalVector_Moab(DM dm,Vec *gvec)
{
  PetscErrorCode  ierr;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(gvec,2);
  ierr = DMMoab_CreateVector_Private(dm,PETSC_NULL,dmmoab->vowned,PETSC_TRUE,PETSC_TRUE,gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_Moab"
PetscErrorCode DMCreateLocalVector_Moab(DM dm,Vec *lvec)
{
  PetscErrorCode  ierr;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(lvec,2);
  ierr = DMMoab_CreateVector_Private(dm,PETSC_NULL,dmmoab->vlocal,PETSC_FALSE,PETSC_TRUE,lvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetVecTag"
/*@
  DMMoabGetVecTag - Get the MOAB tag associated with this Vec

  Input Parameter:
. vec - Vec being queried

  Output Parameter:
. tag - Tag associated with this Vec

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetVecTag(Vec vec,moab::Tag *tag)
{
  PetscContainer  moabdata;
  Vec_MOAB        *vmoab;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(tag,2);

  /* Get the MOAB private data */
  ierr = PetscObjectQuery((PetscObject)vec,"MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(moabdata, (void**) &vmoab);CHKERRQ(ierr);

  *tag = vmoab->tag;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabGetVecRange"
/*@
  DMMoabGetVecRange - Get the MOAB entities associated with this Vec

  Input Parameter:
. vec   - Vec being queried

  Output Parameter:
. range - Entities associated with this Vec

  Level: beginner

.keywords: DMMoab, create
@*/
PetscErrorCode DMMoabGetVecRange(Vec vec,moab::Range *range)
{
  PetscContainer  moabdata;
  Vec_MOAB        *vmoab;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(range,2);

  /* Get the MOAB private data handle */
  ierr = PetscObjectQuery((PetscObject)vec,"MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(moabdata, (void**) &vmoab);CHKERRQ(ierr);

  *range = *vmoab->tag_range;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoab_VecDuplicate"
PetscErrorCode DMMoab_VecDuplicate(Vec x,Vec *y)
{
  PetscErrorCode ierr;
  DM             dm;
  PetscContainer  moabdata;
  Vec_MOAB        *vmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidPointer(y,2);

  /* Get the Vec_MOAB struct for the original vector */
  ierr = PetscObjectQuery((PetscObject)x,"MOABData", (PetscObject*) &moabdata);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(moabdata, (void**)&vmoab);CHKERRQ(ierr);

  ierr = VecGetDM(x, &dm);CHKERRQ(ierr);
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);

  ierr = DMMoab_CreateVector_Private(dm,PETSC_NULL,vmoab->tag_range,vmoab->is_global_vec,PETSC_TRUE,y);CHKERRQ(ierr);
  ierr = VecSetDM(*y, dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoab_VecCreateTagName_Private"
/*  DMMoab_VecCreateTagName_Private
 *
 *  Creates a unique tag name that will be shared across processes. If
 *  pcomm is NULL, then this is a serial vector. A unique tag name
 *  will be returned in tag_name in either case.
 *
 *  The tag names have the format _PETSC_VEC_N where N is some integer.
 *
 *  NOTE: The tag_name is allocated in this routine; The user needs to free 
 *        the character array.
 */
PetscErrorCode DMMoab_VecCreateTagName_Private(moab::ParallelComm *pcomm,char** tag_name)
{
  moab::ErrorCode mberr;
  PetscErrorCode  ierr;
  PetscInt        n,global_n;
  moab::Tag indexTag;

  PetscFunctionBegin;
  const char*       PVEC_PREFIX      = "__PETSC_VEC_";
  ierr = PetscMalloc(PETSC_MAX_PATH_LEN, tag_name);CHKERRQ(ierr);

  /* Check to see if there are any PETSc vectors defined */
  moab::Interface  *mbiface = pcomm->get_moab();
  moab::EntityHandle rootset = mbiface->get_root_set();
  
  /* Create a tag in MOAB mesh to index and keep track of number of Petsc vec tags */
  mberr = mbiface->tag_get_handle("__PETSC_VECS__",1,moab::MB_TYPE_INTEGER,indexTag,
                                  moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT,0);MBERRNM(mberr);
  mberr = mbiface->tag_get_data(indexTag, &rootset, 1, &n);
  if (mberr == moab::MB_TAG_NOT_FOUND) n=0;  /* this is the first temporary vector */
  else MBERRNM(mberr);

  /* increment the new value of n */
  ++n;

  /* Make sure that n is consistent across all processes */
  ierr = MPI_Allreduce(&n,&global_n,1,MPI_INT,MPI_MAX,pcomm->comm());CHKERRQ(ierr);

  /* Set the new name accordingly and return */
  ierr = PetscSNPrintf(*tag_name, PETSC_MAX_PATH_LEN-1, "%s_%D", PVEC_PREFIX, global_n);CHKERRQ(ierr);
  mberr = mbiface->tag_set_data(indexTag, &rootset, 1, (const void*)&global_n);MBERRNM(mberr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoab_VecUserDestroy"
PetscErrorCode DMMoab_VecUserDestroy(void *user)
{
  Vec_MOAB        *vmoab=(Vec_MOAB*)user;
  PetscErrorCode  ierr;
  moab::ErrorCode merr;

  PetscFunctionBegin;
  if(vmoab->new_tag && vmoab->tag) {
    /* Tag was created via a call to VecDuplicate, delete the underlying tag in MOAB */
    merr = vmoab->mbiface->tag_delete(vmoab->tag);MBERRNM(merr);
  }
  delete vmoab->tag_range;
  vmoab->tag = PETSC_NULL;
  vmoab->mbiface = PETSC_NULL;
  vmoab->pcomm = PETSC_NULL;
  ierr = PetscFree(vmoab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabVecGetArray"
/*@
  DMMoabVecGetArray - Returns the writable direct access array to the local representation of MOAB tag data for the underlying vector using locally owned+ghosted range of entities

  Collective on MPI_Comm

  Input Parameter:
. dm              - The DMMoab object being set
. vec             - The Vector whose underlying data is requested

  Output Parameter:
. array           - The local data array

  Level: intermediate

.keywords: MOAB, distributed array

.seealso: DMMoabVecRestoreArray(), DMMoabVecGetArrayRead(), DMMoabVecRestoreArrayRead()
@*/
PetscErrorCode  DMMoabVecGetArray(DM dm,Vec vec,void* array)
{
  DM_Moab        *dmmoab;
  moab::ErrorCode merr;
  PetscErrorCode  ierr;
  PetscInt        count;
  moab::Tag       vtag;
  PetscScalar    **varray;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(array,3);
  dmmoab=(DM_Moab*)dm->data;

  /* Get the MOAB private data */
  ierr = DMMoabGetVecTag(vec,&vtag);CHKERRQ(ierr);

  /* Get the real scalar array handle */
  varray = reinterpret_cast<PetscScalar**>(array);

  /* exchange the data into ghost cells first */
  merr = dmmoab->pcomm->exchange_tags(vtag,*dmmoab->vlocal);MBERRNM(merr);

  /* Get the array data for local entities */
  merr = dmmoab->mbiface->tag_iterate(vtag,dmmoab->vlocal->begin(),dmmoab->vlocal->end(),count,reinterpret_cast<void*&>(*varray),true);MBERRNM(merr);
  if (count!=(PetscInt)dmmoab->vlocal->size()) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mismatch between local vertices and tag partition for Vec. %D != %D.",count,dmmoab->vlocal->size());

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabVecRestoreArray"
/*@
  DMMoabVecRestoreArray - Restores the writable direct access array obtained via DMMoabVecGetArray

  Collective on MPI_Comm

  Input Parameter:
+ dm              - The DMMoab object being set
. vec             - The Vector whose underlying data is requested
- array           - The local data array

  Level: intermediate

.keywords: MOAB, distributed array

.seealso: DMMoabVecGetArray(), DMMoabVecGetArrayRead(), DMMoabVecRestoreArrayRead()
@*/
PetscErrorCode  DMMoabVecRestoreArray(DM dm,Vec v,void* array)
{
  DM_Moab        *moab;
  moab::ErrorCode merr;
  PetscErrorCode  ierr;
  moab::Tag       vtag;
  PetscScalar    **varray;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscValidPointer(array,3);
  moab=(DM_Moab*)dm->data;

  /* Get the MOAB private data */
  ierr = DMMoabGetVecTag(v,&vtag);CHKERRQ(ierr);

  /* Get the real scalar array handle */
  varray = reinterpret_cast<PetscScalar**>(array);
  merr = moab->mbiface->tag_set_data(vtag,*moab->vlocal,reinterpret_cast<void*&>(*varray));MBERRNM(merr);

  /* reduce the tags correctly -> should probably let the user choose how to reduce in the future
     For all FEM residual based assembly calculations, MPI_SUM should serve well */
  merr = moab->pcomm->reduce_tags(vtag,MPI_SUM,*moab->vlocal);MBERRNM(merr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMoabVecGetArrayRead"
/*@
  DMMoabVecGetArrayRead - Returns the read-only direct access array to the local representation of MOAB tag data for the underlying vector using locally owned+ghosted range of entities

  Collective on MPI_Comm

  Input Parameter:
+ dm              - The DMMoab object being set
. vec             - The Vector whose underlying data is requested

  Output Parameter:
. array           - The local data array

  Level: intermediate

.keywords: MOAB, distributed array

.seealso: DMMoabVecRestoreArrayRead(), DMMoabVecGetArray(), DMMoabVecRestoreArray()
@*/
PetscErrorCode  DMMoabVecGetArrayRead(DM dm,Vec vec,void* array)
{
  DM_Moab        *moab;
  moab::ErrorCode merr;
  PetscErrorCode  ierr;
  PetscInt        count;
  moab::Tag       vtag;
  PetscScalar    **varray;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  moab=(DM_Moab*)dm->data;

  /* Get the MOAB private data */
  ierr = DMMoabGetVecTag(vec,&vtag);CHKERRQ(ierr);

  /* Get the real scalar array handle */
  varray = reinterpret_cast<PetscScalar**>(array);

  /* exchange the data into ghost cells first */
  merr = moab->pcomm->exchange_tags(vtag,*moab->vlocal);MBERRNM(merr);

  /* Get the array data for local entities */
  merr = moab->mbiface->tag_iterate(vtag,moab->vlocal->begin(),moab->vlocal->end(),count,reinterpret_cast<void*&>(*varray),true);MBERRNM(merr);
  if (count!=(PetscInt)moab->vlocal->size()) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mismatch between local vertices and tag partition for Vec. %D != %D.",count,moab->vlocal->size());

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoabVecRestoreArrayRead"
/*@
  DMMoabVecRestoreArray - Restores the read-only direct access array obtained via DMMoabVecGetArray

  Collective on MPI_Comm

  Input Parameter:
+ dm              - The DMMoab object being set
. vec             - The Vector whose underlying data is requested
- array           - The local data array

  Level: intermediate

.keywords: MOAB, distributed array

.seealso: DMMoabVecGetArrayRead(), DMMoabVecGetArray(), DMMoabVecRestoreArray()
@*/
PetscErrorCode  DMMoabVecRestoreArrayRead(DM dm,Vec v,void* array)
{
  PetscFunctionBegin;
  /* Nothing to do -> do not free the array memory obtained from tag_iterate */
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMGlobalToLocalBegin_Moab"
PetscErrorCode  DMGlobalToLocalBegin_Moab(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode    ierr;  
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;

  PetscFunctionBegin;
  ierr = VecScatterBegin(dmmoab->ltog_sendrecv,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMGlobalToLocalEnd_Moab"
PetscErrorCode  DMGlobalToLocalEnd_Moab(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode    ierr;  
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;

  PetscFunctionBegin;
  ierr = VecScatterEnd(dmmoab->ltog_sendrecv,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMLocalToGlobalBegin_Moab"
PetscErrorCode  DMLocalToGlobalBegin_Moab(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode    ierr;  
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;

  PetscFunctionBegin;
  ierr = VecScatterBegin(dmmoab->ltog_sendrecv,l,g,mode,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMLocalToGlobalEnd_Moab"
PetscErrorCode  DMLocalToGlobalEnd_Moab(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode    ierr;  
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;

  PetscFunctionBegin;
  ierr = VecScatterEnd(dmmoab->ltog_sendrecv,l,g,mode,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


