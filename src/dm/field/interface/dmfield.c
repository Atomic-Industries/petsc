#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/

const char *const DMFieldContinuities[] = {
  "VERTEX",
  "EDGE",
  "FACET",
  "CELL",
  0
};

PETSC_INTERN PetscErrorCode DMFieldCreate(DM dm,PetscInt numComponents,DMFieldContinuity continuity,DMField *field)
{
  PetscErrorCode ierr;
  DMField        b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(field,2);
  ierr = DMFieldInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(b,DMFIELD_CLASSID,"DMField","Field over DM","DM",PetscObjectComm((PetscObject)dm),DMFieldDestroy,DMFieldView);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);
  b->dm = dm;
  b->continuity = continuity;
  b->numComponents = numComponents;
  *field = b;
  PetscFunctionReturn(0);
}

/*@
   DMFieldDestroy - destroy a DMField

   Collective

   Input Arguments:
.  field - address of DMField

   Level: advanced

.seealso: DMFieldCreate()
@*/
PetscErrorCode DMFieldDestroy(DMField *field)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*field) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*field),DMFIELD_CLASSID,1);
  if (--((PetscObject)(*field))->refct > 0) {*field = 0; PetscFunctionReturn(0);}
  if ((*field)->ops->destroy) {ierr = (*(*field)->ops->destroy)(*field);CHKERRQ(ierr);}
  ierr = DMDestroy(&((*field)->dm));CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(field);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMFieldView - view a DMField

   Collective

   Input Arguments:
+  field - DMField
-  viewer - viewer to display field, for example PETSC_VIEWER_STDOUT_WORLD

   Level: advanced

.seealso: DMFieldCreate()
@*/
PetscErrorCode DMFieldView(DMField field,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)field),&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(field,1,viewer,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)field,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%D components\n",field->numComponents);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%s continuity\n",DMFieldContinuities[field->continuity]);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);
    ierr = DMView(field->dm,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  if (field->ops->view) {ierr = (*field->ops->view)(field,viewer);CHKERRQ(ierr);}
  if (iascii) {
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   DMFieldSetType - set the DMField implementation

   Collective on DMField

   Input Parameters:
+  field - the DMField context
-  type - a known method

   Notes:
   See "include/petscvec.h" for available methods (for instance)
+    DMFIELDDA    - a field defined only by its values at the corners of a DMDA
.    DMFIELDDS    - a field defined by a discretization over a mesh set with DMSetField()
-    DMFIELDSHELL - a field defined by arbitrary callbacks

  Level: advanced

.keywords: DMField, set, type

.seealso: DMFieldType,
@*/
PetscErrorCode DMFieldSetType(DMField field,DMFieldType type)
{
  PetscErrorCode ierr,(*r)(DMField);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)field,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(DMFieldList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested DMField type %s",type);
  /* Destroy the previous private DMField context */
  if (field->ops->destroy) {
    ierr = (*(field)->ops->destroy)(field);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(field->ops,sizeof(*field->ops));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)field,type);CHKERRQ(ierr);
  field->ops->create = r;
  ierr = (*r)(field);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMFieldGetType - Gets the DMField type name (as a string) from the DMField.

  Not Collective

  Input Parameter:
. field  - The DMField context

  Output Parameter:
. type - The DMField type name

  Level: advanced

.keywords: DMField, get, type, name
.seealso: DMFieldSetType()
@*/
PetscErrorCode  DMFieldGetType(DMField field, DMFieldType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, VEC_TAGGER_CLASSID,1);
  PetscValidPointer(type,2);
  ierr = DMFieldRegisterAll();CHKERRQ(ierr);
  *type = ((PetscObject)field)->type_name;
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldGetNumComponents(DMField field, PetscInt *nc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  PetscValidIntPointer(nc,2);
  *nc = field->numComponents;
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldGetDM(DMField field, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  PetscValidPointer(dm,2);
  *dm = field->dm;
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldEvaluate(DMField field, Vec points, PetscDataType datatype, void *B, void *D, void *H)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  PetscValidHeaderSpecific(points,VEC_CLASSID,2);
  if (B) PetscValidPointer(B,3);
  if (D) PetscValidPointer(D,4);
  if (H) PetscValidPointer(H,5);
  if (field->ops->evaluate) {
    ierr = (*field->ops->evaluate) (field, points, datatype, B, D, H);CHKERRQ(ierr);
  } else SETERRQ (PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"Not implemented for this type");
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldEvaluateFE(DMField field, PetscInt numCells, const PetscInt *cells, PetscQuadrature points, PetscDataType datatype, void *B, void *D, void *H)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  if (numCells) PetscValidIntPointer(cells,3);
  PetscValidHeader(points,4);
  if (B) PetscValidPointer(B,5);
  if (D) PetscValidPointer(D,6);
  if (H) PetscValidPointer(H,7);
  if (field->ops->evaluate) {
    ierr = (*field->ops->evaluateFE) (field, numCells, cells, points, datatype, B, D, H);CHKERRQ(ierr);
  } else SETERRQ (PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"Not implemented for this type");
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldEvaluateFV(DMField field, PetscInt numCells, const PetscInt *cells, PetscDataType datatype, void *B, void *D, void *H)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  if (numCells) PetscValidIntPointer(cells,3);
  if (B) PetscValidPointer(B,4);
  if (D) PetscValidPointer(D,5);
  if (H) PetscValidPointer(H,6);
  if (field->ops->evaluate) {
    ierr = (*field->ops->evaluateFV) (field, numCells, cells, datatype, B, D, H);CHKERRQ(ierr);
  } else SETERRQ (PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"Not implemented for this type");
  PetscFunctionReturn(0);
}
