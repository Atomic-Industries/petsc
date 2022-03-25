#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

/*@
  DMPlexSetMigrationSF - Sets the SF for migrating from a parent DM into this DM

  Input Parameters:
+ dm        - The DM
- naturalSF - The PetscSF

  Note: It is necessary to call this in order to have DMCreateSubDM() or DMCreateSuperDM() build the Global-To-Natural map

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexCreateMigrationSF(), DMPlexGetMigrationSF()
@*/
PetscErrorCode DMPlexSetMigrationSF(DM dm, PetscSF migrationSF)
{
  PetscFunctionBegin;
  dm->sfMigration = migrationSF;
  CHKERRQ(PetscObjectReference((PetscObject) migrationSF));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetMigrationSF - Gets the SF for migrating from a parent DM into this DM

  Input Parameter:
. dm          - The DM

  Output Parameter:
. migrationSF - The PetscSF

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexCreateMigrationSF(), DMPlexSetMigrationSF
@*/
PetscErrorCode DMPlexGetMigrationSF(DM dm, PetscSF *migrationSF)
{
  PetscFunctionBegin;
  *migrationSF = dm->sfMigration;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetGlobalToNaturalSF - Sets the SF for mapping Global Vec to the Natural Vec

  Input Parameters:
+ dm          - The DM
- naturalSF   - The PetscSF

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexCreateGlobalToNaturalSF(), DMPlexGetGlobaltoNaturalSF()
@*/
PetscErrorCode DMPlexSetGlobalToNaturalSF(DM dm, PetscSF naturalSF)
{
  PetscFunctionBegin;
  dm->sfNatural = naturalSF;
  CHKERRQ(PetscObjectReference((PetscObject) naturalSF));
  dm->useNatural = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetGlobalToNaturalSF - Gets the SF for mapping Global Vec to the Natural Vec

  Input Parameter:
. dm          - The DM

  Output Parameter:
. naturalSF   - The PetscSF

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexCreateGlobalToNaturalSF(), DMPlexSetGlobaltoNaturalSF
@*/
PetscErrorCode DMPlexGetGlobalToNaturalSF(DM dm, PetscSF *naturalSF)
{
  PetscFunctionBegin;
  *naturalSF = dm->sfNatural;
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateGlobalToNaturalSF - Creates the SF for mapping Global Vec to the Natural Vec

  Input Parameters:
+ dm          - The DM
. section     - The PetscSection describing the Vec before the mesh was distributed,
                or NULL if not available
- sfMigration - The PetscSF used to distribute the mesh, or NULL if it cannot be computed

  Output Parameter:
. sfNatural   - PetscSF for mapping the Vec in PETSc ordering to the canonical ordering

  Note: This is not typically called by the user.

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField()
 @*/
PetscErrorCode DMPlexCreateGlobalToNaturalSF(DM dm, PetscSection section, PetscSF sfMigration, PetscSF *sfNatural)
{
  MPI_Comm       comm;
  Vec            gv, tmpVec;
  PetscSF        sf, sfEmbed, sfSeqToNatural, sfField, sfFieldInv;
  PetscSection   gSection, sectionDist, gLocSection;
  PetscInt      *spoints, *remoteOffsets;
  PetscInt       ssize, pStart, pEnd, p, globalSize;
  PetscLayout    map;
  PetscBool      destroyFlag = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  if (!sfMigration) {
    /* If sfMigration is missing,
    sfNatural cannot be computed and is set to NULL */
    *sfNatural = NULL;
    PetscFunctionReturn(0);
  } else if (!section) {
    /* If the sequential section is not provided (NULL),
    it is reconstructed from the parallel section */
    PetscSF sfMigrationInv;
    PetscSection localSection;

    CHKERRQ(DMGetLocalSection(dm, &localSection));
    CHKERRQ(PetscSFCreateInverseSF(sfMigration, &sfMigrationInv));
    CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section));
    CHKERRQ(PetscSFDistributeSection(sfMigrationInv, localSection, NULL, section));
    CHKERRQ(PetscSFDestroy(&sfMigrationInv));
    destroyFlag = PETSC_TRUE;
  }
  /* CHKERRQ(PetscPrintf(comm, "Point migration SF\n"));
   CHKERRQ(PetscSFView(sfMigration, 0)); */
  /* Create a new section from distributing the original section */
  CHKERRQ(PetscSectionCreate(comm, &sectionDist));
  CHKERRQ(PetscSFDistributeSection(sfMigration, section, &remoteOffsets, sectionDist));
  /* CHKERRQ(PetscPrintf(comm, "Distributed Section\n"));
   CHKERRQ(PetscSectionView(sectionDist, PETSC_VIEWER_STDOUT_WORLD)); */
  CHKERRQ(DMSetLocalSection(dm, sectionDist));
  /* If a sequential section is provided but no dof is affected,
  sfNatural cannot be computed and is set to NULL */
  CHKERRQ(DMCreateGlobalVector(dm, &tmpVec));
  CHKERRQ(VecGetSize(tmpVec, &globalSize));
  CHKERRQ(DMRestoreGlobalVector(dm, &tmpVec));
  if (globalSize) {
  /* Get a pruned version of migration SF */
    CHKERRQ(DMGetGlobalSection(dm, &gSection));
    CHKERRQ(PetscSectionGetChart(gSection, &pStart, &pEnd));
    for (p = pStart, ssize = 0; p < pEnd; ++p) {
      PetscInt dof, off;

      CHKERRQ(PetscSectionGetDof(gSection, p, &dof));
      CHKERRQ(PetscSectionGetOffset(gSection, p, &off));
      if ((dof > 0) && (off >= 0)) ++ssize;
    }
    CHKERRQ(PetscMalloc1(ssize, &spoints));
    for (p = pStart, ssize = 0; p < pEnd; ++p) {
      PetscInt dof, off;

      CHKERRQ(PetscSectionGetDof(gSection, p, &dof));
      CHKERRQ(PetscSectionGetOffset(gSection, p, &off));
      if ((dof > 0) && (off >= 0)) spoints[ssize++] = p;
    }
    CHKERRQ(PetscSFCreateEmbeddedLeafSF(sfMigration, ssize, spoints, &sfEmbed));
    CHKERRQ(PetscFree(spoints));
    /* CHKERRQ(PetscPrintf(comm, "Embedded SF\n"));
    CHKERRQ(PetscSFView(sfEmbed, 0)); */
    /* Create the SF for seq to natural */
    CHKERRQ(DMGetGlobalVector(dm, &gv));
    CHKERRQ(VecGetLayout(gv,&map));
    /* Note that entries of gv are leaves in sfSeqToNatural, entries of the seq vec are roots */
    CHKERRQ(PetscSFCreate(comm, &sfSeqToNatural));
    CHKERRQ(PetscSFSetGraphWithPattern(sfSeqToNatural, map, PETSCSF_PATTERN_GATHER));
    CHKERRQ(DMRestoreGlobalVector(dm, &gv));
    /* CHKERRQ(PetscPrintf(comm, "Seq-to-Natural SF\n"));
    CHKERRQ(PetscSFView(sfSeqToNatural, 0)); */
    /* Create the SF associated with this section */
    CHKERRQ(DMGetPointSF(dm, &sf));
    CHKERRQ(PetscSectionCreateGlobalSection(sectionDist, sf, PETSC_FALSE, PETSC_TRUE, &gLocSection));
    CHKERRQ(PetscSFCreateSectionSF(sfEmbed, section, remoteOffsets, gLocSection, &sfField));
    CHKERRQ(PetscSFDestroy(&sfEmbed));
    CHKERRQ(PetscSectionDestroy(&gLocSection));
    /* CHKERRQ(PetscPrintf(comm, "Field SF\n"));
    CHKERRQ(PetscSFView(sfField, 0)); */
    /* Invert the field SF so it's now from distributed to sequential */
    CHKERRQ(PetscSFCreateInverseSF(sfField, &sfFieldInv));
    CHKERRQ(PetscSFDestroy(&sfField));
    /* CHKERRQ(PetscPrintf(comm, "Inverse Field SF\n"));
    CHKERRQ(PetscSFView(sfFieldInv, 0)); */
    /* Multiply the sfFieldInv with the */
    CHKERRQ(PetscSFComposeInverse(sfFieldInv, sfSeqToNatural, sfNatural));
    CHKERRQ(PetscObjectViewFromOptions((PetscObject) *sfNatural, NULL, "-globaltonatural_sf_view"));
    /* Clean up */
    CHKERRQ(PetscSFDestroy(&sfFieldInv));
    CHKERRQ(PetscSFDestroy(&sfSeqToNatural));
  } else {
    *sfNatural = NULL;
  }
  CHKERRQ(PetscSectionDestroy(&sectionDist));
  CHKERRQ(PetscFree(remoteOffsets));
  if (destroyFlag) CHKERRQ(PetscSectionDestroy(&section));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGlobalToNaturalBegin - Rearranges a global Vector in the natural order.

  Collective on dm

  Input Parameters:
+ dm - The distributed DMPlex
- gv - The global Vec

  Output Parameters:
. nv - Vec in the canonical ordering distributed over all processors associated with gv

  Note: The user must call DMSetUseNatural(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(), DMPlexGlobalToNaturalEnd()
@*/
PetscErrorCode DMPlexGlobalToNaturalBegin(DM dm, Vec gv, Vec nv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(DMPLEX_GlobalToNaturalBegin,dm,0,0,0));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size));
  if (dm->sfNatural) {
    CHKERRQ(VecGetArray(nv, &outarray));
    CHKERRQ(VecGetArrayRead(gv, &inarray));
    CHKERRQ(PetscSFBcastBegin(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray,MPI_REPLACE));
    CHKERRQ(VecRestoreArrayRead(gv, &inarray));
    CHKERRQ(VecRestoreArray(nv, &outarray));
  } else if (size == 1) {
    CHKERRQ(VecCopy(gv, nv));
  } else PetscCheckFalse(dm->useNatural,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  CHKERRQ(PetscLogEventEnd(DMPLEX_GlobalToNaturalBegin,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGlobalToNaturalEnd - Rearranges a global Vector in the natural order.

  Collective on dm

  Input Parameters:
+ dm - The distributed DMPlex
- gv - The global Vec

  Output Parameter:
. nv - The natural Vec

  Note: The user must call DMSetUseNatural(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

 .seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(), DMPlexGlobalToNaturalBegin()
 @*/
PetscErrorCode DMPlexGlobalToNaturalEnd(DM dm, Vec gv, Vec nv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(DMPLEX_GlobalToNaturalEnd,dm,0,0,0));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size));
  if (dm->sfNatural) {
    CHKERRQ(VecGetArrayRead(gv, &inarray));
    CHKERRQ(VecGetArray(nv, &outarray));
    CHKERRQ(PetscSFBcastEnd(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray,MPI_REPLACE));
    CHKERRQ(VecRestoreArrayRead(gv, &inarray));
    CHKERRQ(VecRestoreArray(nv, &outarray));
  } else if (size == 1) {
  } else PetscCheckFalse(dm->useNatural,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  CHKERRQ(PetscLogEventEnd(DMPLEX_GlobalToNaturalEnd,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexNaturalToGlobalBegin - Rearranges a Vector in the natural order to the Global order.

  Collective on dm

  Input Parameters:
+ dm - The distributed DMPlex
- nv - The natural Vec

  Output Parameters:
. gv - The global Vec

  Note: The user must call DMSetUseNatural(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(),DMPlexGlobalToNaturalEnd()
@*/
PetscErrorCode DMPlexNaturalToGlobalBegin(DM dm, Vec nv, Vec gv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(DMPLEX_NaturalToGlobalBegin,dm,0,0,0));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size));
  if (dm->sfNatural) {
    /* We only have access to the SF that goes from Global to Natural.
       Instead of inverting dm->sfNatural, we can call PetscSFReduceBegin/End with MPI_Op MPI_SUM.
       Here the SUM really does nothing since sfNatural is one to one, as long as gV is set to zero first. */
    CHKERRQ(VecZeroEntries(gv));
    CHKERRQ(VecGetArray(gv, &outarray));
    CHKERRQ(VecGetArrayRead(nv, &inarray));
    CHKERRQ(PetscSFReduceBegin(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray, MPI_SUM));
    CHKERRQ(VecRestoreArrayRead(nv, &inarray));
    CHKERRQ(VecRestoreArray(gv, &outarray));
  } else if (size == 1) {
    CHKERRQ(VecCopy(nv, gv));
  } else PetscCheckFalse(dm->useNatural,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  CHKERRQ(PetscLogEventEnd(DMPLEX_NaturalToGlobalBegin,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexNaturalToGlobalEnd - Rearranges a Vector in the natural order to the Global order.

  Collective on dm

  Input Parameters:
+ dm - The distributed DMPlex
- nv - The natural Vec

  Output Parameters:
. gv - The global Vec

  Note: The user must call DMSetUseNatural(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(), DMPlexGlobalToNaturalBegin()
 @*/
PetscErrorCode DMPlexNaturalToGlobalEnd(DM dm, Vec nv, Vec gv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(DMPLEX_NaturalToGlobalEnd,dm,0,0,0));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size));
  if (dm->sfNatural) {
    CHKERRQ(VecGetArrayRead(nv, &inarray));
    CHKERRQ(VecGetArray(gv, &outarray));
    CHKERRQ(PetscSFReduceEnd(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray, MPI_SUM));
    CHKERRQ(VecRestoreArrayRead(nv, &inarray));
    CHKERRQ(VecRestoreArray(gv, &outarray));
  } else if (size == 1) {
  } else PetscCheckFalse(dm->useNatural,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  CHKERRQ(PetscLogEventEnd(DMPLEX_NaturalToGlobalEnd,dm,0,0,0));
  PetscFunctionReturn(0);
}
