#include <petscsys.h>

/*@C
    PetscEqualReal - Returns whether the two real values are equal.

    Input Parameters:
+     a - first real number
-     b - second real number

    Note:
    Equivalent to "a == b". Should be used to prevent compilers from
    emitting floating point comparison warnings (e.g. GCC's -Wfloat-equal flag)
    in PETSc header files or user code.

    Level: developer

.seealso: `PetscIsCloseAtTol()`, `PetscEqualScalar()`
@*/
PetscBool PetscEqualReal(PetscReal a, PetscReal b)
{
  return (a == b) ? PETSC_TRUE : PETSC_FALSE;
}

/*@C
    PetscEqualScalar - Returns whether the two scalar values are equal.

    Input Parameters:
+     a - first scalar value
-     b - second scalar value

    Note:
    Equivalent to "a == b". Should be used to prevent compilers from
    emitting floating point comparison warnings (e.g. GCC's -Wfloat-equal flag)
    in PETSc header files or user code.

    Level: developer

.seealso: `PetscIsCloseAtTol()`, `PetscEqualReal()`
@*/
PetscBool PetscEqualScalar(PetscScalar a, PetscScalar b)
{
  return (a == b) ? PETSC_TRUE : PETSC_FALSE;
}
