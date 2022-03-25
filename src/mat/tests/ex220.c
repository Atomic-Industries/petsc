
#include <petscmat.h>

static char help[PETSC_MAX_PATH_LEN] = "Tests MatLoad() with MatCreateDense() for memory leak ";

int main(int argc, char **argv)
{
    PetscErrorCode      ierr;
    PetscViewer         viewer;
    Mat                 A;
    char                filename[PETSC_MAX_PATH_LEN];
    PetscBool           flg;

    ierr = PetscInitialize(&argc, &argv, (char*)0, help);if (ierr) return ierr;
    CHKERRQ(PetscOptionsGetString(NULL, NULL, "-f", filename, sizeof(filename), &flg));
    PetscCheckFalse(!flg,PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate a filename for input with the -f option");

    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer));
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 36, 36, NULL, &A));
    CHKERRQ(MatLoad(A, viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(MatDestroy(&A));
    ierr = PetscFinalize();
    return ierr;
}

/*TEST

     test:
       requires: double !complex !defined(PETSC_USE_64BIT_INDICES) datafilespath
       args: -f ${DATAFILESPATH}/matrices/small

TEST*/
