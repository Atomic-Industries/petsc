#if !defined(PETSCDMBF_XD_H)
#define PETSCDMBF_XD_H

#include <petsc/private/dmforestimpl.h> /*I "petscdmforest.h" I*/
#include "petsc_p4est_package.h"

#if defined(PETSC_HAVE_P4EST)

#if !defined(P4_TO_P8)
#include <p4est.h>
#include <p4est_extended.h>
#include <p4est_ghost.h>
#include <p4est_bits.h>
#include <p4est_algorithms.h>
#else
#include <p8est.h>
#include <p8est_extended.h>
#include <p8est_ghost.h>
#include <p8est_bits.h>
#include <p8est_algorithms.h>
#endif /* !defined(P4_TO_P8) */

#endif /* defined(PETSC_HAVE_P4EST) */

#endif /* defined(PETSCDMBF_XD_H) */
