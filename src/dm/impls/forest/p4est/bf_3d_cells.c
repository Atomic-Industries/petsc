#include "bf_3d_cells.h"

#if defined(PETSC_HAVE_P4EST)
#include <p4est_to_p8est.h> /* convert to p8est for 3D domains */

/* rename generic functions that are the same for 2D and 3D */
#define DMBF_XD_P4estCreate             DMBF_3D_P4estCreate
#define DMBF_XD_P4estDestroy            DMBF_3D_P4estDestroy
#define DMBF_XD_GhostCreate             DMBF_3D_GhostCreate
#define DMBF_XD_GhostDestroy            DMBF_3D_GhostDestroy

#define DM_BF_XD_Topology               DM_BF_3D_Topology
#define DM_BF_XD_Cells                  DM_BF_3D_Cells
#define _p_DM_BF_XD_Cells               _p_DM_BF_3D_Cells

#define DMBF_XD_TopologyGetConnectivity DMBF_3D_TopologyGetConnectivity
#define DMBF_XD_CellsCreate             DMBF_3D_CellsCreate
#define DMBF_XD_CellsDestroy            DMBF_3D_CellsDestroy
#define DMBF_XD_GetSizes                DMBF_3D_GetSizes
#define DMBF_XD_CellsGetP4est           DMBF_3D_CellsGetP4est
#define DMBF_XD_CellsGetGhost           DMBF_3D_CellsGetGhost

/* include generic functions */
#include "bf_xd_cells.c"

#endif /* defined(PETSC_HAVE_P4EST) */
