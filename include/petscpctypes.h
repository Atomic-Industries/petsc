#ifndef PETSCPCTYPES_H
#define PETSCPCTYPES_H

/* SUBMANSEC = PC */

/*S
     PC - Abstract PETSc object that manages all preconditioners including direct solvers such as `PCLU`

   Level: beginner

.seealso: [](sec_pc), `PCCreate()`, `PCSetType()`, `PCType`
S*/
typedef struct _p_PC *PC;

/*J
    PCType - String with the name of a PETSc preconditioner method.

   Level: beginner

   Note:
   `PCRegister()` is used to register preconditioners that are then accessible via `PCSetType()`

.seealso: [](sec_pc), `PCSetType()`, `PC`, `PCCreate()`, `PCRegister()`, `PCSetFromOptions()`, `PCLU`, `PCJACOBI`, `PCBJACOBI`
J*/
typedef const char *PCType;
#define PCNONE               "none"
#define PCJACOBI             "jacobi"
#define PCSOR                "sor"
#define PCLU                 "lu"
#define PCQR                 "qr"
#define PCSHELL              "shell"
#define PCAMGX               "amgx"
#define PCBJACOBI            "bjacobi"
#define PCMG                 "mg"
#define PCEISENSTAT          "eisenstat"
#define PCILU                "ilu"
#define PCICC                "icc"
#define PCASM                "asm"
#define PCGASM               "gasm"
#define PCKSP                "ksp"
#define PCBJKOKKOS           "bjkokkos"
#define PCCOMPOSITE          "composite"
#define PCREDUNDANT          "redundant"
#define PCSPAI               "spai"
#define PCNN                 "nn"
#define PCCHOLESKY           "cholesky"
#define PCPBJACOBI           "pbjacobi"
#define PCVPBJACOBI          "vpbjacobi"
#define PCMAT                "mat"
#define PCHYPRE              "hypre"
#define PCPARMS              "parms"
#define PCFIELDSPLIT         "fieldsplit"
#define PCTFS                "tfs"
#define PCML                 "ml"
#define PCGALERKIN           "galerkin"
#define PCEXOTIC             "exotic"
#define PCCP                 "cp"
#define PCBFBT               "bfbt"
#define PCLSC                "lsc"
#define PCPYTHON             "python"
#define PCPFMG               "pfmg"
#define PCSMG                "smg"
#define PCSYSPFMG            "syspfmg"
#define PCREDISTRIBUTE       "redistribute"
#define PCSVD                "svd"
#define PCGAMG               "gamg"
#define PCCHOWILUVIENNACL    "chowiluviennacl"
#define PCROWSCALINGVIENNACL "rowscalingviennacl"
#define PCSAVIENNACL         "saviennacl"
#define PCBDDC               "bddc"
#define PCKACZMARZ           "kaczmarz"
#define PCTELESCOPE          "telescope"
#define PCPATCH              "patch"
#define PCLMVM               "lmvm"
#define PCHMG                "hmg"
#define PCDEFLATION          "deflation"
#define PCHPDDM              "hpddm"
#define PCH2OPUS             "h2opus"
#define PCMPI                "mpi"

/*E
    PCSide - If the preconditioner is to be applied to the left, right
     or symmetrically around the operator.

   Level: beginner

.seealso: [](sec_pc), `PC`
E*/
typedef enum {
  PC_SIDE_DEFAULT = -1,
  PC_LEFT,
  PC_RIGHT,
  PC_SYMMETRIC
} PCSide;
#define PC_SIDE_MAX (PC_SYMMETRIC + 1)

/*E
    PCRichardsonConvergedReason - reason a `PCRICHARDSON` `PCApplyRichardson()` method terminated

   Level: advanced

   Developer Note:
    this must match petsc/finclude/petscpc.h and the `KSPConvergedReason` values in petscksp.h

.seealso: [](sec_pc), `PCRICHARDSON`, `PC`, `PCApplyRichardson()`
E*/
typedef enum {
  PCRICHARDSON_CONVERGED_RTOL = 2,
  PCRICHARDSON_CONVERGED_ATOL = 3,
  PCRICHARDSON_CONVERGED_ITS  = 4,
  PCRICHARDSON_DIVERGED_DTOL  = -4
} PCRichardsonConvergedReason;

/*E
    PCJacobiType - What elements are used to form the Jacobi preconditioner

   Values:
+  `PC_JACOBI_DIAGONAL` - use the diagonal entry, if it is zero use one
.  `PC_JACOBI_ROWMAX` - use the maximum absolute value in the row
-  `PC_JACOBI_ROWSUM` - use the sum of the values in the row (not the absolute values)

   Level: intermediate

.seealso: [](sec_pc), `PCJACOBI`, `PC`
E*/
typedef enum {
  PC_JACOBI_DIAGONAL,
  PC_JACOBI_ROWMAX,
  PC_JACOBI_ROWSUM
} PCJacobiType;

/*E
    PCASMType - Type of additive Schwarz method to use

   Values:
+  `PC_ASM_BASIC`        - Symmetric version where residuals from the ghost points are used
                        and computed values in ghost regions are added together.
                        Classical standard additive Schwarz.
.  `PC_ASM_RESTRICT`     - Residuals from ghost points are used but computed values in ghost
                        region are discarded.
                        Default.
.  `PC_ASM_INTERPOLATE`  - Residuals from ghost points are not used, computed values in ghost
                        region are added back in.
-  `PC_ASM_NONE`         - Residuals from ghost points are not used, computed ghost values are
                        discarded.
                        Not very good.

   Level: beginner

.seealso: [](sec_pc), `PC`, `PCASM`, `PCASMSetType()`
E*/
typedef enum {
  PC_ASM_BASIC       = 3,
  PC_ASM_RESTRICT    = 1,
  PC_ASM_INTERPOLATE = 2,
  PC_ASM_NONE        = 0
} PCASMType;

/*E
    PCGASMType - Type of generalized additive Schwarz method to use (differs from `PCASM` in allowing multiple processors per subdomain).

   Each subdomain has nested inner and outer parts.  The inner subdomains are assumed to form a non-overlapping covering of the computational
   domain, while the outer subdomains contain the inner subdomains and overlap with each other.  This preconditioner will compute
   a subdomain correction over each *outer* subdomain from a residual computed there, but its different variants will differ in
   (a) how the outer subdomain residual is computed, and (b) how the outer subdomain correction is computed.

   Values:
+  `PC_GASM_BASIC`      - Symmetric version where the full from the outer subdomain is used, and the resulting correction is applied
                        over the outer subdomains.  As a result, points in the overlap will receive the sum of the corrections
                        from neighboring subdomains.
                        Classical standard additive Schwarz.
.  `PC_GASM_RESTRICT`    - Residual from the outer subdomain is used but the correction is restricted to the inner subdomain only
                        (i.e., zeroed out over the overlap portion of the outer subdomain before being applied).  As a result,
                        each point will receive a correction only from the unique inner subdomain containing it (nonoverlapping covering
                        assumption).
                        Default.
.  `PC_GASM_INTERPOLATE` - Residual is zeroed out over the overlap portion of the outer subdomain, but the resulting correction is
                        applied over the outer subdomain. As a result, points in the overlap will receive the sum of the corrections
                        from neighboring subdomains.

-  `PC_GASM_NONE`       - Residuals and corrections are zeroed out outside the local subdomains.
                        Not very good.

   Level: beginner

.seealso: [](sec_pc), `PCGASM`, `PCASM`, `PC`, `PCGASMSetType()`
E*/
typedef enum {
  PC_GASM_BASIC       = 3,
  PC_GASM_RESTRICT    = 1,
  PC_GASM_INTERPOLATE = 2,
  PC_GASM_NONE        = 0
} PCGASMType;

/*E
    PCCompositeType - Determines how two or more preconditioner are composed

  Values:
+  `PC_COMPOSITE_ADDITIVE` - results from application of all preconditioners are added together
.  `PC_COMPOSITE_MULTIPLICATIVE` - preconditioners are applied sequentially to the residual freshly
                                computed after the previous preconditioner application
.  `PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE` - preconditioners are applied sequentially to the residual freshly
                                computed from first preconditioner to last and then back (Use only for symmetric matrices and preconditioners)
.  `PC_COMPOSITE_SPECIAL` - This is very special for a matrix of the form alpha I + R + S
                         where first preconditioner is built from alpha I + S and second from
                         alpha I + R
.  `PC_COMPOSITE_SCHUR` -  composes the Schur complement of the matrix from two blocks, see `PCFIELDSPLIT`
-  `PC_COMPOSITE_GKB` - the generalized Golub-Kahan bidiagonalization preconditioner, see `PCFIELDSPLIT`

   Level: beginner

.seealso: [](sec_pc), `PCCOMPOSITE`, `PCFIELDSPLIT`, `PC`, `PCCompositeSetType()`
E*/
typedef enum {
  PC_COMPOSITE_ADDITIVE,
  PC_COMPOSITE_MULTIPLICATIVE,
  PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE,
  PC_COMPOSITE_SPECIAL,
  PC_COMPOSITE_SCHUR,
  PC_COMPOSITE_GKB
} PCCompositeType;

/*E
    PCFieldSplitSchurPreType - Determines how to precondition a Schur complement

    Level: intermediate

.seealso: [](sec_pc), `PCFIELDSPLIT`, `PCFieldSplitSetSchurPre()`, `PC`
E*/
typedef enum {
  PC_FIELDSPLIT_SCHUR_PRE_SELF,
  PC_FIELDSPLIT_SCHUR_PRE_SELFP,
  PC_FIELDSPLIT_SCHUR_PRE_A11,
  PC_FIELDSPLIT_SCHUR_PRE_USER,
  PC_FIELDSPLIT_SCHUR_PRE_FULL
} PCFieldSplitSchurPreType;

/*E
    PCFieldSplitSchurFactType - determines which off-diagonal parts of the approximate block factorization to use

    Level: intermediate

 .seealso: [](sec_pc), `PCFIELDSPLIT`, `PCFieldSplitSetSchurFactType()`, `PC`
E*/
typedef enum {
  PC_FIELDSPLIT_SCHUR_FACT_DIAG,
  PC_FIELDSPLIT_SCHUR_FACT_LOWER,
  PC_FIELDSPLIT_SCHUR_FACT_UPPER,
  PC_FIELDSPLIT_SCHUR_FACT_FULL
} PCFieldSplitSchurFactType;

/*E
    PCPARMSGlobalType - Determines the global preconditioner method in `PCPARMS`

    Level: intermediate

 .seealso: [](sec_pc), `PCPARMS`, `PCPARMSSetGlobal()`, `PC`
E*/
typedef enum {
  PC_PARMS_GLOBAL_RAS,
  PC_PARMS_GLOBAL_SCHUR,
  PC_PARMS_GLOBAL_BJ
} PCPARMSGlobalType;

/*E
    PCPARMSLocalType - Determines the local preconditioner method in `PCPARMS`

    Level: intermediate

.seealso: [](sec_pc), `PCPARMS`, `PCPARMSSetLocal()`, `PC`
E*/
typedef enum {
  PC_PARMS_LOCAL_ILU0,
  PC_PARMS_LOCAL_ILUK,
  PC_PARMS_LOCAL_ILUT,
  PC_PARMS_LOCAL_ARMS
} PCPARMSLocalType;

/*J
    PCGAMGType - type of generalized algebraic multigrid `PCGAMG` method

   Values:
+   `PCGAMGAGG` - (the default) smoothed aggregation algorithm, robust, very well tested
.   `PCGAMGGEO` - geometric coarsening, uses mesh generator to produce coarser meshes, limited to triangles, not well tested
-   `PCGAMGCLASSICAL` - classical algebraic multigrid preconditioner, incomplete, poorly tested

     Level: intermediate

.seealso: [](sec_pc), `PCGAMG`, `PCMG`, `PC`, `PCSetType()`, `PCGAMGSetThreshold()`, `PCGAMGSetThreshold()`, `PCGAMGSetReuseInterpolation()`
J*/
typedef const char *PCGAMGType;
#define PCGAMGAGG       "agg"
#define PCGAMGGEO       "geo"
#define PCGAMGCLASSICAL "classical"

typedef const char *PCGAMGClassicalType;
#define PCGAMGCLASSICALDIRECT   "direct"
#define PCGAMGCLASSICALSTANDARD "standard"

/*E
    PCMGType - Determines the type of multigrid method that is run.

   Level: beginner

   Values:
+  `PC_MG_MULTIPLICATIVE` (default) - traditional V or W cycle as determined by `PCMGSetCycleType()`
.  `PC_MG_ADDITIVE` - the additive multigrid preconditioner where all levels are
                smoothed before updating the residual. This only uses the
                down smoother, in the preconditioner the upper smoother is ignored
.  `PC_MG_FULL` - same as multiplicative except one also performs grid sequencing,
            that is starts on the coarsest grid, performs a cycle, interpolates
            to the next, performs a cycle etc. This is much like the F-cycle presented in "Multigrid" by Trottenberg, Oosterlee, Schuller page 49, but that
            algorithm supports smoothing on before the restriction on each level in the initial restriction to the coarsest stage. In addition that algorithm
            calls the V-cycle only on the coarser level and has a post-smoother instead.
-  `PC_MG_KASKADE` - like full multigrid except one never goes back to a coarser level
               from a finer

.seealso: [](sec_pc), `PCMG`, `PC`, `PCMGSetType()`, `PCMGSetCycleType()`, `PCMGSetCycleTypeOnLevel()`
E*/
typedef enum {
  PC_MG_MULTIPLICATIVE,
  PC_MG_ADDITIVE,
  PC_MG_FULL,
  PC_MG_KASKADE
} PCMGType;
#define PC_MG_CASCADE PC_MG_KASKADE;

/*E
    PCMGCycleType - Use V-cycle or W-cycle

   Level: beginner

   Values:
+  `PC_MG_V_CYCLE` - use the v cycle
-  `PC_MG_W_CYCLE` - use the w cycle

.seealso: [](sec_pc), `PCMG`, `PC`, `PCMGSetCycleType()`
E*/
typedef enum {
  PC_MG_CYCLE_V = 1,
  PC_MG_CYCLE_W = 2
} PCMGCycleType;

/*E
    PCMGalerkinType - Determines if the coarse grid operators are computed via the Galerkin process

   Values:
+  `PC_MG_GALERKIN_PMAT` - computes the pmat (matrix from which the preconditioner is built) via the Galerkin process from the finest grid
.  `PC_MG_GALERKIN_MAT` -  computes the mat (matrix used to apply the operator) via the Galerkin process from the finest grid
.  `PC_MG_GALERKIN_BOTH` - computes both the mat and pmat via the Galerkin process (if pmat == mat the construction is only done once
-  `PC_MG_GALERKIN_NONE` - neither operator is computed via the Galerkin process, the user must provide the operator

   Level: beginner

   Note:
   Users should never set `PC_MG_GALERKIN_EXTERNAL`, it is used by `PCHYPRE` and `PCML`

.seealso: [](sec_pc), `PCMG`, `PC`, `PCMGSetCycleType()`
E*/
typedef enum {
  PC_MG_GALERKIN_BOTH,
  PC_MG_GALERKIN_PMAT,
  PC_MG_GALERKIN_MAT,
  PC_MG_GALERKIN_NONE,
  PC_MG_GALERKIN_EXTERNAL
} PCMGGalerkinType;

/*E
    PCExoticType - Face based or wirebasket based coarse grid space

   Level: beginner

.seealso: [](sec_pc), `PCExoticSetType()`, `PCEXOTIC`
E*/
typedef enum {
  PC_EXOTIC_FACE,
  PC_EXOTIC_WIREBASKET
} PCExoticType;

/*E
   PCBDDCInterfaceExtType - Defines how interface balancing is extended into the interior of subdomains.

   Level: intermediate

   Values:
+  `PC_BDDC_INTERFACE_EXT_DIRICHLET` - solves Dirichlet interior problem; this is the standard BDDC algorithm
-  `PC_BDDC_INTERFACE_EXT_LUMP` - skips interior solve; sometimes called M_1 and associated with "lumped FETI-DP"

.seealso: [](sec_pc), `PCBDDC`, `PC`
E*/
typedef enum {
  PC_BDDC_INTERFACE_EXT_DIRICHLET,
  PC_BDDC_INTERFACE_EXT_LUMP
} PCBDDCInterfaceExtType;

/*E
  PCMGCoarseSpaceType - Function space for coarse space for adaptive interpolation

  Level: beginner

.seealso: [](sec_pc), `PCMGSetAdaptCoarseSpaceType()`, `PCMG`, `PC`
E*/
typedef enum {
  PCMG_ADAPT_NONE,
  PCMG_ADAPT_POLYNOMIAL,
  PCMG_ADAPT_HARMONIC,
  PCMG_ADAPT_EIGENVECTOR,
  PCMG_ADAPT_GENERALIZED_EIGENVECTOR,
  PCMG_ADAPT_GDSW
} PCMGCoarseSpaceType;

/*E
    PCPatchConstructType - The algorithm used to construct patches for the preconditioner

   Level: beginner

.seealso: [](sec_pc), `PCPatchSetConstructType()`, `PCPATCH`, `PC`
E*/
typedef enum {
  PC_PATCH_STAR,
  PC_PATCH_VANKA,
  PC_PATCH_PARDECOMP,
  PC_PATCH_USER,
  PC_PATCH_PYTHON
} PCPatchConstructType;

/*E
    PCDeflationSpaceType - Type of deflation

    Values:
+   `PC_DEFLATION_SPACE_HAAR`        - directly assembled based on Haar (db2) wavelet with overflowed filter cuted-off
.   `PC_DEFLATION_SPACE_DB2`         - `MATCOMPOSITE` of 1-lvl matices based on db2 (2 coefficient Daubechies / Haar wavelet)
.   `PC_DEFLATION_SPACE_DB4`         - same as above, but with db4 (4 coefficient Daubechies)
.   `PC_DEFLATION_SPACE_DB8`         - same as above, but with db8 (8 coefficient Daubechies)
.   `PC_DEFLATION_SPACE_DB16`        - same as above, but with db16 (16 coefficient Daubechies)
.   `PC_DEFLATION_SPACE_BIORTH22`    - same as above, but with biorthogonal 2.2 (6 coefficients)
.   `PC_DEFLATION_SPACE_MEYER`       - same as above, but with Meyer/FIR (62 coefficients)
.   `PC_DEFLATION_SPACE_AGGREGATION` - aggregates local indices (given by operator matrix distribution) into a subdomain
-   `PC_DEFLATION_SPACE_USER`        - indicates space set by user

    Level: intermediate

    Note:
    Wavelet-based space (except Haar) can be used in multilevel deflation.

.seealso: [](sec_pc), `PCDeflationSetSpaceToCompute()`, `PCDEFLATION`, `PC`
E*/
typedef enum {
  PC_DEFLATION_SPACE_HAAR,
  PC_DEFLATION_SPACE_DB2,
  PC_DEFLATION_SPACE_DB4,
  PC_DEFLATION_SPACE_DB8,
  PC_DEFLATION_SPACE_DB16,
  PC_DEFLATION_SPACE_BIORTH22,
  PC_DEFLATION_SPACE_MEYER,
  PC_DEFLATION_SPACE_AGGREGATION,
  PC_DEFLATION_SPACE_USER
} PCDeflationSpaceType;

/*E
    PCHPDDMCoarseCorrectionType - Type of coarse correction used by `PCHPDDM`

    Level: intermediate

    Values:
+   `PC_HPDDM_COARSE_CORRECTION_DEFLATED` (default) - eq. (1) in `PCHPDDMShellApply()`
.   `PC_HPDDM_COARSE_CORRECTION_ADDITIVE` - eq. (2)
-   `PC_HPDDM_COARSE_CORRECTION_BALANCED` - eq. (3)

.seealso: [](sec_pc), `PCHPDDM`, `PC`, `PCSetType()`, `PCHPDDMShellApply()`
E*/
typedef enum {
  PC_HPDDM_COARSE_CORRECTION_DEFLATED,
  PC_HPDDM_COARSE_CORRECTION_ADDITIVE,
  PC_HPDDM_COARSE_CORRECTION_BALANCED
} PCHPDDMCoarseCorrectionType;

/*E
    PCFailedReason - indicates type of `PC` failure

    Level: beginner

    Developer Note:
    Any additions/changes here MUST also be made in include/petsc/finclude/petscpc.h

.seealso: [](sec_pc), `PC`
E*/
typedef enum {
  PC_SETUP_ERROR = -1,
  PC_NOERROR,
  PC_FACTOR_STRUCT_ZEROPIVOT,
  PC_FACTOR_NUMERIC_ZEROPIVOT,
  PC_FACTOR_OUTMEMORY,
  PC_FACTOR_OTHER,
  PC_INCONSISTENT_RHS,
  PC_SUBPC_ERROR
} PCFailedReason;

/*E
    PCGAMGLayoutType - Layout for reduced grids

    Level: intermediate

    Developer Note:
    Any additions/changes here MUST also be made in include/petsc/finclude/petscpc.h

.seealso: [](sec_pc), `PCGAMG`, `PC`, `PCGAMGSetCoarseGridLayoutType()`
E*/
typedef enum {
  PCGAMG_LAYOUT_COMPACT,
  PCGAMG_LAYOUT_SPREAD
} PCGAMGLayoutType;

#endif
