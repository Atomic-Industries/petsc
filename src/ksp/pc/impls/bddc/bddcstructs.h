#if !defined(__pcbddc_structs_h)
#define __pcbddc_structs_h

#include <petscksp.h>
#include <petscbt.h>

/* special marks for interface graph: they cannot be enums, since special marks should in principle range from -4 to -max_int */
#define PCBDDCGRAPH_NEUMANN_MARK -1
#define PCBDDCGRAPH_DIRICHLET_MARK -2
#define PCBDDCGRAPH_LOCAL_PERIODIC_MARK -3
#define PCBDDCGRAPH_SPECIAL_MARK -4

/* Structure for local graph partitioning */
struct _PCBDDCGraph {
  ISLocalToGlobalMapping l2gmap;
  PetscInt               nvtxs;
  PetscBT                touched;
  PetscInt               *count;
  PetscInt               **neighbours_set;
  PetscInt               *subset;
  PetscInt               *which_dof;
  PetscInt               *cptr;
  PetscInt               *queue;
  PetscInt               *special_dof;
  PetscInt               *subset_ncc;
  PetscInt               *subset_ref_node;
  PetscInt               *mirrors;
  PetscInt               **mirrors_set;
  PetscInt               ncc;
  PetscInt               n_subsets;
  PetscInt               custom_minimal_size;
  PetscInt               nvtxs_csr;
  PetscInt               *xadj;
  PetscInt               *adjncy;
};
typedef struct _PCBDDCGraph *PCBDDCGraph;

struct _PCBDDCSubSchurs {
  /* local Neumann matrix */
  Mat A;
  /* local Schur complement */
  Mat S;
  /* index sets */
  IS  is_I;
  IS  is_B;
  /* local Schur complements on subsets (principal minors) */
  Mat *S_Ej;
  Mat S_Ej_all;
  Mat sum_S_Ej_all;
  /* work vectors */
  Vec *work1;
  Vec *work2;
  /* IS */
  IS *is_AEj_I;
  IS *is_AEj_B;
  /* number of local subproblems */
  PetscInt n_subs;
  /* number of local sequential subproblems */
  PetscInt n_subs_seq;
  /* number of local parallel subproblems */
  PetscInt n_subs_par;
  /* number of global sequential subproblems */
  PetscInt n_subs_seq_g;
  /* number of global parallel subproblems */
  PetscInt n_subs_par_g;
  /* connected components */
  IS*      is_subs;
  /* some workspace */
  PetscInt *index_sequential;
  PetscInt *index_parallel;
  PetscInt *auxglobal_sequential;
  PetscInt *auxglobal_parallel;
};
typedef struct _PCBDDCSubSchurs *PCBDDCSubSchurs;

/* Structure for deluxe scaling */
struct _PCBDDCDeluxeScaling {
  /* simple scaling on selected dofs (i.e. primal vertices and nodes on interface dirichlet boundaries) */
  PetscInt        n_simple;
  PetscInt*       idx_simple_B;
  /* sequential problems  */
  VecScatter      seq_scctx;
  Vec             seq_work1;
  Vec             seq_work2;
  Mat             seq_mat;
  KSP             seq_ksp;
  /* parallel problems */
  PetscInt        par_colors;
  VecScatter*     par_scctx_s;
  VecScatter*     par_scctx_p;
  Vec*            par_vec;
  KSP*            par_ksp;
  PetscInt*       par_col2sub;
};
typedef struct _PCBDDCDeluxeScaling *PCBDDCDeluxeScaling;

/* inexact solvers with nullspace correction */
struct _NullSpaceCorrection_ctx {
  Mat basis_mat;
  Mat Kbasis_mat;
  Mat Lbasis_mat;
  PC  local_pc;
  Vec work_small_1;
  Vec work_small_2;
  Vec work_full_1;
  Vec work_full_2;
};
typedef struct _NullSpaceCorrection_ctx *NullSpaceCorrection_ctx;

/* change of basis */
struct _PCBDDCChange_ctx {
  Mat original_mat;
  Mat global_change;
  Vec *work;
};
typedef struct _PCBDDCChange_ctx *PCBDDCChange_ctx;

/* feti-dp mat */
struct _FETIDPMat_ctx {
  PetscInt   n_lambda;
  Vec        lambda_local;
  Vec        temp_solution_B;
  Vec        temp_solution_D;
  Mat        B_delta;
  Mat        B_Ddelta;
  VecScatter l2g_lambda;
  PC         pc;
};
typedef struct _FETIDPMat_ctx *FETIDPMat_ctx;

/* feti-dp dirichlet preconditioner */
struct _FETIDPPC_ctx {
  Mat        S_j;
  Vec        lambda_local;
  Mat        B_Ddelta;
  VecScatter l2g_lambda;
  PC         pc;
};
typedef struct _FETIDPPC_ctx *FETIDPPC_ctx;

#endif
