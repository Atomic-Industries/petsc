#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <petsc/private/matimpl.h>               /*I "petscmatcoarsen.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscsf.h>
#include <petsccublas.h>

#define MIS_NOT_DONE 0
#define MIS_COARSE 1
#define MIS_FINE -1
#define MIS_APPENDED -2

__global__ void pmis_init_random(PetscInt *perm_ix,
                                 PetscInt *lid_random,
                                 PetscInt size)
{
  PetscInt lid,i;
  PetscInt global_id   = blockDim.x*blockIdx.x+threadIdx.x;
  PetscInt global_size = gridDim.x*blockDim.x;
  curandState state;

  curand_init(global_id,global_id,0,&state);
  for (i=global_id; i<size; i+=global_size) {
    lid = perm_ix[i];
    lid_random[lid] = curand_uniform(&state)*size;
  }
}

__global__ void pmis_init_workdata(PetscInt my0,
                                   PetscInt *lid_index,
                                   PetscInt *lid_parent_gid,
                                   PetscInt size)
{
  PetscInt i;
  PetscInt global_id   = blockDim.x*blockIdx.x+threadIdx.x;
  PetscInt global_size = gridDim.x*blockDim.x;

  for (i=global_id; i<size; i+=global_size) {
    lid_index[i] = i+my0;
    lid_parent_gid[i] = -1;
  }
}

__global__ void pmis_max_neighborhood(PetscInt const *lid_cprowID,
                                      PetscInt const *lid_state,
                                      PetscInt const *lid_random,
                                      PetscInt const *lid_index,
                                      PetscInt const *lid_state_ghosts,
                                      PetscInt const *lid_random_ghosts,
                                      PetscInt const *lid_index_ghosts,
                                      PetscInt       *lid_state2,
                                      PetscInt       *lid_random2,
                                      PetscInt       *lid_index2,
                                      PetscInt const *ii,
                                      PetscInt const *jj,
                                      PetscInt const *ii_ghosts,
                                      PetscInt const *jj_ghosts,
                                      PetscInt size)
{
  PetscInt global_id   = blockDim.x*blockIdx.x+threadIdx.x;
  PetscInt global_size = gridDim.x*blockDim.x;
  PetscInt i,j,lidj,max_state,max_random,max_index;

  for (i=global_id; i<size; i+=global_size) {
    max_state  = lid_state[i];
    max_random = lid_random[i];
    max_index  = lid_index[i];

    for (j=ii[i]; j<ii[i+1]; j++) { /* matA */
      lidj = jj[j];
      /* lexigraphical triple-max */
      if (max_state < lid_state[lidj]) {
        max_state  = lid_state[lidj];
        max_random = lid_random[lidj];
        max_index  = lid_index[lidj];
      } else if (max_state == lid_state[lidj]) {
        if (max_random < lid_random[lidj]) {
          max_state  = lid_state[lidj];
          max_random = lid_random[lidj];
          max_index  = lid_index[lidj];
        } else if (max_random == lid_random[lidj]) {
          if (max_index < lid_index[lidj]) {
            max_state  = lid_state[lidj];
            max_random = lid_random[lidj];
            max_index  = lid_index[lidj];
          }
        }
      }
    }
    if (lid_cprowID && lid_cprowID[i] != -1) {
      for (j=ii_ghosts[lid_cprowID[i]]; j<ii_ghosts[lid_cprowID[i]+1]; j++) { /* matB */
        lidj = jj_ghosts[j];
        /* lexigraphical triple-max */
        if (max_state < lid_state_ghosts[lidj]) {
          max_state  = lid_state_ghosts[lidj];
          max_random = lid_random_ghosts[lidj];
          max_index  = lid_index_ghosts[lidj];
        } else if (max_state == lid_state_ghosts[lidj]) {
          if (max_random < lid_random_ghosts[lidj]) {
            max_state  = lid_state_ghosts[lidj];
            max_random = lid_random_ghosts[lidj];
            max_index  = lid_index_ghosts[lidj];
          } else if (max_random == lid_random_ghosts[lidj]) {
            if (max_index < lid_index_ghosts[lidj]) {
              max_state  = lid_state_ghosts[lidj];
              max_random = lid_random_ghosts[lidj];
              max_index  = lid_index_ghosts[lidj];
            }
          }
        }
      }
    }
    lid_state2[i]  = max_state;
    lid_random2[i] = max_random;
    lid_index2[i]  = max_index;
  }
}

__global__ void pmis_mark_mis_nodes(PetscInt       my0,
                                    PetscInt const *lid_state2,
                                    PetscInt const *lid_index2,
                                    PetscInt       *lid_state,
                                    PetscInt       *lid_type,
                                    PetscInt       *undecided_buffer,
                                    PetscInt       size)
{
  PetscInt global_id   = blockDim.x*blockIdx.x+threadIdx.x;
  PetscInt global_size = gridDim.x*blockDim.x;
  PetscInt num_undecided = 0;
  PetscInt max_state,max_index,i;

  for (i=global_id; i<size; i+=global_size) {
    max_state = lid_state2[i];
    max_index = lid_index2[i];
    if (lid_type[i] == MIS_NOT_DONE) {
      if (i+my0 == max_index) { /* MIS node */
        lid_type[i] = MIS_COARSE;
        lid_state[i] = 1;
      } else if (max_state == 1) { /* can be removed */
        lid_type[i] = MIS_FINE;
        lid_state[i] = -1;
      } else num_undecided += 1;
    }
  }
  /* reduction of the number of undecided nodes inside a block */
  __shared__ PetscInt shared_buffer[256];
  shared_buffer[threadIdx.x] = num_undecided;
  for (PetscInt stride=blockDim.x/2; stride>0; stride/=2)
  {
    __syncthreads();
    if (threadIdx.x < stride) shared_buffer[threadIdx.x] += shared_buffer[threadIdx.x+stride];
  }
  if (threadIdx.x == 0) undecided_buffer[blockIdx.x] = shared_buffer[0];
}

__global__ void pmis_ghost_nodes_parents(PetscInt const *lid_cprowID,
                                         PetscInt const *lid_state_ghosts,
                                         PetscInt const *lid_type,
                                         PetscInt const *cpcol_gid,
                                         PetscInt       *lid_parent_gid,
                                         PetscInt const *ii_ghosts,
                                         PetscInt const *jj_ghosts,
                                         PetscInt       size)
{
  PetscInt global_id   = blockDim.x*blockIdx.x+threadIdx.x;
  PetscInt global_size = gridDim.x*blockDim.x;
  PetscInt i,j,lidj;

  for (i=global_id; i<size; i+=global_size) {
    if (lid_type[i] == MIS_FINE && lid_cprowID && lid_cprowID[i] != -1) {
      /* check the ghost neighbors */
      for (j=ii_ghosts[lid_cprowID[i]]; j<ii_ghosts[lid_cprowID[i]+1]; j++) { /* matB */
        lidj = jj_ghosts[j];
        if (lid_state_ghosts[lidj] == 1) {
          lid_parent_gid[i] = cpcol_gid[lidj];
          break;
        }
      }
    }
  }
}

/*
   maxIndSetAgg - parallel maximal independent set (MIS) with data locality info. MatAIJ specific!!!

   Input Parameter:
   . perm - serial permutation of rows of local to process in MIS
   . Gmat - glabal matrix of graph (data not defined)
   . strict_aggs - flag for whether to keep strict (non overlapping) aggregates in 'llist';

   Output Parameter:
   . a_selected - IS of selected vertices, includes 'ghost' nodes at end with natural local indices
   . a_locals_llist - array of list of nodes rooted at selected nodes
*/
PETSC_EXTERN PetscErrorCode maxIndSetAggCUDA(IS perm,Mat Gmat,PetscBool strict_aggs,PetscCoarsenData **a_locals_llist)
{
  PetscErrorCode   ierr;
  Mat_SeqAIJ       *matA,*matB=NULL;
  Mat_MPIAIJ       *mpimat=NULL;
  MPI_Comm         comm;
  PetscInt         i,j,num_fine_ghosts,iter,Iend,my0,lid,num_undecided,num_undecided2,nselected;
  PetscInt         *dev_lid_type,*lid_type;
  PetscInt         undecided_buffer[256];
  PetscInt         *lid_gid,*dev_cpcol_gid,*dev_lid_parent_gid,*dev_lid_state,*dev_lid_random,*dev_lid_index,*dev_lid_state2,*dev_lid_random2,*dev_lid_index2,*dev_tmp;
  PetscInt         *dev_lid_state_ghosts = NULL,*dev_lid_random_ghosts = NULL,*dev_lid_index_ghosts = NULL;
  PetscInt         *dev_undecided_buffer;
  PetscInt         *dev_matAi,*dev_matAj,*dev_matBi,*dev_matBj;
  PetscInt         *cpcol_gid,*lid_cprowID = NULL,*dev_lid_cprowID = NULL;
  PetscBool        isMPI,isAIJ;
  const PetscInt   *perm_ix;
  const PetscInt   nloc = Gmat->rmap->n; /* number of local points (exclude ghost points) */
  PetscCoarsenData *agg_lists;
  PetscLayout      layout;
  PetscSF          sf;
  cudaError_t      cerr;

  PetscFunctionBegin;
  if (!strict_aggs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Not support strict_aggs=false on CUDA.");
  ierr = PetscObjectGetComm((PetscObject)Gmat,&comm);CHKERRQ(ierr);
  ierr = PetscCDCreate(nloc,&agg_lists);CHKERRQ(ierr);
  if (a_locals_llist) *a_locals_llist = agg_lists;

  /* get submatrices */
  ierr = PetscObjectBaseTypeCompare((PetscObject)Gmat,MATMPIAIJ,&isMPI);CHKERRQ(ierr);
  if (isMPI) {
    mpimat = (Mat_MPIAIJ*)Gmat->data;
    matA   = (Mat_SeqAIJ*)mpimat->A->data;
    matB   = (Mat_SeqAIJ*)mpimat->B->data;
    /* force compressed storage of B */
    ierr   = MatCheckCompressedRow(mpimat->B,matB->nonzerorowcnt,&matB->compressedrow,matB->i,Gmat->rmap->n,-1.0);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectBaseTypeCompare((PetscObject)Gmat,MATSEQAIJ,&isAIJ);CHKERRQ(ierr);
    if (!isAIJ) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Require AIJ matrix.");
    matA = (Mat_SeqAIJ*)Gmat->data;
  }
  ierr = MatGetOwnershipRange(Gmat,&my0,&Iend);CHKERRQ(ierr);
  ierr = ISGetIndices(perm, &perm_ix);CHKERRQ(ierr);
  cerr = cudaMalloc((void**)&dev_matAi,nloc*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMemcpy(dev_matAi,matA->i,nloc*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&dev_matAj,matA->nz*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMemcpy(dev_matAj,matA->j,matA->nz*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);

  ierr = PetscMalloc1(nloc,&lid_type);CHKERRQ(ierr);
  cerr = cudaMalloc((void**)&dev_lid_type,nloc*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&dev_lid_state,nloc*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&dev_lid_random,nloc*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&dev_lid_index,nloc*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&dev_lid_state2,nloc*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&dev_lid_random2,nloc*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&dev_lid_index2,nloc*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&dev_lid_parent_gid,nloc*sizeof(PetscInt));CHKERRCUDA(cerr);

  /* initialize the data */
  cerr = cudaMemset(dev_lid_type,0,nloc*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMemset(dev_lid_state,0,nloc*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMemcpy(dev_lid_random,perm_ix,nloc*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
  pmis_init_workdata<<<128, 128>>>(my0, 
                                   dev_lid_index,
                                   dev_lid_parent_gid,
                                   nloc
                                  );


  PetscMPIInt     rank;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (mpimat) {
    ierr = VecGetLocalSize(mpimat->lvec,&num_fine_ghosts);CHKERRQ(ierr);
    cerr = cudaMalloc((void**)&dev_lid_state_ghosts,num_fine_ghosts*sizeof(PetscInt));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void**)&dev_lid_random_ghosts,num_fine_ghosts*sizeof(PetscInt));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void**)&dev_lid_index_ghosts,num_fine_ghosts*sizeof(PetscInt));CHKERRCUDA(cerr);
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)Gmat),&sf);CHKERRQ(ierr);
    ierr = MatGetLayouts(Gmat,&layout,NULL);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(sf,layout,num_fine_ghosts,NULL,PETSC_COPY_VALUES,mpimat->garray);CHKERRQ(ierr);
    cerr = cudaMalloc((void**)&dev_matBi,(matB->compressedrow.nrows+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void**)&dev_matBj,matB->nz*sizeof(PetscInt));CHKERRCUDA(cerr);
    cerr = cudaMemcpy(dev_matBi,matB->compressedrow.i,(matB->compressedrow.nrows+1)*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(dev_matBj,matB->j,matB->nz*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);

    /* gid table is needed only for building aggregates */
    ierr = PetscMalloc1(nloc,&lid_gid);CHKERRQ(ierr);
    ierr = PetscMalloc1(num_fine_ghosts,&cpcol_gid);CHKERRQ(ierr);
    cerr = cudaMalloc((void**)&dev_cpcol_gid,num_fine_ghosts*sizeof(PetscInt));CHKERRCUDA(cerr);
    for (i=0; i<nloc; i++) lid_gid[i] = i+my0;
    ierr = PetscSFBcastBegin(sf,MPIU_INT,lid_gid,cpcol_gid);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_INT,lid_gid,cpcol_gid);CHKERRQ(ierr);
    cerr = cudaMemcpy(dev_cpcol_gid,cpcol_gid,num_fine_ghosts*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    ierr = PetscFree(lid_gid);CHKERRQ(ierr);
  } else num_fine_ghosts = 0;

  if (matB) {
    ierr = PetscMalloc1(nloc,&lid_cprowID);CHKERRQ(ierr);
    ierr = cudaMalloc((void**)&dev_lid_cprowID,nloc*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0; i<nloc; i++) lid_cprowID[i] = -1;
    /* set index into cmpressed row 'lid_cprowID' */
    for (i=0; i<matB->compressedrow.nrows; i++) {
      lid = matB->compressedrow.rindex[i];
      lid_cprowID[lid] = i;
    }
    cerr = cudaMemcpy(dev_lid_cprowID,lid_cprowID,nloc*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
  }

  cerr = cudaMalloc((void**)&dev_undecided_buffer,256*sizeof(PetscInt));CHKERRCUDA(cerr);
  /* MIS */
  iter = 0;
  num_undecided2 = nloc;
  while (num_undecided2) {
    PetscInt r;
    iter++;

    if (mpimat) {
      ierr = PetscSFBcastBegin(sf,MPIU_INT,dev_lid_state,dev_lid_state_ghosts);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf,MPIU_INT,dev_lid_state,dev_lid_state_ghosts);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(sf,MPIU_INT,dev_lid_random,dev_lid_random_ghosts);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf,MPIU_INT,dev_lid_random,dev_lid_random_ghosts);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(sf,MPIU_INT,dev_lid_index,dev_lid_index_ghosts);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf,MPIU_INT,dev_lid_index,dev_lid_index_ghosts);CHKERRQ(ierr);
    }

    for (r=0; r<1; r++) { /* only work for MIS(1)*/
      if (r>0) {
        /* copy work array (can be fused into a single kernel if needed. Previous kernel is in most cases sufficiently heavy) */
        dev_tmp = dev_lid_state; dev_lid_state  = dev_lid_state2; dev_lid_state2 = dev_tmp;
        dev_tmp = dev_lid_random; dev_lid_random = dev_lid_random2; dev_lid_random2 = dev_tmp;
        dev_tmp = dev_lid_index; dev_lid_index  = dev_lid_index2; dev_lid_index2 = dev_tmp;
      }
      /* max operation over neighborhood */
      pmis_max_neighborhood<<<128, 128>>>(dev_lid_cprowID,
                                          dev_lid_state,
                                          dev_lid_random,
                                          dev_lid_index,
                                          dev_lid_state_ghosts,
                                          dev_lid_random_ghosts,
                                          dev_lid_index_ghosts,
                                          dev_lid_state2,
                                          dev_lid_random2,
                                          dev_lid_index2,
                                          dev_matAi,
                                          dev_matAj,
                                          dev_matBi,
                                          dev_matBj,
                                          nloc
                                         );

    }

    pmis_mark_mis_nodes<<<128, 128>>>(my0,
                                      dev_lid_state2,
                                      dev_lid_index2,
                                      dev_lid_state,
                                      dev_lid_type,
                                      dev_undecided_buffer,
                                      nloc
                                     );

    cerr = cudaMemcpy(undecided_buffer,dev_undecided_buffer,256*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);

    /* reduction among all blocks */
    num_undecided = 0;
    for (i=0; i<256; i++) {
      num_undecided += undecided_buffer[i];
    }
    if (mpimat) {
      /* all done? */
      ierr = MPIU_Allreduce(&num_undecided,&num_undecided2,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr); /* synchronous version */
      if (!num_undecided2) break;
    } else break; /* all done */
  } /* outer parallel MIS loop */

  cerr = cudaMemcpy(lid_type,dev_lid_type,nloc*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
  for (i=0; i<nloc; i++) {
    if (lid_type[i] == MIS_COARSE) {
      nselected++;
      ierr = PetscCDAppendID(agg_lists,i,i+my0);CHKERRQ(ierr);
      lid_type[i] = MIS_APPENDED;
      /* append local adjacient nodes that are not selected */
      for (j=0; j< matA->i[i+1]-matA->i[i]; j++) {
        PetscInt lidj = matA->j[matA->i[i]+j];
        if (lid_type[lidj] != MIS_APPENDED) {
          ierr = PetscCDAppendID(agg_lists,i,lidj+my0);CHKERRQ(ierr);
          lid_type[lidj] = MIS_APPENDED;
        }
      }
    }
  }

  /* tell adj who my lid_parent_gid vertices belong to - fill in agg_lists selected ghost lists */
  if (matB) {
    PetscInt *cpcol_sel_gid,sgid,gid;

    /* find the parents of the ghost nodes */
    ierr = PetscSFBcastBegin(sf,MPIU_INT,dev_lid_state,dev_lid_state_ghosts);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_INT,dev_lid_state,dev_lid_state_ghosts);CHKERRQ(ierr);
    cerr = cudaMemcpy(dev_lid_type,lid_type,nloc*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    pmis_ghost_nodes_parents<<<128,128>>>(dev_lid_cprowID,
                                          dev_lid_state_ghosts,
                                          dev_lid_type,
                                          dev_cpcol_gid,
                                          dev_lid_parent_gid,
                                          dev_matBi,
                                          dev_matBj,
                                          nloc
                                         );

    ierr = PetscMalloc1(num_fine_ghosts, &cpcol_sel_gid);CHKERRQ(ierr);
    /* get proc of the ghost to be appended */
    ierr = PetscSFBcastBegin(sf,MPIU_INT,dev_lid_parent_gid,cpcol_sel_gid);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_INT,dev_lid_parent_gid,cpcol_sel_gid);CHKERRQ(ierr);
    for (i=0; i<num_fine_ghosts; i++) {
      sgid = cpcol_sel_gid[i];
      gid  = cpcol_gid[i];
      if (sgid >= my0 && sgid < Iend) {
        ierr = PetscCDAppendID(agg_lists, sgid-my0, gid);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(cpcol_sel_gid);CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(perm,&perm_ix);CHKERRQ(ierr);
  ierr = PetscInfo2(Gmat,"\t selected %D of %D vertices.\n",nselected,nloc);CHKERRQ(ierr);

  if (mpimat) {
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
    cerr = cudaFree(dev_lid_state_ghosts);CHKERRCUDA(cerr);
    cerr = cudaFree(dev_lid_random_ghosts);CHKERRCUDA(cerr);
    cerr = cudaFree(dev_lid_index_ghosts);CHKERRCUDA(cerr);
    cerr = cudaFree(dev_matBi);CHKERRCUDA(cerr);
    cerr = cudaFree(dev_matBj);CHKERRCUDA(cerr);
    cerr = cudaFree(dev_cpcol_gid);CHKERRCUDA(cerr);
    cerr = cudaFree(dev_lid_parent_gid);CHKERRCUDA(cerr);
    ierr = PetscFree(cpcol_gid);CHKERRQ(ierr);
  }
  if (matB) {
    ierr = PetscFree(lid_cprowID);CHKERRQ(ierr);
    cerr = cudaFree(dev_lid_cprowID);CHKERRCUDA(cerr);
  }
  ierr = PetscFree(lid_type);CHKERRQ(ierr);
  cerr = cudaFree(dev_lid_type);CHKERRCUDA(cerr);
  cerr = cudaFree(dev_lid_state);CHKERRCUDA(cerr);
  cerr = cudaFree(dev_lid_random);CHKERRCUDA(cerr);
  cerr = cudaFree(dev_lid_index);CHKERRCUDA(cerr);
  cerr = cudaFree(dev_lid_state2);CHKERRCUDA(cerr);
  cerr = cudaFree(dev_lid_random2);CHKERRCUDA(cerr);
  cerr = cudaFree(dev_lid_index2);CHKERRCUDA(cerr);
  cerr = cudaFree(dev_matAi);CHKERRCUDA(cerr);
  cerr = cudaFree(dev_matAj);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}
