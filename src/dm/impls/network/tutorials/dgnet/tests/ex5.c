static const char help[] = "Test of DMPlexAdd_Disconnected";
/*

  TODO ADD EXAMPLE RUNS HERE 

  Contributed by: Aidan Hamilton <aidan@udel.edu>

*/

#include <petscdm.h>
#include <petscdmnetwork.h>
#include "../dgnet.h"

static PetscErrorCode Physics_CreateDummy(DGNetwork dgnet,PetscInt dof, PetscInt *order)
{
  PetscFunctionBeginUser;
  dgnet->physics.dof            = dof;
  dgnet->physics.order          = order; 
  dgnet->physics.destroy        = PhysicsDestroy_SimpleFree_Net;
  PetscFunctionReturn(0);
}
static PetscErrorCode MakeOrder(PetscInt dof, PetscInt *order,PetscInt maxdegree)
{
  PetscInt  i; 
  for(i=0; i<dof; i++) order[i] = maxdegree;
  PetscFunctionReturn(0);
}
int main(int argc,char *argv[])
{
  MPI_Comm          comm;
  DGNetwork         dgnet;
  PetscInt          *order,dof=1,maxdegree = 2,networktype = 0,dx=10,numdm; 
  PetscErrorCode    ierr;
  PetscMPIInt       size,rank;
  PetscViewer       viewer; 
  DM                dmsum,*dmlist; 
  PetscSection      stratumoff; 

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscMalloc1(1,&dgnet));
  PetscCall(PetscMemzero(dgnet,sizeof(*dgnet)));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Set default values */
  dgnet->comm           = comm;

  /* Command Line Options
   
   TODO Create DGNetwork commandline options (same as any other petsc object)
  
   */
  dgnet->ndaughters = 2;
  ierr = PetscOptionsBegin(comm,NULL,"DGNetwork options","");CHKERRQ(ierr);
    PetscCall(PetscOptionsInt("-network","Select from preselected graphs to build the network (0-6)","",networktype,&networktype,NULL));
    PetscCall(PetscOptionsInt("-dx","Number of elements on each edge","",dx,&dx,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscCall(PetscMalloc1(dof,&order));
  MakeOrder(dof,order,maxdegree);
  PetscCall(Physics_CreateDummy(dgnet,dof,order));
  viewer = PETSC_VIEWER_STDOUT_(comm);
  PetscCall(DMNetworkCreate(PETSC_COMM_WORLD,&dgnet->network)); 
  /* dgnet creation needs to be reworked as this is silly */
  PetscCall(DGNetworkCreate(dgnet,networktype,dx));
  PetscCall(DGNetworkSetComponents(dgnet));
  PetscCall(DGNetworkCleanUp(dgnet));
  PetscCall(DGNetworkBuildEdgeDM(dgnet));
  PetscCall(DGNetworkViewEdgeDMs(dgnet,viewer));
  PetscCall(DGNetworkCreateNetworkDMPlex_3D(dgnet,NULL,0,&dmsum,&stratumoff,&dmlist,&numdm));
  PetscCall(PetscSectionView(stratumoff,viewer));
  PetscCall(DMView(dmsum,viewer));

  /* Clean up */
  ierr = PetscFree(dmlist);
  PetscCall(DGNetworkDestroy(dgnet));
  PetscCall(DMDestroy(&dgnet->network));
  ierr = DMDestroy(&dmsum);
  PetscCall(PetscSectionDestroy(&stratumoff));
  PetscCall(PetscFree(dgnet));
  PetscCall(PetscFree(order));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
TEST*/