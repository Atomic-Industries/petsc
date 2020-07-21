#include <petscdmnetwork.h>
#include <petsc.h>

/* Finite Volume Data Strcutures, temporary as they are built right for this application. 
   but they work for now */
#include "finitevolume1d.h"
#include "limiters.h"

/* Finite Volume Data Structures */
typedef enum {JUNCT=1,RESERVOIR=2,VALVE=3,DEMAND=4,INFLOW=5,STAGE=6,TANK=7,OUTFLOW=8} VertexType;

/* Network Data Structures */

/* Component numbers used for accessing data in DMNetWork*/
typedef enum {FVEDGE=0} EdgeCompNum;
typedef enum {JUNCTION=0,FLUX=1} VertexCompNum;   
typedef enum {EDGEIN=0,EDGEOUT=1} EdgeDirection;

struct _p_Junction{
  PetscInt	    id;        /* global index */
  PetscInt      tag;       /* external id */
  VertexType    type;               
  Mat           *jacobian;
  PetscReal     x; /* x-coordinates */
  PetscBool     *dir; /*In the local ordering whether index i point into or out of the vertex. PetscTrue points out. */
  PetscInt      numedges; /* Number of edges connected to this vertex (globally) (it feels like this info should 
                             live in the dmnetwork, but I don't see how to access it.)*/           
  /* Finite Volume Context */
  /*RiemannFunction_2WaySplit couplingflux; Need to figure out how to build a function pointer within a network component in a sensible way. */

  /* boundary data structures - To be added*/
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_Junction *Junction;

struct _p_FVEdge
{
  /* identification variables */
  PetscInt    id;
  PetscInt    offset_vto,offset_vfrom; /* offsets for placing the reconstruction data and setting flux data 
                                                      for the edge cells */
  /* solver objects */
  /* Note that this object holds no solution data. This is held 
     by the DMnetwork. This object merely gives the appropriate context 
     for the data belonging to the given edge */

  PetscInt    nnodes;   /* number of nodes in da discretization */
  Mat         *jacobian;
 /*void                *user;*/ /* user inputted data, need for function evaluations. However not 
                                   sure how do this right, as this data will have to be set after partitioning, 
                                   so the user will have to provide a function to set these based on id I think.
                                   worry about it later */

  /* FV object */
  PetscReal h; /* discretization size, assumes uniform mesh*/

  /* Multirate ODE Context */ 
  PetscInt  tobufferlvl,frombufferlvl; /* Level of the buffer on the to and from ends of the edge. lvl 0 refers to no buffer at all */
  
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_FVEdge *FVEdge;

typedef struct {
  PetscErrorCode                 (*sample)(void*,PetscInt,PetscReal,PetscReal,PetscReal*);
  PetscErrorCode                 (*inflow)(void*,PetscReal,PetscReal,PetscReal*);
  RiemannFunction_2WaySplit      riemann;
  ReconstructFunction_2WaySplit  characteristic;
  PetscErrorCode                 (*destroy)(void*);
  void                           *user;
  PetscInt                       dof;
  char                           *fieldname[16];
} PhysicsCtx_Net;

/* Global FV information on the entire network. */
struct _p_FVNetwork 
{
  MPI_Comm    comm;
  PetscInt    nedge,nvertex;           /* local number of components */
  PetscInt    Nedge,Nvertex;           /* global number of components */
  PetscInt    *edgelist;               /* local edge list */
  Vec         localX,localF;           /* vectors used in local function evalutation */
  Vec         X,Ftmp;                  /* Global vectors used in function evaluations */
  PetscInt    nnodes_loc;              /* num of global and local nodes */
  DM          network;
  PetscBool   monifv;
  PetscReal   ymin,ymax;               
  DMNetworkMonitor  monitor;
  char        prefix[256];
  void        (*limit)(const PetscScalar*,const PetscScalar*,PetscScalar*,PetscInt);
  RiemannFunction_2WaySplit couplingflux; /* Structure for performing the coupling flux. Should be attached 
                                             to a junction instead of the global network structure. Also not sure 
                                             if this is the right function type for this. But we will see. */

  /* Local work arrays */
  PetscScalar *R,*Rinv;         /* Characteristic basis, and it's inverse.  COLUMN-MAJOR */
  PetscScalar *cjmpLR;          /* Jumps at left and right edge of cell, in characteristic basis, len=2*dof uL____cell_i____uR*/
  PetscScalar *cslope;          /* Limited slope, written in characteristic basis */
  PetscScalar *uLR;             /* Solution at left and right of a cell, conservative variables, len=2*dof */
  PetscScalar *flux;            /* Flux across interface */
  PetscReal   *speeds;          /* Speeds of each wave */
  PetscReal   *uPlus;           /* Solution at the left of the interfacce in conservative variables, len = dof  uPlus_|_uL___cell_i___uR_|_ */

  PetscReal   cfl_idt;          /* Max allowable value of 1/Delta t */
  PetscReal   cfl;
  PetscInt    initial,subcase;
  PetscBool   simulation;
  PetscBool   exact;
  PetscInt    hratio;
  PetscInt    Mx;               /* Variable used to specify smallest number of cells for an edge in a problem */
  /* Junction */
  Junction    junction;
  /* Edges */
  FVEdge      fvedge;
  /* FV Context */ 
  /* We assume for efficiency and simplicity that the network has
     a single discretization on all edges/vertices and the same physics. 
     So that context information is stored here in the network object. The 
     solvers and rhs functions in the edges/vertices will call this info when 
     actually performing the cell updates */ 
  PhysicsCtx_Net physics; 
  /* Multirate Context */
  /* All of these IS are on MPI_COMM_SELF*/
  IS          slow_edges,fast_edges,buf_slow_vert,slow_vert, fast_vert;                                                                 
  PetscInt    bufferwidth; 
}PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar)); 
typedef struct _p_FVNetwork *FVNetwork; 

typedef struct{
FVNetwork fvnet; 
IS        edgelist;
IS        vtxlist;
IS        wheretoputstuff;
} RhsCtx; 

PetscErrorCode FVNetCharacteristicLimit(FVNetwork,PetscScalar*,PetscScalar*,PetscScalar*);
/* Set up the FVNetworkComponents and 'blank' network data to be read by the other functions. 
   Allocate the work array data for FVNetwork */
PetscErrorCode FVNetworkCreate(PetscInt,FVNetwork,PetscInt);
/* set the components into the network and the number of variables
   each component requires. Also construct the local ordering for the
   edges of a vertex */ 
PetscErrorCode FVNetworkSetComponents(FVNetwork);
/* Delete the unneeded data built by FVNetworkCreate. Removes 
   the edgelist data, fvedges, junctions, that have been set 
    into the network by FVNetworkSetComponents */
PetscErrorCode FVNetworkCleanUp(FVNetwork);
/* After distributing the network, build the dynamic data required 
   by the components. This includes physics data as well as building 
   the vertex data structures needed for evaluating the edge data they 
   'steal' */ 
PetscErrorCode FVNetworkSetupPhysics(FVNetwork);
/* Create the multirate data structures the components require */
PetscErrorCode FVNetworkSetupMultirate(FVNetwork,PetscInt*,PetscInt*,PetscInt*); 
/* Destroy allocated data */
PetscErrorCode FVNetworkDestroy(FVNetwork);
/* Set Initial Solution */\
PetscErrorCode FVNetworkSetInitial(FVNetwork,Vec);
/*RHS Function*/
PetscErrorCode FVNetRHS(TS,PetscReal,Vec,Vec,void*);
/* Multirate Functions */
PetscErrorCode FVNetworkGenerateMultiratePartition_HValue(FVNetwork,PetscReal);
PetscErrorCode FVNetworkGenerateMultiratePartition_Preset(FVNetwork);
PetscErrorCode FVNetworkFinalizePartition(FVNetwork);
PetscErrorCode FVNetworkBuildMultirateIS(FVNetwork,IS*,IS*,IS*);

PetscErrorCode FVNetRHS_Buffer(TS,PetscReal,Vec,Vec,void*);
PetscErrorCode FVNetRHS_Multirate(TS,PetscReal,Vec,Vec,void*);