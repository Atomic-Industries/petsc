
static char help[] = "Tests options database monitoring and precedence.\n\n";

#include <petscsys.h>
#include <petscviewer.h>

PetscErrorCode PetscOptionsMonitorCustom(const char name[],const char value[],void *ctx)
{
  PetscViewer    viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  if (!value) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"* Removing option: %s\n",name));
  } else if (!value[0]) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"* Setting option: %s (no value)\n",name));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"* Setting option: %s = %s\n",name,value));
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscViewer       viewer=NULL;
  PetscViewerFormat format;

  CHKERRQ(PetscInitialize(&argc,&argv,"ex55options",help));
  CHKERRQ(PetscOptionsInsertString(NULL,"-option1 1 -option2 -option3 value3"));
  CHKERRQ(PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-options_monitor_viewer",&viewer,&format,NULL));
  if (viewer) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(PetscOptionsMonitorSet(PetscOptionsMonitorCustom,viewer,NULL));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  CHKERRQ(PetscOptionsInsertString(NULL,"-option4 value4 -option5"));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      localrunfiles: ex55options .petscrc petscrc
      filter: egrep -v -e "(malloc|nox|display|saws_port|vecscatter|options_left|check_pointer_intensity|cuda_initialize|error_output_stdout|use_gpu_aware_mpi|checkstack)"
      args: -options_left 0 -options_view -options_monitor_viewer ascii
      args: -skip_petscrc {{0 1}separate output} -options_monitor_cancel {{0 1}separate output}
      test:
        suffix: 1
      test:
        suffix: 2
        args: -options_monitor
      test:
        suffix: 3
        args: -options_monitor -option_cmd_1 option_cmd_1_val -option_cmd_2
   test:
      # test effect of -skip_petscrc in ex55options file
      suffix: 4
      localrunfiles: ex55options .petscrc petscrc
      filter: egrep -v -e "(malloc|nox|display|saws_port|vecscatter|options_left|check_pointer_intensity|cuda_initialize|error_output_stdout|use_gpu_aware_mpi|checkstack)"
      args: -options_left 0 -options_view -options_monitor
   testset:
      # test -help / -help intro / -version from command line
      localrunfiles: ex55options .petscrc petscrc
      filter: egrep -e "(version|help|^See)"
      args: -options_left -options_view -options_monitor
      test:
        suffix: 5a
        args: -help
      test:
        suffix: 5b
        args: -help intro
      test:
        suffix: 5c
        args: -version
   testset:
      # test -help / -help intro / -version from file
      localrunfiles: ex55options rc_help rc_help_intro rc_version
      filter: egrep -e "(version|help|^See)"
      args: -skip_petscrc
      args: -options_left -options_view -options_monitor
      test:
        suffix: 6a
        args: -options_file rc_help
      test:
        suffix: 6b
        args: -options_file rc_help_intro
      test:
        suffix: 6c
        args: -options_file rc_version

TEST*/
