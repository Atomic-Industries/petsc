#!/bin/sh
# $Id: solid.make,v 1.12 1998/06/08 16:52:36 curfman Exp curfman $ 

# Defaults
hme="/home/petsc/petsc-2.0.22"
src_dir=""
action="lib"

# process the command line arguments
for arg in "$@" ; do
#    echo procs sing arg $arg
    case "$arg" in 
        -echo)
        set -x
        ;;

        -help | -h)
        echo "Description: "
        echo " This program is used to build petsc.solid libraries on the variety"
        echo " of platforms on which it is built."
        echo " "
        echo "Options:"
        echo "  PETSC_DIR=petsc_dir : the current installation of petsc"
        echo "  SRC_DIR=src_dir     : the petsc src dir where make should be invoked"
        echo "  ACTION=action       : defaults to \"lib\" "
        echo " "
        echo "Example Usage:"
        echo "  - To update the libraries with changes in src/sles/interface"
        echo "  solid.make PETSC_DIR=/home/petsc/petsc-2.0.22 SRC_DIR=src/sles/interface ACTION=lib"
        echo "  - To rebuild a new version of PETSC on all the machines"
        echo "  solid.make PETSC_DIR=/home/petsc/petsc-2.0.22 SRC_DIR=\"\" ACTION=\"all\" "
        echo " "
        echo "Defaults:"
        echo "  PETSC_DIR=$hme SRC_DIR=$src_dir ACTION=$action"
        echo " "
        echo "Notes:"
        echo " To avoid problems with file permissions, this script is restricted"
        echo " to be run by the user petsc"
        exit 1
        ;;

        PETSC_DIR=*)
        hme=`echo $arg|sed 's/PETSC_DIR=//g'`
        ;;

        SRC_DIR=*)
        src_dir=`echo $arg|sed 's/SRC_DIR=//g'`
        ;;

        ACTION=*)
        action=`echo $arg|sed 's/ACTION=//g'`
        ;;

        *) 
        echo " ignoring option $arg"
        ;;
    esac
done

user=`whoami`
if [ ${user} != petsc ]; then
    echo 'Run this script as user petsc'
    exit
fi

set -x

# solaris
arch=solaris
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action shared"
rsh -n fire "cd $hme/$src_dir; $make BOPT=g"
rsh -n fire "cd $hme/$src_dir; $make BOPT=O"
rsh -n fire "cd $hme/$src_dir; $make BOPT=g_c++"
#rsh -n fire "cd $hme/$src_dir; $make BOPT=O_c++"
rsh -n fire "cd $hme/$src_dir; $make BOPT=g_complex"
rsh -n fire "cd $hme/$src_dir; $make BOPT=O_complex"

arch=IRIX64
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action shared"
rsh -n yukon "cd $hme/$src_dir; $make BOPT=g"
rsh -n yukon "cd $hme/$src_dir; $make BOPT=O"
rsh -n yukon "cd $hme/$src_dir; $make BOPT=g_complex"
rsh -n yukon "cd $hme/$src_dir; $make BOPT=O_complex"

# rs6000
arch=rs6000
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action shared"
rsh -n ico09 "cd $hme/$src_dir; $make BOPT=g"
rsh -n ico09 "cd $hme/$src_dir; $make BOPT=O"
rsh -n ico09 "cd $hme/$src_dir; $make BOPT=g_c++"
rsh -n ico09 "cd $hme/$src_dir; $make BOPT=O_c++"
rsh -n ico09 "cd $hme/$src_dir; $make BOPT=g_complex"
rsh -n ico09 "cd $hme/$src_dir; $make BOPT=O_complex"

# IRIX
arch=IRIX
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action shared"
#make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action"
rsh -n violet "cd $hme/$src_dir; $make BOPT=g"
rsh -n violet "cd $hme/$src_dir; $make BOPT=O"

# sun4
arch=sun4
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action shared"
rsh -n merlin "cd $hme/$src_dir; $make BOPT=g"
rsh -n merlin "cd $hme/$src_dir; $make BOPT=O"
rsh -n merlin "cd $hme/$src_dir; $make BOPT=g_c++"


# rs6000_p4
arch=rs6000_p4
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action shared"
#rsh -n doc "cd $hme/$src_dir; $make BOPT=g"
#rsh -n doc "cd $hme/$src_dir; $make BOPT=O"

# rs6000_shmem is used by Tom Canfeild on octa nodes
arch=rs6000_shmem
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action shared"
rsh -n octa01 "cd $hme/$src_dir; $make BOPT=O"
