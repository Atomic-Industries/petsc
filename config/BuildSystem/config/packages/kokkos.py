import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = '3.1.01'
    self.versionname      = 'KOKKOS_VERSION'
    self.download         = ['git://https://github.com/kokkos/kokkos.git']
    self.downloaddirnames = ['kokkos']
    self.includes         = ['Kokkos_Macros.hpp']
    self.liblist          = [['libkokkoscontainers.a','libkokkoscore.a']]
    self.functions        = ['']
    self.functionsCxx     = [1,'namespace Kokkos {void initialize(int&,char*[]);}','int one = 1;char* args[1];Kokkos::initialize(one,args);']
    self.cxx              = 1
    self.requirescxx11    = 1
    self.downloadonWindows= 0
    self.hastests         = 1
    self.requiresrpath    = 1
    self.precisions       = ['double']
    return

  def __str__(self):
    output  = config.package.Package.__str__(self)
    if hasattr(self,'system'): output += '  Backend: '+self.system+'\n'
    return output

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('Kokkos', '-with-kokkos-cuda-arch', nargs.ArgString(None, 0, 'One of KEPLER30, KEPLER32, KEPLER35, KEPLER37, MAXWELL50, MAXWELL52, MAXWELL53, PASCAL60, PASCAL61, VOLTA70, VOLTA72, TURING75 (Titan V is Volta), use nvidia-smi'))
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.externalpackagesdir = framework.require('PETSc.options.externalpackagesdir',self)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.flibs           = framework.require('config.packages.flibs',self)
    self.cxxlibs         = framework.require('config.packages.cxxlibs',self)
    self.mathlib         = framework.require('config.packages.mathlib',self)
    self.deps            = [self.mpi,self.blasLapack,self.flibs,self.cxxlibs,self.mathlib]
    self.openmp          = framework.require('config.packages.openmp',self)
    self.pthread         = framework.require('config.packages.pthread',self)
    self.cuda            = framework.require('config.packages.cuda',self)
    self.hwloc           = framework.require('config.packages.hwloc',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.odeps           = [self.openmp,self.hwloc,self.cuda,self.pthread]
    return

  def versionToStandardForm(self,ver):
    '''Converts from kokkos 30101 notation to standard notation 3.1.01'''
    return ".".join(map(str,[int(ver)//10000, int(ver)//100%100, int(ver)%100]))

  # duplicate from Trilinos.py
  def toString(self,string):
    string    = self.libraries.toString(string)
    if self.requiresrpath: return string
    newstring = ''
    for i in string.split(' '):
      if i.find('-rpath') == -1:
        newstring = newstring+' '+i
    return newstring.strip()

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_XSDK_DEFAULTS=YES')
    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=DEBUG')
    else:
      args.append('-DCMAKE_BUILD_TYPE=RELEASE')
      args.append('-DXSDK_ENABLE_DEBUG=NO')

    # Trilinos cmake does not set this variable (as it should) so cmake install does not properly reset the -id and rpath of --prefix installed Trilinos libraries
    args.append('-DCMAKE_INSTALL_NAME_DIR:STRING="'+os.path.join(self.installDir,self.libdir)+'"')

    if self.mpi.found:
      args.append('-DKokkos_ENABLE_MPI=ON')

    if self.hwloc.found:
      args.append('-DKokkos_ENABLE_HWLOC=ON')
      args.append('-DKokkos_HWLOC_DIR='+self.hwloc.directory)

    # looks for pthread by default so need to turn it off unless specifically requested
    pthreadfound = self.pthread.found
    if not 'with-pthread' in self.framework.clArgDB:
      pthreadfound = 0

    if self.openmp.found + pthreadfound + self.cuda.found > 1:
      raise RuntimeError("Kokkos only supports a single parallel system during its configuration")

    args.append('-DKokkos_ENABLE_SERIAL=ON')
    if self.openmp.found:
      args.append('-DKokkos_ENABLE_OPENMP=ON')
      self.system = 'OpenMP'
    if pthreadfound:
      args.append('-DKokkos_ENABLE_PTHREAD=ON')
      self.system = 'PThread'
    if self.cuda.found:
      args.append('-DKokkos_ENABLE_CUDA=ON')
      self.system = 'CUDA'
      self.pushLanguage('CUDA')
      petscNvcc = self.getCompiler()
      cudaFlags = self.getCompilerFlags()
      self.popLanguage()
      args.append('-DKOKKOS_CUDA_OPTIONS="'+cudaFlags.replace(' ',';')+'"')
      # Kokkos must be compiled with its horrible nvcc_wrapper script when using nvcc
      # cannot find way to set nvcc exectuable
      # NVCC_WRAPPER_DEFAULT_COMPILER
      args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_COMPILER=')
      dir = self.externalpackagesdir.dir
      args.append('-DCMAKE_CXX_COMPILER='+os.path.join(dir,'git.kokkos','bin','nvcc_wrapper'))
      if not 'with-kokkos-cuda-arch' in self.framework.clArgDB:
        raise RuntimeError('You must set -with-kokkos-cuda-arch=PASCAL61, VOLTA70, VOLTA72, TURING75 etc.')
      args.append('-DKokkos_ARCH_'+self.argDB['with-kokkos-cuda-arch']+'=ON')
      args.append('-DKokkos_ENABLE_CUDA_LAMBDA:BOOL=ON')
      self.addMakeMacro('KOKKOS_BIN',os.path.join(self.installDir,'bin'))
      #  Kokkos nvcc_wrapper REQUIRES nvcc be visible in the PATH!
      path = os.getenv('PATH')
      nvccpath = os.path.dirname(petscNvcc)
      if nvccpath:
         os.environ['PATH'] = path+':'+nvccpath
    return args
