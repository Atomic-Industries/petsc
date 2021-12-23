import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = '6c786cd6900b977e7b3b19a1f88c0c433a49cbcc' # develop of 2021-12-20
    self.versionname      = 'KOKKOS_KERNELS_VERSION'  # It looks kokkos-kernels does not yet have a macro for version number
    self.download         = ['git://https://github.com/kokkos/kokkos-kernels.git']
    self.includes         = ['KokkosBlas.hpp','KokkosSparse_CrsMatrix.hpp']
    self.liblist          = [['libkokkoskernels.a']]
    self.functions        = ['']
    # I don't know how to make it work since all KK routines are templated and always need Kokkos::View. So I cheat here and use functionCxx from Kokkos.
    self.functionsCxx     = [1,'namespace Kokkos {void initialize(int&,char*[]);}','int one = 1;char* args[1];Kokkos::initialize(one,args);']
    self.buildLanguages   = ['Cxx']
    self.downloadonWindows= 0
    self.hastests         = 1
    self.requiresrpath    = 1
    return

  def __str__(self):
    output  = config.package.CMakePackage.__str__(self)
    if hasattr(self,'system'): output += '  Backend: '+self.system+'\n'
    return output

  def setupHelp(self, help):
    import nargs
    config.package.CMakePackage.setupHelp(self, help)
    help.addArgument('KOKKOS-KERNELS', '-with-kokkos-kernels-tpl=<bool>', nargs.ArgBool(None, 1, 'Indicate if you wish to let Kokkos-Kernels use Third-Party Libraries (TPLs)'))
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.externalpackagesdir = framework.require('PETSc.options.externalpackagesdir',self)
    self.scalarTypes         = framework.require('PETSc.options.scalarTypes',self)
    self.kokkos              = framework.require('config.packages.kokkos',self)
    self.deps                = [self.kokkos]
    self.cuda                = framework.require('config.packages.cuda',self)
    self.hip                 = framework.require('config.packages.hip',self)
    self.sycl                = framework.require('config.packages.sycl',self)
    self.odeps               = [self.cuda,self.hip,self.sycl]
    return

  def versionToStandardForm(self,ver):
    '''Converts from Kokkos kernels 30101 notation to standard notation 3.1.01'''
    return ".".join(map(str,[int(ver)//10000, int(ver)//100%100, int(ver)%100]))

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
    KokkosRoot = self.kokkos.directory
    args.append('-DKokkos_ROOT='+KokkosRoot)
    if self.scalarTypes.scalartype == 'complex':
      if self.scalarTypes.precision == 'double':
        args.append('-DKokkosKernels_INST_COMPLEX_DOUBLE=ON')
      elif self.scalarTypes.precision == 'single':
        args.append('-DKokkosKernels_INST_COMPLEX_FLOAT=ON')

    # By default it installs in lib64, change it to lib
    if self.checkSharedLibrariesEnabled():
      args.append('-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON')
      args.append('-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON')
    if self.cuda.found:
      args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_COMPILER=')
      args.append('-DCMAKE_CXX_COMPILER='+self.getCompiler('Cxx')) # use the host CXX compiler, let Kokkos handle the nvcc_wrapper business
      # as of version 3.2.00 Cuda 11 is not supported, e.g., identifier "cusparseXcsrgemmNnz" is undefined
      if not self.argDB['with-kokkos-kernels-tpl'] or self.cuda.version_tuple >= (11,0):
        args.append('-DKokkosKernels_ENABLE_TPL_CUBLAS=OFF')
        args.append('-DKokkosKernels_ENABLE_TPL_CUSPARSE=OFF')
    elif self.hip.found:
      self.system = 'HIP'
      with self.Language('HIP'):
        petscHipc = self.getCompiler()
        self.getExecutable(petscHipc,getFullPath=1,resultName='systemHipc')
      if not hasattr(self,'systemHipc'):
        raise RuntimeError('HIP error: could not find path of hipc')
      args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_COMPILER=')
      args.append('-DCMAKE_CXX_COMPILER='+self.systemHipc)
    elif self.sycl.found:
      args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_COMPILER=')
      args.append('-DCMAKE_CXX_COMPILER='+self.kokkos.systemSyclc)

    # -DCMAKE_CXX_STANDARD= will be taken from Kokkos
    args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_STANDARD=')
    return args

  def configureLibrary(self):
    self.buildLanguages= self.kokkos.buildLanguages
    config.package.CMakePackage.configureLibrary(self)
