import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = '5dc20f1424206f2a09b001e2585fe5c794e60dbf'
    self.giturls          = ['https://github.com/elemental/Elemental']
    self.download         = ['http://libelemental.org/pub/releases/Elemental-0.85.tgz']
    self.liblist          = [['libEl.a','libpmrrr.a']]
    self.includes         = ['El.hpp']
    self.cxx              = 1
    self.requirescxx11    = 1
    self.downloadonWindows= 0
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.deps            = [self.mpi,self.blasLapack]
    self.parmetis        = framework.require('config.packages.parmetis',self)
    self.metis           = framework.require('config.packages.metis',self)
    #
    # also requires the ./configure option --with-cxx-dialect=C++11
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DMATH_LIBS:STRING="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    # temporary patch
    args.append('-DINSTALL_PYTHON_PACKAGE=OFF -DBUILD_KISSFFT=OFF')
    self.framework.pushLanguage('C')
    args.append('-DMPI_C_COMPILER="'+self.framework.getCompiler()+'"')
    if self.framework.argDB['with-64-bit-indices']:
      args.append('-DEL_USE_64BIT_INTS=ON')
    self.framework.popLanguage()

    self.framework.pushLanguage('Cxx')
    if config.setCompilers.Configure.isSolaris():
       raise RuntimeError('Sorry, Elemental does not compile with Oracle/Solaris/Sun compilers')
    args.append('-DMPI_CXX_COMPILER="'+self.framework.getCompiler()+'"')
    self.framework.popLanguage()

    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('-DMPI_Fortran_COMPILER="'+self.framework.getCompiler()+'"')
      self.framework.popLanguage()

    # how to check for GKLIB?
    if self.metis.found:
      args.append('-DBUILD_METIS=OFF -DMANUAL_METIS=ON -DMETIS_ROOT="'+self.metis.installDir+'"')
    else:
      args.append('-DBUILD_METIS=OFF -DMANUAL_METIS=OFF')
    if self.parmetis.found:
      args.append('-DBUILD_PARMETIS=OFF -DMANUAL_PARMETIS=ON -DPARMETIS_ROOT="'+self.parmetis.installDir+'"')
    else:
      args.append('-DBUILD_PARMETIS=OFF -DMANUAL_PARMETIS=OFF')

    return args




