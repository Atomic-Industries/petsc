import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.minversion       = '2.0'
    self.versionname      = 'HIP_VERSION'
    # Does not seem to include version
    #self.versioninclude  = 'hip/hip_runtime.h'
    #self.requiresversion = 2
    self.functions        = ['hipblasCreate']
    # hipfft and hipsolver aren't available really (hipfft is close).
    #self.includes        = ['hipblas.h','hipfft.h','hipsparse.h','hipsolver.h']
    #self.liblist         = [['libhipblas.a','libhiprtc.a','libhipsparse.a','libhipsolver.a'],
    #                         ['hipfft.lib','hipblas.lib','hiprtc.lib','hipsparse.lib','hipsolver.lib']]
    self.includes         = ['hipblas.h','hipsparse.h']
    self.liblist          = [['libhipsparse.a','libhipblas.a','librocsparse.a','librocblas.a','libamdhip64.a'],
                             ['hipsparse.lib','hipblas.lib','rocsparse.lib','rocblas.lib','amdhip64.lib'],]
    #self.liblist          = [['libhipsparse.a','libhipblas.a','libhiprtc.a'],
    #                         ['hipsparse.lib','hipblas.lib','hiprtc.lib']]
    self.precisions       = ['single','double']
    self.cxx              = 1
    self.complex          = 1
    self.hastests         = 0
    self.hastestsdatafiles= 0
    # Handle the platform issues
    if 'HIP_PLATFORM' in os.environ:
      self.platform = os.environ['HIP_PLATFORM']
    elif hasttr('CUDA', config.compile):
      self.platform = 'nvcc'
    else:
      self.platform = 'hcc'

    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers',self)
    self.headers      = framework.require('config.headers',self)
    return

  def getSearchDirectories(self):
    import os
    self.pushLanguage('HIP')
    petscHip = self.getCompiler()
    self.popLanguage()
    self.getExecutable(petscHip,getFullPath=1,resultName='systemHipcc')
    if hasattr(self,'systemHipcc'):
      hipccDir = os.path.dirname(self.systemHipcc)
      hipDir = os.path.split(hipccDir)[0]
      yield hipDir
    return

  def checkSizeofVoidP(self):
    '''Checks if the HIPCC compiler agrees with the C compiler on what size of void * should be'''
    self.log.write('Checking if sizeof(void*) in HIP is the same as with regular compiler\n')
    size = self.types.checkSizeof('void *', (8, 4), lang='HIP', save=False)
    if size != self.types.sizes['void-p']:
      raise RuntimeError('HIP Error: sizeof(void*) with HIP compiler is ' + str(size) + ' which differs from sizeof(void*) with C compiler')
    return

  def configureTypes(self):
    import config.setCompilers
    if not self.getDefaultPrecision() in ['double', 'single']:
      raise RuntimeError('Must use either single or double precision with HIP')
    self.checkSizeofVoidP()
    return

  def checkHIPCCDoubleAlign(self):
    if 'known-hip-align-double' in self.argDB:
      if not self.argDB['known-hip-align-double']:
        raise RuntimeError('HIP error: PETSC currently requires that HIP double alignment match the C compiler')
    else:
      typedef = 'typedef struct {double a; int b;} teststruct;\n'
      hip_size = self.types.checkSizeof('teststruct', (16, 12), lang='HIP', codeBegin=typedef, save=False)
      c_size = self.types.checkSizeof('teststruct', (16, 12), lang='C', codeBegin=typedef, save=False)
      if c_size != hip_size:
        raise RuntimeError('HIP compiler error: memory alignment doesn\'t match C compiler (try adding -malign-double to compiler options)')
    return

  def configureLibrary(self):
    self.setCompilers.pushLanguage('HIP')
    self.addDefine('HAVE_HIP','1')
    # May need more checks/defines/work here
    if self.platform == 'nvcc': 
        self.pushLanguage('CUDA')
        petscNvcc = self.getCompiler()
        cudaFlags = self.getCompilerFlags()
        self.popLanguage()
        self.getExecutable(petscNvcc,getFullPath=1,resultName='systemNvcc')
        if hasattr(self,'systemNvcc'):
          nvccDir = os.path.dirname(self.systemNvcc)
          cudaDir = os.path.split(nvccDir)[0]
        else:
          raise RuntimeError('Unable to locate CUDA NVCC compiler')
        self.includedir = ['include',os.path.join(cudaDir,'include')]
        self.delDefine('HAVE_CUDA')
        self.addDefine('HAVE_HIPCUDA',1)
    else:
        self.addDefine('HAVE_HIPROCM',1)
    self.setCompilers.popLanguage()

    config.package.Package.configureLibrary(self)
    #self.checkHIPDoubleAlign()
    self.configureTypes()
    return
