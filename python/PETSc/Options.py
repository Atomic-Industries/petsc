import config.base
import os
import re

class Options(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    return

  def getCFlags(self, compiler, bopt):
    import config.setCompilers

    flags = []
    # GNU gcc
    if config.setCompilers.Configure.isGNU(compiler):
      if bopt == '':
        flags.append('-Wall')
        if 'USER' in os.environ and os.environ['USER'] in ['barrysmith','bsmith','knepley','buschelm','balay','petsc']:
          flags.extend(['-Wshadow', '-Wwrite-strings'])
      elif bopt == 'g':
        flags.append('-g3')
      elif bopt == 'O':
        flags.extend(['-O', '-fomit-frame-pointer', '-Wno-strict-aliasing'])
    # Alpha
    elif re.match(r'alphaev[5-9]', self.framework.host_cpu):
      # Compaq C
      if compiler == 'cc':
        if bopt == 'O':
          flags.extend(['-O2', '-Olimit 2000'])
    # MIPS
    elif re.match(r'mips', self.framework.host_cpu):
      # MIPS Pro C
      if compiler == 'cc':
        if bopt == '':
          flags.extend(['-woff 1164', '-woff 1552', '-woff 1174'])
        elif bopt == 'g':
          flags.append('-g')
        elif bopt == 'O':
          flags.extend(['-O2', '-OPT:Olimit=6500'])
    # Intel
    elif re.match(r'i[3-9]86', self.framework.host_cpu):
      # Linux Intel
      if compiler == 'icc':
        if bopt == 'g':
          flags.append('-g')
        elif bopt == 'O':
          flags.append('-O3')
      # Windows Intel
      elif compiler.find('win32fe icl') >= 0:
        if bopt == '':
          flags.append('-MT')
        elif bopt == 'g':
          flags.append('-Z7')
        elif bopt == 'O':
          flags.extend(['-O3', '-QxW'])
      # Windows Microsoft
      elif compiler.find('win32fe cl') >= 0:
        if bopt == '':
          flags.append('-MT')
        elif bopt == 'g':
          flags.append('-Z7')
        elif bopt == 'O':
          flags.extend(['-O3', '-QxW'])
    # Generic
    if not len(flags):
      if bopt == 'g':
        flags.append('-g')
      elif bopt == 'O':
        flags.append('-O')
    return flags

  def getCxxFlags(self, compiler, bopt):
    import config.setCompilers

    flags = []
    # GNU g++
    if config.setCompilers.Configure.isGNU(compiler):
      if bopt == '':
        flags.append('-Wall')
      elif bopt in ['g', 'g_complex']:
        flags.append('-g3')
      elif bopt in ['O', 'O_complex']:
        if os.environ.has_key('USER'):
          if os.environ['USER'] in ['barrysmith', 'bsmith', 'knepley', 'buschelm', 'petsc', 'balay']:
            flags.extend(['-Wshadow', '-Wwrite-strings', '-Wno-strict-aliasing'])
          flags.extend(['-O', '-fomit-frame-pointer'])
    # Alpha
    elif re.match(r'alphaev[0-9]', self.framework.host_cpu):
      # Compaq C++
      if compiler == 'cxx':
        if bopt in ['O', 'O_complex']:
          flags.append('-O2')
    # MIPS
    elif re.match(r'mips', self.framework.host_cpu):
      # MIPS Pro C++
      if compiler == 'cc':
        if bopt == '':
          flags.extend(['-woff 1164', '-woff 1552', '-woff 1174'])
        elif bopt in ['g', 'g_complex']:
          flags.append('-g')
        elif bopt in ['O', 'O_complex']:
          flags.extend(['-O2', '-OPT:Olimit=6500'])
    # Intel
    elif re.match(r'i[3-9]86', self.framework.host_cpu):
      # Linux Intel
      if compiler == 'icc':
        if bopt == 'g':
          flags.append('-g')
        elif bopt == 'O':
          flags.append('-O3')
      # Windows Intel
      elif compiler.find('win32fe icl') >= 0:
        if bopt == '':
          flags.append('-MT -GX -GR')
        elif bopt in ['g', 'g_complex']:
          flags.append('-Z7')
        elif bopt in ['O', 'O_complex']:
          flags.extend(['-O3', '-QxW'])
      # Windows Microsoft
      elif compiler.find('win32fe cl') >= 0:
        if bopt == '':
          flags.append('-MT -GX -GR')
        elif bopt == 'g':
          flags.append('-Z7')
        elif bopt == 'O':
          flags.extend(['-O2', '-QxW'])
        elif bopt == 'g_complex':
          flags.extend(['-Z7', '-Zm200'])
        elif bopt == 'O_complex':
          flags.extend(['-O2', '-Zm200'])
    # Generic
    if not len(flags):
      if bopt in ['g', 'g_complex']:
        flags.append('-g')
      elif bopt in ['O', 'O_complex']:
        flags.append('-O')
    return flags

  def getFortranFlags(self, compiler, bopt):
    flags = []
    # Alpha
    if re.match(r'alphaev[0-9]', self.framework.host_cpu):
      # Compaq Fortran
      if compiler == 'fort':
        if bopt == 'O':
          flags.append('-O2')
    # Intel
    elif re.match(r'i[3-9]86', self.framework.host_cpu):
      # Portland Group Fortran 90
      if compiler == 'pgf90':
        if bopt == 'O':
          flags.extend(['-fast', '-tp p6', '-Mnoframe'])
      # Linux Intel
      elif compiler in ['ifc', 'ifort']:
        if bopt == 'g':
          flags.append('-g')
        elif bopt == 'O':
          flags.append('-O3')
      # Windows Intel
      elif compiler.find('win32fe ifl') >= 0 or compiler.find('win32fe ifort') >= 0:
        if bopt == '':
          flags.append('-MT')
        elif bopt == 'g':
          flags.append('-Z7')
        elif bopt == 'O':
          flags.extend(['-O3', '-QxW'])
      # Compaq Visual FORTRAN
      elif compiler.find('win32fe f90') >= 0 or compiler.find('win32fe df') >= 0:
        if bopt == '':
          flags.append('-threads')
        elif bopt == 'g':
          flags.extend(['-dbglibs', '-debug:full'])
        elif bopt == 'O':
          flags.extend(['-optimize:5', '-fast'])
        elif bopt == 'g_complex':
          flags.extend(['-dbglibs', '-debug:full'])
        elif bopt == 'O_complex':
          flags.append('-optimize:4')
    # MIPS
    elif re.match(r'mips', self.framework.host_cpu):
      # MIPS Pro Fortran
      if compiler == 'f90':
        if bopt == '':
          flags.append('-cpp')
        elif bopt == 'g':
          flags.extend(['-g', '-trapuv'])
        elif bopt == 'O':
          flags.extend(['-O2', '-IPA:cprop=OFF', '-OPT:IEEE_arithmetic=1'])
    # Generic
    if not len(flags):
      if bopt == 'g':
        flags.append('-g')
      elif bopt == 'O':
        flags.append('-O')
    return flags

  def getCompilerFlags(self, language, compiler, bopt):
    flags = ''
    if language == 'C':
      flags = self.getCFlags(compiler, bopt)
    elif language == 'Cxx':
      flags = self.getCxxFlags(compiler, bopt)
    elif language in ['Fortran', 'F77']:
      flags = self.getFortranFlags(compiler, bopt)
    return flags

  def getCompilerVersion(self, language, compiler):
    if compiler is None:
      raise RuntimeError('Invalid compiler for version determination')
    version = 'Unknown'
    try:
      if language == 'C':
        if re.match(r'alphaev[0-9]', self.framework.host_cpu) and compiler == 'cc':
          flags = '-V'
        elif re.match(r'mips', self.framework.host_cpu) and compiler == 'cc':
          flags = '-version'
        else:
          flags = '--version'
      elif language == 'Cxx':
        if re.match(r'alphaev[0-9]', self.framework.host_cpu) and compiler == 'cxx':
          flags = '-V'
        elif re.match(r'mips', self.framework.host_cpu) and compiler == 'cc':
          flags = '-version'
        else:
          flags = '--version'
      elif language in ['Fortran', 'F77']:
        if re.match(r'alphaev[0-9]', self.framework.host_cpu) and compiler == 'fort':
          flags = '-version'
        elif re.match(r'i[3-9]86', self.framework.host_cpu) and compiler == 'f90':
          flags = '-V'
        elif re.match(r'i[3-9]86', self.framework.host_cpu) and compiler == 'pgf90':
          flags = '-V'
        elif re.match(r'mips', self.framework.host_cpu) and compiler == 'f90':
          flags = '-version'
        else:
          flags = '--version'
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' '+flags, log = self.framework.log)
      if not status:
        if compiler.find('win32fe'):
          version = '\\n'.join(output.split('\n')[0:2])
        else:
          version = output.split('\n')[0]
    except RuntimeError, e:
      self.framework.log.write('Could not determine compiler version: '+str(e))
    return version
