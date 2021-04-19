from __future__ import absolute_import
import logger

import os
try:
  from urllib import urlretrieve
except ImportError:
  from urllib.request import urlretrieve
try:
  import urlparse as urlparse_local # novermin
except ImportError:
  from urllib import parse as urlparse_local
import config.base
import socket
import shutil

# Fix parsing for nonstandard schemes
urlparse_local.uses_netloc.extend(['bk', 'ssh', 'svn'])

class Retriever(logger.Logger):
  def __init__(self, sourceControl, clArgs = None, argDB = None):
    logger.Logger.__init__(self, clArgs, argDB)
    self.sourceControl = sourceControl
    self.stamp = None
    return

  def isDirectoryGitRepo(self, directory):
    from config.base import Configure
    for loc in ['.git','']:
      cmd = '%s rev-parse --resolve-git-dir  %s'  % (self.sourceControl.git, os.path.join(directory,loc))
      (output, error, ret) = Configure.executeShellCommand(cmd, checkCommand = Configure.passCheckCommand, log = self.log)
      if not ret:
        return True
    return False

  @staticmethod
  def removeTarget(t):
    if os.path.islink(t) or os.path.isfile(t):
      os.unlink(t) # same as os.remove(t)
    elif os.path.isdir(t):
      shutil.rmtree(t)

  @staticmethod
  def getDownloadFailureMessage(package, url, filename=None):
    slashFilename = '/'+filename if filename else ''
    return '''\
Unable to download package %s from: %s
* If URL specified manually - perhaps there is a typo?
* If your network is disconnected - please reconnect and rerun ./configure
* Or perhaps you have a firewall blocking the download
* You can run with --with-packages-download-dir=/adirectory and ./configure will instruct you what packages to download manually
* or you can download the above URL manually, to /yourselectedlocation%s
  and use the configure option:
  --download-%s=/yourselectedlocation%s
    ''' % (package.upper(), url, slashFilename, package, slashFilename)

  @staticmethod
  def removePrefix(url,prefix):
    '''Replacement for str.removeprefix() supported only since Python 3.9'''
    if url.startswith(prefix):
      return url[len(prefix):]
    return url

  def genericRetrieve(self, url, root, package):
    '''Fetch the gzipped tarfile indicated by url and expand it into root
       - All the logic for removing old versions, updating etc. must move'''

    # copy a directory
    if url.startswith('dir://'):
      import shutil
      dir = url[6:]
      if not os.path.isdir(dir): raise RuntimeError('Url begins with dir:// but is not a directory')

      if os.path.isdir(os.path.join(root,os.path.basename(dir))): shutil.rmtree(os.path.join(root,os.path.basename(dir)))
      if os.path.isfile(os.path.join(root,os.path.basename(dir))): os.unlink(os.path.join(root,os.path.basename(dir)))

      shutil.copytree(dir,os.path.join(root,os.path.basename(dir)))
      return

    if url.startswith('link://'):
      import shutil
      dir = url[7:]
      if not os.path.isdir(dir): raise RuntimeError('Url begins with link:// but it is not pointing to a directory')

      if os.path.islink(os.path.join(root,os.path.basename(dir))): os.unlink(os.path.join(root,os.path.basename(dir)))
      if os.path.isfile(os.path.join(root,os.path.basename(dir))): os.unlink(os.path.join(root,os.path.basename(dir)))
      if os.path.isdir(os.path.join(root,os.path.basename(dir))): shutil.rmtree(os.path.join(root,os.path.basename(dir)))
      os.symlink(os.path.abspath(dir),os.path.join(root,os.path.basename(dir)))
      return

    if url.startswith('git://'):
      if not hasattr(self.sourceControl, 'git'): return
      import shutil
      dir = url[6:]
      if os.path.isdir(dir):
        if not os.path.isdir(os.path.join(dir,'.git')): raise RuntimeError('Url begins with git:// and is a directory but but does not have a .git subdirectory')

      newgitrepo = os.path.join(root,'git.'+package)
      if os.path.isdir(newgitrepo): shutil.rmtree(newgitrepo)
      if os.path.isfile(newgitrepo): os.unlink(newgitrepo)

      try:
        config.base.Configure.executeShellCommand(self.sourceControl.git+' clone --recursive '+dir+' '+newgitrepo, log = self.log)
      except  RuntimeError as e:
        self.logPrint('ERROR: '+str(e))
        err = str(e)
        failureMessage = '''\
Unable to download package %s from: %s
* If URL specified manually - perhaps there is a typo?
* If your network is disconnected - please reconnect and rerun ./configure
* Or perhaps you have a firewall blocking the download
* You can run with --with-packages-download-dir=/adirectory and ./configure will instruct you what packages to download manually
* or you can download the above URL manually, to /yourselectedlocation
  and use the configure option:
  --download-%s=/yourselectedlocation
''' % (package.upper(), url, package)
        raise RuntimeError('Unable to download '+package+'\n'+err+failureMessage)
      return

    if url.startswith('hg://'):
      if not hasattr(self.sourceControl, 'hg'): return

      newgitrepo = os.path.join(root,'hg.'+package)
      if os.path.isdir(newgitrepo): shutil.rmtree(newgitrepo)
      if os.path.isfile(newgitrepo): os.unlink(newgitrepo)
      try:
        config.base.Configure.executeShellCommand(self.sourceControl.hg+' clone '+url[5:]+' '+newgitrepo)
      except  RuntimeError as e:
        self.logPrint('ERROR: '+str(e))
        err = str(e)
        failureMessage = '''\
Unable to download package %s from: %s
* If URL specified manually - perhaps there is a typo?
* If your network is disconnected - please reconnect and rerun ./configure
* Or perhaps you have a firewall blocking the download
* You can run with --with-packages-download-dir=/adirectory and ./configure will instruct you what packages to download manually
* or you can download the above URL manually, to /yourselectedlocation
  and use the configure option:
  --download-%s=/yourselectedlocation
''' % (package.upper(), url, package)
        raise RuntimeError('Unable to download '+package+'\n'+err+failureMessage)
      return

    if url.startswith('ssh://hg@'):
      if not hasattr(self.sourceControl, 'hg'): return

      newgitrepo = os.path.join(root,'hg.'+package)
      if os.path.isdir(newgitrepo): shutil.rmtree(newgitrepo)
      if os.path.isfile(newgitrepo): os.unlink(newgitrepo)
      try:
        config.base.Configure.executeShellCommand(self.sourceControl.hg+' clone '+url+' '+newgitrepo)
      except  RuntimeError as e:
        self.logPrint('ERROR: '+str(e))
        err = str(e)
        failureMessage = '''\
Unable to download package %s from: %s
* If URL specified manually - perhaps there is a typo?
* If your network is disconnected - please reconnect and rerun ./configure
* Or perhaps you have a firewall blocking the download
* You can run with --with-packages-download-dir=/adirectory and ./configure will instruct you what packages to download manually
* or you can download the above URL manually, to /yourselectedlocation
  and use the configure option:
  --download-%s=/yourselectedlocation
''' % (package.upper(), url, package)
        raise RuntimeError('Unable to download '+package+'\n'+err+failureMessage)
      return

    # get the tarball file name from the URL
    filename = os.path.basename(urlparse_local.urlparse(url)[2])
    localFile = os.path.join(root,'_d_'+filename)
    self.logPrint('Retrieving %s as tarball to %s' % (url,localFile) , 3, 'install')
    ext =  os.path.splitext(localFile)[1]
    if ext not in ['.bz2','.tbz','.gz','.tgz','.zip','.ZIP']:
      raise RuntimeError('Unknown compression type in URL: '+ url)

    self.removeTarget(localFile)

    if parsed[0] == 'file' and not parsed[1]:
      url = parsed[2]
    if os.path.exists(url):
      if not os.path.isfile(url):
        raise RuntimeError('Local path exists but is not a regular file: '+ url)
      # copy local file
      shutil.copyfile(url, localFile)
    else:
      # fetch remote file
      try:
        sav_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(30)
        urlretrieve(url, localFile)
        socket.setdefaulttimeout(sav_timeout)
      except Exception as e:
        socket.setdefaulttimeout(sav_timeout)
        failureMessage = self.getDownloadFailureMessage(package, url, filename)
        raise RuntimeError(failureMessage)

    self.logPrint('Extracting '+localFile)
    if ext in ['.zip','.ZIP']:
      config.base.Configure.executeShellCommand('cd '+root+'; unzip '+localFile, log = self.log)
      output = config.base.Configure.executeShellCommand('cd '+root+'; zipinfo -1 '+localFile+' | head -n 1', log = self.log)
      dirname = os.path.normpath(output[0].strip())
    else:
      failureMessage = '''\
Downloaded package %s from: %s is not a tarball.
[or installed python cannot process compressed files]
* If you are behind a firewall - please fix your proxy and rerun ./configure
  For example at LANL you may need to set the environmental variable http_proxy (or HTTP_PROXY?) to  http://proxyout.lanl.gov
* You can run with --with-packages-download-dir=/adirectory and ./configure will instruct you what packages to download manually
* or you can download the above URL manually, to /yourselectedlocation/%s
  and use the configure option:
  --download-%s=/yourselectedlocation/%s
''' % (package.upper(), url, filename, package, filename)
      import tarfile
      try:
        tf  = tarfile.open(os.path.join(root, localFile))
      except tarfile.ReadError as e:
        raise RuntimeError(str(e)+'\n'+failureMessage)
      if not tf: raise RuntimeError(failureMessage)
      #git puts 'pax_global_header' as the first entry and some tar utils process this as a file
      firstname = tf.getnames()[0]
      if firstname == 'pax_global_header':
        firstmember = tf.getmembers()[1]
      else:
        firstmember = tf.getmembers()[0]
      # some tarfiles list packagename/ but some list packagename/filename in the first entry
      if firstmember.isdir():
        dirname = firstmember.name
      else:
        dirname = os.path.dirname(firstmember.name)
      tf.extractall(root)
      tf.close()

    # fix file permissions for the untared tarballs.
    try:
      # check if 'dirname' is set'
      if dirname:
        config.base.Configure.executeShellCommand('cd '+root+'; chmod -R a+r '+dirname+';find  '+dirname + ' -type d -name "*" -exec chmod a+rx {} \;', log = self.log)
      else:
        self.logPrintBox('WARNING: Could not determine dirname extracted by '+localFile+' to fix file permissions')
    except RuntimeError as e:
      raise RuntimeError('Error changing permissions for '+dirname+' obtained from '+localFile+ ' : '+str(e))
    os.unlink(localFile)
