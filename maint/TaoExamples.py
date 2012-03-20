import os
import sys
import subprocess
import signal
import string

def findPetscVariable(key):
    import re
    value = None
    configfile = os.path.join(os.environ['PETSC_DIR'],os.environ['PETSC_ARCH'],'conf','petscvariables')
    if os.access(configfile,os.R_OK):
        f = open(configfile)
        lines = f.readlines()
        for l in lines:
            m = re.match(key+'\s*=\s*(\S+)',l)
            if m and len(m.groups()):
                value = m.groups()[0]
    return value
            


class Example:
    def __init__(self,example,nprocs=1,options="",method=None,tags=[],name="",section=None):
        self.example=example
        self.nprocs=nprocs
        self.options=options.split(" ")
        self.method=method
        self.tags=tags
        self.name=name
        self.section=section

    def executableName(self,version=2):
        if version==1 or self.section is not None:
            return self.example
        elif version==2:
            return 'test_'+self.example
        else:
            return "Bad TAO version (%d)" % version

    def buildCommand(self,version=2):
        if version ==1 or version==2:
            return ['make', self.executableName(version)]

        
    def runCommand(self,version=2):
        if "single" in self.tags:
            c=[]
        else:
            mpiexec = findPetscVariable('MPIEXEC')
            if mpiexec is None:
                mpiexec = "mpiexec"
            c = [mpiexec,"-np","%s" %self.nprocs]
        c.extend( [os.path.join('.',self.executableName())])
        c.extend(self.options)
        c.extend(['-tao_method','tao_'+self.method])
        return c

    def getTao1Directory(self):
        section=""
        for t in ["unconstrained","bound"]:
            if self.hasTag(t):
                section=t
                break
        if t:
            return os.path.join("src",t,"examples","tutorials")
        if self.hasTag("meshrefinement"):
            return os.path.join("src","petsctao","gridapplication","examples")
            

    def hasTag(self,tag):
        return (tag in self.tags)
    def hasAllTags(self,tlist):
        if (tlist is None or len(tlist)==0):
            return True
        if len(tlist)==1:
            return self.hasTag(tlist[0])
        return (self.hasTag(tlist[0]) and self.hasAllTags(tlist[1:]))
    def hasNoTags(self,tlist):
        if (tlist is None or len(tlist)==0):
            return True
        if (len(tlist)==1):
            return not self.hasTag(tlist[0])
        else:
            return (not self.hasTag(tlist[0]) and self.hasNoTags(tlist[1:]))

class ExampleList:
    def __init__(self):
        self.list = []

    def get_children(self,pid):
        p = subprocess.Popen('ps --no-heading -o pid --ppid %d' % pid, shell = True,
              stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        stdout, stderr = p.communicate()
        pidlist = []
        for child in [int(p) for p in stdout.split()]:
            pidlist.append(child)
            pidlist.extend(self.get_children(child))
        return pidlist

    def execute(self,command, cwd = None, echo=False,timeout = 60):
        class Alarm(Exception):
            pass
        def alarm_handler(sgnl, frame):
            raise Alarm
        if (echo):
            sys.stdout.write(string.join(command," ")+"\n")
        sys.stdout.flush()
        try:
            p = subprocess.Popen(command, shell = False, cwd = cwd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        except OSError, e:
            sys.stderr.write("Could not execute '%s':\n  %s\n" % (string.join(command),str(e)))
            return (-1,"","")
        if timeout > 0:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(timeout)
        try:
            stdout, stderr = p.communicate()
            if timeout > 0:
                signal.alarm(0)
        except Alarm:
            pids = [p.pid]
            pids.extend(self.get_children(p.pid))
            for pid in pids:
                os.kill(pid, signal.SIGKILL)
            return -9, p.stdout.read(), 'ERROR: %s timed out after %d seconds\n' % (string.join(command," "),timeout)
        return p.returncode, stdout, stderr





    def add(self,example):
        if example not in self.list:
            self.list.append(example)
    
    def withTag(self,tag):
        l = []
        for e in self.list:
            if e.hasTag(tag):
                l.append(e)
        return l

    def setWithTags(self,taglist):
        negtags = []
        postags = []
        retlist = []
        
        if len(taglist) == 0: # No inclusion tags defaults to entire list
            retlist = self.list[:]
            return

        # First check for names
        for t in taglist:
            for e in self.list:
                if e.name == t:
                    retlist.append(e)
        if len(retlist)>0:
            self.list = retlist
            return
            
        for t in taglist:
            if t.startswith('-'):
                negtags.append(t[1:])
            else:
                postags.append(t)

        for e in self.list:
            if e.hasAllTags(postags) and e.hasNoTags(negtags):
                retlist.append(e)

        self.list = retlist

class TaoExamples(ExampleList):
    """
    Tests to see if installation is ok.  Less thorough than TaoTests
    """
    def __init__(self):
        self.list = []
        self.add(Example(example="minsurf1",nprocs=1,options="-tao_smonitor -mx 10 -my 8",method="nls",tags=["minsurf","single","unconstrained","c","nls"],name="minsurf1",section="unconstrained"))
        self.add(Example(example="minsurf2",nprocs=2,options="-tao_smonitor",method="nls",tags=["minsurf","multiprocessor","unconstrained","dm","c","nls"],name="minsurf2_2",section="unconstrained"))
        self.add(Example(example="rosenbrock1",nprocs=1,options="-tao_smonitor",method="ntr",tags=["rosenbrock","single","unconstrained","c","ntr"],name="rosenbrock1_3",section="unconstrained"))
        self.add(Example(example="rosenbrock1f",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["rosenbrock","single","unconstrained","fortran","lmvm"],name="rosenbrock1f",section="unconstrained"))
        self.add(Example(example="eptorsion2",nprocs=1,options="-tao_smonitor",method="nls",tags=["eptorsion","single","unconstrained","c","nls","dm"],name="eptorsion2",section="unconstrained"))
        self.add(Example(example="eptorsion2f",nprocs=2,options="-tao_smonitor",method="nls",tags=["eptorsion","multiprocessor","unconstrained","fortran","nls","dm"],name="eptorsion2f_2",section="unconstrained"))
        self.add(Example(example="plate2",nprocs=3,options="-tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type mask",method="tron",tags=["bound","plate","multiprocessor","c","tron","dm"],name="plate2_4",section="bound"))
        self.add(Example(example="plate2f",nprocs=1,options="-tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2",method="blmvm",tags=["bound","plate","single","fortran","tron","dm"],name="plate2f",section="bound"))

class TaoTests(ExampleList):
    def __init__(self):
        self.list = []

            
                 
        # Unconstrained
        self.add(Example(example="minsurf1",nprocs=1,options="-tao_smonitor -mx 10 -my 8",method="nls",tags=["minsurf","single","unconstrained","c","nls"],name="minsurf1",section="unconstrained"))
        self.add(Example(example="minsurf2",nprocs=1,options="-tao_smonitor -mx 10 -my 8",method="lmvm",tags=["minsurf","single","unconstrained","dm","c","lmvm"],name="minsurf2",section="unconstrained"))
        self.add(Example(example="minsurf2",nprocs=2,options="-tao_smonitor",method="nls",tags=["minsurf","multiprocessor","unconstrained","dm","c","nls"],name="minsurf2_2",section="unconstrained"))
        self.add(Example(example="minsurf2",nprocs=3,options="-tao_smonitor -mx 10 -my 10 -tao_cg_type fr",method="cg",tags=["minsurf","multiprocessor","unconstrained","dm","c","cg"],name="minsurf2_3",section="unconstrained"))
        self.add(Example(example="minsurf2",nprocs=2,options="-tao_smonitor -my 6 -my 8",method="ntr",tags=["minsurf","multiprocessor","unconstrained","dm","c","ntr"],name="minsurf2_4",section="unconstrained"))
        self.add(Example(example="minsurf2",nprocs=3,options="-tao_smonitor -my 23 -my 17",method="nls",tags=["minsurf","multiprocessor","unconstrained","dm","c","nls"],name="minsurf2_5",section="unconstrained"))
        self.add(Example(example="minsurf2",nprocs=1,options="-tao_smonitor -mx 4 -my 20 -random 2",method="ntr",tags=["minsurf","single","unconstrained","dm","c","ntr"],name="minsurf2_6",section="unconstrained"))


        self.add(Example(example="rosenbrock1",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["rosenbrock","single","unconstrained","c","lmvm"],name="rosenbrock1",section="unconstrained"))
        self.add(Example(example="rosenbrock1",nprocs=1,options="-tao_smonitor",method="ntr",tags=["rosenbrock","single","unconstrained","c","ntr"],name="rosenbrock1_3",section="unconstrained"))
        self.add(Example(example="rosenbrock1f",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["rosenbrock","single","unconstrained","fortran","lmvm"],name="rosenbrock1f",section="unconstrained"))

        self.add(Example(example="limit_feval",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["rosenbrock","single","unconstrained","c","lmvm"],name="limit_feval"))
        self.add(Example(example="limit_minf",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["rosenbrock","single","unconstrained","c","lmvm"],name="limit_minf"))
        self.add(Example(example="limit_iter",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["rosenbrock","single","unconstrained","c","lmvm"],name="limit_iter"))
        self.add(Example(example="limit_fevalf",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["rosenbrock","single","unconstrained","fortran","lmvm"],name="limit_fevalf"))
        self.add(Example(example="limit_minff",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["rosenbrock","single","unconstrained","fortran","lmvm"],name="limit_minff"))
        self.add(Example(example="limit_iterf",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["rosenbrock","single","unconstrained","fortran","lmvm"],name="limit_iterf"))

        self.add(Example(example="eptorsion1",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["eptorsion","single","unconstrained","c","lmvm"],name="eptorsion1",section="unconstrained"))
        self.add(Example(example="eptorsion1",nprocs=1,options="-tao_smonitor",method="nls",tags=["eptorsion","single","unconstrained","c","nls"],name="eptorsion1_2",section="unconstrained"))
        self.add(Example(example="eptorsion1",nprocs=1,options="-tao_smonitor -tao_cg_type prp",method="cg",tags=["eptorsion","single","unconstrained","c","cg"],name="eptorsion1_3",section="unconstrained"))
        self.add(Example(example="eptorsion1",nprocs=1,options="-tao_smonitor",method="ntr",tags=["eptorsion","single","unconstrained","c","ntr"],name="eptorsion1_4",section="unconstrained"))
        self.add(Example(example="eptorsion2",nprocs=1,options="-tao_smonitor",method="nls",tags=["eptorsion","single","unconstrained","c","nls","dm"],name="eptorsion2",section="unconstrained"))
        self.add(Example(example="eptorsion2",nprocs=2,options="-tao_smonitor",method="nls",tags=["eptorsion","multiprocessor","unconstrained","c","nls","dm"],name="eptorsion2_2",section="unconstrained"))
        self.add(Example(example="eptorsion2",nprocs=1,options="-tao_smonitor",method="ntr",tags=["eptorsion","multiprocessor","unconstrained","c","ntr","dm"],name="eptorsion2_3",section="unconstrained"))
        self.add(Example(example="eptorsion2",nprocs=3,options="-tao_smonitor -mx 16 -my 16",method="ntr",tags=["eptorsion","multiprocessor","unconstrained","c","ntr","dm"],name="eptorsion2_4",section="unconstrained"))
        self.add(Example(example="eptorsion2",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["eptorsion","single","unconstrained","c","lmvm","dm"],name="eptorsion2_5",section="unconstrained"))
        self.add(Example(example="eptorsion2",nprocs=3,options="-tao_smonitor -mx 16 -my 16",method="lmvm",tags=["eptorsion","multiprocessor","unconstrained","c","lmvm","dm"],name="eptorsion2_6",section="unconstrained"))
        self.add(Example(example="eptorsion2f",nprocs=1,options="-tao_smonitor",method="nls",tags=["eptorsion","single","unconstrained","fortran","nls","dm"],name="eptorsion2f",section="unconstrained"))
        self.add(Example(example="eptorsion2f",nprocs=2,options="-tao_smonitor",method="nls",tags=["eptorsion","multiprocessor","unconstrained","fortran","nls","dm"],name="eptorsion2f_2",section="unconstrained"))
        self.add(Example(example="eptorsion2f",nprocs=3,options="-tao_smonitor -mx 16 -my 16",method="lmvm",tags=["eptorsion","multiprocessor","unconstrained","fortran","lmvm","dm"],name="eptorsion2f_3",section="unconstrained"))
        self.add(Example(example="eptorsion2f",nprocs=3,options="-tao_smonitor -mx 16 -my 16 -testmonitor",method="lmvm",tags=["eptorsion","multiprocessor","unconstrained","fortran","lmvm","dm","monitor"],name="eptorsion2f_4",section="unconstrained"))
        self.add(Example(example="eptorsion2f",nprocs=3,options="-tao_smonitor -mx 16 -my 16 -testconvergence",method="lmvm",tags=["eptorsion","multiprocessor","unconstrained","fortran","lmvm","dm","convergence"],name="eptorsion2f_5",section="unconstrained"))

        # Bound constrained
        self.add(Example(example="plate2",nprocs=1,options="-tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2",method="tron",tags=["bound","plate","single","c","tron","dm"],name="plate2",section="bound"))
        self.add(Example(example="plate2",nprocs=2,options="-tao_smonitor -mx 8 -my 8 -bmx 2 -bmy 5 -bheight 0.3",method="blmvm",tags=["bound","plate","multiprocessor","c","blmvm","dm"],name="plate2_2",section="bound"))
        self.add(Example(example="plate2",nprocs=3,options="-tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1",method="tron",tags=["bound","plate","multiprocessor","c","tron","dm"],name="plate2_3",section="bound"))
        # test subset types
        self.add(Example(example="plate2",nprocs=3,options="-tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type mask",method="tron",tags=["bound","plate","multiprocessor","c","tron","dm"],name="plate2_4",section="bound"))
        self.add(Example(example="plate2",nprocs=3,options="-tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type matrixfree -matrixfree -pc_type none",method="tron",tags=["bound","plate","multiprocessor","c","tron","dm","matrixfree"],name="plate2_5",section="bound"))
        self.add(Example(example="plate2",nprocs=3,options="-tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type matrixfree -matrixfree -pc_type none",method="tron",tags=["bound","plate","single","c","tron","dm","matrixfree"],name="plate2_6",section="bound"))
        self.add(Example(example="plate2",nprocs=3,options="-tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type matrixfree -pc_type none",method="gpcg",tags=["bound","plate","multiprocessor","c","gpcg","dm","matrixfree"],name="plate2_7",section="bound"))
        self.add(Example(example="plate2",nprocs=3,options="-tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type mask",method="gpcg",tags=["bound","plate","multiprocessor","c","gpcg","dm","mask"],name="plate2_8",section="bound"))
        self.add(Example(example="plate2",nprocs=1,options="-tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type matrixfree -matrixfree -pc_type none",method="tron",tags=["bound","plate","single","c","tron","dm","matrixfree"],name="plate2_9",section="bound"))
        self.add(Example(example="plate2",nprocs=1,options="-tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type mask -mask",method="tron",tags=["bound","plate","single","c","tron","dm","mask"],name="plate2_10",section="bound"))



        self.add(Example(example="plate2f",nprocs=1,options="-tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2",method="blmvm",tags=["bound","plate","single","fortran","tron","dm"],name="plate2f",section="bound"))
        self.add(Example(example="plate2f",nprocs=2,options="-tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2",method="blmvm",tags=["bound","plate","multiprocessor","fortran","blmvm","dm"],name="plate2f_2",section="bound"))

        self.add(Example(example="jbearing2",nprocs=1,options="-tao_smonitor -mx 8 -my 12",method="tron",tags=["bound","jbearing","single","c","tron","dm"],name="jbearing2",section="bound"))
        self.add(Example(example="jbearing2",nprocs=2,options="-tao_smonitor -mx 50 -my 50 -ecc 0.99",method="gpcg",tags=["bound","jbearing","multiprocessor","c","gpcg","dm"],name="jbearing2_2",section="bound"))
        self.add(Example(example="jbearing2",nprocs=2,options="-tao_smonitor -mx 10 -my 16 -ecc 0.9",method="bqpip",tags=["bound","jbearing","multiprocessor","c","bqpip","dm"],name="jbearing2_3",section="bound"))
        self.add(Example(example="jbearing2",nprocs=2,options="-tao_smonitor -mx 10 -my 16 -ecc 0.9 -testmonitor",method="bqpip",tags=["bound","jbearing","multiprocessor","c","bqpip","dm","monitor"],name="jbearing2_4",section="bound"))
        self.add(Example(example="jbearing2",nprocs=2,options="-tao_smonitor -mx 10 -my 16 -ecc 0.9 -testconvergence",method="bqpip",tags=["bound","jbearing","multiprocessor","c","bqpip","dm","convergence"],name="jbearing2_5",section="bound"))
        

        # Least squares
        self.add(Example(example="chwirut1",nprocs=1,options="-tao_smonitor -tao_fatol 0 -tao_frtol 0 -printhistory",method="pounders",tags=["leastsquares","chwirut","single","c","pounders"],name="chwirut1",section="leastsquares"))
        self.add(Example(example="chwirut2",nprocs=3,options="-tao_fatol 0 -tao_frtol 0",method="pounders",tags=["leastsquares","chwirut","multiprocessor","c","pounders"],name="chwirut2",section="leastsquares"))
        self.add(Example(example="chwirut1f",nprocs=1,options="-tao_smonitor -tao_fatol 0 -tao_frtol 0",method="pounders",tags=["leastsquares","chwirut","single","fortran","pounders"],name="chwirut1f",section="leastsquares"))
        self.add(Example(example="chwirut2f",nprocs=3,options="-tao_smonitor -tao_fatol 0 -tao_frtol 0",method="pounders",tags=["leastsquares","chwirut","fortran","multiprocessor","pounders"],name="chwirut2f",section="leastsquares"))


        # Check gradients and hessians of examples
        self.add(Example(example="minsurf1",nprocs=1,options="-tao_fd_test_gradient -tao_fd_test_hessian",method="fd_test",tags=["fd_test"],name="minsurf1_fd_test",section="unconstrained"))
        self.add(Example(example="eptorsion1",nprocs=1,options="-tao_fd_test_gradient -tao_fd_test_hessian",method="fd_test",tags=["fd_test"],name="eptorsion1_fd_test",section="unconstrained"))
        self.add(Example(example="rosenbrock1",nprocs=1,options="-tao_fd_test_gradient -tao_fd_test_hessian",method="fd_test",tags=["fd_test"],name="rosenbrock1_fd_test",section="unconstrained"))
        self.add(Example(example="rosenbrock1f",nprocs=1,options="-tao_fd_test_gradient -tao_fd_test_hessian",method="fd_test",tags=["fd_test"],name="rosenbrock1f_fd_test",section="unconstrained"))
        
                 
                 
        self.add(Example(example="plate2",nprocs=2,options="-tao_fd_test_gradient -tao_fd_test_hessian",method="fd_test",tags=["fd_test"],name="plate2_fd_test",section="bound"))
        self.add(Example(example="plate2f",nprocs=2,options="-tao_fd_test_gradient -tao_fd_test_hessian",method="fd_test",tags=["fd_test"],name="plate2f_fd_test",section="bound"))
        self.add(Example(example="eptorsion2",nprocs=2,options="-tao_fd_test_gradient -tao_fd_test_hessian",method="fd_test",tags=["fd_test"],name="eptorsion2_fd_test",section="unconstrained"))
        self.add(Example(example="eptorsion2",nprocs=2,options="-tao_fd_test_gradient -tao_fd_test_hessian",method="fd_test",tags=["fd_test"],name="eptorsion2_fd_test",section="unconstrained"))
            
        
        # test line search options
        self.add(Example(example="gts",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["jbearing","lmvm","single","c","unconstrained","gts"],name="gts"))
        self.add(Example(example="gtsf",nprocs=2,options="-tao_smonitor",method="nls",tags=["nls","eptorsion","multiprocessor","fortran","bound","gts"],name="gtsf"))
        self.add(Example(example="linesearch_fg",nprocs=1,options="-tao_smonitor",method="blmvm",tags=["jbearing","blmvm","single","c","unconstrained","linesearch_fg"],name="linesearch_fg"))

        # Complementarity
        self.add(Example(example="blackscholes",nprocs=1,options="-tao_monitor",method="ssils",tags=["complementarity","single","blackscholes","c"],name="blackscholes",section="complementarity"))
        self.add(Example(example="blackscholes",nprocs=1,options="-tao_monitor",method="ssfls",tags=["complementarity","single","blackscholes","c"],name="blackscholes_2",section="complementarity"))
        self.add(Example(example="minsurf1",nprocs=1,options="-tao_monitor -tao_view",method="ssils",tags=["complementarity","single","minsurf","c"],name="comp_minsurf1",section="complementarity"))
        self.add(Example(example="minsurf1",nprocs=1,options="-tao_monitor -tao_view",method="ssfls",tags=["complementarity","single","minsurf","c"],name="comp_minsurf2",section="complementarity"))
        self.add(Example(example="blackscholes",nprocs=1,options="-tao_monitor -tao_subset_type subvec",method="asils",tags=["complementarity","single","blackscholes","c"],name="blackscholes_3",section="complementarity"))
        self.add(Example(example="blackscholes",nprocs=1,options="-tao_monitor -tao_subset_type mask",method="asils",tags=["complementarity","single","blackscholes","c"],name="blackscholes_4",section="complementarity"))
        self.add(Example(example="blackscholes",nprocs=1,options="-tao_monitor -tao_subset_type matrixfree -pc_type jacobi",method="asils",tags=["complementarity","single","blackscholes","c"],name="blackscholes_5",section="complementarity"))
        self.add(Example(example="blackscholes",nprocs=1,options="-tao_monitor -tao_subset_type subvec",method="asfls",tags=["complementarity","single","blackscholes","c"],name="blackscholes_6",section="complementarity"))
        self.add(Example(example="blackscholes",nprocs=1,options="-tao_monitor -tao_subset_type mask",method="asfls",tags=["complementarity","single","blackscholes","c"],name="blackscholes_7",section="complementarity"))
        self.add(Example(example="blackscholes",nprocs=1,options="-tao_monitor -tao_subset_type matrixfree -pc_type jacobi",method="asfls",tags=["complementarity","single","blackscholes","c"],name="blackscholes_8",section="complementarity"))




        # PDE-constrained
        self.add(Example(example="elliptic",nprocs=1,options="-tao_cmonitor -ns 1",method="lcl",tags=["pde","elliptic","single","c"],name="elliptic",section="pde_constrained"))
        self.add(Example(example="elliptic",nprocs=2,options="-tao_cmonitor -tao_max_it 10 -use_ptap -use_lrc -ns 1",method="lcl",tags=["pde","elliptic","multiprocessor","c"],name="elliptic2",section="pde_constrained" ))
        self.add(Example(example="parabolic",nprocs=1,options="-tao_cmonitor",method="lcl",tags=["pde","parabolic","single","c"],name="parabolic",section="pde_constrained"))
        self.add(Example(example="hyperbolic",nprocs=1,options="-tao_cmonitor -tao_max_funcs 10",method="lcl",tags=["pde","hyperbolic","single","c"],name="hyperbolic",section="pde_constrained"))
