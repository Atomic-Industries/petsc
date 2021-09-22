import torch
from petsc4py import PETSc, PETSC_COMM_WORLD

class TAOtorch(torch.optim.Optimizer):
    r"""PyTorch.Optimizer() wrapper for TAO solvers.

    This class makes TAO solvers mimic traditional PyTorch.Optimizer() objects
    by performing single-iteration tao.solve() calls for each optimizer.step()
    in a training cycle.

    This implementation incorporates an adaptive learning rate based on
    `RMSprop <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_
    and `AMSgrad <https://openreview.net/forum?id=ryQu7f-RZ>`_. The learning rate
    is baked into the TAO gradient evaluation, and the TAO line search is set to accept
    unit step length.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        tao (PETSc.TAO, optional): PETSc.TAO solver object (default: TAOBNCG)
        adaptive (str, optional): choose between 'rmsgrad, 'amsgrad', 'norm', and None (default: 'amsgrad')
        rhp (float, optional): RMS decay rate (default: 0.9)
        eps (float, optional): RMS zero safeguard (default: 1e-8)
    """

    def __init__(self, params, tao=None, adaptive='amsgrad', rho=0.9, eps=1e-8):
        if not 0.0 <= rho <= 1.0:
            raise ValueError("Invalid step direction RMS decay rate: {}".format(rho))
        if not 0.0 < eps:
            raise ValueError("Invalid zero safeguard: {}".format(eps))
        if adaptive not in ['rmsgrad', 'amsgrad', 'norm', None]:
            raise ValueError("Invalid adaptive LR method: {}".format(adaptive))

        if tao is None:
            tao = PETSc.TAO().create(self.comm)
            opts = PETSc.Options().create()
            opts.setValue('-tao_type', 'bncg')
            opts.setValue('-tao_bncg_type', 'gd')
            opts.setValue('-tao_bncg_no_scaling','')
            opts.setValue('-tao_ls_type', 'unit')
            tao.setFromOptions()
            opts.destroy()

        paramvec = PETSc.Vec().createWithDLPack(torch.utils.dlpack.to_dlpack(self._getParams()))
        tao.setInitial(paramvec)
        tao.setObjectiveGradient(self._evalObjGrad)
        tao.setMaximumIterations(1)
        tao.setTolerances(gatol=0.0, gttol=0.0, grtol=0.0)
        
        defaults = dict(
            paramvec=paramvec,
            tao=tao,
            rho=rho,
            eps=eps,
            adaptive=adaptive,
            dir_sq_avg=None,
            dir_sq_avg_max=None)
        super(TAOtorch, self).__init__(params, defaults)

    def getTao(self):
        return self.defaults['tao']

    def setTao(self, tao):
        tao.setInitial(self.defaults['paramvec'])
        tao.setObjectiveGradient(self._evalObjGrad)
        tao.setMaximumIterations(1)
        tao.setTolerances(gatol=0.0, gttol=0.0, grtol=0.0)
        opts = PETSc.Options().create()
        opts.setValue('-tao_ls_type', 'unit')
        tao.setFromOptions()
        opts.destroy()
        self.defaults['tao'].destroy()
        self.defaults['tao'] = tao

    def _getParams(self, zero=False, grad=False):
        flatpar = []
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.requires_grad:
                    flatpar.append(p.detach().clone().view(-1))
                    if zero:
                        flatpar[-1][:] = 0.0
                    elif grad:
                        if p.grad is not None:
                            if p.grad.is_sparse:
                                flatpar[-1][:] = p.grad.to_dense().view(-1)[:]
                            else:
                                flatpar[-1][:] = p.grad.view(-1)[:]
                        else:
                            flatpar[-1][:] = 0.0
        return torch.cat(flatpar, 0)

    def _setParams(self, flatpar):
        begin = 0
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.requires_grad:
                    end = begin + len(p.view(-1))
                    p.copy_(torch.reshape(flatpar[begin:end], p.shape))

    def _computeAdaptiveLR(self, ds):
        lr_type = self.defaults['adaptive']
        if lr_type in ['rmsgrad', 'amsgrad']:
            rho = self.defaults['rho']
            eps = self.defaults['eps']
            dir_sq_avg = self.defaults['dir_sq_avg']
            dir_sq_avg_max = self.defaults['dir_sq_avg_max']
            if dir_sq_avg is None:
                # this is the first step so create data structures to track average mean squares
                dir_sq_avg = torch.zeros_like(ds)
                dir_sq_avg.addcmul_(ds, ds, value=1.-rho)
                if lr_type == 'amsgrad':
                    dir_sq_avg_max = dir_sq_avg.detach().clone()
                else:
                    dir_sq_avg_max = dir_sq_avg
            else:
                dir_sq_avg.mul_(rho).addcmul_(ds, ds, value=1.-rho)
                if lr_type == 'amsgrad':
                    torch.maximum(dir_sq_avg_max, dir_sq_avg, out=dir_sq_avg_max)
                else:
                    dir_sq_avg_max = dir_sq_avg
            return 1.0/dir_sq_avg_max.add(eps).sqrt_()
        elif lr_type == 'norm':
            return 1.0/ds.norm(2)
        else:
            return 1.0

    def _evalObjGrad(self, tao, x, G):
        # assume that loss.backward() has already been called before tao.step()
        # create a flattened parameter tensor that shares memory with the TAO gradient vector
        flatgrad = torch.utils.dlpack.from_dlpack(G.toDLPack())
        # copy NN gradients into the flattened tensor
        flatgrad.copy_(self.getParams(grad=True))
        # scale the gradient with RMSgrad/AMSGrad adaptive learning rate
        lr = self._computeAdaptiveLR(flatgrad)
        flatgrad.mul_(lr)

    def step(self, closure=None):
        # first create a flattened parameter tensor that shares memory with the TAO solution
        flatpar = torch.utils.dlpack.from_dlpack(self.defaults['paramvec'].toDLPack())
        # get the NN parameters and write into the flattened tensor
        flatpar.copy_(self._getParams())
        # trigger the tao solution (for 1 iteration) and then write it to NN parameters
        self.tao.solve()
        self._setParams(flatpar)

    def destroy(self):
        self.defaults['tao'].destroy()
        self.defaults['paramvec'].destroy()