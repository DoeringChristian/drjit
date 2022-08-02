import drjit as _dr
import numpy as _np


def to_torch(arg):
    is_llvm = _dr.is_llvm_v(arg)
    is_cuda = _dr.is_cuda_v(arg)

    import torch
    import torch.autograd

    class ToTorch(torch.autograd.Function):
        @staticmethod
        def forward(ctx, arg, handle):
            ctx.drjit_arg = arg
            if is_llvm:
                return torch.tensor(_np.array(arg))
            elif is_cuda:
                return torch.tensor(_np.array(arg)).cuda()
            else:
                raise TypeError("to_torch(): only cuda and llvm is supported")

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_output):
            _dr.set_grad(ctx.drjit_arg, grad_output)
            _dr.enqueue(_dr.ADMode.Backward, ctx.drjit_arg)
            _dr.traverse(type(ctx.drjit_arg), _dr.ADMode.Backward)
            del ctx.drjit_arg
            return None, None

    handle = torch.empty(0, requires_grad=True)
    return ToTorch.apply(arg, handle)


def from_torch(dtype, arg):
    import torch
    if not _dr.is_diff_v(dtype) or not _dr.is_array_v(dtype):
        raise TypeError(
            "from_torch(): expected a differentiable DrJit array type!")

    class FromTorch(_dr.CustomOp):
        def eval(self, arg, handle):
            self.torch_arg = arg
            if _dr.is_cuda_v(dtype):
                return dtype(arg.cuda())
            elif _dr.is_llvm_v(dtype):
                return dtype(arg.cpu())
            else:
                raise TypeError(
                    "from_torch(): only cuda and llvm is supported")

        def forward(self):
            raise TypeError("from_torch(): forward-mode AD is not supported!")

        def backward(self):
            if _dr.is_cuda_v(dtype):
                grad = torch.tensor(_np.array(self.grad_out())).cuda()
            elif _dr.is_llvm_v(dtype):
                grad = torch.tensor(_np.array(self.grad_out())).cuda().cpu()
            self.torch_arg.backward(grad)

    handle = _dr.zeros(dtype)
    _dr.enable_grad(handle)
    return _dr.custom(FromTorch, arg, handle)
