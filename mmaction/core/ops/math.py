import torch
import torch.nn.functional as F


class OnnxLpNormalization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, axis=0, p=2, eps=1e-12):
        denom = x.norm(2, axis, True).clamp_min(eps).expand_as(x)
        return x / denom

    @staticmethod
    def symbolic(g, x, axis=0, p=2, eps=1e-12):
        return g.op('LpNormalization', x, axis_i=int(axis), p_i=int(p))


def normalize(x, dim, p=2, eps=1e-12):
    if torch.onnx.is_in_onnx_export():
        return OnnxLpNormalization.apply(x, dim, p, eps)
    else:
        return F.normalize(x, dim=dim)
