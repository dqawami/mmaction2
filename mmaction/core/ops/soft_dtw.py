"""
The original repo: https://github.com/Maghoumi/pytorch-softdtw-cuda
"""


import numpy as np
import torch


def compute_softdtw(D, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    scale = -1.0 / gamma

    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0

    for b in range(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                r0 = scale * R[b, i - 1, j - 1]
                r1 = scale * R[b, i - 1, j]
                r2 = scale * R[b, i, j - 1]

                r_max = max(max(r0, r1), r2)
                r_sum = np.exp(r0 - r_max) + np.exp(r1 - r_max) + np.exp(r2 - r_max)
                soft_min = - gamma * (np.log(r_sum) + r_max)

                R[b, i, j] = D[b, i - 1, j - 1] + soft_min

    return R


def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]

    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))

    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]

    for k in range(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                a = np.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma)
                b = np.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma)
                c = np.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma)

                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

    return E[:, 1:N + 1, 1:M + 1]


class SoftDTW(torch.autograd.Function):
    """
    CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
    """

    @staticmethod
    def forward(ctx, cost_matrix, gamma, bandwidth):
        dev = cost_matrix.device
        dtype = cost_matrix.dtype

        ctx.gamma = gamma
        ctx.bandwidth = bandwidth

        cost_matrix_ = cost_matrix.detach().cpu().numpy()
        R = torch.Tensor(compute_softdtw(cost_matrix_, gamma, bandwidth)).to(dev).type(dtype)

        ctx.save_for_backward(cost_matrix, R)

        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype

        cost_matrix, R = ctx.saved_tensors
        gamma = ctx.gamma
        bandwidth = ctx.bandwidth

        cost_matrix_ = cost_matrix.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        E = torch.Tensor(compute_softdtw_backward(cost_matrix_, R_, gamma, bandwidth)).to(dev).type(dtype)

        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


soft_dtw = SoftDTW.apply
