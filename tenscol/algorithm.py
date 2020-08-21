import numpy as np
import torch


class GpGrad:
    def __init__(
            self,
            graph: np.array,  # input instance
            d: int,  # dimension count
            k: int,  # color
            lr: float,  # learning rate
            device):
        self.device = device

        # instance
        self.A = torch.from_numpy(graph.astype(np.float32)).to(device)

        # weight tensor dimensions
        self.d = d
        self.n = self.A.shape[0]
        self.k = k

        # weight matrix
        self.W = torch.normal(mean=0.0,
                              std=0.001,
                              size=(self.d, self.n, self.k),
                              dtype=torch.float32,
                              requires_grad=True,
                              device=self.device)

        # optimizer
        self.optimizer = torch.optim.Adam([self.W], lr=lr)

    def _step(self):
        self.optimizer.zero_grad()

        # binary conversion
        W_softmax = torch.nn.functional.softmax(self.W, dim=2)
        W_max_values, W_max_indices = W_softmax.max(2)
        W_onehot = torch.zeros(
            W_softmax.shape, device=self.device).scatter(-1, W_max_indices.view((self.d, self.n, 1)), 1.0)
        W_onehot = W_onehot + W_softmax - W_softmax.data

        self.step()

        self.optimizer.step()

    def step(self):
        pass