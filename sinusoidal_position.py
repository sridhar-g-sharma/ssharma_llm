import torch
import numpy as np


class SinusoidalPosition(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        n = 10000

        weight = torch.zeros(
            (num_embeddings, embedding_dim),
            dtype=torch.float32
        )

        for k in range(num_embeddings):
            for i in range(int(embedding_dim / 2)):
                denominator = n ** (2 * i / embedding_dim)

                weight[k, 2 * i] = np.sin(k / denominator)
                weight[k, 2 * i + 1] = np.cos(k / denominator)

        # Automatically moves with model.to(device)
        self.register_buffer("weight", weight)

    def forward(self, x):
        return self.weight[x]