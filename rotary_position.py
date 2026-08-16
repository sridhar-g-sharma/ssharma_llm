import torch


class RotaryPosition(torch.nn.Module):
    def __init__(self, d_model, theta=10000.0):
        super().__init__()

        self.theta = theta
        self.d_model = d_model

        # Precompute inverse frequencies
        inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, d_model, 2).float() / d_model)
        )

        #  moves with model.to(device)
        self.register_buffer("inv_freq", inv_freq)

    def rotate(self, x):
        # x shape: (batch_size, num_heads, seq_len, head_dim)
        seq_len = x.shape[-2]

        #  create positions on same device as x
        pos = torch.arange(
            seq_len,
            dtype=torch.float32,
            device=x.device
        )

        # Calculate angles
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        xu = x[..., :self.d_model // 2]
        xd = x[..., self.d_model // 2:]

        hatx = torch.cat([-xd, xu], dim=-1)

        rotated_x = x * cos + hatx * sin

        return rotated_x.to(torch.float32)