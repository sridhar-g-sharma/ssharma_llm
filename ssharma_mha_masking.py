import math
import numpy as np
import torch


class CustomMHA(torch.nn.Module):
    """
    param d_model : (int) the length of vectors used in this model
    param n_heads : (int) the number of attention heads. You can assume that
    this even divides d_model.
    param rotary_emb Rotary Embedding Matrix if required
    param RANDOM_WEIGHTS: Random/Deterministic Weights. deterministic for testing
    """

    def __init__(self, d_model, n_heads, rotary_emb, RANDOM_WEIGHTS=True, llm_type="Autoregressive"):
        super().__init__()
        self.verbose = False

        if RANDOM_WEIGHTS:
            self.W_kqv = torch.nn.Parameter(0.01 * torch.rand(3 * d_model, d_model))
            self.W_o = torch.nn.Parameter(0.01 * torch.rand(d_model, d_model))
        else:
            D = d_model
            tensor1 = torch.tensor(
                np.reshape(np.linspace(-2.0, 1.5, D * D * 3), (D * 3, D)),
                dtype=torch.float32,
            )
            tensor2 = torch.tensor(
                np.reshape(np.linspace(-1.0, 2.0, D * D), (D, D)),
                dtype=torch.float32,
            )
            self.W_kqv = torch.nn.Parameter(tensor1)
            self.W_o = torch.nn.Parameter(tensor2)

        self.n_heads = n_heads
        self.embed_dim = d_model
        self.rotary_emb = rotary_emb

        if self.rotary_emb is None and self.verbose:
            print("Empty Rotary Embedding Matrix")

    def forward(self, x):
        # Multiply W_kqv with x: (B, S, D) @ (D, 3D) -> (B, S, 3D)
        T = x @ self.W_kqv.T
        if self.verbose:
            print("T Shape ", T.shape)

        seq_length = x.shape[1]
        self.D_h = int(self.embed_dim / self.n_heads)

        # Split into Query, Key, Value
        [Q, K, V] = torch.chunk(T, chunks=3, dim=2)

        # Handle Rotary Embedding
        if self.rotary_emb is not None:
            Q = self.rotary_emb.rotate(Q)
            K = self.rotary_emb.rotate(K)

        # Reshape to (B, S, h, D/h) -> (B, h, S, D/h)
        K = K.reshape(x.shape[0], seq_length, self.n_heads, self.D_h).permute(0, 2, 1, 3)
        Q = Q.reshape(x.shape[0], seq_length, self.n_heads, self.D_h).permute(0, 2, 1, 3)
        V = V.reshape(x.shape[0], seq_length, self.n_heads, self.D_h).permute(0, 2, 1, 3)

        KT = K.permute(0, 1, 3, 2)

        # compute QK^T / sqrt(d) -> (B, h, S, S)
        QKT = Q @ KT
        QKT = QKT / math.sqrt(float(self.D_h))

        # Causal Attention Mask created directly on x.device
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        expanded_causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

        if self.llm_type == "Autoregressive":
            causal_mask = torch.triu(
                torch.ones(seq_length, seq_length, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            expanded_causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
            QKT_masked = QKT.masked_fill(expanded_causal_mask, float("-inf"))
        else:
            QKT_masked = QKT

        # Apply Causal Mask
        QKT_masked = QKT.masked_fill(expanded_causal_mask, float("-inf"))

        # Softmax & Attention aggregation
        SQKT = torch.nn.functional.softmax(QKT_masked, dim=-1)
        attention_1 = SQKT @ V

        # Reshape back into (B, S, D)
        attention_2 = attention_1.permute(0, 2, 1, 3).contiguous()
        attention_3 = attention_2.reshape(x.shape[0], x.shape[1], self.n_heads * self.D_h)

        # Output projection
        y = attention_3 @ self.W_o.T
        return y


class CustomMHA_solution(torch.nn.Module):
    def __init__(self, d_model, n_heads, RANDOM_WEIGHTS=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        if RANDOM_WEIGHTS:
            self.W_qkv = torch.nn.Parameter(0.01 * torch.randn((3 * d_model, d_model)))
            self.W_o = torch.nn.Parameter(0.01 * torch.randn((d_model, d_model)))
        else:
            D = d_model
            tensor1 = torch.tensor(
                np.reshape(np.linspace(-2.0, 1.5, D * D * 3), (D * 3, D)),
                dtype=torch.float32,
            )
            tensor2 = torch.tensor(
                np.reshape(np.linspace(-1.0, 2.0, D * D), (D, D)),
                dtype=torch.float32,
            )
            self.W_qkv = torch.nn.Parameter(tensor1)
            self.W_o = torch.nn.Parameter(tensor2)

    def forward(self, x):
        added_batch = False
        if len(x.shape) == 2:
            added_batch = True
            x = x[None, :, :]

        B, S, D = x.shape
        QKV = x @ self.W_qkv.T  # (B, S, 3D)
        Q, K, V = torch.chunk(QKV, 3, -1)

        dh = D // self.n_heads

        # Reshape into 4D: (B, n_heads, S, dh)
        q_heads = Q.reshape(B, S, self.n_heads, dh).transpose(1, 2)
        k_heads = K.reshape(B, S, self.n_heads, dh).transpose(1, 2)
        v_heads = V.reshape(B, S, self.n_heads, dh).transpose(1, 2)

        # compute QK^T / sqrt(dh) -> (B, n_heads, S, S)
        qkt = (q_heads @ k_heads.transpose(-2, -1)) / math.sqrt(float(dh))

        # Causal mask on matching device
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
        )
        expanded_causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

        qkt_masked = qkt.masked_fill(expanded_causal_mask, float("-inf"))

        # Softmax & output projection
        attn = torch.nn.functional.softmax(qkt_masked, dim=-1)
        out = attn @ v_heads

        out = out.transpose(1, 2).contiguous().reshape(B, S, D)
        x = out @ self.W_o.T

        if added_batch:
            x = x[0]

        return x


def test_MHA(use_solution=False):
    D = 6
    H = 2
    if use_solution:
        mha = CustomMHA_solution(D, H, RANDOM_WEIGHTS=False)
    else:
        mha = CustomMHA(D, H, rotary_emb=None, RANDOM_WEIGHTS=False)

    B = 2
    S = 3
    x = torch.tensor(
        np.reshape(np.linspace(-1.0, 0.5, B * S * D), (B, S, D)), dtype=torch.float32
    )
    print("Input shape ", x.shape)
    print("=================================")
    y1 = mha.forward(x)
    print("Output shape: ", y1.shape)
    print("\n Output \n", y1)
    return y1


if __name__ == "__main__":
    expected = torch.tensor(
        [
            [
                [21.7331, 6.4755, -8.7821, -24.0397, -39.2973, -54.5549],
                [19.6692, 6.0497, -7.5698, -21.1893, -34.8087, -48.4282],
                [17.6900, 5.6398, -6.4105, -18.4607, -30.5110, -42.5612],
            ],
            [
                [2.8558, 0.8462, -1.1635, -3.1731, -5.1827, -7.1924],
                [-1.2608, -0.5160, 0.2287, 0.9735, 1.7182, 2.4630],
                [-5.6875, -2.0716, 1.5444, 5.1603, 8.7762, 12.3922],
            ],
        ]
    )

    print("Testing CustomMHA...")
    actual_0 = test_MHA(use_solution=False)
    error_0 = actual_0 - expected
    print("\n Error b/w Actual & Expected (CustomMHA): \n", error_0)

    print("\nTesting CustomMHA_solution...")
    actual_1 = test_MHA(use_solution=True)
    error_1 = actual_1 - expected
    print("\n Error b/w Actual & Expected (CustomMHA_solution): \n", error_1)