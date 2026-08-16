

import math
import torch
import numpy as np

class CustomMQA(torch.nn.Module):
    def __init__(self, d_model, n_heads, rotary_emb, llm_type="Autoregressive"):
        super().__init__()
        self.verbose = False
        self.n_heads = n_heads
        self.embed_dim = d_model
        self.head_dim = d_model // n_heads
        self.rotary_emb = rotary_emb

        # Projections registered as Parameters
        self.query_proj = torch.nn.Parameter(0.01 * torch.rand(d_model, d_model))
        self.key_proj = torch.nn.Parameter(0.01 * torch.rand(d_model, self.head_dim))
        self.value_proj = torch.nn.Parameter(0.01 * torch.rand(d_model, self.head_dim))
        self.W_o = torch.nn.Parameter(0.01 * torch.rand(d_model, d_model))

    def forward(self, x):
        B, S, D = x.shape
        
        # Build W_kqv dynamically so it automatically stays on x.device
        key_expanded = self.key_proj.repeat(1, self.n_heads)
        val_expanded = self.value_proj.repeat(1, self.n_heads)
        W_kqv = torch.cat((self.query_proj, key_expanded, val_expanded), dim=0)

        # Multiply W_kqv with x
        T = x @ W_kqv.T  # (B, S, 3 * D)

        [Q, K, V] = torch.chunk(T, chunks=3, dim=2)

        # Handle Rotary Embedding
        if self.rotary_emb is not None:
            Q = self.rotary_emb.rotate(Q)
            K = self.rotary_emb.rotate(K)

        # Reshape for multi-head attention: (B, S, n_heads, head_dim)
        Q = Q.reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (B, h, S, dh)
        K = K.reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (B, h, S, dh)
        V = V.reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (B, h, S, dh)

        # Compute QK^T / sqrt(d_h)
        KT = K.permute(0, 1, 3, 2)
        QKT = (Q @ KT) / math.sqrt(float(self.head_dim))  # (B, h, S, S)

        # Conditionally apply Causal Masking
        if self.llm_type == "Autoregressive":
            causal_mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
            expanded_causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
            QKT_masked = QKT.masked_fill(expanded_causal_mask, float('-inf'))
        else:
            QKT_masked = QKT
        
        QKT_masked = QKT.masked_fill(expanded_causal_mask, float('-inf'))

        # Softmax & Attention Value aggregation
        SQKT = torch.nn.functional.softmax(QKT_masked, dim=-1)
        attention_1 = SQKT @ V  # (B, h, S, dh)

        # Reshape back to (B, S, D)
        attention_2 = attention_1.permute(0, 2, 1, 3).contiguous()
        attention_3 = attention_2.reshape(B, S, D)

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
            tensor1 = torch.tensor(np.reshape(np.linspace(-2.0, 1.5, D * D * 3), (D * 3, D)), dtype=torch.float32)
            tensor2 = torch.tensor(np.reshape(np.linspace(-1.0, 2.0, D * D), (D, D)), dtype=torch.float32)
            self.W_qkv = torch.nn.Parameter(tensor1)
            self.W_o = torch.nn.Parameter(tensor2)

    def forward(self, x):
        added_batch = False
        if len(x.shape) == 2:
            added_batch = True
            x = x[None, :, :]

        B, S, D = x.shape
        QKV = x @ self.W_qkv.T  # (B, S, 3D)
        Q, K, V = torch.chunk(QKV, 3, dim=-1)

        dh = D // self.n_heads
        
        # Reshape into (B, n_heads, S, dh)
        q_heads = Q.reshape(B, S, self.n_heads, dh).transpose(1, 2)
        k_heads = K.reshape(B, S, self.n_heads, dh).transpose(1, 2)
        v_heads = V.reshape(B, S, self.n_heads, dh).transpose(1, 2)

        # Compute QK^T / sqrt(dh)
        qkt = (q_heads @ k_heads.transpose(-2, -1)) / math.sqrt(float(dh)) # (B, h, S, S)

        # Causal mask created ON x.device matching 4D shape (1, 1, S, S)
        causal_mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        expanded_causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        qkt_masked = qkt.masked_fill(expanded_causal_mask, float('-inf'))

        # Softmax and attention output
        attn = torch.nn.functional.softmax(qkt_masked, dim=-1)
        out = attn @ v_heads  # (B, h, S, dh)

        # Reshape back to (B, S, D)
        out = out.transpose(1, 2).contiguous().reshape(B, S, D)
        x = out @ self.W_o.T

        if added_batch:
            x = x[0]

        return x

if __name__ == "__main__":
    expected=torch.tensor([[[ 21.7331,   6.4755,  -8.7821, -24.0397, -39.2973, -54.5549],
         [ 19.6692,   6.0497,  -7.5698, -21.1893, -34.8087, -48.4282],
         [ 17.6900,   5.6398,  -6.4105, -18.4607, -30.5110, -42.5612]],

        [[  2.8558,   0.8462,  -1.1635,  -3.1731,  -5.1827,  -7.1924],
         [ -1.2608,  -0.5160,   0.2287,   0.9735,   1.7182,   2.4630],
         [ -5.6875,  -2.0716,   1.5444,   5.1603,   8.7762,  12.3922]]])
    print("Use  Solution for Assignment 4 of MHA")
    actual=test_MHA(use_solution=False)
    error_0=actual-expected
    print("\n Error b/w Actual & Expected from Assignment 4 with Mask \n")
    print(error_0)

    print("Use Provided Solution of MHA")
    actual=test_MHA(use_solution=True)

    
    error_1=actual-expected
    print("\n Error b/w Actual & Expected with solution provided for assignment 4\n")
    print(error_1)

    print("\n Differences between two solutions \n",error_0-error_1)


# In[ ]:




