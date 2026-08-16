

import sys
import torch
from ssharma_mha_masking import CustomMHA
from ssharma_mqa_masking import CustomMQA
from sinusoidal_position import SinusoidalPosition
from rotary_position import RotaryPosition



class TransformerBlock(torch.nn.Module):
    """
    Unified Transformer Block serving both:
      - Decoder (Autoregressive): Causal masking enabled
      - Encoder (Diffusion): Bidirectional full attention (no causal mask)
    """
    def __init__(
        self, 
        d_model, 
        n_heads, 
        rotary_emb, 
        attention_type="Multi_Head_attention", 
        ffn_hidden_dim=4,
        llm_type="Autoregressive",
        activation="relu"
    ):
        super().__init__()
        self.layerNorm1 = torch.nn.LayerNorm(d_model)
        self.llm_type = llm_type
        
        # Dispatch Attention with llm_type
        if attention_type == "Multi_Head_attention":
            self.attention = CustomMHA(d_model, n_heads, rotary_emb, llm_type=llm_type)
        elif attention_type == "Multi_Query_attention":
            self.attention = CustomMQA(d_model, n_heads, rotary_emb, llm_type=llm_type)
        else:
            raise ValueError(f"Illegal Attention Type: {attention_type}")
            
        self.layerNorm2 = torch.nn.LayerNorm(d_model)
        self.ffn_hidden_dim = ffn_hidden_dim
        
        act_layer = torch.nn.GELU() if activation == "gelu" else torch.nn.ReLU()
        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(d_model, self.ffn_hidden_dim * d_model),
            act_layer,
            torch.nn.Linear(self.ffn_hidden_dim * d_model, d_model),
        )
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):       
        # Pre-LN Self-Attention with residual connection
        residual1 = x
        x = self.layerNorm1(x)
        x = self.attention(x)
        x1 = x + residual1

        # Pre-LN Feed-Forward with residual connection
        residual2 = x1
        x1 = self.layerNorm2(x1)
        x1 = self.FFN(x1)
        x1 = self.dropout(x1)
        y = x1 + residual2
        return y
        

"""
        GPT Model supporting Learned, Sinusoidal, or Rotary (RoPE) position embeddings.
        Supports bidirectional attention when llm_type != "Autoregressive".
        Also Supports Non Causal models
        Supports Discrete Space Diffusion based models. Masked Based Diffusion models
"""

class GPTModel(torch.nn.Module):
    def __init__(
        self, 
        d_model, 
        n_heads, 
        layers, 
        vocab_size, 
        max_seq_len,
        positional_embedding="learned",
        theta=10000.0,
        attention_type="Multi_Head_attention",
        ffn_hidden_dim=4,
        llm_type="Autoregressive",
        activation="relu",
        diffusion_steps=128
    ):
        """
        Unified Model supporting both Autoregressive (Decoder) and Diffusion (Encoder) architectures.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.layers = layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.positional_embedding = positional_embedding
        self.attention_type = attention_type
        self.theta = theta
        self.llm_type = llm_type
        self.activation=activation
        self.diffusion_steps=diffusion_steps
        
        # Token Embedding
        self.embed1 = torch.nn.Embedding(vocab_size, d_model)
        
        # Positional Embedding setup
        if positional_embedding == "learned":
            self.pos_emb = torch.nn.Embedding(max_seq_len, d_model)
            self.rotary_emb = None
        elif positional_embedding == "sinusoidal":
            self.pos_emb = SinusoidalPosition(max_seq_len, d_model)
            self.pos_emb.requires_grad_(False)
            self.rotary_emb = None
        elif positional_embedding == "rotary":
            self.register_buffer("pos_emb", torch.zeros((max_seq_len, d_model)), persistent=False)
            self.rotary_emb = RotaryPosition(d_model, self.theta)
        else:
            raise ValueError(f"Illegal Embedding: {positional_embedding}")

        if llm_type == "Diffusion":
            self.time_emb = torch.nn.Embedding(self.diffusion_steps + 1, self.d_model)

        # Transformer Blocks (Encoder vs Decoder dynamically selected via llm_type)
        self.transformer_layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model, 
                n_heads=n_heads, 
                rotary_emb=self.rotary_emb, 
                attention_type=self.attention_type, 
                ffn_hidden_dim=ffn_hidden_dim,
                llm_type=self.llm_type,
                activation=self.activation
            ) for _ in range(layers)
        ])
        
        # Final Output Head
        self.final_output = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, S = x.shape
        tokens = self.embed1(x)
        
        if self.positional_embedding != "rotary":
            positions = torch.arange(S, device=x.device)
            pos_embeddings = self.pos_emb(positions)
            X = tokens + pos_embeddings
        else:
            X = tokens
        
        for layer in self.transformer_layers:
            X = layer(X)
            
        logits = self.final_output(X)
        return logits
        
        


if __name__ == "__main__":

    # example of building the model and doing a forward pass
    D = 128
    H = 8
    L = 4
    llm_type="Autoregressive"
    model = GPTModel( D, H, L, 1000, 512,"learned",llm_type)
    B = 32
    S = 48 # this can be less than 512, it just cant be more than 512
    x = torch.randint(1000, (B, S))
    y = model(x) # this should give us logits over the vocab for all positions

    # should be size (B, S, 1000)
    print(y)
    print(y.shape)





