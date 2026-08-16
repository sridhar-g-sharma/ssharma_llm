

import sys
import torch
from ssharma_mha_masking import CustomMHA
from ssharma_mqa_masking import CustomMQA
from sinusoidal_position import SinusoidalPosition
from rotary_position import RotaryPosition



class TransformerDecoderBlock(torch.nn.Module):

    def __init__(self, d_model, n_heads,rotary_emb,attention_type="Multi_Head_attention",\
                 ffn_hidden_dim=4,llm_type="Autoregressive"):
        super().__init__()
        self.layerNorm1= torch.nn.LayerNorm(d_model)
        if attention_type =="Multi_Head_attention":
            self.attention=CustomMHA(d_model,n_heads,rotary_emb,llm_type)
            print("Multi Head Attention")
        elif attention_type =="Multi_Query_attention":
            self.attention=CustomMQA(d_model,n_heads,rotary_emb,llm_type)
            print("Multi Query Attention")
        else:
            print("Illegal Attention Type ",attention_type)
            sys.exit(-1)
        self.ffn_hidden_dim=ffn_hidden_dim
        self.layerNorm2 = torch.nn.LayerNorm(d_model)
        print("FFN Hidden Dim ", ffn_hidden_dim)
        self.FFN=torch.nn.Sequential(
            torch.nn.Linear(d_model, self.ffn_hidden_dim * d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(self.ffn_hidden_dim * d_model, d_model),
        )
        self.dropout=torch.nn.Dropout(0.1)

    '''
        param x : (tensor) a tensor of size (batch_size, sequence_length, d_model)
        returns the computed output of the block with the same size.
    '''
    def forward(self, x):       
        self.residual1=x
        x=self.layerNorm1(x)
        x=self.attention(x)
        x1=x+self.residual1

        self.residual2=x1
        x1=self.layerNorm2(x1)
        x1=self.FFN(x1)
        x1=self.dropout(x1)
        y=x1+self.residual2
        return y
        
        

        
        

import sys
import torch

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
        llm_type="Autoregressive"
    ):
        """
        GPT Model supporting Learned, Sinusoidal, or Rotary (RoPE) position embeddings.
        Supports bidirectional attention when llm_type == "Diffusion".
        Support Discrete Space Diffusion based models. Masked Based Diffusion models
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
        
        # Token Embedding
        self.embed1 = torch.nn.Embedding(vocab_size, d_model)
        
        # Positional Embedding setup
        if positional_embedding == "learned":
            self.pos_emb = torch.nn.Embedding(max_seq_len, d_model)
            self.rotary_emb = None
            print("Learned Embedding")
        elif positional_embedding == "sinusoidal":
            self.pos_emb = SinusoidalPosition(max_seq_len, d_model)
            self.pos_emb.requires_grad_(False)
            self.rotary_emb = None
            print("Sinusoidal Embedding")
        elif positional_embedding == "rotary":
            # Registered as a non-trainable buffer to move automatically with model.to(device)
            self.register_buffer("pos_emb", torch.zeros((max_seq_len, d_model)), persistent=False)
            self.rotary_emb = RotaryPosition(d_model, self.theta)
            print("Rotary Embedding")
        else:
            print("Illegal Embedding ", positional_embedding)
            sys.exit(-1)
        print("LLM TYPE: ",llm_type)
        # Decoder Layers
        self.transformer_layers = torch.nn.ModuleList([
            TransformerDecoderBlock(
                d_model, n_heads, self.rotary_emb, self.attention_type, ffn_hidden_dim
            ) for _ in range(layers)
        ])
        
        # Final Output Head
        self.final_output = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Batch size (B) and Sequence length (S)
        B, S = x.shape
        
        # Look up token embeddings: (B, S) -> (B, S, d_model)
        tokens = self.embed1(x)
        
        if self.positional_embedding != "rotary":
            # Generate position indices on the SAME device as x
            positions = torch.arange(S, device=x.device)
            
            # Look up position embeddings and add to token embeddings
            # (1, S, d_model) broadcasts seamlessly across batch dimension
            pos_embeddings = self.pos_emb(positions)
            X = tokens + pos_embeddings
        else:
            # RoPE handles positional embeddings inside self-attention layers
            X = tokens
        
        # Pass through transformer decoder blocks
        for layer in self.transformer_layers:
            X = layer(X)
            
        # Project back to vocabulary logits: (B, S, d_model) -> (B, S, vocab_size)
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





