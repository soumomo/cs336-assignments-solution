"""
CS336 Assignment 1 — Transformer Model Components

All components are torch.nn.Module subclasses (except softmax).
DO NOT use torch.nn.Linear, torch.nn.Embedding, or torch.nn.functional.
Only allowed: nn.Parameter, nn.Module, nn.ModuleList, torch.sigmoid.

Build in this order (each depends on the ones above):

1. Linear (1 pt)
   - y = Wx (no bias)
   - Init: truncated normal, variance = 2 / (d_in + d_out)
   - Test: pytest -k test_linear

2. Embedding (1 pt)
   - Lookup into (vocab_size, d_model) weight matrix
   - Init: truncated normal, variance = 1
   - Test: pytest -k test_embedding

3. RMSNorm (1 pt)
   - Root Mean Square Layer Normalization with learnable gain
   - Upcast to float32 for stability, then downcast back
   - Test: pytest -k test_rmsnorm

4. SwiGLU Feed-Forward Network (2 pts)
   - FFN(x) = W2(SiLU(W1 * x) ⊙ W3 * x)
   - Three Linear layers (no bias), d_ff ≈ (8/3) * d_model rounded to multiple of 64
   - Test: pytest -k test_swiglu

5. Softmax (1 pt)
   - Numerically stable: subtract max before exp
   - Standalone function, not a Module
   - Test: pytest -k test_softmax

6. RoPE — Rotary Position Embeddings (2 pts)
   - Apply pairwise rotations to Q/K vectors based on position
   - Pre-compute sin/cos with register_buffer in __init__
   - Must handle arbitrary batch dimensions
   - Test: pytest -k test_rope

7. Scaled Dot-Producat Attention (5 pts)
   - attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
   - Support optional boolean mask
   - Handle arbitrary batch-like leading dimensions
   - Test: pytest -k test_scaled_dot_product_attention

8. Causal Multi-Head Self-Attention (5 pts)
   - Project x -> Q, K, V (three separate Linear layers)
   - Split into heads, apply RoPE to Q and K
   - Apply scaled dot-product attention with causal mask
   - Concat heads, project output
   - Test: pytest -k test_multihead_self_attention

9. Transformer Block (3 pts)
   - Pre-norm residual connections:
     z = x + MHSA(RMSNorm(x))
     output = z + FFN(RMSNorm(z))
   - Test: pytest -k test_transformer_block

10. Full Transformer LM (3 pts)
    - Token Embedding -> N x TransformerBlock -> RMSNorm -> Linear head -> logits
    - Test: pytest -k test_transformer_lm
"""
#imports
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self , d_in: int , d_out: int):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_out , d_in))
        variance = 2 / (d_out + d_in)
        std = variance ** 0.5

        nn.init.trunc_normal_(self.weight , mean=0.0, std=std, a=-2.0*std, b=2.0*std)
    
    def forward(self , x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x , self.weight.t())


class Embedding(nn.Module):
    def __init__(self , vocab_size: int , d_model :int):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        #defining the weight matrix
        self.weight = nn.Parameter(torch.empty(vocab_size , d_model))

        variance = 1
        std = variance ** 0.5
        nn.init.trunc_normal_(self.weight , mean=0.0, std=std, a=-2.0*std, b=2.0*std)


    def forward(self ,token_ids: torch.Tensor ) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self , d_model: int , eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.d_model = d_model

        self.weight = nn.Parameter(torch.ones(d_model ,))

    def forward(self , x: torch.Tensor):
        orig_dtype = x.dtype
        x = x.float()
        return ((x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))*self.weight).to(orig_dtype)

class SwiGLU(nn.Module):
    def __init__(self , d_model: int ,d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model , d_ff)
        self.w2 = Linear(d_ff , d_model)
        self.w3 = Linear(d_model , d_ff)

    def forward(self , x):
        out1 = self.w1(x)
        out3 = self.w3(x)
        silu_out1 = out1 * torch.sigmoid(out1)
        gated_mul = silu_out1 * out3
        return self.w2(gated_mul)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_val = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_val)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

class RoPE(nn.Module):
    def __init__(self , d_k: int , max_seq_len: int , theta: float):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

        # compute position vectors
        # create a 1D tensor containing values from 0 to max_seq_len - 1
        positions = torch.arange(max_seq_len)

        # compute frequency bands
        pair_idx = torch.arange(0 , d_k , 2).float()
        freq = self.theta ** (-pair_idx/d_k)

        #calculate the angles
        angles = torch.outer(positions , freq)

        #storing cached
        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)

        #register as buffer
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """ 
        indexing the cox/sin cache
        precomputed lookup table of shape (batch, seq_len, d_k // 2)
        """
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        #broadcasting
        """
        if the input /key tensor x has shape [batch, heads, seq_len, d_k] then our cos/sin are missing the heads dimension
        we insert a dimension of 1 before the seq_len dimension so that it becomes (batch, 1, seq_len, d_k // 2) and can broadcasr over any head
        """
        #this hardcoded is giving dimension mismatch
        # cos = cos.unsqueeze(1)
        # sin = sin.unsqueeze(1)

        #initiate the dimension with all 1s
        view_shape = [1]*x.ndim

        #fill in the sequence length and head dimension
        view_shape[-1] = x.shape[-1] // 2
        view_shape[-2] = x.shape[-2]

        if token_positions.ndim > 1:
            view_shape[0] = token_positions.shape[0]

        cos = cos.view(view_shape)
        sin = sin.view(view_shape)

        #performing pair wise rotation
        #sliced into even and odd components
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        #compute the rotated components
        x_even_rotated = x_even*cos - x_odd*sin
        x_odd_rotated = x_odd*cos + x_even*sin

        #reassembling the final output tensor
        out = torch.empty_like(x)
        out[..., ::2] = x_even_rotated
        out[..., 1::2] = x_odd_rotated

        return out


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        #compute scaled dot product
        K_t = K.transpose(-1,-2)
        scaled_dot_product = torch.matmul(Q , K_t)/(d_k ** 0.5)

        if mask is not None:
        # anywhere mask is False (or 0), fill with negative infinity
            scaled_dot_product = scaled_dot_product.masked_fill(mask == False, float('-inf'))

        attn_weights = softmax(scaled_dot_product , dim = -1)
        output = torch.matmul(attn_weights , V)


        return output

class Multi_Head_Attention(nn.Module):
    '''
    there will be two versions —— with and without RoPE
    '''
    def __init__(self, d_model: int , num_heads: int , max_seq_len: int|None = None , theta: float | None = None):
        super().__init__()
        self.q_proj = Linear(d_model , d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.attention = Attention()
        self.num_heads = num_heads
        self.d_k = d_model//num_heads
        if max_seq_len is not None:
            self.rope = RoPE(d_model//num_heads, max_seq_len, theta)

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # (batch_size, seq_len, d_model))
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # reshape and transpose
        Q = Q.view(*Q.shape[:-1], self.num_heads , self.d_k).transpose(-3,-2)
        K = K.view(*K.shape[:-1], self.num_heads , self.d_k).transpose(-3,-2)
        V = V.view(*V.shape[:-1], self.num_heads , self.d_k).transpose(-3,-2)

        # new shape for Q, K, V: [..., num_heads, seq_len, d_k]

        if hasattr(self , 'rope'):
            Q = self.rope(Q , token_positions )
            K = self.rope(K , token_positions )

        # causal masking time
        # torch.tril creates a lower triangular matrix of 1s (past positions) and 0s (future positions)
        seq_len = x.shape[-2]
        mask  = torch.tril(torch.ones(seq_len , seq_len , device=x.device , dtype=bool))
        out =  self.attention(Q , K , V , mask) #(... , num_heads , seq_len , d_k)
        out = out.transpose(-3,-2)
        out = out.reshape(*out.shape[:-2] , -1)
        return self.output_proj(out)

class TransformerBlock(nn.Module):
    '''
    Sub-Layer 1: Multi-Head Self-Attention

    x → RMSNorm → Multi_Head_Attention → + x  →  z
                                      ↑
                              (residual skip)

    Sub-Layer 2: Feed-Forward Network (SwiGLU)

    z → RMSNorm → SwiGLU → + z  →  output
                         ↑
                 (residual skip)


    
    '''

    def __init__(self ,d_model, num_heads , d_ff, max_seq_len, theta ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = Multi_Head_Attention(d_model , num_heads ,max_seq_len ,theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model , d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        norm_x = self.ln1(x)
        attn_layer = self.attn(norm_x , token_positions)
        z = x + attn_layer


        norm_z = self.ln2(z)
        passed_z = self.ffn(norm_z)
        out = z + passed_z

        return out
        

class TransformerLM(nn.Module):
    def __init__(self , vocab_size: int,context_length: int,d_model: int,num_layers: int,num_heads: int,d_ff: int,rope_theta: float):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size , d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model , vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-1]
        token_positions = torch.arange(seq_len , device=x.device)
        h = self.token_embeddings(x)
        for layer in self.layers:
            h = layer(h , token_positions)
        h = self.ln_final(h)
        return self.lm_head(h)
    
    

        
