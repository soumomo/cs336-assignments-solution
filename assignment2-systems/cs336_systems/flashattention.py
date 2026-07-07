import torch
import math
from einops import rearrange
torch.manual_seed(42)

N = 8 #seq_len
d = 4 #head_dim
M = 32 #SRAM Capacity

Bc = math.ceil(M/(4*d))
Br = min(math.ceil(M / (4 * d)), d)
print(f"Br = {Br}")
print(f"Bc = {Bc}")

Q = torch.randn(N,d)
K = torch.randn(N, d)
V = torch.randn(N, d)

O = torch.zeros((N, d))
l = torch.zeros(N)
m = torch.full((N,), -float("inf"))

Tr = math.ceil(N / Br)
Tc = math.ceil(N / Bc)
print(f"Tr: {Tr}")
print(f"Tc: {Tc}")


Q_blocks = torch.split(Q, Br, dim=0) #(Br, d)
V_blocks = torch.split(V, Bc, dim=0) #(Bc, d)
K_blocks = torch.split(K, Bc, dim=0) #(Bc, d)

O_blocks = list(torch.split(O, Br, dim=0)) #(Br, d)
l_blocks = list(torch.split(l, Br, dim=0)) #(Br,)
m_blocks = list(torch.split(m, Br, dim=0)) #(Br,)

for Kj, Vj in zip(K_blocks, V_blocks):
    for i, (Qi, Oi, li, mi) in enumerate(zip(Q_blocks, O_blocks, l_blocks, m_blocks)):
        # line 9
        Sij = Qi @ Kj.T #(Br, Bc)
        # line 10
        m_tilde = Sij.max(dim=1).values #(max along the rows)
        P_tilde = torch.exp(Sij - rearrange(m_tilde, 'br -> br 1'))
        l_tilde = P_tilde.sum(dim=1)
        
        # line 11
        m_new = torch.maximum(mi, m_tilde)
        alpha = torch.exp(mi - m_new)
        beta = torch.exp(m_tilde - m_new)
        l_new = alpha * li + beta * l_tilde

        # line 12
        PV = P_tilde @ Vj
        alpha = rearrange(alpha, 'br-> br 1')
        beta = rearrange(beta, 'br -> br 1')
        li = rearrange(li, 'br -> br 1')
        l_new = rearrange(l_new, 'br -> br 1')
        O_new = (li*alpha*Oi + beta* PV)/l_new

        O_blocks[i] = O_new
        l_blocks[i] = l_new.squeeze(1)
        m_blocks[i] = m_new

O = torch.cat(O_blocks, dim=0)

scores = Q @ K.T
P = torch.softmax(scores, dim=-1)
expected = P @ V

print("All Close:", torch.allclose(O, expected, atol=1e-5))
print("Max Error:", (O - expected).abs().max())

print("\nFlashAttention")
print(O)

print("\nVanilla")
print(expected)