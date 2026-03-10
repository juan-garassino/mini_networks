import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """Expert module: 2-layer MLP with GELU activation"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class MoELayer(nn.Module):
    """Mixture of Experts layer with common expert and routing"""
    def __init__(self, d_model, num_experts, d_ff):
        super().__init__()
        self.common_expert = Expert(d_model, d_ff)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.num_experts = num_experts
        self.d_model = d_model

    def forward(self, x):
        # Common expert processes all tokens
        common_out = self.common_expert(x)
        
        # Flatten for expert processing
        orig_shape = x.shape
        x_flat = x.view(-1, self.d_model)
        
        # Gating network determines expert weights
        gate_logits = self.gate(x_flat)
        weights = F.softmax(gate_logits, dim=-1)
        expert_weights, expert_indices = weights.topk(1, dim=-1)
        
        # Process tokens with selected experts
        expert_out = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            token_mask = (expert_indices.squeeze() == expert_idx)
            if token_mask.any():
                expert_out[token_mask] = self.experts[expert_idx](x_flat[token_mask])
        
        # Combine outputs and reshape
        expert_out = expert_out.view(orig_shape)
        return common_out + expert_out

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project to query, key, value
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)
        
        # Combine heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_linear(attn_out)

class TransformerBlock(nn.Module):
    """Transformer block with MHA and MoE FFN"""
    def __init__(self, d_model, num_heads, num_experts, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, num_experts, d_ff)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Multi-head attention
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # MoE FFN
        moe_out = self.moe(self.norm2(x))
        return x + self.dropout(moe_out)

class PositionalEncoding(nn.Module):
    """Positional encoding using fixed frequencies"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MoETransformer(nn.Module):
    """Complete MoE Transformer with 3 layers and 3 experts"""
    def __init__(self, vocab_size, d_model=256, num_heads=3, num_layers=3, num_experts=3, d_ff=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, num_experts, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 10000
    d_model = 256
    num_heads = 3
    num_layers = 3
    num_experts = 3
    d_ff = 1024  # Feed-forward dimension
    
    # Create model
    model = MoETransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        num_experts=num_experts,
        d_ff=d_ff
    )
    
    # Sample input (batch_size=4, seq_len=32)
    input_tokens = torch.randint(0, vocab_size, (4, 32))
    
    # Forward pass
    output = model(input_tokens)
    print("Output shape:", output.shape)  # Should be [4, 32, 256]