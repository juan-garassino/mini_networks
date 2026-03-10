class ExpertMLP(nn.Module):
    """Expert module: Same as original MLP but without dropout in return"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x  # Dropout will be applied in MoE layer

class MoE(nn.Module):
    """Mixture of Experts with Common Expert"""
    def __init__(self, config, num_experts=3):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        
        # Common expert (shared across all tokens)
        self.common_expert = ExpertMLP(config)
        
        # Specialized experts
        self.experts = nn.ModuleList([
            ExpertMLP(config) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(config.n_embd, num_experts, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize gate weights to small values
        torch.nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # Common expert processes all tokens
        common_out = self.common_expert(x)
        
        # Reshape for expert processing: [batch, seq, dim] -> [batch*seq, dim]
        batch_size, seq_len, n_embd = x.shape
        x_flat = x.view(-1, n_embd)
        
        # Gating network
        gate_logits = self.gate(x_flat)
        gate_scores = F.softmax(gate_logits, dim=-1)
        
        # Top-1 expert selection
        expert_weights, expert_indices = torch.topk(gate_scores, k=1, dim=-1)
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        
        # Process tokens with selected experts
        expert_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Get tokens assigned to this expert
            token_idx = (expert_mask[:, :, i] == 1).nonzero(as_tuple=True)[0]
            if token_idx.numel() > 0:
                tokens = x_flat[token_idx]
                expert_out[token_idx] = expert(tokens)
        
        # Weight expert outputs by gating scores
        expert_out = expert_out * expert_weights
        
        # Reshape back to original dimensions
        expert_out = expert_out.view(batch_size, seq_len, n_embd)
        
        # Combine common + specialized expert outputs
        out = common_out + expert_out
        return self.dropout(out)

class Block(nn.Module):
    """Modified Transformer Block with MoE instead of MLP"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.moe = MoE(config, num_experts=3)  # Using 3 experts

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.moe(self.ln_2(x))
        return x