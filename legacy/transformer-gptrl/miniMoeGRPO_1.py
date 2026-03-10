import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
from dataclasses import dataclass

# ========== Modified MoE Implementation ==========
class ExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)

class MoE(nn.Module):
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
        
        # Initialize gate weights
        torch.nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # Common expert processes all tokens
        common_out = self.common_expert(x)
        
        # Reshape for expert processing
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
            token_idx = (expert_mask[:, :, i] == 1).nonzero(as_tuple=True)[0]
            if token_idx.numel() > 0:
                expert_out[token_idx] = expert(x_flat[token_idx])
        
        # Weight expert outputs by gating scores
        expert_out = expert_out * expert_weights
        
        # Reshape back to original dimensions
        expert_out = expert_out.view(batch_size, seq_len, n_embd)
        
        # Combine common + specialized expert outputs
        out = common_out + expert_out
        return self.dropout(out)

# ========== Modified Transformer Block ==========
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.moe = MoE(config, num_experts=3)  # MoE instead of MLP

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.moe(self.ln_2(x))
        return x

# ========== GRPO Optimization ==========
class GRPO(torch.optim.Optimizer):
    """Group Relative Positional Optimization"""
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-2, 
                 group_size=0.1, min_lr=1e-7, warmup_steps=1000, decay_steps=100000):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                        group_size=group_size, min_lr=min_lr, warmup_steps=warmup_steps,
                        decay_steps=decay_steps)
        super().__init__(params, defaults)
        self.step_count = 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        self.step_count += 1
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('GRPO does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                step = state['step']
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute learning rate
                lr = group['lr']
                min_lr = group['min_lr']
                warmup_steps = group['warmup_steps']
                decay_steps = group['decay_steps']
                
                # Warmup and decay
                if step < warmup_steps:
                    lr = min_lr + (lr - min_lr) * step / warmup_steps
                else:
                    progress = (step - warmup_steps) / decay_steps
                    lr = lr * math.exp(-0.5 * progress)
                    lr = max(lr, min_lr)
                
                # Group-wise normalization
                group_size = max(1, int(group['group_size'] * p.numel()))
                p_flat = p.data.view(-1)
                grad_flat = grad.view(-1)
                
                # Sort parameters and gradients by magnitude
                sorted_indices = torch.argsort(p_flat.abs(), descending=True)
                sorted_p = p_flat[sorted_indices]
                sorted_grad = grad_flat[sorted_indices]
                
                # Apply group-wise updates
                for i in range(0, len(sorted_p), group_size):
                    group_p = sorted_p[i:i+group_size]
                    group_grad = sorted_grad[i:i+group_size]
                    
                    if len(group_p) == 0:
                        continue
                    
                    # Adam update within group
                    exp_avg_group = exp_avg.view(-1)[sorted_indices[i:i+group_size]]
                    exp_avg_sq_group = exp_avg_sq.view(-1)[sorted_indices[i:i+group_size]]
                    
                    denom = exp_avg_sq_group.sqrt().add_(group['eps'])
                    step_size = lr
                    
                    update = exp_avg_group / denom
                    
                    # Apply weight decay
                    if group['weight_decay'] != 0:
                        update.add_(group_p, alpha=group['weight_decay'])
                    
                    # Apply update
                    sorted_p[i:i+group_size].add_(-step_size * update)
                
                # Reconstruct the parameter tensor
                p_flat[sorted_indices] = sorted_p
                p.data.copy_(p_flat.view_as(p.data))
        
        return loss

# ========== GPT Model with MoE ==========
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    num_experts: int = 3  # New config for MoE

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # ... (same as original) ...

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, use_grpo=False):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        if use_grpo:
            print("Using GRPO optimizer")
            return GRPO(
                optim_groups, 
                lr=learning_rate, 
                betas=betas,
                weight_decay=weight_decay,
                group_size=0.1,
                min_lr=learning_rate/10,
                warmup_steps=1000,
                decay_steps=100000
            )
        else:
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # ... (same as original) ...

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # ... (same as original) ...