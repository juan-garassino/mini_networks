import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tiktoken
import time
from collections import defaultdict

# ========== GRPO Implementation ==========
class GRPOTrainer:
    def __init__(self, model, ref_model, tokenizer, device, config):
        self.model = model
        self.ref_model = ref_model  # Reference model for KL divergence
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
            param.detach_()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            betas=(0.9, 0.95),
            weight_decay=config['weight_decay']
        )
        
        # Group definitions (customize based on your needs)
        self.group_definitions = {
            'rare_tokens': list(range(0, 100)),         # Infrequent tokens
            'common_tokens': list(range(100, 500)),     # Common tokens
            'structural_tokens': list(range(500, 1000)) # Structural tokens
        }
        
    def compute_reward(self, logits, ref_logits, labels, group_mask):
        """
        Compute reward with KL penalty and group-specific bonuses
        """
        # KL divergence penalty
        kl_penalty = F.kl_div(
            F.log_softmax(ref_logits, dim=-1),
            F.softmax(logits, dim=-1),
            reduction='none'
        ).sum(dim=-1)
        
        # Per-token reward
        rewards = -self.config['kl_coef'] * kl_penalty
        
        # Add group-specific bonuses
        for group_name, token_ids in self.group_definitions.items():
            group_bonus = self.config.get(f'{group_name}_bonus', 0.0)
            group_indices = group_mask[group_name]
            rewards[group_indices] += group_bonus
        
        return rewards
    
    def create_group_mask(self, input_ids):
        """
        Create masks for different token groups
        """
        group_mask = {}
        for group_name, token_ids in self.group_definitions.items():
            group_mask[group_name] = torch.isin(input_ids, torch.tensor(token_ids, device=self.device))
        return group_mask
    
    def train_step(self, batch):
        """
        Perform a single GRPO training step
        """
        self.model.train()
        input_ids, labels = batch
        input_ids, labels = input_ids.to(self.device), labels.to(self.device)
        
        # Get model outputs
        logits, _ = self.model(input_ids)
        
        # Get reference model outputs
        with torch.no_grad():
            ref_logits, _ = self.ref_model(input_ids)
        
        # Create group masks
        group_mask = self.create_group_mask(input_ids)
        
        # Compute rewards
        rewards = self.compute_reward(logits, ref_logits, labels, group_mask)
        
        # Compute policy loss
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        
        # Compute advantages per group
        advantages = torch.zeros_like(rewards)
        for group_name in self.group_definitions:
            group_indices = group_mask[group_name]
            if group_indices.any():
                group_rewards = rewards[group_indices]
                group_advantages = group_rewards - group_rewards.mean()
                group_advantages = group_advantages / (group_rewards.std() + 1e-8)
                advantages[group_indices] = group_advantages
        
        # PPO-style clipped loss
        ratios = torch.exp(selected_log_probs - selected_log_probs.detach())
        clipped_ratios = torch.clamp(ratios, 1.0 - self.config['clip_epsilon'], 
                                     1.0 + self.config['clip_epsilon'])
        
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages)
        policy_loss = policy_loss.mean()
        
        # Backpropagate
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
        self.optimizer.step()
        
        return policy_loss.item()
    
    def train(self, dataset, epochs):
        """
        Full GRPO training loop
        """
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        print(f"Starting GRPO training for {epochs} epochs")
        for epoch in range(epochs):
            epoch_loss = 0
            start_time = time.time()
            
            for batch_idx, batch in enumerate(dataloader):
                loss = self.train_step(batch)
                epoch_loss += loss
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss:.4f}")
            
            avg_loss = epoch_loss / len(dataloader)
            elapsed = time.time() - start_time
            print(f"GRPO Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

# ========== Shakespeare Dataset ==========
class ShakespeareDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def prepare_datasets():
    # Pretraining dataset (basic Shakespeare)
    with open('input.txt', 'r', encoding='utf-8') as f:
        pretrain_text = f.read()
    
    # Finetuning dataset (higher quality Shakespeare)
    with open('finetune.txt', 'r', encoding='utf-8') as f:
        finetune_text = f.read()
    
    # Tokenize
    enc = tiktoken.get_encoding("gpt2")
    pretrain_data = enc.encode(pretrain_text)
    finetune_data = enc.encode(finetune_text)
    
    return pretrain_data, finetune_data, enc

# ========== Training Pipeline ==========
def main():
    # Configuration
    config = {
        'n_layer': 3, 
        'n_head': 3, 
        'n_embd': 192, 
        'block_size': 128,
        'dropout': 0.1,
        'bias': True,
        'vocab_size': 50304,  # Will be set after tokenizer
        'num_experts': 3
    }
    
    # Prepare datasets
    pretrain_data, finetune_data, enc = prepare_datasets()
    config['vocab_size'] = enc.n_vocab
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpt_config = GPTConfig(**config)
    model = GPT(gpt_config).to(device)
    ref_model = GPT(gpt_config).to(device)  # Reference model for KL divergence
    
    # Create data loaders
    pretrain_dataset = ShakespeareDataset(pretrain_data, block_size=config['block_size'])
    finetune_dataset = ShakespeareDataset(finetune_data, block_size=config['block_size'])
    
    # ===== Stage 1: Pretraining =====
    print("Starting pretraining...")
    optimizer = model.configure_optimizers(
        weight_decay=1e-2,
        learning_rate=6e-4,
        betas=(0.9, 0.95),
        device_type=device,
        use_grpo=False
    )
    
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True)
    for epoch in range(3):
        train_epoch(model, pretrain_loader, optimizer, device)
    
    # ===== Stage 2: Fine-tuning =====
    print("\nStarting fine-tuning...")
    optimizer = model.configure_optimizers(
        weight_decay=1e-2,
        learning_rate=1e-5,
        betas=(0.9, 0.95),
        device_type=device,
        use_grpo=False
    )
    
    finetune_loader = DataLoader(finetune_dataset, batch_size=32, shuffle=True)
    for epoch in range(2):
        train_epoch(model, finetune_loader, optimizer, device)
    
    # Save fine-tuned model
    torch.save(model.state_dict(), 'shakespeare_finetuned.pth')
    print("Fine-tuning complete. Model saved.")
    
    # ===== Stage 3: GRPO Optimization =====
    print("\nStarting GRPO optimization...")
    
    # Load reference model with fine-tuned weights
    ref_model.load_state_dict(model.state_dict())
    
    # GRPO configuration
    grpo_config = {
        'lr': 1e-6,
        'weight_decay': 1e-3,
        'kl_coef': 0.1,              # KL penalty coefficient
        'clip_epsilon': 0.2,          # PPO clipping parameter
        'max_grad_norm': 1.0,         # Gradient clipping
        'batch_size': 16,             # Smaller batch size for RL
        'rare_tokens_bonus': 0.1,     # Reward bonus for rare tokens
        'common_tokens_bonus': -0.05, # Small penalty for common tokens
        'structural_tokens_bonus': 0.2 # Bonus for structural tokens
    }
    
    # Create GRPO trainer
    grpo_trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=enc,
        device=device,
        config=grpo_config
    )
    
    # Train with GRPO
    grpo_trainer.train(finetune_dataset, epochs=5)
    
    # Save final model
    torch.save(model.state_dict(), 'shakespeare_grpo_tuned.pth')
    print("GRPO training complete. Model saved.")

def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    losses = []
    
    for batch_idx, (X, Y) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        
        _, loss = model(X, Y)
        
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
    
    return np.mean(losses)

# ========== Model Architecture (from previous implementation) ==========
# [Include all the previous model classes: ExpertMLP, MoE, LayerNorm, 
#  CausalSelfAttention, Block, GPTConfig, GPT]

if __name__ == "__main__":
    main()