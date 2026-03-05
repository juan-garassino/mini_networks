import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# ==================== Reward Model ====================
class ShakespeareRewardModel(nn.Module):
    def __init__(self, base_model, hidden_size=128):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Sequential(
            nn.Linear(base_model.config.n_embd, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids):
        # Get base model embeddings
        _, hidden_states = self.base_model(input_ids)
        # Use last token's embedding for reward prediction
        last_hidden = hidden_states[:, -1, :]
        return self.reward_head(last_hidden)
    
    def compute_reward(self, text, tokenizer):
        """Compute reward for a given text"""
        input_ids = tokenizer.encode(text, return_tensors='pt').to(self.base_model.device)
        with torch.no_grad():
            reward = self.forward(input_ids)
        return reward.item()

# ==================== Preference Dataset ====================
class ShakespearePreferenceDataset(Dataset):
    def __init__(self, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = deque(maxlen=10000)  # Circular buffer
        
        # Shakespeare-specific features for reward simulation
        self.archaic_words = ["thou", "thee", "thy", "thine", "hath", "doth", 
                              "art", "wert", "wherefore", "hither", "yon"]
        self.rhyme_pairs = [("love", "dove"), ("heart", "part"), ("eyes", "skies"),
                            ("night", "light"), ("day", "away"), ("mind", "find")]
    
    def add_sample(self, prompt, response_chosen, response_rejected):
        """Add a human preference sample"""
        self.data.append({
            'prompt': prompt,
            'chosen': response_chosen,
            'rejected': response_rejected
        })
    
    def simulate_preferences(self, model, num_samples=1000):
        """Simulate human preferences using Shakespearean rules"""
        print("Simulating Shakespearean preferences...")
        prompts = [
            "To be, or not to be",
            "Shall I compare thee",
            "Romeo, Romeo, wherefore",
            "All the world's a stage",
            "What light through yonder window breaks"
        ]
        
        for _ in range(num_samples):
            prompt = random.choice(prompts)
            
            # Generate two responses
            response1 = self.generate_response(model, prompt)
            response2 = self.generate_response(model, prompt)
            
            # Score responses using Shakespearean rules
            score1 = self.shakespearean_score(response1)
            score2 = self.shakespearean_score(response2)
            
            # Create preference pair
            if score1 >= score2:
                self.add_sample(prompt, response1, response2)
            else:
                self.add_sample(prompt, response2, response1)
    
    def shakespearean_score(self, text):
        """Score text based on Shakespearean characteristics"""
        score = 0
        
        # 1. Archaic words usage
        for word in self.archaic_words:
            if word in text.lower():
                score += 1
        
        # 2. Rhyme scheme detection
        lines = text.split('\n')
        for i in range(len(lines)-1):
            last_word1 = lines[i].strip().split()[-1] if lines[i].strip() else ""
            last_word2 = lines[i+1].strip().split()[-1] if lines[i+1].strip() else ""
            
            for rhyme1, rhyme2 in self.rhyme_pairs:
                if (rhyme1 in last_word1.lower() and rhyme2 in last_word2.lower()) or \
                   (rhyme2 in last_word1.lower() and rhyme1 in last_word2.lower()):
                    score += 2
        
        # 3. Iambic pentameter detection (simplified)
        for line in lines:
            words = line.split()
            if 8 <= len(words) <= 12:  # Approximate word count
                score += 1
        
        # 4. Theatrical elements
        if any(term in text for term in ["O ", "Alas", "Prithee", "Hark"]):
            score += 2
            
        return score
    
    def generate_response(self, model, prompt, max_new_tokens=32):
        """Generate response from model"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.8)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt_ids = self.tokenizer.encode(
            sample['prompt'], 
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).squeeze(0)
        
        chosen_ids = self.tokenizer.encode(
            sample['chosen'], 
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).squeeze(0)
        
        rejected_ids = self.tokenizer.encode(
            sample['rejected'], 
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).squeeze(0)
        
        return {
            'prompt_ids': prompt_ids,
            'chosen_ids': chosen_ids,
            'rejected_ids': rejected_ids
        }

# ==================== PPO Trainer ====================
class ShakespearePPOTrainer:
    def __init__(self, actor_model, critic_model, reward_model, tokenizer, config):
        self.actor = actor_model
        self.critic = critic_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Optimizers
        self.actor_optim = torch.optim.AdamW(
            self.actor.parameters(), 
            lr=config['actor_lr'],
            weight_decay=config['weight_decay']
        )
        
        self.critic_optim = torch.optim.AdamW(
            self.critic.parameters(), 
            lr=config['critic_lr'],
            weight_decay=config['weight_decay']
        )
        
        # KL divergence controller
        self.kl_ctl = AdaptiveKLController(config['kl_target'])
        
    def train_step(self, batch):
        # Unpack batch
        prompt_ids = batch['prompt_ids'].to(self.actor.device)
        chosen_ids = batch['chosen_ids'].to(self.actor.device)
        rejected_ids = batch['rejected_ids'].to(self.actor.device)
        
        # Generate responses with current policy
        with torch.no_grad():
            actor_output = self.actor(prompt_ids)
            logits = actor_output.logits
            values = self.critic(prompt_ids).squeeze(-1)
            
        # Compute rewards
        chosen_rewards = self.compute_rewards(chosen_ids)
        rejected_rewards = self.compute_rewards(rejected_ids)
        
        # Compute advantages
        advantages = chosen_rewards - rejected_rewards
        
        # Compute policy loss
        log_probs = self.log_probs(logits, chosen_ids)
        ratio = torch.exp(log_probs - log_probs.detach())
        clipped_ratio = torch.clamp(ratio, 1.0 - self.config['clip_epsilon'], 
                                    1.0 + self.config['clip_epsilon'])
        
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
        
        # Compute KL penalty
        with torch.no_grad():
            ref_logits = self.actor(prompt_ids).logits
        kl_div = self.kl_divergence(logits, ref_logits)
        kl_penalty = self.kl_ctl.value * kl_div
        
        # Total loss
        total_loss = policy_loss.mean() + kl_penalty.mean()
        
        # Update actor
        self.actor_optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['max_grad_norm'])
        self.actor_optim.step()
        
        # Update KL controller
        self.kl_ctl.update(kl_div.mean().item())
        
        # Update critic
        critic_loss = F.mse_loss(values, chosen_rewards)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config['max_grad_norm'])
        self.critic_optim.step()
        
        return {
            'policy_loss': policy_loss.mean().item(),
            'critic_loss': critic_loss.item(),
            'kl_div': kl_div.mean().item(),
            'mean_reward': chosen_rewards.mean().item()
        }
    
    def compute_rewards(self, response_ids):
        """Compute rewards using reward model"""
        return self.reward_model(response_ids)
    
    def log_probs(self, logits, labels):
        """Compute log probabilities of labels given logits"""
        log_probs = F.log_softmax(logits, dim=-1)
        return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    
    def kl_divergence(self, policy_logits, ref_logits):
        """Compute KL divergence between policy and reference"""
        policy_probs = F.softmax(policy_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        return F.kl_div(torch.log(policy_probs), ref_probs, reduction='none').sum(dim=-1)

# ==================== KL Controller ====================
class AdaptiveKLController:
    def __init__(self, target_kl):
        self.target_kl = target_kl
        self.value = 0.1
        self.step_size = 0.01
        
    def update(self, current_kl):
        if current_kl < self.target_kl / 1.5:
            # KL too low - increase penalty
            self.value += self.step_size
        elif current_kl > self.target_kl * 1.5:
            # KL too high - decrease penalty
            self.value = max(0.0, self.value - self.step_size)

# ==================== RLHF Training Pipeline ====================
def train_rlhf(model, tokenizer, device, config):
    print("Starting RLHF training pipeline...")
    
    # Step 1: Prepare models
    base_model = model  # Our pretrained MoE Shakespeare model
    reward_model = ShakespeareRewardModel(base_model).to(device)
    critic_model = ShakespeareRewardModel(base_model).to(device)  # Separate critic
    actor_model = base_model  # We'll fine-tune the base model
    
    # Step 2: Create preference dataset
    pref_dataset = ShakespearePreferenceDataset(tokenizer)
    pref_dataset.simulate_preferences(model, num_samples=5000)
    
    # Step 3: Train reward model
    print("Training reward model...")
    reward_model = train_reward_model(reward_model, pref_dataset, tokenizer, device)
    
    # Step 4: PPO Training
    print("Starting PPO training...")
    ppo_trainer = ShakespearePPOTrainer(
        actor_model=actor_model,
        critic_model=critic_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config['ppo']
    )
    
    dataloader = DataLoader(pref_dataset, batch_size=config['ppo']['batch_size'], shuffle=True)
    
    for epoch in range(config['ppo']['epochs']):
        epoch_losses = []
        for batch in dataloader:
            metrics = ppo_trainer.train_step(batch)
            epoch_losses.append(metrics)
        
        # Print epoch summary
        avg_policy_loss = np.mean([m['policy_loss'] for m in epoch_losses])
        avg_critic_loss = np.mean([m['critic_loss'] for m in epoch_losses])
        avg_kl = np.mean([m['kl_div'] for m in epoch_losses])
        avg_reward = np.mean([m['mean_reward'] for m in epoch_losses])
        
        print(f"Epoch {epoch+1}/{config['ppo']['epochs']} | "
              f"Policy Loss: {avg_policy_loss:.4f} | "
              f"Critic Loss: {avg_critic_loss:.4f} | "
              f"KL Div: {avg_kl:.4f} | "
              f"Avg Reward: {avg_reward:.4f}")
        
        # Generate sample after each epoch
        sample_prompt = "To be, or not to be"
        generated = ppo_trainer.actor.generate(
            tokenizer.encode(sample_prompt, return_tensors='pt').to(device),
            max_new_tokens=32,
            temperature=0.7
        )
        print(f"Sample: {tokenizer.decode(generated[0], skip_special_tokens=True)}")
    
    return actor_model

def train_reward_model(model, dataset, tokenizer, device, epochs=3, lr=1e-4):
    """Train the reward model on preference data"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            prompt_ids = batch['prompt_ids'].to(device)
            chosen_ids = batch['chosen_ids'].to(device)
            rejected_ids = batch['rejected_ids'].to(device)
            
            # Compute rewards
            chosen_rewards = model(chosen_ids)
            rejected_rewards = model(rejected_ids)
            
            # Ranking loss
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
            
            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Reward Model Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
    
    return model

# ==================== Main Training Integration ====================
def main():
    # Configuration
    config = {
        'model': {
            'n_layer': 3, 
            'n_head': 3, 
            'num_experts': 3,
            'n_embd': 192, 
            'block_size': 128,
            'dropout': 0.1,
            'bias': True
        },
        'ppo': {
            'actor_lr': 1e-6,
            'critic_lr': 1e-5,
            'weight_decay': 1e-4,
            'kl_target': 0.1,
            'clip_epsilon': 0.2,
            'max_grad_norm': 1.0,
            'batch_size': 8,
            'epochs': 5
        }
    }
    
    # Prepare tokenizer and device
    enc = tiktoken.get_encoding("gpt2")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model_config = GPTConfig(**config['model'])
    model_config.vocab_size = enc.n_vocab
    model = GPT(model_config).to(device)
    
    # Load pretrained weights
    model.load_state_dict(torch.load('shakespeare_finetuned.pth'))
    model.eval()
    
    # RLHF Training
    rlhf_model = train_rlhf(
        model=model,
        tokenizer=enc,
        device=device,
        config=config
    )
    
    # Save RLHF-tuned model
    torch.save(rlhf_model.state_dict(), 'shakespeare_rlhf_tuned.pth')
    print("RLHF training complete. Model saved.")

if __name__ == "__main__":
    main()