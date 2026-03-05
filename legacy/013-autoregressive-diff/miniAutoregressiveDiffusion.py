import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from einops import rearrange

# ===== Hyperparameters =====
config = {
    "image_size": 32,
    "channels": 1,  # Grayscale
    "batch_size": 64,
    "lr": 3e-4,
    "num_epochs": 30,
    "T": 1000,  # Diffusion timesteps
    "beta_schedule": "cosine",  # Noise schedule
    "embed_dim": 64,
    "num_res_blocks": 3,
    "attention_resolutions": [16],
    "dropout": 0.1,
    "ema_decay": 0.9999,
    "grad_clip": 1.0,
    "warmup_steps": 500,
    "curriculum_factor": 0.8
}

# ===== Diffusion Noise Scheduler =====
class NoiseScheduler:
    def __init__(self, T, beta_schedule="cosine"):
        self.T = T
        
        if beta_schedule == "linear":
            self.betas = torch.linspace(1e-4, 0.02, T)
        elif beta_schedule == "cosine":
            t = torch.arange(T, dtype=torch.float32)
            s = 0.008
            f = torch.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
            self.betas = torch.clip(1 - f[1:] / f[:-1], 0, 0.999)
        
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x_0, t):
        """Diffusion forward process"""
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

# ===== Model Architecture =====
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        h = self.dropout(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        q = self.q(h).view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        k = self.k(h).view(B, C, -1)  # [B, C, H*W]
        v = self.v(h).view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        attn = torch.softmax(torch.bmm(q, k) / np.sqrt(C), dim=-1)
        attn_out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        
        return x + self.proj_out(attn_out)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb
        self.register_buffer('emb', emb)
    
    def forward(self, t):
        emb = t[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class HybridImageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(config["embed_dim"]),
            nn.Linear(config["embed_dim"], config["embed_dim"]),
            nn.SiLU(),
            nn.Linear(config["embed_dim"], config["embed_dim"])
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(config["channels"], config["embed_dim"], 3, padding=1)
        
        # Downsample blocks
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        ch = config["embed_dim"]
        
        for _ in range(config["num_res_blocks"]):
            self.down_blocks.append(ResidualBlock(ch, ch * 2, config["dropout"]))
            ch *= 2
            
            if config["image_size"] in config["attention_resolutions"]:
                self.down_attns.append(AttentionBlock(ch))
        
        # Middle blocks
        self.mid_block1 = ResidualBlock(ch, ch, config["dropout"])
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch, config["dropout"])
        
        # Upsample blocks
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        
        for _ in range(config["num_res_blocks"]):
            if config["image_size"] in config["attention_resolutions"]:
                self.up_attns.append(AttentionBlock(ch))
            
            self.up_blocks.append(ResidualBlock(ch * 2, ch // 2, config["dropout"]))
            ch //= 2
        
        # Final layers
        self.final_norm = nn.GroupNorm(8, ch)
        self.final_conv = nn.Conv2d(ch, config["channels"], 3, padding=1)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Initial convolution
        h = self.init_conv(x)
        hs = [h]
        
        # Downsample path
        for block, attn in zip(self.down_blocks, self.down_attns):
            h = block(h)
            if attn is not None:
                h = attn(h)
            hs.append(h)
        
        # Middle blocks
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # Upsample path
        for block, attn in zip(self.up_blocks, self.up_attns):
            h = torch.cat([h, hs.pop()], dim=1)
            if attn is not None:
                h = attn(h)
            h = block(h)
        
        # Final layers
        h = F.silu(self.final_norm(h))
        return self.final_conv(h)

# ===== Training Utilities =====
class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}
    
    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

class WarmupLR(optim.lr_scheduler._LRScheduler):
    """Linear learning rate warmup"""
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [base_lr * min(1.0, self.last_epoch / self.warmup_steps) 
                for base_lr in self.base_lrs]

class CurriculumSampler:
    """Gradually increases image complexity during training"""
    def __init__(self, dataset, factor=0.8):
        self.dataset = dataset
        self.factor = factor
        self.current_threshold = 0.0
    
    def update(self, epoch):
        self.current_threshold = min(1.0, epoch * self.factor / len(self.dataset))
    
    def __iter__(self):
        indices = []
        for idx, (img, _) in enumerate(self.dataset):
            # Simple complexity measure: image variance
            complexity = img.var()
            if complexity > self.current_threshold:
                indices.append(idx)
        return iter(indices)

# ===== Training Loop with SOTA Techniques =====
def train_model(config):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize(config["image_size"]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    
    # Curriculum sampler
    sampler = CurriculumSampler(train_set, factor=config["curriculum_factor"])
    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], 
        sampler=sampler, num_workers=2
    )
    
    # Initialize model and optimizer
    model = HybridImageModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    
    # Learning rate scheduler
    scheduler = WarmupLR(optimizer, warmup_steps=config["warmup_steps"])
    
    # EMA
    ema = EMA(config["ema_decay"])
    ema.register(model)
    
    # Noise scheduler
    noise_scheduler = NoiseScheduler(config["T"], config["beta_schedule"])
    
    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        sampler.update(epoch)  # Update curriculum sampling
        
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            # Sample timesteps
            t = torch.randint(0, config["T"], (images.size(0), device=device)
            
            # Add noise
            noisy_images, noise = noise_scheduler.add_noise(images, t)
            
            # Predict noise
            pred_noise = model(noisy_images, t)
            
            # Loss with EMA smoothing
            loss = F.mse_loss(pred_noise, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            
            optimizer.step()
            scheduler.step()
            ema.update(model)
            
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{config['num_epochs']}] Batch [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")
    
    # Apply EMA weights
    ema.apply(model)
    
    # Save model
    torch.save(model.state_dict(), "hybrid_image_gen.pth")
    print("Training complete. Model saved.")
    
    return model

# ===== Sampling Functions =====
def sample(model, config, num_images=8, device="cuda"):
    """Sampling using diffusion reverse process"""
    model.eval()
    noise_scheduler = NoiseScheduler(config["T"], config["beta_schedule"])
    
    # Start from pure noise
    x = torch.randn(num_images, config["channels"], 
                   config["image_size"], config["image_size"], device=device)
    
    # Reverse diffusion process
    for t in range(config["T"]-1, -1, -1):
        t_tensor = torch.full((num_images,), t, device=device, dtype=torch.long)
        
        with torch.no_grad():
            # Predict noise
            pred_noise = model(x, t_tensor)
            
            # Reverse diffusion step
            alpha_t = noise_scheduler.alphas[t]
            alpha_bar_t = noise_scheduler.alpha_bars[t]
            beta_t = noise_scheduler.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            
            # Update x
            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise
            ) + torch.sqrt(beta_t) * noise
    
    # Denormalize and return
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    return x

# ===== Main Execution =====
if __name__ == "__main__":
    # Train the model
    trained_model = train_model(config)
    
    # Generate samples
    samples = sample(trained_model, config, num_images=8)
    
    # Display samples
    grid = torchvision.utils.make_grid(samples, nrow=4)
    torchvision.utils.save_image(grid, "generated_samples.png")
    print("Samples saved to generated_samples.png")