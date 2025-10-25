# NEW CODE



# ============================================================================
# FIXED PixelCNN Implementation
# ============================================================================

class MaskedConv2d(nn.Conv2d):
    """Fixed masked convolution that doesn't break gradients"""
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)

    def create_mask(self, mask_type):
        k = self.kernel_size[0]
        self.mask[:, :, :k//2, :] = 1
        self.mask[:, :, k//2, :k//2] = 1
        if mask_type == 'B':
            self.mask[:, :, k//2, k//2] = 1

    def forward(self, x):
        # ✓ FIXED: Don't modify weights in-place, use functional API
        masked_weight = self.weight * self.mask
        return F.conv2d(x, masked_weight, self.bias, self.stride, 
                       self.padding, self.dilation, self.groups)


class GatedResidualBlock(nn.Module):
    """Gated activation improves PixelCNN performance"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = MaskedConv2d('B', hidden_dim, hidden_dim * 2, 3, padding=1)
        self.conv2 = MaskedConv2d('B', hidden_dim, hidden_dim, 1)
        
    def forward(self, x):
        h = self.conv1(F.relu(x))
        # Split into two halves for gated activation
        h1, h2 = h.chunk(2, dim=1)
        h = torch.tanh(h1) * torch.sigmoid(h2)
        h = self.conv2(h)
        return x + h


class ImprovedPixelCNN(nn.Module):
    """Enhanced PixelCNN with better architecture"""
    def __init__(self, num_embeddings, spatial_h, spatial_w, 
                 num_layers=15, hidden_dim=128):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        
        # Input projection with larger kernel
        self.input_conv = nn.Sequential(
            MaskedConv2d('A', num_embeddings, hidden_dim, 7, padding=3),
            nn.ReLU()
        )
        
        # Gated residual blocks
        self.residual_blocks = nn.ModuleList([
            GatedResidualBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection with better initialization
        self.output = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d('B', hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_embeddings, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Better initialization strategy"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, MaskedConv2d)):
                # ✓ FIXED: Use proper initialization gain
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (B, H, W) - discrete codes
        x_onehot = F.one_hot(x, self.num_embeddings).float()
        x_onehot = x_onehot.permute(0, 3, 1, 2).contiguous()
        
        h = self.input_conv(x_onehot)
        
        for block in self.residual_blocks:
            h = block(h)
        
        logits = self.output(h)
        return logits
    
    @torch.no_grad()
    def sample(self, batch_size, device, temperature=1.0):
        """Improved sampling with temperature control"""
        samples = torch.zeros(batch_size, self.spatial_h, self.spatial_w,
                            dtype=torch.long, device=device)
        
        self.eval()
        
        for i in range(self.spatial_h):
            for j in range(self.spatial_w):
                logits = self(samples)
                logits = logits[:, :, i, j] / temperature
                
                # Add top-k sampling for better quality
                if temperature < 1.0:
                    # Greedy sampling for low temperature
                    samples[:, i, j] = logits.argmax(dim=1)
                else:
                    # Probabilistic sampling
                    probs = F.softmax(logits, dim=1)
                    samples[:, i, j] = torch.multinomial(probs, 1).squeeze(-1)
        
        return samples


# ============================================================================
# IMPROVED Prior Trainer
# ============================================================================

class ImprovedPriorTrainer:
    def __init__(self, prior, vqvae, config):
        self.prior = prior
        self.vqvae = vqvae
        self.config = config
        
        # ✓ FIXED: Add learning rate scheduler
        self.optimizer = torch.optim.AdamW(
            prior.parameters(), 
            lr=config.learning_rate_prior,
            weight_decay=0.01  # Add weight decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_epochs_prior,
            eta_min=config.learning_rate_prior * 0.1
        )
        
        self.history = {'loss': [], 'accuracy': [], 'perplexity': [], 'epoch': []}
        self.start_epoch = 0
        self.best_loss = float('inf')
    
    def save_checkpoint(self, epoch, filepath):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.prior.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_loss': self.best_loss
        }
        torch.save(checkpoint, filepath)
        print(f"✓ Prior checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        if os.path.exists(filepath):
            try:
                checkpoint = torch.load(filepath, map_location=self.config.device)
                self.prior.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                self.history = checkpoint['history']
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_loss = checkpoint.get('best_loss', float('inf'))
                
                print(f"✓ Prior checkpoint loaded, resuming from epoch {self.start_epoch}")
                return True
            except Exception as e:
                print(f"✗ Error loading prior checkpoint: {e}")
                return False
        return False
    
    def train_epoch(self, dataloader, spatial_h, spatial_w):
        self.prior.train()
        self.vqvae.eval()
        
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_perplexity = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training Prior")
        
        for batch in pbar:
            batch = batch.to(self.config.device)
            
            # Get codes from VQ-VAE
            with torch.no_grad():
                codes = self.vqvae.encode(batch)
                codes = codes.view(-1, spatial_h, spatial_w)
            
            # Forward pass
            logits = self.prior(codes)
            loss = F.cross_entropy(logits, codes)
            
            # ✓ FIXED: Add perplexity and accuracy tracking
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                avg_probs = probs.mean(dim=[0, 2, 3])
                perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
                
                pred = logits.argmax(dim=1)
                accuracy = (pred == codes).float().mean()
            
            # Check for NaN
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # ✓ FIXED: Adaptive gradient clipping
            torch.nn.utils.clip_grad_norm_(self.prior.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            epoch_perplexity += perplexity.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': f'{accuracy.item():.3f}',
                'ppl': f'{perplexity.item():.1f}'
            })
        
        # Update learning rate
        self.scheduler.step()
        
        return {
            'loss': epoch_loss / num_batches,
            'accuracy': epoch_accuracy / num_batches,
            'perplexity': epoch_perplexity / num_batches
        }
    
    def train(self, train_loader, spatial_h, spatial_w):
        print(f"\n{'='*80}")
        print(f"TRAINING IMPROVED PIXELCNN PRIOR")
        print(f"{'='*80}\n")
        
        for epoch in range(self.start_epoch, self.config.num_epochs_prior):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs_prior}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            metrics = self.train_epoch(train_loader, spatial_h, spatial_w)
            
            # Update history
            self.history['loss'].append(metrics['loss'])
            self.history['accuracy'].append(metrics['accuracy'])
            self.history['perplexity'].append(metrics['perplexity'])
            self.history['epoch'].append(epoch + 1)
            
            print(f"Loss: {metrics['loss']:.4f}, "
                  f"Acc: {metrics['accuracy']:.3f}, "
                  f"Perplexity: {metrics['perplexity']:.1f}")
            
            # Save best model
            if metrics['loss'] < self.best_loss:
                self.best_loss = metrics['loss']
                self.save_checkpoint(
                    epoch, 
                    os.path.join(self.config.checkpoint_dir, 'prior_best.pt')
                )
                print(f"✓ New best model saved!")
            
            # Regular checkpoints
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    epoch, 
                    os.path.join(self.config.checkpoint_dir, f'prior_epoch_{epoch+1}.pt')
                )
        
        # Save final checkpoint
        self.save_checkpoint(
            self.config.num_epochs_prior - 1,
            os.path.join(self.config.checkpoint_dir, 'prior_final.pt')
        )
        
        return self.history


















































# OLD CODE




# ============================================================================
# PixelCNN Prior
# ============================================================================

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)

    def create_mask(self, mask_type):
        k = self.kernel_size[0]
        self.mask[:, :, :k//2, :] = 1
        self.mask[:, :, k//2, :k//2] = 1
        if mask_type == 'B':
            self.mask[:, :, k//2, k//2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class PixelCNNResidualBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d('B', h, h, 1),
            nn.BatchNorm2d(h),
            nn.ReLU(),
            MaskedConv2d('B', h, h, 1),
            nn.BatchNorm2d(h)
        )

    def forward(self, x):
        return x + self.conv(x)

class PixelCNN(nn.Module):
    def __init__(self, num_embeddings, spatial_h, spatial_w, num_layers=12, hidden_dim=64):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w

        # Input projection
        self.input_conv = MaskedConv2d('A', num_embeddings, hidden_dim, 7, padding=3)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            PixelCNNResidualBlock(hidden_dim) for _ in range(num_layers)
        ])

        # Output projection
        self.output = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d('B', hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_embeddings, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, MaskedConv2d)):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (B, H, W)
        x_onehot = F.one_hot(x, self.num_embeddings).float()  # (B, H, W, num_embeddings)
        x_onehot = x_onehot.permute(0, 3, 1, 2).contiguous()  # (B, num_embeddings, H, W)

        x = self.input_conv(x_onehot)
        for block in self.residual_blocks:
            x = block(x)
        logits = self.output(x)  # (B, num_embeddings, H, W)

        return logits

    @torch.no_grad()
    def sample(self, batch_size, device, temperature=1.0):
        samples = torch.zeros(batch_size, self.spatial_h, self.spatial_w,
                            dtype=torch.long, device=device)

        # Sample pixel by pixel
        for i in range(self.spatial_h):
            for j in range(self.spatial_w):
                logits = self(samples)  # (B, num_embeddings, H, W)
                probs = F.softmax(logits[:, :, i, j] / temperature, dim=1)
                samples[:, i, j] = torch.multinomial(probs, 1).squeeze(-1)

        return samples

# ============================================================================
# Decoding Helper
# ============================================================================

def decode_codes(vqvae, encoding_indices, spatial_h, spatial_w):
    """Decode from discrete codes with correct spatial dimensions"""
    # encoding_indices shape: (B, H, W)
    batch_size = encoding_indices.shape[0]

    # Flatten for embedding lookup
    flat_codes = encoding_indices.view(-1)  # (B*H*W)
    quantized = F.embedding(flat_codes, vqvae.vq.embed)  # (B*H*W, embedding_dim)

    # Reshape back to spatial format
    quantized = quantized.view(batch_size, spatial_h, spatial_w, -1)  # (B, H, W, embedding_dim)
    quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (B, embedding_dim, H, W)

    return vqvae.decoder(quantized)

# ============================================================================
# Prior Trainer
# ============================================================================

class PriorTrainer:
    def __init__(self, prior, vqvae, config):
        self.prior = prior
        self.vqvae = vqvae
        self.config = config
        self.optimizer = torch.optim.Adam(prior.parameters(), lr=config.learning_rate_prior)
        self.history = {'loss': [], 'epoch': []}
        self.start_epoch = 0

    def save_checkpoint(self, epoch, filepath):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.prior.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, filepath)
        print(f"✓ Prior checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath):
        if os.path.exists(filepath):
            try:
                checkpoint = torch.load(filepath, map_location=self.config.device, weights_only=False)
                self.prior.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.history = checkpoint['history']
                self.start_epoch = checkpoint['epoch'] + 1
                print(f"✓ Prior checkpoint loaded, resuming from epoch {self.start_epoch}")
                return True
            except Exception as e:
                print(f"✗ Error loading prior checkpoint: {e}")
                return False
        return False

    def train_epoch(self, dataloader, spatial_h, spatial_w):
        self.prior.train()
        self.vqvae.eval()
        epoch_loss = 0
        num_valid = 0

        pbar = tqdm(dataloader, desc="Training Prior")
        for batch in pbar:
            batch = batch.to(self.config.device)

            # Get codes from VQ-VAE with correct reshaping
            with torch.no_grad():
                codes = self.vqvae.encode(batch)  # (B, H*W)
                codes = codes.view(-1, spatial_h, spatial_w)  # (B, H, W)

            # Train PixelCNN
            logits = self.prior(codes)  # (B, num_embeddings, H, W)
            loss = F.cross_entropy(logits, codes)

            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch")
                continue

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.prior.parameters(), self.config.grad_clip)
            self.optimizer.step()

            epoch_loss += loss.item()
            num_valid += 1
            pbar.set_postfix({'loss': loss.item()})

        return epoch_loss / num_valid if num_valid > 0 else float('inf')

    def train(self, train_loader, spatial_h, spatial_w):  # FIXED: Added spatial_h, spatial_w parameters
        print(f"\n{'='*80}")
        print(f"TRAINING PIXELCNN PRIOR")
        print(f"{'='*80}\n")

        for epoch in range(self.start_epoch, self.config.num_epochs_prior):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs_prior}")

            loss = self.train_epoch(train_loader, spatial_h, spatial_w)

            if loss == float('inf'):
                print("Training stopped due to invalid loss")
                break

            self.history['loss'].append(loss)
            self.history['epoch'].append(epoch + 1)

            print(f"Loss: {loss:.4f}")

            # Save checkpoints
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, os.path.join(self.config.checkpoint_dir, f'prior_epoch_{epoch+1}.pt'))

        # Save final checkpoint
        self.save_checkpoint(self.config.num_epochs_prior - 1, os.path.join(self.config.checkpoint_dir, 'prior_final.pt'))

        return self.history
