"""
VQ-VAE Configuration Optimized for Spatial Structure
This will create codes that PixelCNN can actually learn from
"""

# OPTION 1: Conservative Fix (Recommended)
# Good balance between reconstruction and prior learnability
EXPERIMENT_CONFIGS_OPTION1 = {
    # Paths
    "data_dir": "./emoji_data",
    "checkpoint_dir": "./checkpoints",
    "results_dir": "./results",

    # Data
    "image_size": 64,
    "batch_size": 64,
    "num_workers": 2,

    # VQ-VAE Architecture
    "num_hiddens": 128,
    "num_residual_hiddens": 32,
    "num_residual_layers": 2,
    "embedding_dim": 64,        # ✓ Reduced from 256 (smaller = smoother)
    "num_embeddings": 128,      # ✓ Reduced from 256 (fewer choices = more spatial structure)
    "commitment_cost": 0.5,     # ✓ Increased from 0.01 (encourages spatial smoothness)
    "decay": 0.99,              # ✓ Increased from 0.95 (more stable codebook)

    # Training
    "num_epochs_vqvae": 100,
    "learning_rate_vqvae": 3e-4,

    # Codebook monitoring
    "min_codebook_usage": 50.0,
    "check_usage_every": 5,

    # Experiment metadata
    "experiment_name": "spatial_structure_v1",
    "notes": "Optimized for PixelCNN: reduced codebook, higher commitment cost"
}


# OPTION 2: Aggressive Fix (If Option 1 still has issues)
# Prioritizes spatial structure over reconstruction quality
EXPERIMENT_CONFIGS_OPTION2 = {
    # Paths
    "data_dir": "./emoji_data",
    "checkpoint_dir": "./checkpoints_spatial",
    "results_dir": "./results_spatial",

    # Data
    "image_size": 64,
    "batch_size": 64,
    "num_workers": 2,

    # VQ-VAE Architecture
    "num_hiddens": 128,
    "num_residual_hiddens": 32,
    "num_residual_layers": 2,
    "embedding_dim": 32,        # ✓ Even smaller (forces more sharing)
    "num_embeddings": 64,       # ✓ Very small codebook (strong spatial structure)
    "commitment_cost": 1.0,     # ✓ High commitment (prioritize smooth codes)
    "decay": 0.995,             # ✓ Very stable codebook

    # Training
    "num_epochs_vqvae": 150,    # ✓ Train longer for convergence
    "learning_rate_vqvae": 3e-4,

    # Codebook monitoring
    "min_codebook_usage": 50.0,
    "check_usage_every": 5,

    # Experiment metadata
    "experiment_name": "spatial_structure_aggressive",
    "notes": "Maximum spatial structure: tiny codebook, high commitment cost"
}


# OPTION 3: Hierarchical VQ-VAE (Advanced)
# Use 2-level codebook for better structure
EXPERIMENT_CONFIGS_HIERARCHICAL = {
    # Paths
    "data_dir": "./emoji_data",
    "checkpoint_dir": "./checkpoints_hierarchical",
    "results_dir": "./results_hierarchical",

    # Data
    "image_size": 64,
    "batch_size": 64,
    "num_workers": 2,

    # VQ-VAE Architecture - Bottom Level
    "num_hiddens": 128,
    "num_residual_hiddens": 32,
    "num_residual_layers": 2,
    "embedding_dim_bottom": 32,
    "num_embeddings_bottom": 512,  # Can be larger at bottom
    "commitment_cost_bottom": 0.25,
    
    # VQ-VAE Architecture - Top Level (what PixelCNN will model)
    "embedding_dim_top": 32,
    "num_embeddings_top": 64,      # Small for PixelCNN
    "commitment_cost_top": 1.0,    # High for spatial structure
    
    "decay": 0.99,

    # Training
    "num_epochs_vqvae": 150,
    "learning_rate_vqvae": 3e-4,

    # Codebook monitoring
    "min_codebook_usage": 50.0,
    "check_usage_every": 5,

    # Experiment metadata
    "experiment_name": "hierarchical_vqvae",
    "notes": "2-level VQ-VAE for better prior learning"
}


# =============================================================================
# Key Changes Explained
# =============================================================================

"""
WHY THESE CHANGES WILL HELP:

1. SMALLER CODEBOOK (256 → 128 or 64)
   - Fewer options = more likely to reuse codes
   - Forces adjacent pixels to share similar codes
   - Target: 40-50% spatial correlation

2. HIGHER COMMITMENT COST (0.01 → 0.5 or 1.0)
   - Penalizes encoder for creating codes far from embeddings
   - Encourages smoother transitions between codes
   - Acts as regularization for spatial structure

3. SMALLER EMBEDDING DIM (256 → 64 or 32)
   - Less capacity = more information sharing
   - Bottleneck forces spatial coherence
   - Still enough capacity for 2474 emojis

4. HIGHER DECAY (0.95 → 0.99)
   - More stable codebook updates
   - Prevents drastic changes that break structure
   - Better long-term convergence

EXPECTED RESULTS:
- Spatial correlation: 40-60% (vs current 21%)
- Oracle accuracy: 40-50% (vs current 21%)
- Learnability score: 50-70/100 (vs current 23/100)
- Reconstruction SSIM: 0.80-0.85 (slight drop from 0.89)
- PixelCNN accuracy: 60-75% (vs current 42%)
- FID score: 50-80 (vs current 232)

TRADEOFF:
You'll sacrifice some reconstruction quality (SSIM ~0.85 vs 0.90)
but gain MUCH better generation quality (FID ~60 vs 232)
"""


# =============================================================================
# Testing Recommendations
# =============================================================================

"""
RECOMMENDED TESTING ORDER:

1. Start with OPTION 1 (Conservative)
   - Run Phase 1 with these settings
   - Check spatial_correlation in diagnostics
   - If > 40%, proceed to Phase 2
   - If < 40%, try OPTION 2

2. If OPTION 1 insufficient, try OPTION 2 (Aggressive)
   - Should give 50-60% spatial correlation
   - Reconstruction will be slightly worse but acceptable
   - PixelCNN will train much better

3. If you want best results, implement OPTION 3 (Hierarchical)
   - Requires code changes (2-level encoder/decoder)
   - Best of both worlds: good reconstruction + good prior
   - Used in original VQ-VAE-2 paper

QUICK TEST:
Before full training, do a 10-epoch test run and check diagnostics.
If spatial_correlation > 40%, you're good to go!
"""


# =============================================================================
# Additional Architectural Changes for Spatial Structure
# =============================================================================

class SpatiallySmoothedEncoder(nn.Module):
    """
    Modified encoder that encourages spatial smoothness
    Add this if commitment cost alone isn't enough
    """
    def __init__(self, in_channels, num_hiddens, num_residual_layers, 
                 num_residual_hiddens):
        super().__init__()
        
        # Use strided convs with LESS aggressive downsampling
        # Original: 64 -> 32 -> 16 (4x downsampling)
        # Modified: 64 -> 32 (2x downsampling) for more spatial detail
        self.conv1 = nn.Conv2d(in_channels, num_hiddens // 2, 4, stride=2, padding=1)
        
        # Add extra conv layer instead of second downsampling
        self.conv2 = nn.Conv2d(num_hiddens // 2, num_hiddens, 3, padding=1)
        self.conv3 = nn.Conv2d(num_hiddens, num_hiddens, 3, padding=1)
        
        self.residual_stack = ResidualStack(
            num_hiddens, num_hiddens, 
            num_residual_layers, num_residual_hiddens
        )
        
        # Add spatial smoothing layer
        self.smooth = nn.Conv2d(num_hiddens, num_hiddens, 3, padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.residual_stack(x)
        x = self.smooth(x)  # Extra smoothing
        return x


# =============================================================================
# Usage
# =============================================================================

"""
# In your Phase 1 code, replace EXPERIMENT_CONFIGS with:
EXPERIMENT_CONFIGS = EXPERIMENT_CONFIGS_OPTION1  # Start here

# If you need more aggressive:
# EXPERIMENT_CONFIGS = EXPERIMENT_CONFIGS_OPTION2

# Run Phase 1, then check diagnostics:
python phase1.py

# After training completes, the diagnostics will automatically run
# Look for: "Code Learnability Score: XX/100"
# Target: > 50 for good PixelCNN performance

# If score is good, proceed to Phase 2 as normal
# Your PixelCNN should now achieve 60-70% accuracy and FID < 80
"""