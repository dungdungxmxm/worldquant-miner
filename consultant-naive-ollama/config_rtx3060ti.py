"""
RTX 3060 Ti (8GB VRAM) Optimized Configuration
This file contains optimized settings for running the alpha generation system on RTX 3060 Ti.

Author: AI Assistant
Date: 2025-12-19
"""

# ========================================
# GPU Configuration for RTX 3060 Ti (8GB VRAM)
# ========================================

GPU_CONFIG = {
    # Model Fleet - Optimized for 8GB VRAM
    # Priority order: smallest to largest (to avoid VRAM issues)
    "model_fleet": [
        {
            "name": "deepseek-r1:7b",
            "vram_mb": 1100,
            "priority": 1,
            "description": "DeepSeek-R1 1.5B - Primary model for RTX 3060 Ti (lightweight)",
            "recommended": True
        },
        {
            "name": "llama3.2:3b",
            "vram_mb": 2048,
            "priority": 2,
            "description": "Llama 3.2 3B - Fallback model (good balance)",
            "recommended": True
        },
        {
            "name": "phi3:mini",
            "vram_mb": 2200,
            "priority": 3,
            "description": "Phi3 mini - Secondary fallback",
            "recommended": False
        },
        {
            "name": "deepseek-r1:7b",
            "vram_mb": 4700,
            "priority": 4,
            "description": "DeepSeek-R1 7B - Use only if VRAM allows",
            "recommended": False
        }
    ],
    
    # Ollama Settings
    "ollama": {
        "gpu_layers": 20,              # Number of layers to load on GPU (reduced for 8GB)
        "num_parallel": 1,             # Number of parallel requests (keep at 1)
        "gpu_memory_utilization": 0.8, # 55% of 8GB = ~4.4GB
        "gpu_memory_fraction": 0.8,   # Same as utilization
        "max_loaded_models": 1,        # Only keep 1 model in VRAM
        "num_gpu": 1                   # Single GPU
    },
    
    # Docker Resource Limits
    "docker": {
        "memory_gb": 10,    # RAM limit (reduced from 8G)
        "cpus": 4.0       # CPU cores (reduced from 3.0)
    },
    
    # VRAM Monitoring
    "vram_monitoring": {
        "enabled": True,
        "check_interval_seconds": 30,
        "max_vram_errors": 3,          # Downgrade after 2 errors (reduced from 3)
        "vram_threshold_percent": 90,  # Alert if VRAM usage > 90%
        "auto_downgrade": True
    }
}

# ========================================
# Alpha Generation Configuration
# ========================================

ALPHA_CONFIG = {
    # Batch Processing
    "batch_size": 1,           # Number of alphas to generate per batch (keep small for 8GB VRAM)
    "max_concurrent": 1,       # Max concurrent simulations (keep at 1 to avoid VRAM spikes)
    
    # Generation Settings
    "target_alpha_count": 50,  # Reduced from 100 to 50 per cycle (faster iterations)
    "temperature": 0.3,        # AI model temperature (low for consistency)
    "top_p": 0.9,             # Nucleus sampling
    "max_tokens": 800,        # Reduced from 1000 to 800 (faster generation)
    
    # Simulation Settings (Base Configuration)
    "simulation": {
        "region": "USA",
        "universe": "TOP3000",
        "delay": 1,
        "decay": 0,
        "neutralization": "INDUSTRY",
        "truncation": 0.08,
        "pasteurization": "ON",
        "nanHandling": "OFF"
    },
    
    # Hopeful Alpha Criteria
    "hopeful_criteria": {
        "min_fitness": 0.5,    # Minimum fitness to save as hopeful
        "min_sharpe": 1.0,     # Minimum Sharpe ratio
        "max_turnover": 0.7,   # Maximum turnover
        "min_turnover": 0.01   # Minimum turnover
    }
}

# ========================================
# Mining Configuration
# ========================================

MINING_CONFIG = {
    # Parameter Variation
    "auto_mode": True,         # Use automatic parameter ranges
    "integer_range_percent": 0.2,  # ±20% for integer parameters
    "float_range_percent": 0.1,    # ±10% for float parameters
    "variation_steps": 5,          # Number of steps in parameter range
    
    # Configuration Testing
    "test_all_configs": False,     # Set to False to reduce testing (use smart selection)
    "max_configs_per_alpha": 50,   # Reduced from 1000+ to 50 (prioritize best configs)
    "prioritize_configs": True,    # Use historical success rate to prioritize
    
    # Early Stopping
    "early_stopping": {
        "enabled": True,
        "min_fitness_threshold": 0.3,  # Stop if fitness < 0.3 after 10 configs
        "check_after_n_configs": 10
    }
}

# ========================================
# Monitoring & Logging
# ========================================

MONITORING_CONFIG = {
    "log_level": "INFO",
    "log_file": "alpha_orchestrator.log",
    "dashboard_port": 5000,
    "ollama_webui_port": 3000,
    
    # Performance Metrics
    "track_metrics": True,
    "metrics_file": "performance_metrics.json",
    "metrics_include": [
        "vram_usage",
        "generation_time",
        "simulation_time",
        "success_rate",
        "model_performance"
    ]
}

# ========================================
# Helper Functions
# ========================================

def get_recommended_model():
    """Get the recommended model for RTX 3060 Ti."""
    for model in GPU_CONFIG["model_fleet"]:
        if model.get("recommended"):
            return model["name"]
    return "deepseek-r1:7b"  # Default fallback

def get_vram_safe_batch_size():
    """Calculate safe batch size based on VRAM."""
    return ALPHA_CONFIG["batch_size"]

def should_downgrade_model(current_vram_mb, model_vram_mb):
    """Check if model should be downgraded based on VRAM usage."""
    total_vram_mb = 8192  # RTX 3060 Ti total VRAM
    usage_percent = (current_vram_mb / total_vram_mb) * 100
    return usage_percent > GPU_CONFIG["vram_monitoring"]["vram_threshold_percent"]

