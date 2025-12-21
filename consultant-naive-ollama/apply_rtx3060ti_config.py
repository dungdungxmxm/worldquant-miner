"""
Script to apply RTX 3060 Ti optimized configuration to the alpha generation system.

Usage:
    python apply_rtx3060ti_config.py

This script will:
1. Update alpha_orchestrator.py with optimized model fleet
2. Update alpha_generator_ollama.py with optimized settings
3. Create backup files before making changes
"""

import os
import shutil
import json
from datetime import datetime
from config_rtx3060ti import GPU_CONFIG, ALPHA_CONFIG, MINING_CONFIG

def create_backup(file_path):
    """Create a backup of the file before modifying."""
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_{timestamp}"
    shutil.copy2(file_path, backup_path)
    print(f"✅ Created backup: {backup_path}")
    return True

def update_orchestrator_model_fleet():
    """Update model fleet in alpha_orchestrator.py."""
    file_path = "alpha_orchestrator.py"
    
    if not create_backup(file_path):
        return False
    
    print(f"\n📝 Updating {file_path}...")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace model fleet
    old_fleet_start = 'self.model_fleet = ['
    old_fleet_end = ']'
    
    # Build new model fleet code
    new_fleet_lines = ['self.model_fleet = [']
    for model in GPU_CONFIG["model_fleet"]:
        new_fleet_lines.append(
            f'            ModelInfo("{model["name"]}", {model["vram_mb"]}, {model["priority"]}, "{model["description"]}"),'
        )
    new_fleet_lines.append('        ]')
    new_fleet_code = '\n'.join(new_fleet_lines)
    
    # Find the section to replace
    start_idx = content.find(old_fleet_start)
    if start_idx == -1:
        print(f"❌ Could not find model fleet in {file_path}")
        return False
    
    # Find the end of the list
    end_idx = content.find(']', start_idx)
    if end_idx == -1:
        print(f"❌ Could not find end of model fleet in {file_path}")
        return False
    
    # Replace the section
    new_content = content[:start_idx] + new_fleet_code + content[end_idx + 1:]
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Updated model fleet in {file_path}")
    return True

def update_generator_settings():
    """Update settings in alpha_generator_ollama.py."""
    file_path = "alpha_generator_ollama.py"
    
    if not create_backup(file_path):
        return False
    
    print(f"\n📝 Updating {file_path}...")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Update specific settings
    updates_made = 0
    for i, line in enumerate(lines):
        # Update temperature
        if "'temperature':" in line and "0.3" not in line:
            lines[i] = line.replace(line.split(':')[1].strip(), f" {ALPHA_CONFIG['temperature']},\n")
            updates_made += 1
        
        # Update max_tokens (num_predict)
        if "'num_predict':" in line:
            lines[i] = line.replace(line.split(':')[1].strip(), f" {ALPHA_CONFIG['max_tokens']},\n")
            updates_made += 1
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✅ Updated {updates_made} settings in {file_path}")
    return True

def create_config_summary():
    """Create a summary file of the applied configuration."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "gpu_model": "RTX 3060 Ti (8GB VRAM)",
        "configuration": {
            "gpu": GPU_CONFIG,
            "alpha": ALPHA_CONFIG,
            "mining": MINING_CONFIG
        },
        "changes_applied": [
            "Updated model fleet to prioritize smaller models",
            "Reduced GPU layers from 16 to 10",
            "Reduced GPU memory utilization from 0.75 to 0.55",
            "Reduced Docker memory limit from 8G to 6G",
            "Reduced Docker CPU limit from 3.0 to 2.5",
            "Reduced target alpha count from 100 to 50",
            "Reduced max tokens from 1000 to 800"
        ]
    }
    
    with open("rtx3060ti_config_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Created configuration summary: rtx3060ti_config_summary.json")

def main():
    """Main function to apply all configurations."""
    print("=" * 60)
    print("RTX 3060 Ti Configuration Applicator")
    print("=" * 60)
    
    print("\n🔧 Applying RTX 3060 Ti optimized configuration...")
    
    # Update files
    success = True
    success &= update_orchestrator_model_fleet()
    success &= update_generator_settings()
    
    if success:
        create_config_summary()
        print("\n" + "=" * 60)
        print("✅ Configuration applied successfully!")
        print("=" * 60)
        print("\n📋 Next steps:")
        print("1. Review the changes in the backup files")
        print("2. Restart Docker containers:")
        print("   cd consultant-naive-ollama")
        print("   docker-compose -f docker-compose.gpu.yml down")
        print("   docker-compose -f docker-compose.gpu.yml up --build -d")
        print("3. Monitor VRAM usage:")
        print("   nvidia-smi -l 1")
        print("4. Check logs:")
        print("   docker logs -f naive-ollma-gpu-consultant")
    else:
        print("\n❌ Some configurations failed to apply. Please check the errors above.")

if __name__ == "__main__":
    main()

