"""
Real-time system monitoring script for consultant-naive-ollama.

This script monitors:
1. Docker container status
2. GPU VRAM usage
3. Ollama API status
4. Alpha generation progress
5. Module statistics

Author: AI Assistant
Date: 2025-12-19
"""

import subprocess
import json
import time
import sys
from datetime import datetime

def run_command(cmd):
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def check_containers():
    """Check Docker container status."""
    print("\n" + "=" * 80)
    print("🐳 DOCKER CONTAINERS STATUS")
    print("=" * 80)
    
    cmd = 'docker ps --filter "name=consultant" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'
    output = run_command(cmd)
    print(output)

def check_gpu():
    """Check GPU VRAM usage."""
    print("\n" + "=" * 80)
    print("🎮 GPU STATUS")
    print("=" * 80)
    
    cmd = "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits"
    output = run_command(cmd)
    
    if "Error" not in output:
        parts = output.split(", ")
        if len(parts) >= 6:
            name = parts[0]
            mem_used = int(parts[1])
            mem_total = int(parts[2])
            gpu_util = int(parts[3])
            temp = int(parts[4])
            power = float(parts[5])
            
            mem_percent = (mem_used / mem_total) * 100
            
            print(f"GPU: {name}")
            print(f"VRAM: {mem_used} MB / {mem_total} MB ({mem_percent:.1f}%)")
            print(f"Utilization: {gpu_util}%")
            print(f"Temperature: {temp}°C")
            print(f"Power: {power:.1f}W")
            
            # Warning thresholds
            if mem_percent > 90:
                print("⚠️  WARNING: VRAM usage > 90%!")
            if temp > 80:
                print("⚠️  WARNING: Temperature > 80°C!")
    else:
        print(output)

def check_ollama():
    """Check Ollama API status."""
    print("\n" + "=" * 80)
    print("🤖 OLLAMA API STATUS")
    print("=" * 80)
    
    cmd = 'docker exec naive-ollma-gpu-consultant curl -s http://localhost:11434/api/tags'
    output = run_command(cmd)
    
    try:
        data = json.loads(output)
        models = data.get('models', [])
        
        print(f"Total models: {len(models)}")
        for model in models:
            name = model.get('name', 'Unknown')
            size_gb = model.get('size', 0) / (1024**3)
            print(f"  - {name} ({size_gb:.2f} GB)")
    except:
        print("❌ Ollama API not responding or error")

def check_modules():
    """Check improvement modules status."""
    print("\n" + "=" * 80)
    print("🔧 IMPROVEMENT MODULES STATUS")
    print("=" * 80)
    
    # Check RAG system
    cmd = 'docker exec naive-ollma-gpu-consultant python -c "from alpha_rag_system import AlphaRAGSystem; rag = AlphaRAGSystem(); stats = rag.get_statistics(); print(f\\"RAG: {stats[\'total_alphas\']} alphas, {stats[\'hopeful_alphas\']} hopeful\\")"'
    output = run_command(cmd)
    print(f"✅ {output}")
    
    # Check smart config selector
    cmd = 'docker exec naive-ollma-gpu-consultant python -c "from smart_config_selector import SmartConfigSelector; sel = SmartConfigSelector(); stats = sel.get_statistics(); print(f\\"Smart Config: {stats.get(\'total_configs_tested\', 0)} configs tested, {stats.get(\'overall_success_rate\', 0):.1%} success rate\\")"'
    output = run_command(cmd)
    print(f"✅ {output}")
    
    # Check feedback loop
    cmd = 'docker exec naive-ollma-gpu-consultant python -c "from feedback_loop_system import FeedbackLoopSystem; fb = FeedbackLoopSystem(); stats = fb.get_statistics(); print(f\\"Feedback Loop: {stats.get(\'total_alphas\', 0)} alphas, {stats.get(\'success_rate\', 0):.1%} success rate\\")"'
    output = run_command(cmd)
    print(f"✅ {output}")

def check_files():
    """Check important files."""
    print("\n" + "=" * 80)
    print("📁 IMPORTANT FILES")
    print("=" * 80)
    
    files = [
        'hopeful_alphas.json',
        'submission_log.json',
        'config_success_history.json',
        'feedback_history.json'
    ]
    
    for filename in files:
        cmd = f'docker exec naive-ollma-gpu-consultant test -f {filename} && echo "EXISTS" || echo "NOT FOUND"'
        output = run_command(cmd)
        
        if "EXISTS" in output:
            # Get file size
            cmd_size = f'docker exec naive-ollma-gpu-consultant stat -f%z {filename} 2>/dev/null || docker exec naive-ollma-gpu-consultant stat -c%s {filename} 2>/dev/null'
            size = run_command(cmd_size)
            try:
                size_kb = int(size) / 1024
                print(f"✅ {filename} ({size_kb:.1f} KB)")
            except:
                print(f"✅ {filename}")
        else:
            print(f"⏳ {filename} (not created yet)")

def check_logs():
    """Check recent logs."""
    print("\n" + "=" * 80)
    print("📋 RECENT LOGS (Last 5 lines)")
    print("=" * 80)
    
    cmd = 'docker logs naive-ollma-gpu-consultant --tail 5'
    output = run_command(cmd)
    print(output)

def main():
    """Main monitoring loop."""
    print("\n" + "=" * 80)
    print("🚀 CONSULTANT-NAIVE-OLLAMA SYSTEM MONITOR")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        check_containers()
        check_gpu()
        check_ollama()
        check_modules()
        check_files()
        check_logs()
        
        print("\n" + "=" * 80)
        print("✅ Monitoring complete!")
        print("=" * 80)
        print("\nTips:")
        print("  - Dashboard: http://localhost:5000")
        print("  - Ollama WebUI: http://localhost:3000")
        print("  - Run 'nvidia-smi -l 1' for continuous GPU monitoring")
        print("  - Run 'docker logs -f naive-ollma-gpu-consultant' for live logs")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

