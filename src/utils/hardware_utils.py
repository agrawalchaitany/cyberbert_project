import gc
import psutil
import platform
import subprocess
import sys
from typing import Dict, Any, Optional
import os

def check_cuda_system_available() -> bool:
    """Check if CUDA is available at system level without requiring torch"""
    try:
        import ctypes
        ctypes.CDLL("nvcuda.dll" if platform.system() == "Windows" else "libcuda.so.1")
        return True
    except:
        return False

def check_gpu_availability() -> bool:
    """Check if GPU is available and working"""
    if not check_cuda_system_available():
        return False
        
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        
        # Test CUDA with small operation
        test_tensor = torch.tensor([1.]).cuda()
        test_tensor = test_tensor + 1
        return True
    except Exception as e:
        print(f"GPU check failed: {e}")
        return False

def get_torch_requirements() -> str:
    """Generate appropriate PyTorch requirements based on hardware"""
    if check_gpu_availability():
        return """
torch>=2.6.0
torchvision>=0.15.0
torchaudio>=2.0.0"""
    else:
        return """
--find-links https://download.pytorch.org/whl/cpu/torch_stable.html
torch==2.6.0+cpu
torchvision==0.15.0+cpu
torchaudio==2.0.0+cpu"""

def install_torch_dependencies():
    """Install correct PyTorch version based on hardware"""
    print("Checking PyTorch installation...")
    if check_gpu_availability():
        print("GPU detected, installing CUDA-enabled PyTorch")
        cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
    else:
        print("No GPU detected, installing CPU-only PyTorch")
        cmd = [sys.executable, "-m", "pip", "install", 
               "torch", "torchvision", "torchaudio", 
               "--index-url", "https://download.pytorch.org/whl/cpu"]
    
    subprocess.run(cmd, check=True)

def get_hardware_info() -> Dict[str, Any]:
    """Get detailed hardware information and optimal device"""
    import torch
    
    info = {
        'device': None,
        'batch_size': 8,
        'memory_gb': 0,
        'description': '',
        'supports_cuda': False,
        'supports_mps': False,  # For Apple Silicon
        'total_ram': psutil.virtual_memory().total / 1e9,  # GB
        'cpu_count': psutil.cpu_count(logical=False) or 1,  # Avoid None
        'cpu_threads': psutil.cpu_count(logical=True) or 2  # Avoid None
    }
    
    # Check for CUDA first
    if check_gpu_availability():
        info['device'] = torch.device('cuda')
        info['supports_cuda'] = True
        info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['batch_size'] = min(32, max(4, int(info['memory_gb'] / 1.5)))
        info['description'] = f"GPU: {info['gpu_name']} ({info['memory_gb']:.1f}GB)"
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
    # Then check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['device'] = torch.device('mps')
        info['supports_mps'] = True
        info['memory_gb'] = info['total_ram']  # Use system RAM as estimate
        info['batch_size'] = min(16, max(2, int(info['total_ram'] / 3)))
        info['description'] = f"Apple Silicon MPS ({info['memory_gb']:.1f}GB RAM)"
    # Default to CPU
    else:
        info['device'] = torch.device('cpu')
        info['memory_gb'] = info['total_ram']
        # More conservative batch size on CPU
        info['batch_size'] = min(8, max(1, int(info['total_ram'] / 4)))
        info['description'] = f"CPU: {platform.processor()}"
    
    # Set optimized parameters based on device
    info['half_precision'] = info['supports_cuda']  # Use half precision on CUDA
    info['gradient_checkpointing'] = info['memory_gb'] < 12  # Use gradient checkpointing on lower memory
    info['worker_threads'] = min(4, max(0, info['cpu_threads'] - 2))  # Leave some cores for system
    
    return info

def optimize_memory(hw_info: Dict[str, Any], model=None):
    """Apply memory optimizations based on hardware"""
    gc.collect()
    import torch
    
    # CUDA-specific optimizations
    if hw_info['supports_cuda']:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if model:
            model.cuda()
            torch.cuda.empty_cache()
    # MPS-specific optimizations (Apple Silicon)
    elif hw_info['supports_mps']:
        if model:
            model.to(hw_info['device'])
    # CPU-specific optimizations
    else:
        torch.backends.cudnn.enabled = False
        torch.set_num_threads(hw_info['cpu_threads'])
        if model:
            model.cpu()
            
    # Set memory efficient options for model if available
    if model and hasattr(model, 'gradient_checkpointing_enable') and hw_info['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

def print_hardware_summary(hw_info: Dict[str, Any]):
    """Print hardware configuration summary"""
    print("\nHardware Configuration:")
    print(f"- {hw_info['description']}")
    print(f"- CPU Cores: {hw_info['cpu_count']} (Threads: {hw_info['cpu_threads']})")
    print(f"- Available Memory: {hw_info['memory_gb']:.1f} GB")
    print(f"- Device Type: {hw_info['device'].type}")
    
    if hw_info['supports_cuda']:
        print(f"- CUDA Available: {hw_info['supports_cuda']}")
        print(f"- CUDA Version: {hw_info['cuda_version']}")
        print(f"- cuDNN Version: {hw_info['cudnn_version']}")
    elif hw_info.get('supports_mps', False):
        print(f"- Apple Silicon MPS Available: {hw_info['supports_mps']}")
    
    print(f"- Recommended Batch Size: {hw_info['batch_size']}")
    print(f"- Gradient Checkpointing: {'Enabled' if hw_info['gradient_checkpointing'] else 'Disabled'}")
    print(f"- Half Precision: {'Enabled' if hw_info['half_precision'] else 'Disabled'}")
    print(f"- DataLoader Workers: {hw_info['worker_threads']}")

def get_optimized_settings(hw_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate optimal hyperparameters based on hardware configuration
    """
    settings = {
        # Training settings
        'batch_size': hw_info['batch_size'],
        'max_length': 128,  # Consider shorter sequences for speed
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'mixed_precision': hw_info['supports_cuda'],
        'gradient_accumulation_steps': 1,
        
        # DataLoader settings
        'num_workers': hw_info['worker_threads'],
        'pin_memory': hw_info['supports_cuda'],
        
        # Trainer settings
        'eval_steps': 100 if hw_info['supports_cuda'] else 500,
        'warmup_ratio': 0.1,
        'gradient_checkpointing': hw_info['gradient_checkpointing'],
        'early_stopping_patience': 3,
    }
    
    # Adjust for different device types
    if hw_info['supports_cuda']:
        # On CUDA, we can use larger batches and faster evaluation
        settings['learning_rate'] = 5e-5  # Slightly higher learning rate
    elif hw_info.get('supports_mps', False):
        # Apple Silicon specifics
        settings['mixed_precision'] = False  # MPS doesn't support mixed precision yet
    else:
        # CPU optimizations
        settings['max_length'] = 96  # Even shorter sequences on CPU
        settings['learning_rate'] = 1e-5  # Lower learning rate for stability
        settings['batch_size'] = max(1, settings['batch_size'])  # Ensure batch size is at least 1
        
    return settings
