import gc
import psutil
import platform
import subprocess
import sys
from typing import Dict, Any

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
        'total_ram': psutil.virtual_memory().total / 1e9,  # GB
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_threads': psutil.cpu_count(logical=True)
    }
    
    if check_gpu_availability():
        info['device'] = torch.device('cuda')
        info['supports_cuda'] = True
        info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info['batch_size'] = min(32, int(info['memory_gb'] / 2))
        info['description'] = f"GPU: {torch.cuda.get_device_name(0)} ({info['memory_gb']:.1f}GB)"
        info['cuda_version'] = torch.version.cuda
    else:
        info['device'] = torch.device('cpu')
        info['batch_size'] = min(16, int(info['total_ram'] / 2))
        info['description'] = f"CPU: {platform.processor()}"
        info['memory_gb'] = info['total_ram']
    
    return info

def optimize_memory(hw_info: Dict[str, Any], model=None):
    """Apply memory optimizations based on hardware"""
    gc.collect()
    import torch
    
    if hw_info['supports_cuda']:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if model:
            model.cuda()
            torch.cuda.empty_cache()
    else:
        torch.backends.cudnn.enabled = False
        if model:
            model.cpu()
            
    # Set memory efficient options
    if model:
        model.gradient_checkpointing_enable()

def print_hardware_summary(hw_info: Dict[str, Any]):
    """Print hardware configuration summary"""
    print("\nHardware Configuration:")
    print(f"- {hw_info['description']}")
    print(f"- CPU Cores: {hw_info['cpu_count']} (Threads: {hw_info['cpu_threads']})")
    print(f"- Available Memory: {hw_info['memory_gb']:.1f} GB")
    print(f"- Device Type: {hw_info['device'].type}")
    print(f"- CUDA Available: {hw_info['supports_cuda']}")
    if hw_info['supports_cuda']:
        print(f"- CUDA Version: {hw_info['cuda_version']}")
    print(f"- Recommended Batch Size: {hw_info['batch_size']}")
