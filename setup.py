import os
import sys
import subprocess
import platform

def install_package(package: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def uninstall_torch():
    """Uninstall all PyTorch-related packages"""
    print("Removing existing PyTorch installations...")
    packages = ["torch", "torchvision", "torchaudio"]
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])
        except subprocess.CalledProcessError:
            pass

def check_cuda_available():
    """Check if CUDA is available on the system without requiring torch"""
    try:
        # Try importing CUDA toolkit
        import ctypes
        ctypes.CDLL("nvcuda.dll" if platform.system() == "Windows" else "libcuda.so.1")
        return True
    except:
        return False

def setup_environment():
    """Setup the project environment with correct dependencies"""
    # First install basic requirements
    print("Installing basic requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    install_package("psutil")
    
    # Remove any existing PyTorch installations
    uninstall_torch()
    
    # Check CUDA availability before installing PyTorch
    has_cuda = check_cuda_available()
    
    # Install PyTorch based on CUDA availability
    if has_cuda:
        print("CUDA detected, installing GPU version of PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio"
        ])
    else:
        print("No CUDA detected, installing CPU-only PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
    
    # Now install remaining requirements
    print("Installing remaining requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_base.txt"])
    
    # Final verification using torch
    import torch
    print(f"\nSetup complete! Using {'GPU' if torch.cuda.is_available() else 'CPU'} version of PyTorch")

if __name__ == "__main__":
    setup_environment()
