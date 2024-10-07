# https://raw.githubusercontent.com/dwancin/pycuts/main/pycuts/torch.py
import torch
from typing import Optional

def device(specified_device: Optional[torch.device] = None) -> torch.device:
    """
    Returns the appropriate device (cuda, mps, or cpu).
    
    Args:
        specified_device (torch.device, optional): A device explicitly specified by the user.
    
    Returns:
        torch.device: The best available device or the specified device.
    """
    return specified_device or torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )

def gpu(specified_device: Optional[torch.device] = None) -> bool:
    """
    Checks whether a GPU is available or not.
    
    Args:
        specified_device (torch.device, optional): A device explicitly specified by the user.

    Returns:
        bool: `True` if a GPU is available, `False` if only CPU is available.
    """
    current_device = device(specified_device)
    return current_device.type in ["mps", "cuda"]

def torch_dtype(specified_device: Optional[torch.device] = None) -> torch.dtype:
    """
    Returns the appropriate torch dtype (precision) based on the device.

    For CUDA and MPS devices, it defaults to torch.float16 for optimization.
    For CPU devices, it defaults to torch.float32.

    Args:
        specified_device (torch.device, optional): A device explicitly specified by the user.
    
    Returns:
        torch.dtype: The optimal torch data type for the device.
    """
    current_device = device(specified_device)

    if current_device.type == "cuda" or current_device.type == "mps":
        # Use half precision (float16) on CUDA and MPS devices for speed and memory efficiency
        return torch.float16
    else:
        # Use float32 on CPU (default for many operations)
        return torch.float32

def empty_cache(specified_device: Optional[torch.device] = None) -> None:
    """
    Clears the GPU memory to prevent out-of-memory errors.
    
    Args:
        specified_device (torch.device, optional): A device explicitly specified by the user.
    """
    current_device = device(specified_device)
    if current_device.type == "cuda":
        torch.cuda.empty_cache()
    elif current_device.type == "mps":
        torch.mps.empty_cache()

def synchronize(specified_device: Optional[torch.device] = None) -> None:
    """
    Waits for all kernels in all streams on the given device to complete.
    
    Args:
        specified_device (torch.device, optional): A device explicitly specified by the user.
    """
    current_device = device(specified_device)
    if current_device.type == "cuda":
        torch.cuda.synchronize()
    elif current_device.type == "mps":
        torch.mps.synchronize()

def device_count(specified_device: Optional[torch.device] = None) -> int:
    """
    Returns the number of available devices.
    
    Args:
        specified_device (torch.device, optional): A device explicitly specified by the user.
    
    Returns:
        int: The number of devices available.
    """
    current_device = device(specified_device)
    if current_device.type == "cuda":
        return torch.cuda.device_count()
    elif current_device.type == "mps":
        return 1  # Only one MPS device available on macOS
    else:
        return 1

def manual_seed(seed: int, specified_device: Optional[torch.device] = None) -> None:
    """
    Sets the seed for generating random numbers for reproducible behavior.
    
    Args:
        seed (int): The desired seed value.
        specified_device (torch.device, optional): A device explicitly specified by the user.
    
    Raises:
        ImportError: If random or numpy cannot be imported.
    """
    try:
        import random
        import numpy as np
    except ImportError as e:
        raise ImportError(f"Required module not found: {e.name}. Please ensure it is installed.") from e
    
    current_device = device(specified_device)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if current_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif current_device.type == "mps":
        torch.mps.manual_seed(seed)

