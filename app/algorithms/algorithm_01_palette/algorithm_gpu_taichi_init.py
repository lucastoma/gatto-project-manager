# /app/algorithms/algorithm_01_palette/algorithm_gpu_taichi_init.py
# Module for Taichi initialization and management.

import os
import warnings
from typing import Optional, Tuple
import logging

# Global Taichi objects
ti = None
tm = None
TAICHI_AVAILABLE = False
GPU_BACKEND = None

def safe_taichi_init() -> bool:
    """Safely initialize Taichi with GPU support if available."""
    global ti, tm, TAICHI_AVAILABLE, GPU_BACKEND
    
    if TAICHI_AVAILABLE:
        return GPU_BACKEND is not None
        
    try:
        import taichi as _ti
        import taichi.math as _tm
        
        # Configure Taichi to use GPU if available
        if os.environ.get('TI_ENABLE_CUDA', '0') == '1':
            _ti.init(arch=_ti.gpu, debug=False, log_level='error')
            GPU_BACKEND = 'cuda'
        elif os.environ.get('TI_ENABLE_METAL', '0') == '1':
            _ti.init(arch=_ti.metal, debug=False, log_level='error')
            GPU_BACKEND = 'metal'
        else:
            _ti.init(arch=_ti.cpu, debug=False, log_level='error')
            GPU_BACKEND = 'cpu'
            
        ti = _ti
        tm = _tm
        TAICHI_AVAILABLE = True
        return True
        
    except ImportError:
        warnings.warn("Taichi not available. Falling back to CPU implementation.")
        return False
    except Exception as e:
        warnings.warn(f"Failed to initialize Taichi: {e}. Falling back to CPU implementation.")
        return False

def _safe_taichi_cleanup():
    """Clean up Taichi resources if initialized."""
    global ti, TAICHI_AVAILABLE
    if TAICHI_AVAILABLE and ti is not None:
        try:
            ti.reset()
        except Exception as e:
            logging.warning(f"Error during Taichi cleanup: {e}")
        finally:
            ti = None
            TAICHI_AVAILABLE = False

# Initialize Taichi on module import
safe_taichi_init()

# Clean up on program exit
import atexit
atexit.register(_safe_taichi_cleanup)
