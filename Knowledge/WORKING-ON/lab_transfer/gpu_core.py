"""
OpenCL accelerated core for LAB Color Transfer.
"""
import numpy as np
import pyopencl as cl
import os

from .logger import get_logger

class LABColorTransferGPU:
    """
    GPU-accelerated version of LABColorTransfer using OpenCL.
    """
    def __init__(self):
        self.logger = get_logger("LABTransferGPU")
        self.context = None
        self.queue = None
        self.program = None
        self._initialize_opencl()

    def _initialize_opencl(self):
        """
        Initializes OpenCL context, queue, and compiles the kernel.
        """
        try:
            # Find a GPU device
            platform = cl.get_platforms()[0]
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if not devices:
                raise RuntimeError("No GPU device found for OpenCL.")
            
            self.context = cl.Context(devices)
            properties = cl.command_queue_properties.PROFILING_ENABLE
            self.queue = cl.CommandQueue(self.context, properties=properties)
            
            # Load and compile the kernel
            kernel_path = os.path.join(os.path.dirname(__file__), 'kernels.cl')
            with open(kernel_path, 'r') as f:
                kernel_code = f.read()
            
            self.program = cl.Program(self.context, kernel_code).build()
            self.logger.info("OpenCL initialized and kernel compiled successfully.")

        except Exception as e:
            self.logger.error(f"Failed to initialize OpenCL: {e}")
            self.context = None # Ensure we fallback to CPU

    def is_gpu_available(self) -> bool:
        """Check if GPU context is successfully initialized."""
        return self.context is not None

    def basic_lab_transfer_gpu(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """
        Performs statistical transfer on all LAB channels using OpenCL.
        """
        if not self.is_gpu_available():
            raise RuntimeError("GPU not available. Cannot perform GPU transfer.")

        h, w, _ = source_lab.shape
        total_pixels = h * w
        
        # Ensure data is float32, as OpenCL kernels often work best with this type
        source_lab_f32 = source_lab.astype(np.float32)
        target_lab_f32 = target_lab.astype(np.float32)
        result_lab_f32 = np.empty_like(source_lab_f32)

        # Create buffers on the device and explicitly copy data
        mf = cl.mem_flags
        source_buf = cl.Buffer(self.context, mf.READ_ONLY, source_lab_f32.nbytes)
        result_buf = cl.Buffer(self.context, mf.WRITE_ONLY, result_lab_f32.nbytes)
        cl.enqueue_copy(self.queue, source_buf, source_lab_f32) # Non-blocking copy

        # Calculate stats on the float32 arrays to ensure type consistency
        s_mean_l, s_std_l = np.mean(source_lab_f32[:,:,0]), np.std(source_lab_f32[:,:,0])
        t_mean_l, t_std_l = np.mean(target_lab_f32[:,:,0]), np.std(target_lab_f32[:,:,0])
        s_mean_a, s_std_a = np.mean(source_lab_f32[:,:,1]), np.std(source_lab_f32[:,:,1])
        t_mean_a, t_std_a = np.mean(target_lab_f32[:,:,1]), np.std(target_lab_f32[:,:,1])
        s_mean_b, s_std_b = np.mean(source_lab_f32[:,:,2]), np.std(source_lab_f32[:,:,2])
        t_mean_b, t_std_b = np.mean(target_lab_f32[:,:,2]), np.std(target_lab_f32[:,:,2])

        # Execute the kernel
        kernel = self.program.basic_lab_transfer
        kernel(self.queue, (total_pixels,), None, source_buf, result_buf,
               np.float32(s_mean_l), np.float32(s_std_l), np.float32(t_mean_l), np.float32(t_std_l),
               np.float32(s_mean_a), np.float32(s_std_a), np.float32(t_mean_a), np.float32(t_std_a),
               np.float32(s_mean_b), np.float32(s_std_b), np.float32(t_mean_b), np.float32(t_std_b),
               np.int32(total_pixels))

        # Add a hard synchronization point to ensure kernel completion
        self.queue.finish()

        # Read back the result
        cl.enqueue_copy(self.queue, result_lab_f32, result_buf).wait()

        return result_lab_f32.astype(source_lab.dtype) # Convert back to original dtype
