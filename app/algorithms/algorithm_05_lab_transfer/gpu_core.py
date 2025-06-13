"""
OpenCL accelerated core for LAB Color Transfer.
"""
import numpy as np
import warnings

# Ignoruj specyficzne ostrzeżenie z PyOpenCL dotyczące cache'owania kerneli.
# Musi być wywołane PRZED importem pyopencl, aby zadziałało.
warnings.filterwarnings("ignore", category=UserWarning, message=".*pytools.persistent_dict.*")

try:
    import pyopencl as cl
    import pyopencl.tools
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
import os
import logging

from .logger import get_logger

class LABColorTransferGPU:
    """
    GPU-accelerated version of LABColorTransfer using OpenCL.
    """
    def __init__(self):
        if not PYOPENCL_AVAILABLE:
            raise ImportError("PyOpenCL not found. GPU acceleration is not available.")

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
            
            self.device = devices[0]
            self.context = cl.Context([self.device])
            logging.info(f"Successfully initialized OpenCL on device: {self.device.name}")
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

    def _calculate_histogram_gpu(self, lab_image_buf: 'cl.Buffer', total_pixels: int) -> np.ndarray:
        """Helper to calculate luminance histogram on the GPU."""
        hist_bins = 101
        host_hist = np.zeros(hist_bins, dtype=np.int32)
        hist_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_hist)

        kernel = self.program.calculate_histogram
        kernel(self.queue, (total_pixels,), None, lab_image_buf, hist_buf, np.int32(total_pixels))
        cl.enqueue_copy(self.queue, host_hist, hist_buf).wait()
        return host_hist

    def _calculate_stats(self, lab_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculates mean and std dev for each channel of a LAB image."""
        # Ensure calculation is done on float64 for precision, like in the CPU version
        lab_image_f64 = lab_image.astype(np.float64)
        mean = np.mean(lab_image_f64, axis=(0, 1))
        std = np.std(lab_image_f64, axis=(0, 1))
        return mean, std

    def _unified_lab_transfer_gpu(self, source_lab: np.ndarray, target_lab: np.ndarray, 
                                    weights: tuple = (1.0, 1.0, 1.0), selective_mode: bool = False) -> np.ndarray:
        """Internal method to run the unified OpenCL kernel."""
        if not self.is_gpu_available():
            raise RuntimeError("GPU not available. Cannot perform GPU transfer.")

        h, w, _ = source_lab.shape
        total_pixels = h * w

        s_mean, s_std = self._calculate_stats(source_lab)
        t_mean, t_std = self._calculate_stats(target_lab)

        source_lab_f32 = source_lab.astype(np.float32)
        result_lab_f32 = np.empty_like(source_lab_f32)

        mf = cl.mem_flags
        source_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source_lab_f32)
        result_buf = cl.Buffer(self.context, mf.WRITE_ONLY, result_lab_f32.nbytes)

        kernel = self.program.unified_lab_transfer
        kernel(self.queue, (total_pixels,), None,
               source_buf, result_buf,
               np.float32(s_mean[0]), np.float32(s_std[0]), np.float32(t_mean[0]), np.float32(t_std[0]),
               np.float32(s_mean[1]), np.float32(s_std[1]), np.float32(t_mean[1]), np.float32(t_std[1]),
               np.float32(s_mean[2]), np.float32(s_std[2]), np.float32(t_mean[2]), np.float32(t_std[2]),
               np.float32(weights[0]), np.float32(weights[1]), np.float32(weights[2]),
               np.int32(1 if selective_mode else 0),
               np.int32(total_pixels))

        cl.enqueue_copy(self.queue, result_lab_f32, result_buf).wait()
        return result_lab_f32.astype(source_lab.dtype)

    def basic_lab_transfer_gpu(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """GPU-accelerated basic LAB color transfer."""
        return self._unified_lab_transfer_gpu(source_lab, target_lab)

    def selective_lab_transfer_gpu(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """GPU-accelerated selective transfer (preserves source L channel)."""
        return self._unified_lab_transfer_gpu(source_lab, target_lab, selective_mode=True)

    def weighted_lab_transfer_gpu(self, source_lab: np.ndarray, target_lab: np.ndarray, 
                                  weights: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
        """GPU-accelerated weighted transfer."""
        return self._unified_lab_transfer_gpu(source_lab, target_lab, weights=weights)

    def adaptive_lab_transfer_gpu(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """
        Performs adaptive transfer by segmenting the image based on luminance.
        """
        if not self.is_gpu_available():
            raise RuntimeError("GPU not available. Cannot perform GPU transfer.")

        h, w, _ = source_lab.shape
        total_pixels = h * w
        mf = cl.mem_flags

        # --- Create buffers for source and target images ---
        source_lab_f32 = source_lab.astype(np.float32)
        target_lab_f32 = target_lab.astype(np.float32)
        source_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source_lab_f32)
        target_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_lab_f32)

        # --- Calculate histograms and percentiles ---
        source_hist = self._calculate_histogram_gpu(source_buf, total_pixels)
        target_hist = self._calculate_histogram_gpu(target_buf, total_pixels)

        def get_percentiles(hist):
            cdf = np.cumsum(hist)
            p33 = np.searchsorted(cdf, cdf[-1] * 0.33)
            p66 = np.searchsorted(cdf, cdf[-1] * 0.66)
            return float(p33), float(p66)

        s_p33, s_p66 = get_percentiles(source_hist)
        t_p33, t_p66 = get_percentiles(target_hist)
        self.logger.info(f"Source thresholds: {s_p33:.2f}, {s_p66:.2f}")
        self.logger.info(f"Target thresholds: {t_p33:.2f}, {t_p66:.2f}")

        # --- Create segmentation masks on GPU ---
        source_mask_buf = cl.Buffer(self.context, mf.READ_WRITE, total_pixels * np.dtype(np.int32).itemsize)
        target_mask_buf = cl.Buffer(self.context, mf.READ_WRITE, total_pixels * np.dtype(np.int32).itemsize)

        mask_kernel = self.program.create_segmentation_mask
        mask_kernel(self.queue, (total_pixels,), None, source_buf, source_mask_buf, np.float32(s_p33), np.float32(s_p66), np.int32(total_pixels))
        mask_kernel(self.queue, (total_pixels,), None, target_buf, target_mask_buf, np.float32(t_p33), np.float32(t_p66), np.int32(total_pixels))
        
        # --- Calculate stats for each segment (Hybrid approach) ---
        source_mask = np.empty(total_pixels, dtype=np.int32)
        target_mask = np.empty(total_pixels, dtype=np.int32)
        cl.enqueue_copy(self.queue, source_mask, source_mask_buf).wait()
        cl.enqueue_copy(self.queue, target_mask, target_mask_buf).wait()

        def _calculate_segment_stats(lab_image, mask):
            lab_image_flat = lab_image.reshape(-1, 3)
            stats = np.zeros(3 * 6, dtype=np.float32)
            for i in range(3):
                segment_pixels = lab_image_flat[mask == i]
                if segment_pixels.size > 0:
                    for j in range(3):
                        stats[i * 6 + j * 2 + 0] = np.mean(segment_pixels[:, j])
                        stats[i * 6 + j * 2 + 1] = np.std(segment_pixels[:, j])
            return stats

        s_stats = _calculate_segment_stats(source_lab_f32, source_mask)
        t_stats = _calculate_segment_stats(target_lab_f32, target_mask)

        # --- Apply segmented transfer on GPU ---
        s_stats_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=s_stats)
        t_stats_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_stats)
        result_buf = cl.Buffer(self.context, mf.WRITE_ONLY, source_lab_f32.nbytes)

        transfer_kernel = self.program.apply_segmented_transfer
        transfer_kernel(self.queue, (total_pixels,), None, 
                        source_buf, source_mask_buf, result_buf, 
                        s_stats_buf, t_stats_buf, np.int32(total_pixels))

        # Read result back to host
        result_lab_f32 = np.empty_like(source_lab_f32)
        cl.enqueue_copy(self.queue, result_lab_f32, result_buf).wait()

        return result_lab_f32.astype(source_lab.dtype)
