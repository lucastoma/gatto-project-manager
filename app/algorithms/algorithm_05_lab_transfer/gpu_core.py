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
            platform = cl.get_platforms()[0]
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if not devices:
                raise RuntimeError("No GPU device found for OpenCL.")
            
            self.device = devices[0]
            self.context = cl.Context([self.device])
            logging.info(f"Successfully initialized OpenCL on device: {self.device.name}")
            properties = cl.command_queue_properties.PROFILING_ENABLE
            self.queue = cl.CommandQueue(self.context, properties=properties)
            
            kernel_path = os.path.join(os.path.dirname(__file__), 'kernels.cl')
            with open(kernel_path, 'r') as f:
                kernel_code = f.read()
            
            self.program = cl.Program(self.context, kernel_code).build()
            self.logger.info("OpenCL initialized and kernel compiled successfully.")
        except Exception as e:
            self.logger.error(f"OpenCL initialization failed: {e}")
            self.context = None

    def is_gpu_available(self):
        return self.context is not None

    def _calculate_stats(self, lab_image_buf, total_pixels, data_offset_pixels: int = 0):
        """
        Calculates mean and std dev for a LAB image buffer on the GPU using parallel reduction.
        """
        mf = cl.mem_flags
        work_group_size = 256
        num_groups = (total_pixels + work_group_size - 1) // work_group_size
        global_size = num_groups * work_group_size

        partial_sums_buf = cl.Buffer(self.context, mf.WRITE_ONLY, num_groups * 6 * 4)
        local_sums = cl.LocalMemory(work_group_size * 6 * 4)

        if total_pixels == 0:
            # Return zero means and stds if the segment (or image) is empty
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        kernel = self.program.stats_partial_reduce
        kernel(self.queue, (global_size,), (work_group_size,),
               lab_image_buf, partial_sums_buf, local_sums, np.int32(total_pixels), np.int32(data_offset_pixels))

        partial_sums = np.empty((num_groups, 6), dtype=np.float32)
        cl.enqueue_copy(self.queue, partial_sums, partial_sums_buf).wait()

        final_sums = np.sum(partial_sums, axis=0)
        
        mean = np.array([final_sums[0], final_sums[2], final_sums[4]]) / total_pixels
        
        mean_sq = mean**2
        var = np.array([
            (final_sums[1] / total_pixels) - mean_sq[0],
            (final_sums[3] / total_pixels) - mean_sq[1],
            (final_sums[5] / total_pixels) - mean_sq[2]
        ])
        var = np.maximum(var, 0)
        std = np.sqrt(var)

        return mean.astype(np.float32), std.astype(np.float32)

    def basic_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray, **kwargs) -> np.ndarray:
        mf = cl.mem_flags
        source_lab_f32 = source_lab.astype(np.float32)
        target_lab_f32 = target_lab.astype(np.float32)
        h, w, _ = source_lab.shape
        total_pixels = h * w

        source_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source_lab_f32)
        target_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_lab_f32)
        result_buf = cl.Buffer(self.context, mf.WRITE_ONLY, source_lab_f32.nbytes)

        s_mean, s_std = self._calculate_stats(source_buf, total_pixels)
        t_mean, t_std = self._calculate_stats(target_buf, total_pixels)

        self.program.basic_transfer(self.queue, (total_pixels,), None, 
                                    source_buf, result_buf, 
                                    cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=s_mean),
                                    cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=s_std),
                                    cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_mean),
                                    cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_std),
                                    np.int32(total_pixels))

        result_lab_f32 = np.empty_like(source_lab_f32)
        cl.enqueue_copy(self.queue, result_lab_f32, result_buf).wait()
        return result_lab_f32.astype(source_lab.dtype)

    def hybrid_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray, **kwargs) -> np.ndarray:
        """
        Performs hybrid LAB color transfer on the GPU.
        Currently, this delegates to the adaptive_lab_transfer method on the GPU.
        kwargs are accepted to match the signature in core.py but are not used by adaptive_lab_transfer_gpu.
        """
        if not self.is_gpu_available():
            self.logger.error("GPU not available, cannot perform hybrid_transfer_gpu.")
            raise RuntimeError("GPU not available for hybrid transfer")
        
        self.logger.info("Executing GPU hybrid_transfer (delegating to adaptive_lab_transfer_gpu).")
        # Adaptive LAB transfer on GPU handles its parameters internally or uses defaults.
        # It does not currently accept external kwargs like num_segments, delta_e_threshold etc.
        # If these were to be passed, adaptive_lab_transfer_gpu would need modification.
        return self.adaptive_lab_transfer(source_lab, target_lab)

    def selective_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray, **kwargs) -> np.ndarray:
        mask = kwargs.get('mask')
        if mask is None:
            raise ValueError("Mask is required for selective transfer on GPU.")
        
        selective_channels = kwargs.get('selective_channels', ['L', 'a', 'b'])
        blend_factor = float(kwargs.get('blend_factor', 1.0))

        mf = cl.mem_flags
        source_lab_f32 = source_lab.astype(np.float32)
        target_lab_f32 = target_lab.astype(np.float32)
        h, w, _ = source_lab.shape
        total_pixels = h * w

        if mask.shape[:2] != (h, w):
            raise ValueError("Mask must have the same dimensions as the source image.")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_u8 = mask.astype(np.uint8)

        source_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source_lab_f32)
        target_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_lab_f32)
        mask_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mask_u8)
        result_buf = cl.Buffer(self.context, mf.WRITE_ONLY, source_lab_f32.nbytes)

        s_mean, s_std = self._calculate_stats(source_buf, total_pixels)
        t_mean, t_std = self._calculate_stats(target_buf, total_pixels)

        process_l = 1 if 'L' in selective_channels else 0
        process_a = 1 if 'a' in selective_channels else 0
        process_b = 1 if 'b' in selective_channels else 0

        self.program.selective_transfer(self.queue, (total_pixels,), None,
                                        source_buf, result_buf, mask_buf,
                                        cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=s_mean),
                                        cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=s_std),
                                        cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_mean),
                                        cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_std),
                                        np.float32(blend_factor),
                                        np.int32(process_l), np.int32(process_a), np.int32(process_b),
                                        np.int32(total_pixels))

        result_lab_f32 = np.empty_like(source_lab_f32)
        cl.enqueue_copy(self.queue, result_lab_f32, result_buf).wait()
        return result_lab_f32.astype(source_lab.dtype)

    def adaptive_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray, **kwargs) -> np.ndarray:
        mf = cl.mem_flags
        source_lab_f32 = source_lab.astype(np.float32)
        target_lab_f32 = target_lab.astype(np.float32)
        h, w, _ = source_lab.shape
        total_pixels = h * w

        source_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source_lab_f32)
        target_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_lab_f32)
        source_mask_buf = cl.Buffer(self.context, mf.READ_WRITE, total_pixels * 4)
        target_mask_buf = cl.Buffer(self.context, mf.READ_WRITE, total_pixels * 4)

        l_source = source_lab[:, :, 0].ravel()
        l_target = target_lab[:, :, 0].ravel()
        s_p33, s_p66 = np.percentile(l_source, [33, 66])
        t_p33, t_p66 = np.percentile(l_target, [33, 66])

        mask_kernel = self.program.create_luminance_mask
        mask_kernel(self.queue, (total_pixels,), None, source_buf, source_mask_buf, np.float32(s_p33), np.float32(s_p66), np.int32(total_pixels))
        mask_kernel(self.queue, (total_pixels,), None, target_buf, target_mask_buf, np.float32(t_p33), np.float32(t_p66), np.int32(total_pixels))
        
        # GPU-side segment statistics calculation
        num_segments = 3 # For dark, mid, bright segments

        # Buffers for segment counts (output of count_pixels_per_segment)
        s_segment_counts_np = np.zeros(num_segments, dtype=np.int32)
        t_segment_counts_np = np.zeros(num_segments, dtype=np.int32)
        s_segment_counts_buf = cl.Buffer(self.context, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=s_segment_counts_np)
        t_segment_counts_buf = cl.Buffer(self.context, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=t_segment_counts_np)
        # Initialize with zeros by filling from host (or use cl.enqueue_fill_buffer if preferred for larger arrays)
        cl.enqueue_copy(self.queue, s_segment_counts_buf, np.zeros(num_segments, dtype=np.int32)).wait() 
        cl.enqueue_copy(self.queue, t_segment_counts_buf, np.zeros(num_segments, dtype=np.int32)).wait()

        # Call count_pixels_per_segment kernel
        self.program.count_pixels_per_segment(self.queue, (total_pixels,), None, source_mask_buf, s_segment_counts_buf, np.int32(total_pixels))
        self.program.count_pixels_per_segment(self.queue, (total_pixels,), None, target_mask_buf, t_segment_counts_buf, np.int32(total_pixels))

        # Retrieve segment counts to CPU
        cl.enqueue_copy(self.queue, s_segment_counts_np, s_segment_counts_buf).wait()
        cl.enqueue_copy(self.queue, t_segment_counts_np, t_segment_counts_buf).wait()

        # Calculate segment offsets on CPU (prefix sum)
        s_segment_offsets_np = np.concatenate(([0], np.cumsum(s_segment_counts_np)[:-1])).astype(np.int32)
        t_segment_offsets_np = np.concatenate(([0], np.cumsum(t_segment_counts_np)[:-1])).astype(np.int32)

        # Buffers for segment offsets (input to scatter_pixels_by_segment)
        s_segment_offsets_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=s_segment_offsets_np)
        t_segment_offsets_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_segment_offsets_np)

        # Buffers for scatter counters (temporary, atomic, for scatter_pixels_by_segment)
        s_scatter_counters_buf = cl.Buffer(self.context, mf.READ_WRITE, num_segments * np.dtype(np.int32).itemsize)
        t_scatter_counters_buf = cl.Buffer(self.context, mf.READ_WRITE, num_segments * np.dtype(np.int32).itemsize)
        cl.enqueue_fill_buffer(self.queue, s_scatter_counters_buf, np.int32(0), 0, num_segments * np.dtype(np.int32).itemsize).wait()
        cl.enqueue_fill_buffer(self.queue, t_scatter_counters_buf, np.int32(0), 0, num_segments * np.dtype(np.int32).itemsize).wait()

        # Buffers for compacted LAB data (output of scatter, input to _calculate_stats for segments)
        s_compacted_lab_data_buf = cl.Buffer(self.context, mf.READ_WRITE, source_lab_f32.nbytes)
        t_compacted_lab_data_buf = cl.Buffer(self.context, mf.READ_WRITE, target_lab_f32.nbytes)

        # Call scatter_pixels_by_segment kernel
        self.program.scatter_pixels_by_segment(self.queue, (total_pixels,), None, 
                                               source_buf, source_mask_buf, s_segment_offsets_buf, 
                                               s_scatter_counters_buf, s_compacted_lab_data_buf, np.int32(total_pixels))
        self.program.scatter_pixels_by_segment(self.queue, (total_pixels,), None, 
                                               target_buf, target_mask_buf, t_segment_offsets_buf, 
                                               t_scatter_counters_buf, t_compacted_lab_data_buf, np.int32(total_pixels))

        # Calculate stats for each segment on GPU
        s_stats_gpu = np.zeros(num_segments * 6, dtype=np.float32) # Lmn, Lsd, amn, asd, bmn, bsd for each segment
        t_stats_gpu = np.zeros(num_segments * 6, dtype=np.float32)

        for i in range(num_segments):
            # Source segment stats
            s_mean_seg, s_std_seg = self._calculate_stats(
                s_compacted_lab_data_buf, 
                s_segment_counts_np[i], 
                data_offset_pixels=s_segment_offsets_np[i]
            )
            for ch_idx in range(3): # L, a, b
                s_stats_gpu[i * 6 + ch_idx * 2 + 0] = s_mean_seg[ch_idx]
                s_stats_gpu[i * 6 + ch_idx * 2 + 1] = s_std_seg[ch_idx]

            # Target segment stats
            t_mean_seg, t_std_seg = self._calculate_stats(
                t_compacted_lab_data_buf, 
                t_segment_counts_np[i], 
                data_offset_pixels=t_segment_offsets_np[i]
            )
            for ch_idx in range(3):
                t_stats_gpu[i * 6 + ch_idx * 2 + 0] = t_mean_seg[ch_idx]
                t_stats_gpu[i * 6 + ch_idx * 2 + 1] = t_std_seg[ch_idx]

        s_stats_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=s_stats_gpu)
        t_stats_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_stats_gpu)
        result_buf = cl.Buffer(self.context, mf.WRITE_ONLY, source_lab_f32.nbytes)

        transfer_kernel = self.program.apply_segmented_transfer
        transfer_kernel(self.queue, (total_pixels,), None, 
                        source_buf, source_mask_buf, result_buf, 
                        s_stats_buf, t_stats_buf, np.int32(total_pixels))

        result_lab_f32 = np.empty_like(source_lab_f32)
        cl.enqueue_copy(self.queue, result_lab_f32, result_buf).wait()

        return result_lab_f32.astype(source_lab.dtype)

    def selective_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray, 
                               mask: np.ndarray, selective_channels: list, 
                               blend_factor: float) -> np.ndarray:
        """
        Performs selective LAB color transfer on the GPU using a mask.
        """
        if not self.is_gpu_available():
            self.logger.error("GPU not available, cannot perform selective_lab_transfer_gpu.")
            # Fallback or raise error, for now, let's assume an error or specific handling is needed
            raise RuntimeError("GPU not available for selective transfer")

        mf = cl.mem_flags
        h, w = source_lab.shape[:2]
        total_pixels = h * w

        # Ensure images are float32 for GPU processing
        source_lab_f32 = source_lab.astype(np.float32, copy=False)
        target_lab_f32 = target_lab.astype(np.float32, copy=False)
        mask_ui8 = mask.astype(np.uint8, copy=False) # Mask typically uint8

        source_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source_lab_f32)
        target_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_lab_f32)
        mask_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mask_ui8)
        result_buf = cl.Buffer(self.context, mf.WRITE_ONLY, source_lab_f32.nbytes)

        # Convert selective_channels list (e.g., ['L', 'b']) to integer flags for the kernel
        # L=0, a=1, b=2. Example: [1,0,1] for L and b.
        channel_flags = np.array([1 if 'L' in selective_channels else 0,
                                  1 if 'a' in selective_channels else 0,
                                  1 if 'b' in selective_channels else 0], dtype=np.int32)
        
        kernel_args = (
            source_buf, 
            target_buf, 
            mask_buf, 
            result_buf,
            channel_flags[0], # process_L
            channel_flags[1], # process_a
            channel_flags[2], # process_b
            np.float32(blend_factor),
            np.int32(w),
            np.int32(h) # height, though kernel uses total_pixels from width*height
        )

        try:
            selective_kernel = self.program.selective_transfer_kernel
            selective_kernel(self.queue, (total_pixels,), None, *kernel_args).wait()
        except Exception as e:
            self.logger.error(f"Error executing selective_transfer_kernel: {e}")
            # Fallback or re-raise, for now re-raise to make issues visible
            raise

        result_lab_f32 = np.empty_like(source_lab_f32)
        cl.enqueue_copy(self.queue, result_lab_f32, result_buf).wait()

        return result_lab_f32.astype(source_lab.dtype)
