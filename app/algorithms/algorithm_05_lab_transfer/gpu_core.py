import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array # If used
import os
import warnings # if used
from typing import List, Tuple, Union, Optional, Dict # Added typing
from .logger import get_logger # Added logger import
from .config import LABTransferConfig # Changed import

# Placeholder for PYOPENCL_AVAILABLE, ensure it's defined based on successful cl import
PYOPENCL_AVAILABLE = True
try:
    if os.environ.get("PYOPENCL_TEST", "0") == "1": # For testing fallback
        raise ImportError("PYOPENCL_TEST is set, simulating no OpenCL")
    cl.create_some_context()
except Exception:
    PYOPENCL_AVAILABLE = False
import warnings

# Ignoruj specyficzne ostrzeżenie z PyOpenCL dotyczące cache'owania kerneli.
# Musi być wywołane PRZED importem pyopencl, aby zadziałało.
warnings.filterwarnings("ignore", category=UserWarning, message=".*pytools.persistent_dict.*")

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    import pyopencl.tools
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
import os
import logging

from .logger import get_logger

class LABColorTransferGPU:
    def __init__(self, config: LABTransferConfig):
        if not PYOPENCL_AVAILABLE:
            # This check might be redundant if ImageProcessor already handles PYOPENCL_AVAILABLE
            # However, it's a good safeguard if LABColorTransferGPU is instantiated directly.
            self.logger = get_logger(self.__class__.__name__) # Initialize logger early for this message
            self.logger.error("PyOpenCL not available. LABColorTransferGPU cannot be initialized.")
            raise RuntimeError("PyOpenCL is not available or context creation failed.")
        
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self._initialize_opencl()
        self._load_kernel()
        self.gpu_mask_buffers_cache = {} # Cache for mask buffers
        
        # Method dispatch for GPU operations
        self._GPU_METHOD_DISPATCH = {
            "basic": "_execute_basic_on_gpu_buffers",
            "linear_blend": "_execute_linear_blend_on_gpu_buffers",
            "selective": "_execute_selective_on_gpu_buffers",
            "adaptive": "_execute_adaptive_on_gpu_buffers",
        }
        
        # Method dispatch for GPU operations
        self._GPU_METHOD_DISPATCH = {
            "basic": "_execute_basic_on_gpu_buffers",
            "linear_blend": "_execute_linear_blend_on_gpu_buffers",
            "selective": "_execute_selective_on_gpu_buffers",
            "adaptive": "_execute_adaptive_on_gpu_buffers",
        }
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
            # TODO: Add configuration for platform/device selection from config.py
            platform_idx = int(os.environ.get('PYOPENCL_CTX', '0').split(':')[0]) \
                if ':' in os.environ.get('PYOPENCL_CTX', '0') else 0
            device_idx = int(os.environ.get('PYOPENCL_CTX', '0').split(':')[1]) \
                if ':' in os.environ.get('PYOPENCL_CTX', '0') else 0

            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found.")
            platform = platforms[platform_idx]
            
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if not devices:
                self.logger.warning("No GPU device found, trying CPU OpenCL device.")
                devices = platform.get_devices(device_type=cl.device_type.CPU)
                if not devices:
                    raise RuntimeError("No GPU or CPU OpenCL device found.")
            
            self.device = devices[device_idx]
            self.context = cl.Context([self.device])
            self.logger.info(f"Successfully initialized OpenCL on device: {self.device.name}")
            
            properties = cl.command_queue_properties.PROFILING_ENABLE
            self.queue = cl.CommandQueue(self.context, properties=properties)
            
            kernel_path = os.path.join(os.path.dirname(__file__), 'kernels.cl')
            with open(kernel_path, 'r', encoding='utf-8') as f:
                kernel_code = f.read()
            
            self.program = cl.Program(self.context, kernel_code).build()
            self.logger.info("OpenCL initialized and kernel compiled successfully.")

        except Exception as e:
            self.logger.error(f"Error initializing OpenCL: {e}", exc_info=True)
            self.context = None # Ensure it's None if initialization fails
            raise RuntimeError(f"OpenCL Initialization Error: {e}")


    def _calculate_stats_gpu(self, lab_image_g, total_pixels, 
                             data_offset_pixels=0, num_pixels_in_segment=None):
        """
        Calculates sum and sum_sq for L, a, b channels using GPU.
        Can operate on a full image or a segment of a compacted image.
        """
        if num_pixels_in_segment is None:
            num_pixels_in_segment = total_pixels # For full image, segment size is total_pixels

        if num_pixels_in_segment == 0: # Avoid division by zero if segment is empty
            return np.zeros(6, dtype=np.float32) # mean_l, std_l, mean_a, std_a, mean_b, std_b

        mf = cl.mem_flags
        
        max_work_group_size = self.device.max_work_group_size 
        work_group_size = min(max_work_group_size, 256) 
        
        num_groups = min(1024, (num_pixels_in_segment + work_group_size -1) // work_group_size) 
        if num_groups == 0: num_groups = 1
        
        global_work_size = (num_groups * work_group_size,)
        local_work_size = (work_group_size,)

        partial_sums_g = cl.Buffer(self.context, mf.WRITE_ONLY, num_groups * 6 * np.float32().itemsize)
        local_sums_g = cl.LocalMemory(work_group_size * 6 * np.float32().itemsize)

        self.program.stats_partial_reduce(
            self.queue, global_work_size, local_work_size,
            lab_image_g,
            partial_sums_g,
            local_sums_g,
            np.int32(num_pixels_in_segment),
            np.int32(data_offset_pixels)
        ).wait()  # Closing parenthesis for stats_partial_reduce
        # Removed duplicate function definition and implementation of _execute_basic_on_gpu_buffers

        partial_sums_h = np.empty(num_groups * 6, dtype=np.float32)
        cl.enqueue_copy(self.queue, partial_sums_h, partial_sums_g).wait()

        total_sums = np.sum(partial_sums_h.reshape(num_groups, 6), axis=0)

        stats = np.zeros(6, dtype=np.float32)
        
        stats[0] = total_sums[0] / num_pixels_in_segment
        stats[1] = np.sqrt(max(0, total_sums[1] / num_pixels_in_segment - stats[0]**2))
        stats[2] = total_sums[2] / num_pixels_in_segment
        stats[3] = np.sqrt(max(0, total_sums[3] / num_pixels_in_segment - stats[2]**2))
        stats[4] = total_sums[4] / num_pixels_in_segment
        stats[5] = np.sqrt(max(0, total_sums[5] / num_pixels_in_segment - stats[4]**2))
        
        return stats

    def basic_lab_transfer(self, source_lab, target_lab, **kwargs):
        mf = cl.mem_flags
        h, w = source_lab.shape[:2]
        total_pixels = h * w

        source_lab_f32 = source_lab.astype(np.float32)
        target_lab_f32 = target_lab.astype(np.float32)

        source_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source_lab_f32)
        target_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_lab_f32)
        result_g = cl.Buffer(self.context, mf.WRITE_ONLY, source_lab_f32.nbytes)

        src_stats = self._calculate_stats_gpu(source_g, total_pixels)
        tgt_stats = self._calculate_stats_gpu(target_g, total_pixels)

        self.program.basic_transfer_kernel(
            self.queue, (total_pixels,), None,
            source_g, result_g,
            np.float32(src_stats[0]), np.float32(src_stats[1]),
            np.float32(src_stats[2]), np.float32(src_stats[3]),
            np.float32(src_stats[4]), np.float32(src_stats[5]),
            np.float32(tgt_stats[0]), np.float32(tgt_stats[1]),
            np.float32(tgt_stats[2]), np.float32(tgt_stats[3]),
            np.float32(tgt_stats[4]), np.float32(tgt_stats[5]),
            np.int32(w), np.int32(h)
        ).wait()

        result_lab_f32 = np.empty_like(source_lab_f32)
        cl.enqueue_copy(self.queue, result_lab_f32, result_g).wait()
        return result_lab_f32.astype(source_lab.dtype)

    def weighted_lab_transfer(self, source_lab, target_lab, **kwargs):
        weights = kwargs.get('weights', {'L': 1.0, 'a': 1.0, 'b': 1.0})
        weight_l = np.float32(weights.get('L', 1.0))
        weight_a = np.float32(weights.get('a', 1.0))
        weight_b = np.float32(weights.get('b', 1.0))

        mf = cl.mem_flags
        h, w = source_lab.shape[:2]
        total_pixels = h * w

        source_lab_f32 = source_lab.astype(np.float32)
        target_lab_f32 = target_lab.astype(np.float32)

        source_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source_lab_f32)
        target_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_lab_f32)
        result_g = cl.Buffer(self.context, mf.WRITE_ONLY, source_lab_f32.nbytes)

        src_stats = self._calculate_stats_gpu(source_g, total_pixels)
        tgt_stats = self._calculate_stats_gpu(target_g, total_pixels)
        
        self.program.weighted_transfer_kernel(
            self.queue, (total_pixels,), None,
            source_g, result_g,
            np.float32(src_stats[0]), np.float32(src_stats[1]),
            np.float32(src_stats[2]), np.float32(src_stats[3]),
            np.float32(src_stats[4]), np.float32(src_stats[5]),
            np.float32(tgt_stats[0]), np.float32(tgt_stats[1]),
            np.float32(tgt_stats[2]), np.float32(tgt_stats[3]),
            np.float32(tgt_stats[4]), np.float32(tgt_stats[5]),
            weight_l, weight_a, weight_b,
            np.int32(w), np.int32(h)
        ).wait()

        result_lab_f32 = np.empty_like(source_lab_f32)
        cl.enqueue_copy(self.queue, result_lab_f32, result_g).wait()
        return result_lab_f32.astype(source_lab.dtype)

    def selective_lab_transfer(self, source_lab, target_lab, mask, **kwargs):
        selective_channels = kwargs.get('selective_channels', ['L', 'a', 'b'])
        blend_factor = np.float32(kwargs.get('blend_factor', 1.0))

        mf = cl.mem_flags
        h, w = source_lab.shape[:2]
        total_pixels = h * w

        source_lab_f32 = source_lab.astype(np.float32)
        target_lab_f32 = target_lab.astype(np.float32)
        mask_ui8 = mask.astype(np.uint8) 

        source_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source_lab_f32)
        target_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_lab_f32)
        mask_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mask_ui8)
        result_g = cl.Buffer(self.context, mf.WRITE_ONLY, source_lab_f32.nbytes)

        channel_flags = np.array([1 if 'L' in selective_channels else 0,
                                  1 if 'a' in selective_channels else 0,
                                  1 if 'b' in selective_channels else 0], dtype=np.int32)
        
        self.program.selective_transfer_kernel(
            self.queue, (total_pixels,), None,
            source_g, target_g, mask_g, result_g,
            channel_flags[0], channel_flags[1], channel_flags[2],
            blend_factor,
            np.int32(w), np.int32(h)
        ).wait()

        result_lab_f32 = np.empty_like(source_lab_f32)
        cl.enqueue_copy(self.queue, result_lab_f32, result_g).wait()
        return result_lab_f32.astype(source_lab.dtype)

    def _create_luminance_mask_and_segment_info(self, lab_image_g, width, height, num_segments):
        mf = cl.mem_flags
        total_pixels = width * height

        segment_indices_map_g = cl.Buffer(self.context, mf.READ_WRITE, total_pixels * np.int32().itemsize)
        
        segment_pixel_counts_h = np.zeros(num_segments, dtype=np.int32)
        segment_pixel_counts_g = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=segment_pixel_counts_h)
        
        self.program.create_luminance_mask(
            self.queue, (total_pixels,), None,
            lab_image_g,
            segment_indices_map_g,
            np.int32(num_segments),
            np.int32(width),
            np.int32(height)
        ).wait()

        self.program.count_pixels_per_segment(
            self.queue, (total_pixels,), None,
            segment_indices_map_g,
            segment_pixel_counts_g,
            np.int32(num_segments),
            np.int32(total_pixels)
        ).wait()
        
        cl.enqueue_copy(self.queue, segment_pixel_counts_h, segment_pixel_counts_g).wait()

        segment_offsets_h = np.zeros(num_segments, dtype=np.int32)
        segment_offsets_h[0] = 0
        for i in range(1, num_segments):
            segment_offsets_h[i] = segment_offsets_h[i-1] + segment_pixel_counts_h[i-1]
        segment_offsets_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=segment_offsets_h)

        compacted_lab_data_g = cl.Buffer(self.context, mf.READ_WRITE, total_pixels * 3 * np.float32().itemsize)
        temp_segment_counters_h = np.zeros(num_segments, dtype=np.int32)
        temp_segment_counters_g = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=temp_segment_counters_h)

        self.program.scatter_pixels_by_segment(
            self.queue, (total_pixels,), None,
            lab_image_g,
            segment_indices_map_g,
            segment_offsets_g,
            compacted_lab_data_g,
            temp_segment_counters_g,
            np.int32(num_segments),
            np.int32(width),
            np.int32(height)
        ).wait()
        
        return segment_indices_map_g, compacted_lab_data_g, segment_pixel_counts_h, segment_offsets_h


    def adaptive_lab_transfer(self, source_lab, target_lab, **kwargs):
        mf = cl.mem_flags
        h, w = source_lab.shape[:2]
        total_pixels = h * w

        num_segments = int(kwargs.get('num_segments', 3))
        if num_segments <= 0:
            self.logger.warning(f"num_segments must be positive, got {num_segments}. Defaulting to 3.")
            num_segments = 3
        
        self.logger.info(f"Adaptive GPU transfer with {num_segments} segments.")

        source_lab_f32 = source_lab.astype(np.float32)
        target_lab_f32 = target_lab.astype(np.float32)

        source_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source_lab_f32)
        target_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_lab_f32)
        
        (src_segment_indices_map_g, 
         src_compacted_lab_g, 
         src_segment_pixel_counts_h, 
         src_segment_offsets_h) = self._create_luminance_mask_and_segment_info(source_g, w, h, num_segments)

        (tgt_segment_indices_map_g, 
         tgt_compacted_lab_g, 
         tgt_segment_pixel_counts_h, 
         tgt_segment_offsets_h) = self._create_luminance_mask_and_segment_info(target_g, w, h, num_segments)

        source_segment_stats_h = np.zeros(num_segments * 6, dtype=np.float32)
        target_segment_stats_h = np.zeros(num_segments * 6, dtype=np.float32)

        for i in range(num_segments):
            if src_segment_pixel_counts_h[i] > 0:
                stats_s = self._calculate_stats_gpu(src_compacted_lab_g, 
                                                    total_pixels, 
                                                    data_offset_pixels=src_segment_offsets_h[i], 
                                                    num_pixels_in_segment=src_segment_pixel_counts_h[i])
                source_segment_stats_h[i*6:(i+1)*6] = stats_s
            
            if tgt_segment_pixel_counts_h[i] > 0:
                stats_t = self._calculate_stats_gpu(tgt_compacted_lab_g,
                                                    total_pixels,
                                                    data_offset_pixels=tgt_segment_offsets_h[i],
                                                    num_pixels_in_segment=tgt_segment_pixel_counts_h[i])
                target_segment_stats_h[i*6:(i+1)*6] = stats_t

        source_segment_stats_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source_segment_stats_h)
        target_segment_stats_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_segment_stats_h)
        
        result_g = cl.Buffer(self.context, mf.WRITE_ONLY, source_lab_f32.nbytes)
        
        self.program.apply_segmented_transfer(
            self.queue, (total_pixels,), None,
            source_g, 
            result_g,
            src_segment_indices_map_g, 
            source_segment_stats_g,
            target_segment_stats_g,
            np.int32(num_segments),
            np.int32(w), np.int32(h)
        ).wait()

        result_lab_f32 = np.empty_like(source_lab_f32)
        cl.enqueue_copy(self.queue, result_lab_f32, result_g).wait()
        return result_lab_f32.astype(source_lab.dtype)

        # Completely removed duplicate _execute_basic_on_gpu_buffers function
        
    def _execute_linear_blend_on_gpu_buffers(self, src_buffer: cl.Buffer, tgt_buffer: cl.Buffer,
                                            width: int, height: int, **kwargs) -> cl.Buffer:
        """Execute linear blend transfer directly on GPU buffers."""
        weights = kwargs.get('weights', [0.3, 0.5, 0.5])
        weights_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                  hostbuf=np.array(weights, dtype=np.float32))
        
        result_buffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, src_buffer.size)
        
        self.program.linear_blend_transfer(
            self.queue, (width * height,), None,
            src_buffer, tgt_buffer, result_buffer, weights_buffer,
            np.int32(width), np.int32(height)
        ).wait()
        
        return result_buffer
        
    def process_image_hybrid_gpu(self,
                                source_lab: np.ndarray,
                                target_lab: np.ndarray,
                                hybrid_pipeline: list | None = None,
                                return_intermediate_steps: bool = False,
                                **kwargs) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        """
        Execute configurable pipeline of color transfer methods on GPU.
        Uses ping-pong buffers to minimize CPU-GPU transfers between steps.
        """
        pipeline_config = hybrid_pipeline if hybrid_pipeline is not None else self.config.get("hybrid_pipeline", [])
        if not isinstance(pipeline_config, list):
            raise ValueError(f"Hybrid pipeline configuration must be a list, got {type(pipeline_config)}.")
            
        # Convert inputs to GPU buffers
        h, w = source_lab.shape[:2]
        src_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=source_lab.astype(np.float32))
        tgt_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=target_lab.astype(np.float32))
        
        # Ping-pong buffers
        buffer_a = src_buffer
        buffer_b = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, src_buffer.size)
        
        intermediate_results = []
        
        for idx, step_config in enumerate(pipeline_config):
            method_name_key = step_config["method"]
            method_params = {**kwargs, **step_config.get("params", {})}
            
            method_to_call_name = self._GPU_METHOD_DISPATCH[method_name_key]
            method_to_call = getattr(self, method_to_call_name)
            
            self.logger.info(f"Hybrid GPU step {idx + 1}/{len(pipeline_config)}: Applying '{method_name_key}'")
            
            # Execute step on GPU buffers
            result_buffer = method_to_call(
                buffer_a, tgt_buffer,
                width=w, height=h,
                **method_params
            )
            
            if return_intermediate_steps:
                # Copy intermediate result back to CPU
                intermediate_result = np.empty_like(source_lab, dtype=np.float32)
                cl.enqueue_copy(self.queue, intermediate_result, result_buffer).wait()
                intermediate_results.append(intermediate_result)
            
            # Swap buffers for next step
            buffer_a, buffer_b = result_buffer, buffer_a
            
        # Get final result
        final_result = np.empty_like(source_lab, dtype=np.float32)
        cl.enqueue_copy(self.queue, final_result, buffer_a).wait()
        
        if return_intermediate_steps:
            return final_result, intermediate_results
        return final_result
        
    def _execute_basic_on_gpu_buffers(self, src_buffer: cl.Buffer, tgt_buffer: cl.Buffer, 
                                  width: int, height: int, **kwargs) -> cl.Buffer:
        """Execute basic transfer directly on GPU buffers."""
        result_buffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, src_buffer.size)
        
        # Calculate stats if needed
        stats_src = self._calculate_stats_for_buffer(src_buffer, width, height)
        stats_tgt = self._calculate_stats_for_buffer(tgt_buffer, width, height)
        
        self.program.basic_transfer(
            self.queue, (width * height,), None,
            src_buffer, tgt_buffer, result_buffer,
            np.float32(stats_src[0]), np.float32(stats_src[1]),
            np.float32(stats_src[2]), np.float32(stats_src[3]),
            np.float32(stats_src[4]), np.float32(stats_src[5]),
            np.float32(stats_tgt[0]), np.float32(stats_tgt[1]),
            np.float32(stats_tgt[2]), np.float32(stats_tgt[3]),
            np.float32(stats_tgt[4]), np.float32(stats_tgt[5]),
            np.int32(width), np.int32(height)
        ).wait()
        
        return result_buffer
        
    def _execute_selective_on_gpu_buffers(self, src_buffer: cl.Buffer, tgt_buffer: cl.Buffer,
                                         width: int, height: int, **kwargs) -> cl.Buffer:
        """Execute selective transfer directly on GPU buffers."""
        result_buffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, src_buffer.size)
        
        # Handle mask
        mask_param = kwargs.get('mask')
        mask_buffer = None
        if mask_param is not None:
            mask_np = self._ensure_mask_is_numpy(mask_param, height, width)
            mask_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                   hostbuf=mask_np)
        
        process_L = 1 if 'L' in kwargs.get('selective_channels', []) else 0
        process_a = 1 if 'a' in kwargs.get('selective_channels', []) else 0
        process_b = 1 if 'b' in kwargs.get('selective_channels', []) else 0
        blend_factor = float(kwargs.get('blend_factor', 1.0))
        
        self.program.selective_transfer(
            self.queue, (width * height,), None,
            src_buffer, tgt_buffer, mask_buffer if mask_buffer else src_buffer, result_buffer,
            np.int32(process_L), np.int32(process_a), np.int32(process_b),
            np.float32(blend_factor),
            np.int32(width), np.int32(height)
        ).wait()
        
        return result_buffer
        
    def process_image_hybrid_gpu(self,
                                source_lab: np.ndarray,
                                target_lab: np.ndarray,
                                hybrid_pipeline: list | None = None,
                                return_intermediate_steps: bool = False,
                                **kwargs) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        """
        Execute configurable pipeline of color transfer methods on GPU.
        Uses ping-pong buffers to minimize CPU-GPU transfers between steps.
        """
        pipeline_config = hybrid_pipeline if hybrid_pipeline is not None else self.config.get("hybrid_pipeline", [])
        if not isinstance(pipeline_config, list):
            raise ValueError(f"Hybrid pipeline configuration must be a list, got {type(pipeline_config)}.")
            
        # Convert inputs to GPU buffers
        h, w = source_lab.shape[:2]
        src_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=source_lab.astype(np.float32))
        tgt_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=target_lab.astype(np.float32))
        
        # Ping-pong buffers
        buffer_a = src_buffer
        buffer_b = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, src_buffer.size)
        
        intermediate_results = []
        
        for idx, step_config in enumerate(pipeline_config):
            method_name_key = step_config["method"]
            method_params = {**kwargs, **step_config.get("params", {})}
            
            method_to_call_name = self._GPU_METHOD_DISPATCH[method_name_key]
            method_to_call = getattr(self, method_to_call_name)
            
            self.logger.info(f"Hybrid GPU step {idx + 1}/{len(pipeline_config)}: Applying '{method_name_key}'")
            
            # Execute step on GPU buffers
            result_buffer = method_to_call(
                buffer_a, tgt_buffer,
                width=w, height=h,
                **method_params
            )
            
            if return_intermediate_steps:
                # Copy intermediate result back to CPU
                intermediate_result = np.empty_like(source_lab, dtype=np.float32)
                cl.enqueue_copy(self.queue, intermediate_result, result_buffer).wait()
                intermediate_results.append(intermediate_result)
            
            # Swap buffers for next step
            buffer_a, buffer_b = result_buffer, buffer_a
            
        # Get final result
        final_result = np.empty_like(source_lab, dtype=np.float32)
        cl.enqueue_copy(self.queue, final_result, buffer_a).wait()
        
        if return_intermediate_steps:
            return final_result, intermediate_results
        return final_result
        
    def hybrid_transfer(self, source_lab, target_lab, **kwargs):
        """Legacy hybrid transfer, now using the new pipeline processor."""
        return self.process_image_hybrid_gpu(source_lab, target_lab, **kwargs)
