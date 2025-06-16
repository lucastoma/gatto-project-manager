import os
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from .logger import get_logger
# Removed incorrect import
from PIL import Image
import skimage.color
from functools import lru_cache
from typing import Optional, Dict, List

from .config import LABTransferConfig
from .metrics import calculate_delta_e_lab, histogram_matching
from .logger import get_logger
from .gpu_core import LABColorTransferGPU

class LABColorTransfer:
    """
    Base class implementing core LAB color transfer methods.
    It now uses scikit-image for robust color conversions and includes
    optimized and refactored transfer methods.
    """
    def __init__(self, config: LABTransferConfig = None, strict_gpu: bool = False):
        self.logger = get_logger()
        self.config = config or LABTransferConfig()
        self.gpu_transfer = None
        if self.config.use_gpu:
            try:
                self.gpu_transfer = LABColorTransferGPU()
                if not self.gpu_transfer.is_gpu_available():
                    self.logger.warning("GPU requested, but OpenCL initialization failed.")
                    if strict_gpu:
                        raise RuntimeError("Strict GPU mode failed: GPU not available or OpenCL initialization failed.")
                    self.gpu_transfer = None # Fallback to CPU
            except Exception as e:
                self.logger.error(f"Failed to initialize GPU context: {e}.")
                if strict_gpu:
                    raise RuntimeError(f"Strict GPU mode failed during context initialization: {e}")
                self.gpu_transfer = None # Fallback to CPU

    @staticmethod
    @lru_cache(maxsize=16)
    def _rgb_to_lab_cached(rgb_bytes: bytes, shape: tuple) -> np.ndarray:
        """Helper for caching RGB to LAB conversion."""
        rgb_array = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(shape)
        return skimage.color.rgb2lab(rgb_array)

    def rgb_to_lab_optimized(self, rgb_array: np.ndarray) -> np.ndarray:
        """
        Convert an RGB image array to LAB color space with caching.
        """
        if not rgb_array.flags['C_CONTIGUOUS']:
            rgb_array = np.ascontiguousarray(rgb_array)
        return self._rgb_to_lab_cached(rgb_array.tobytes(), rgb_array.shape)

    def lab_to_rgb_optimized(self, lab_array: np.ndarray) -> np.ndarray:
        """
        Convert a LAB image array back to RGB uint8 format.
        """
        if lab_array.dtype != np.float64:
            lab_array = lab_array.astype(np.float64)
        rgb_result = skimage.color.lab2rgb(lab_array)
        return (np.clip(rgb_result, 0, 1) * 255).astype(np.uint8)

    def _calculate_stats(self, lab_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculates mean and std dev for each channel of a LAB image."""
        lab_image_f64 = lab_image.astype(np.float64)
        mean = np.mean(lab_image_f64, axis=(0, 1))
        std = np.std(lab_image_f64, axis=(0, 1))
        return mean, std

    def basic_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray, **kwargs) -> np.ndarray:
        # Dtype validation: weighted transfer requires float64
        if source_lab.dtype != np.float64 or target_lab.dtype != np.float64:
            raise ValueError("Input arrays must be float64 for weighted transfer")
        if source_lab.shape != target_lab.shape:
            raise ValueError("Source and target must have the same shape")
        if not (isinstance(source_lab, np.ndarray) and isinstance(target_lab, np.ndarray)):
            raise ValueError("Inputs must be numpy arrays")
            
        # Convert uint8 inputs to float64
        if source_lab.dtype == np.uint8:
            source_lab = source_lab.astype(np.float64) / 255
        if target_lab.dtype == np.uint8:
            target_lab = target_lab.astype(np.float64) / 255
            
        if source_lab.dtype not in (np.float32, np.float64) or target_lab.dtype not in (np.float32, np.float64):
            raise ValueError("Input arrays must be float32 or float64")
            
        original_dtype = source_lab.dtype
        s_mean, s_std = self._calculate_stats(source_lab)
        t_mean, t_std = self._calculate_stats(target_lab)
        src = source_lab.astype(np.float64, copy=False)
        result = np.empty_like(src)
        for i in range(3):
            if s_std[i] < 1e-6:
                result[..., i] = src[..., i] + (t_mean[i] - s_mean[i])
            else:
                std_ratio = t_std[i] / s_std[i]
                result[..., i] = (src[..., i] - s_mean[i]) * std_ratio + t_mean[i]
        return result.astype(original_dtype, copy=False)

    def linear_blend_lab(self, source_lab: np.ndarray, target_lab: np.ndarray, **kwargs) -> np.ndarray:
        weights = kwargs.get('weights', {})
        if not isinstance(weights, dict):
            raise ValueError("Weights must be a dictionary")
        if not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in weights.items()):
            raise ValueError("Weights dictionary keys must be strings and values must be numbers")
        if source_lab.shape != target_lab.shape:
            raise ValueError("Source and target must have the same shape")
        original_dtype = source_lab.dtype
        src = source_lab.astype(np.float64, copy=False)
        s_mean, s_std = self._calculate_stats(src)
        t_mean, t_std = self._calculate_stats(target_lab)
        w = np.array([weights.get('L', 0.5), weights.get('a', 0.5), weights.get('b', 0.5)])
        blended_mean = s_mean * (1 - w) + t_mean * w
        blended_std = s_std * (1 - w) + t_std * w
        result = np.empty_like(src)
        for i in range(3):
            if s_std[i] < 1e-6:
                result[..., i] = src[..., i] + (blended_mean[i] - s_mean[i])
            else:
                std_ratio = blended_std[i] / s_std[i]
                result[..., i] = (src[..., i] - s_mean[i]) * std_ratio + blended_mean[i]
        return result.astype(original_dtype, copy=False)

        def selective_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray, **kwargs) -> np.ndarray:
        """Perform selective LAB transfer using a mask over specified channels."""
        # Determine mask usage
        mask_provided = 'mask' in kwargs
        if mask_provided:
            mask = kwargs['mask']
            if mask is None:
                raise ValueError("Mask parameter is required")
            if not isinstance(mask, np.ndarray):
                raise ValueError("Mask must be a numpy array")
            # Convert mask to boolean 2D array
            mask_bool = mask > 128 if mask.dtype == np.uint8 else mask > 0.5
            if mask_bool.ndim == 3:
                mask_bool = mask_bool[..., 0]
        else:
            # Default full mask
            mask_bool = np.ones(source_lab.shape[:2], dtype=bool)
        # Channel and blend settings
        selective_channels = kwargs.get('selective_channels', ['a', 'b'])
        blend_factor = kwargs.get('blend_factor', 1.0)
        # Validate shapes
        if not (source_lab.shape[:2] == target_lab.shape[:2] == mask_bool.shape):
            raise ValueError("Source, target, and mask must have the same height and width")
        # Prepare result array
        result_lab = source_lab.astype(np.float64, copy=True)
        s_mean, s_std = self._calculate_stats(source_lab)
        t_mean, t_std = self._calculate_stats(target_lab)
        channel_map = {'L': 0, 'a': 1, 'b': 2}
        channels_to_process = [channel_map[c] for c in selective_channels if c in channel_map]
        # Apply transfer per channel
        for i in channels_to_process:
            if np.all(~mask_bool):
                continue
            src_chan = source_lab[..., i]
            if s_std[i] < 1e-6:
                transferred = src_chan + (t_mean[i] - s_mean[i])
            else:
                std_ratio = t_std[i] / s_std[i]
                transferred = (src_chan - s_mean[i]) * std_ratio + t_mean[i]
            blended_chan = transferred * blend_factor + src_chan * (1 - blend_factor)
            np.copyto(result_lab[..., i], blended_chan, where=mask_bool)
        return result_lab.astype(source_lab.dtype, copy=False)

        mask_provided = 'mask' in kwargs
        mask = kwargs.get('mask', None)
        if mask_provided and mask is None:
            raise ValueError("Mask parameter is required")
        # Default mask: full if not provided
        if not mask_provided:
            mask_bool = np.ones(source_lab.shape[:2], dtype=bool)
        else:
            if not hasattr(mask, 'shape'):
                raise ValueError("Mask must be a numpy array")
            mask_bool = mask > 128 if mask.dtype == np.uint8 else mask > 0.5
            if mask_bool.ndim == 3:
                mask_bool = mask_bool[..., 0]
        # Default full mask
        mask = kwargs.get('mask', None)
        # Default mask: full
        if mask is None:
            mask_bool = np.ones(source_lab.shape[:2], dtype=bool)
        else:
            if not hasattr(mask, 'shape'):
                raise ValueError("Mask must be a numpy array")
            mask_bool = mask > 128 if mask.dtype == np.uint8 else mask > 0.5
            if mask_bool.ndim == 3:
                mask_bool = mask_bool[..., 0]
            mask_bool = np.ones(source_lab.shape[:2], dtype=bool)
        else:
            if not hasattr(mask, 'shape'):
                raise ValueError("Mask must be a numpy array")
            
        if mask is None:
            raise ValueError("Mask parameter is required")
        if not hasattr(mask, 'shape'):
            raise ValueError("Mask must be a numpy array")
            
        selective_channels = kwargs.get('selective_channels')
        blend_factor = kwargs.get('blend_factor', 1.0)
        if not (source_lab.shape[:2] == target_lab.shape[:2] == mask.shape[:2]):
            raise ValueError("Source, target, and mask must have the same height and width")
        original_dtype = source_lab.dtype
        result_lab = source_lab.astype(np.float64, copy=True)
        s_mean, s_std = self._calculate_stats(source_lab)
        t_mean, t_std = self._calculate_stats(target_lab)
        mask_bool = mask > 128 if mask.dtype == np.uint8 else mask > 0.5
        if mask_bool.ndim == 3:
            mask_bool = mask_bool[..., 0]
        channel_map = {'L': 0, 'a': 1, 'b': 2}
        channels_to_process = [channel_map[c] for c in (selective_channels or ['a', 'b']) if c in channel_map]
        for i in channels_to_process:
            if np.all(~mask_bool):
                continue
            source_channel = source_lab[..., i]
            if s_std[i] < 1e-6:
                transferred_channel = source_channel + (t_mean[i] - s_mean[i])
            else:
                std_ratio = t_std[i] / s_std[i]
                transferred_channel = (source_channel - s_mean[i]) * std_ratio + t_mean[i]
            blended_channel = (transferred_channel * blend_factor) + (source_channel * (1 - blend_factor))
            np.copyto(result_lab[..., i], blended_channel, where=mask_bool)
        return result_lab

    def weighted_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
        """
        Performs LAB color transfer by blending source and target channels based on specified weights.
        result_channel = source_channel * (1 - weight) + target_channel * weight
        """
        if source_lab.shape != target_lab.shape:
            self.logger.warning(f"Source and target LAB images have different shapes: {source_lab.shape} vs {target_lab.shape}. Resizing target to match source.")
            pil_target_lab_rgb = Image.fromarray(self.lab_to_rgb_optimized(target_lab))
            pil_target_lab_resized_rgb = pil_target_lab_rgb.resize((source_lab.shape[1], source_lab.shape[0]), Image.Resampling.LANCZOS)
            target_lab_resized_rgb_np = np.array(pil_target_lab_resized_rgb)
            target_lab = self.rgb_to_lab_optimized(target_lab_resized_rgb_np)

        original_dtype = source_lab.dtype
        source_lab_f = source_lab.astype(np.float32, copy=False)
        target_lab_f = target_lab.astype(np.float32, copy=False)
        
        result_lab_f = np.copy(source_lab_f)
        
        # Default weights are handled by the config, here we expect weights to be passed.
        w_l = weights.get('L', 0.5) # Default to 0.5 if a specific channel weight is missing
        w_a = weights.get('a', 0.5)
        w_b = weights.get('b', 0.5)

        result_lab_f[:,:,0] = source_lab_f[:,:,0] * (1 - w_l) + target_lab_f[:,:,0] * w_l
        result_lab_f[:,:,1] = source_lab_f[:,:,1] * (1 - w_a) + target_lab_f[:,:,1] * w_a
        result_lab_f[:,:,2] = source_lab_f[:,:,2] * (1 - w_b) + target_lab_f[:,:,2] * w_b
        
        return result_lab_f.astype(original_dtype, copy=False)

    def adaptive_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray, **kwargs) -> np.ndarray:
        if source_lab.shape != target_lab.shape:
            raise ValueError("Source and target must have the same shape")
        original_dtype = source_lab.dtype
        src = source_lab.astype(np.float64, copy=False)
        tgt = target_lab.astype(np.float64, copy=False)
        s_mean, s_std = self._calculate_stats(src)
        t_mean, t_std = self._calculate_stats(tgt)
        l_src, a_src, b_src = src[:, :, 0], src[:, :, 1], src[:, :, 2]
        l_tgt = tgt[:, :, 0]
        # Create temporary 3-channel LAB images for histogram matching L channel
        temp_source_lab_for_l = np.zeros_like(src) # Use 'src' which is source_lab.astype(np.float64)
        temp_target_lab_for_l = np.zeros_like(tgt) # Use 'tgt' which is target_lab.astype(np.float64)
        temp_source_lab_for_l[:,:,0] = l_src
        temp_target_lab_for_l[:,:,0] = l_tgt
        # Match only the L channel
        matched_l_temp_source = histogram_matching(temp_source_lab_for_l, temp_target_lab_for_l, channels=['L'])
        l_src_matched = matched_l_temp_source[:,:,0] # Extract the matched L channel
        a_res = np.empty_like(a_src)
        if s_std[1] < 1e-6:
            a_res = a_src + (t_mean[1] - s_mean[1])
        else:
            a_res = (a_src - s_mean[1]) * (t_std[1] / s_std[1]) + t_mean[1]
        b_res = np.empty_like(b_src)
        if s_std[2] < 1e-6:
            b_res = b_src + (t_mean[2] - s_mean[2])
        else:
            b_res = (b_src - s_mean[2]) * (t_std[2] / s_std[2]) + t_mean[2]
        result_lab = np.stack([l_src_matched, a_res, b_res], axis=-1)
        return result_lab.astype(original_dtype, copy=False)

    def hybrid_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray, **kwargs) -> np.ndarray:
        adaptive_result = self.adaptive_lab_transfer(source_lab, target_lab)
        delta_e = calculate_delta_e_lab(source_lab, adaptive_result)
        threshold = self.config.delta_e_threshold
        blend_mask = np.clip(delta_e / threshold, 0, 1)[:, :, np.newaxis]
        basic_result = self.basic_lab_transfer(source_lab, target_lab)
        final_result = (basic_result * blend_mask) + (adaptive_result * (1 - blend_mask))
        return final_result

    def process_image(self, source_img: np.ndarray, target_img: np.ndarray, method: str, **kwargs) -> np.ndarray:
        """
        Main entry point for processing images.
        Handles color space conversions, selects CPU/GPU implementation,
        and routes to the correct transfer method.
        """
        if source_img.shape != target_img.shape:
            raise ValueError("Source and target must have the same shape")
        if source_img.ndim != 3 or source_img.shape[2] != 3:
            raise ValueError("Images must be (H, W, 3)")

        is_rgb = source_img.dtype == np.uint8
        if is_rgb:
            src_lab = self.rgb_to_lab_optimized(source_img)
            tgt_lab = self.rgb_to_lab_optimized(target_img)
        else:
            src_lab = source_img.astype(np.float64, copy=False)
            tgt_lab = target_img.astype(np.float64, copy=False)

        impl = self
        use_gpu = self.gpu_transfer is not None
        transfer_func_name = f"{method}_lab_transfer"

        if use_gpu and hasattr(self.gpu_transfer, transfer_func_name):
            self.logger.info(f"Attempting to use GPU for method: {method}")
            impl = self.gpu_transfer
        else:
            if use_gpu:
                self.logger.warning(f"Method '{method}' not available on GPU. Falling back to CPU.")
            self.logger.info(f"Using CPU for method: {method}")

        transfer_func = getattr(impl, transfer_func_name, None)
        if not transfer_func:
            raise ValueError(f"Invalid or unsupported method: {method}")

        try:
            result_lab = transfer_func(src_lab, tgt_lab, **kwargs)
        except Exception as e:
            if impl is self.gpu_transfer:
                self.logger.error(f"GPU processing failed for method '{method}': {e}. Falling back to CPU.")
                cpu_transfer_func = getattr(self, transfer_func_name)
                result_lab = cpu_transfer_func(src_lab, tgt_lab, **kwargs)
            else:
                self.logger.error(f"CPU processing failed for method '{method}': {e}")
                raise e

        if is_rgb:
            return self.lab_to_rgb_optimized(result_lab)
        return result_lab

    def process_large_image(self, source_img: np.ndarray, target_img: np.ndarray, method: str = 'basic', **kwargs) -> np.ndarray:
        """
        Process large images by tiling, applying the selected transfer method to each tile.
        """
        tile_size = self.config.tile_size
        overlap = self.config.overlap
        h, w, _ = source_img.shape
        
        result_img = np.zeros((h, w, 3), dtype=np.float64)
        weight_map = np.zeros((h, w, 3), dtype=np.float32)

        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                src_tile = source_img[y:y+tile_size, x:x+tile_size]
                tgt_tile = target_img[y:y+tile_size, x:x+tile_size]
                
                if src_tile.shape[0] < tile_size or src_tile.shape[1] < tile_size:
                    # Handle edge tiles that are smaller than tile_size
                    # A simple approach is to process them as is
                    pass

                processed_tile_lab = self.process_image(src_tile, tgt_tile, method=method, **kwargs)
                
                # Create a weight mask for blending
                tile_weight = self.blend_tile_overlap(np.ones_like(processed_tile_lab, dtype=np.float32), overlap)

                # Add the processed tile to the result image, weighted by the blend mask
                th, tw, _ = processed_tile_lab.shape
                result_img[y:y+th, x:x+tw] += processed_tile_lab * tile_weight
                weight_map[y:y+th, x:x+tw] += tile_weight

        # Normalize the result by the weight map to average overlapping areas
        # Avoid division by zero
        weight_map[weight_map == 0] = 1
        result_img /= weight_map

        return self.lab_to_rgb_optimized(result_img)

    def _validate_pipeline_step(self, step: dict, step_idx: int):
        """Validate a single step in the hybrid pipeline."""
        if not isinstance(step, dict):
            raise ValueError(f"Hybrid pipeline step {step_idx} must be a dictionary, got {type(step)}.")
        if "method" not in step:
            raise ValueError(f"Hybrid pipeline step {step_idx} is missing 'method' key.")
        if not isinstance(step["method"], str):
            raise ValueError(f"Hybrid pipeline step {step_idx} 'method' must be a string, got {type(step['method'])}.")
        if step["method"] not in self._CPU_METHOD_DISPATCH:
            raise ValueError(f"Hybrid pipeline step {step_idx} has unknown method '{step['method']}'. "
                             f"Available methods: {list(self._CPU_METHOD_DISPATCH.keys())}")
        if "params" in step and not isinstance(step["params"], dict):
            raise ValueError(f"Hybrid pipeline step {step_idx} 'params' must be a dictionary, got {type(step['params'])}.")

    def blend_tile_overlap(self, tile: np.ndarray, overlap_size: int) -> np.ndarray:
        """Apply linear alpha blending to tile edges based on overlap size"""
        if overlap_size <= 0:
            return tile
            
        blended = np.ones_like(tile, dtype=np.float32)
        h, w, _ = blended.shape
        overlap_size = min(overlap_size, h//2, w//2)  # Ensure overlap is not larger than half the tile
        
        # Create a linear gradient for one edge
        alpha_h = np.linspace(0, 1, overlap_size)
        alpha_v = np.linspace(0, 1, overlap_size)
        
        # Apply to horizontal edges
        if w > overlap_size * 2:
            blended[:, :overlap_size] *= alpha_h[np.newaxis, :, np.newaxis]
            blended[:, -overlap_size:] *= alpha_h[::-1][np.newaxis, :, np.newaxis]
        
        # Apply to vertical edges
        if h > overlap_size * 2:
            blended[:overlap_size, :] *= alpha_v[:, np.newaxis, np.newaxis]
            blended[-overlap_size:, :] *= alpha_v[::-1][:, np.newaxis, np.newaxis]
                
        return tile * blended
        
    def process_image_hybrid(self,
                             source_lab: np.ndarray,
                             target_lab: np.ndarray,
                             hybrid_pipeline: list | None = None,
                             return_intermediate_steps: bool = False,
                             **kwargs) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        """
        Executes a configurable pipeline of color transfer methods on CPU.
        Each step in `hybrid_pipeline` is a dictionary:
            { "method": "<name>", "params": { ... } }
        `kwargs` are passed to each step unless overridden in `step['params']`.
        """
        pipeline_config = hybrid_pipeline if hybrid_pipeline is not None else self.config.get("hybrid_pipeline", [])
        if not isinstance(pipeline_config, list):
            raise ValueError(f"Hybrid pipeline configuration must be a list, got {type(pipeline_config)}.")

        intermediate_results = []
        current_src_lab = source_lab.copy() # Work on a copy

        for idx, step_config in enumerate(pipeline_config):
            self._validate_pipeline_step(step_config, idx)
            method_name_key = step_config["method"]
            method_params = {**kwargs, **step_config.get("params", {})}

            method_to_call_name = self._CPU_METHOD_DISPATCH[method_name_key]
            method_to_call = getattr(self, method_to_call_name)
            
            self.logger.info(f"Hybrid CPU step {idx + 1}/{len(pipeline_config)}: Applying '{method_name_key}' with params: {method_params.get('selective_channels', method_params.get('num_segments', 'N/A'))}")

            current_src_lab = method_to_call(
                current_src_lab, # Result of previous step is input to current
                target_lab,      # target_lab is constant for all steps
                **method_params
            )
            if return_intermediate_steps:
                intermediate_results.append(current_src_lab.copy())

        if return_intermediate_steps:
            return current_src_lab, intermediate_results
        return current_src_lab
