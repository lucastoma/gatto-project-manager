from typing import Dict, Any, Optional
import os
import json
import time
import numpy as np
from PIL import Image

from ...core.performance_profiler import get_profiler, PerformanceProfiler
from ...core.development_logger import get_logger
from .core import LABColorTransfer, LABTransferConfig

class LABTransferAlgorithm:
    """
    LAB Color Transfer Algorithm
    
    Core functionality:
    1. Convert images to LAB color space for perceptual accuracy
    2. Apply various LAB transfer methods (basic, weighted, selective, adaptive, hybrid)
    3. Support for CPU/GPU processing
    4. Handle tiling for large images
    """
    
    def __init__(self, algorithm_id: str = "algorithm_05_lab_transfer"):
        self.algorithm_id = algorithm_id
        self.logger = get_logger()
        self.profiler: PerformanceProfiler = get_profiler()
        
        self.logger.info(f"Initialized {self.algorithm_id}")
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information for monitoring and documentation."""
        return {
            'algorithm_id': self.algorithm_id,
            'name': 'LAB Color Transfer',
            'description': 'Advanced LAB color space transfer with multiple methods',
            'version': '1.0.1', # Incremented version due to significant update
            'color_space': 'LAB',
            'parameters': {
                'methods': ['basic', 'weighted', 'selective', 'adaptive', 'hybrid'], # Added 'adaptive'
                'channels': ['L', 'a', 'b'],
                'processing': ['cpu', 'gpu', 'hybrid'] # Note: core.py handles CPU/GPU via config
            },
            'supported_formats': ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp'],
            'complexity': 'Varies by method',
            'memory_usage': 'O(n)'
        }
    
    def process(self, master_path: str, target_path: str, **kwargs) -> str:
        """
        Main processing method - applies LAB color transfer algorithm.
        
        Args:
            master_path: Path to master image (source of color statistics)
            target_path: Path to target image (will be color-matched)
            **kwargs: Additional parameters including 'processing_method', 'use_gpu',
                      'tile_size', 'overlap', 'mask_path', method-specific JSON strings etc.
        
        Returns:
            Path to the temporary result image file.
        
        Raises:
            FileNotFoundError: If input images don't exist.
            ValueError: If parameters are invalid.
            RuntimeError: If processing fails.
        """
        self.logger.info(f"Starting LAB Transfer ({self.algorithm_id}) for master: '{master_path}', target: '{target_path}'")
        self.logger.debug(f"Received kwargs: {kwargs}")

        processing_method = kwargs.get('processing_method', 'basic')
        use_gpu = kwargs.get('use_gpu', False)
        tile_size = kwargs.get('tile_size', 512)
        overlap = kwargs.get('overlap', 64)
        mask_path = kwargs.get('mask_path', None)

        try:
            # Create config
            lab_config = LABTransferConfig(
                use_gpu=use_gpu,
                tile_size=tile_size,
                overlap=overlap,
                method=processing_method # method in config can be for general reference
            )
            
            algorithm = LABColorTransfer(config=lab_config)

            # Load images
            source_img_pil = Image.open(master_path).convert('RGB')
            target_img_pil = Image.open(target_path).convert('RGB')
            source_img_np = np.array(source_img_pil)
            target_img_np = np.array(target_img_pil)
            mask_img_np = None
            if mask_path:
                if not os.path.exists(mask_path):
                    self.logger.error(f"Mask file not found at path: {mask_path}")
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")
                mask_img_pil = Image.open(mask_path).convert('RGB') # Assuming RGB mask, can be 'L' too
                mask_img_np = np.array(mask_img_pil)

            # Convert to LAB
            src_lab = algorithm.rgb_to_lab_optimized(source_img_np)
            tgt_lab = algorithm.rgb_to_lab_optimized(target_img_np)

            result_lab = None
            method_params_for_core = {} # For params passed directly to core methods

            self.logger.info(f"Executing LAB transfer with method: '{processing_method}'")

            if processing_method == 'basic':
                result_lab = algorithm.basic_lab_transfer(src_lab, tgt_lab)
            elif processing_method == 'weighted':
                weights_json = kwargs.get('channel_weights_json', '{}')
                channel_weights = json.loads(weights_json) if weights_json else {'L': 0.5, 'a': 0.5, 'b': 0.5}
                result_lab = algorithm.weighted_lab_transfer(src_lab, tgt_lab, weights=channel_weights)
            elif processing_method == 'selective':
                if mask_img_np is None:
                    self.logger.error("Selective method requires a mask, but no valid mask was provided.")
                    raise ValueError("Selective method requires a mask image.")
                selective_channels_json = kwargs.get('selective_channels_json', '["L", "a", "b"]')
                selective_channels = json.loads(selective_channels_json) if selective_channels_json else ['L', 'a', 'b']
                blend_factor = float(kwargs.get('blend_factor', 0.5))
                result_lab = algorithm.selective_lab_transfer(src_lab, tgt_lab, mask_img_np, 
                                                              selective_channels=selective_channels, 
                                                              blend_factor=blend_factor)
            elif processing_method == 'adaptive':
                method_params_for_core['adaptation_method'] = kwargs.get('adaptation_method', 'none')
                method_params_for_core['num_segments'] = int(kwargs.get('num_segments', 100))
                method_params_for_core['delta_e_threshold'] = float(kwargs.get('delta_e_threshold', 10.0))
                method_params_for_core['min_segment_size_perc'] = float(kwargs.get('min_segment_size_perc', 0.01))
                result_lab = algorithm.adaptive_lab_transfer(src_lab, tgt_lab, **method_params_for_core)
            elif processing_method == 'hybrid':
                hybrid_pipeline_json = kwargs.get('hybrid_pipeline_json', '[]')
                pipeline_config = json.loads(hybrid_pipeline_json) if hybrid_pipeline_json else []
                if not pipeline_config:
                     self.logger.warning("Hybrid method called without 'hybrid_pipeline_json', falling back to basic transfer.")
                     result_lab = algorithm.basic_lab_transfer(src_lab, tgt_lab)
                else:
                     result_lab = algorithm.hybrid_lab_transfer(src_lab, tgt_lab, pipeline_config=pipeline_config)
            else:
                self.logger.error(f"Unknown processing_method: {processing_method}")
                raise ValueError(f"Unknown processing_method: {processing_method}")

            if result_lab is None:
                self.logger.error(f"Processing method '{processing_method}' failed to produce a LAB result.")
                raise RuntimeError(f"Image processing with method '{processing_method}' failed.")

            # Convert back to RGB
            result_rgb_np = algorithm.lab_to_rgb_optimized(result_lab)
            result_img_pil = Image.fromarray(result_rgb_np.astype(np.uint8))

            # Save result to a temporary path. API route will move it.
            target_dir = os.path.dirname(target_path) # Assumes target_path is in a writable temp dir like UPLOAD_FOLDER
            timestamp_ms = int(time.time() * 1000)
            original_target_name, original_target_ext = os.path.splitext(os.path.basename(target_path))
            if not original_target_ext: original_target_ext = '.png' # Default extension
            
            temp_result_filename = f"temp_algo05_{processing_method}_{timestamp_ms}{original_target_ext}"
            output_path = os.path.join(target_dir, temp_result_filename)
            
            result_img_pil.save(output_path)
            self.logger.success(f"LAB Transfer ({processing_method}) result saved temporarily to: {output_path}")
            return output_path

        except FileNotFoundError as e:
            self.logger.error(f"File not found during LAB transfer: {str(e)}", exc_info=True)
            raise
        except ValueError as e:
            self.logger.error(f"Value error during LAB transfer: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during LAB transfer ({processing_method}): {str(e)}", exc_info=True)
            raise RuntimeError(f"Algorithm processing failed with method '{processing_method}': {str(e)}") from e
    
    def process_images(self, master_path: str, target_path: str, output_path: Optional[str] = None, **kwargs) -> str:
        """
        Alternative processing method with explicit output path.
        If output_path is provided in kwargs, it will be used.
        """
        if output_path:
            kwargs['output_path'] = output_path # Ensure it's in kwargs for the main process method
        return self.process(master_path, target_path, **kwargs)

def create_lab_transfer_algorithm():
    """Factory function to create LAB Transfer Algorithm instance."""
    return LABTransferAlgorithm()

