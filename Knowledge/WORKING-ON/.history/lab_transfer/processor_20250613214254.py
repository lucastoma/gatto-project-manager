"""
Image batch and large image processing for LAB Color Transfer.
This module provides parallel processing capabilities for handling multiple images
or very large images efficiently.
"""
import os
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .config import LABTransferConfig
from .advanced import LABColorTransferAdvanced # Use the advanced class
from .logger import get_logger

class ImageBatchProcessor:
    """
    Handles batch and large-image processing using LABColorTransfer.
    """
    def __init__(self, config: LABTransferConfig = None):
        self.config = config or LABTransferConfig()
        self.config.validate()
        self.transfer = LABColorTransferAdvanced(self.config)
        self.logger = get_logger()

    def _process_single_image(self, args):
        """A helper method to be run in a separate process."""
        path, target_lab, method = args
        try:
            source_image = Image.open(path).convert('RGB')
            source_lab = self.transfer.rgb_to_lab_optimized(np.array(source_image))
            
            # Apply the selected transfer method based on the config
            if method == 'basic':
                result_lab = self.transfer.basic_lab_transfer(source_lab, target_lab)
            elif method == 'linear_blend':
                result_lab = self.transfer.linear_blend_lab(source_lab, target_lab, self.config.channel_weights)
            elif method == 'selective':
                result_lab = self.transfer.selective_lab_transfer(source_lab, target_lab, self.config.selective_channels)
            elif method == 'adaptive':
                result_lab = self.transfer.adaptive_lab_transfer(source_lab, target_lab)
            elif method == 'hybrid':
                result_lab = self.transfer.hybrid_transfer(source_lab, target_lab)
            else: # Fallback to basic
                result_lab = self.transfer.basic_lab_transfer(source_lab, target_lab)

            result_rgb = self.transfer.lab_to_rgb_optimized(result_lab)
            
            output_dir = os.path.dirname(path) # Save in the same directory for simplicity
            output_filename = f"processed_{os.path.basename(path)}"
            output_path = os.path.join(output_dir, output_filename)
            Image.fromarray(result_rgb).save(output_path)
            
            return {'input': path, 'output': output_path, 'success': True}
        except Exception as e:
            self.logger.exception(f"Failed to process image {path}")
            return {'input': path, 'output': None, 'success': False, 'error': str(e)}

    def process_image_batch(self, image_paths, target_path, max_workers: int = None):
        """
        Batch process images in parallel using ProcessPoolExecutor.
        """
        # Load and convert target once
        target_image = Image.open(target_path).convert('RGB')
        target_lab = self.transfer.rgb_to_lab_optimized(np.array(target_image))
        
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 8)

        self.logger.info(f"Starting parallel batch processing on {max_workers} workers for {len(image_paths)} images.")
        
        args_list = [(path, target_lab, self.config.method) for path in image_paths]
        total = len(image_paths)
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_single_image, args): args for args in args_list}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    path = futures[future][0]
                    self.logger.exception(f"Image {path} generated an exception: {exc}")
                
                if i % 10 == 0 or i == total:
                    self.logger.info(f"Progress: {i}/{total} images processed.")

        success_count = sum(1 for r in results if r.get('success'))
        self.logger.info(f"Batch processing complete: {success_count}/{total} succeeded.")
        return results

    def process_large_image(self, source_path, target_path, output_path):
        """
        Process a large image by tiling and smoothing overlaps.
        """
        src_img = Image.open(source_path).convert('RGB')
        tgt_img = Image.open(target_path).convert('RGB')
        
        src_arr = np.array(src_img)
        tgt_lab = self.transfer.rgb_to_lab_optimized(np.array(tgt_img))
        
        h, w, _ = src_arr.shape
        out_arr = np.zeros_like(src_arr)
        
        tile_size = self.config.tile_size
        overlap = self.config.overlap

        self.logger.info(f"Processing large image ({w}x{h}) with tile size {tile_size} and overlap {overlap}.")

        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                # Define tile boundaries, ensuring they don't exceed image dimensions
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                tile_src_lab = self.transfer.rgb_to_lab_optimized(src_arr[y:y_end, x:x_end])
                tile_tgt_lab = self.transfer.rgb_to_lab_optimized(np.array(tgt_img.resize(tile_src_lab.shape[1::-1])))
                
                # Use a fixed, robust method for tiling
                result_tile_lab = self.transfer.basic_lab_transfer(tile_src_lab, tile_tgt_lab)
                rgb_tile = self.transfer.lab_to_rgb_optimized(result_tile_lab)
                
                blended_tile = self.transfer.blend_tile_overlap(rgb_tile, out_arr, x, y, overlap)
                out_arr[y:y_end, x:x_end] = blended_tile

        Image.fromarray(out_arr).save(output_path)
        self.logger.info(f"Large image processing complete. Result saved to {output_path}")
