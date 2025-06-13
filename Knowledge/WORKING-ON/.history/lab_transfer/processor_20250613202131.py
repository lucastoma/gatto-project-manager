"""
Image batch and large image processing for LAB Color Transfer.
"""
import os
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .config import LABTransferConfig
from .core import LABColorTransfer
from .logger import get_logger

class ImageBatchProcessor:
    """
    Handles batch and large-image processing using LABColorTransfer.
    """
    def __init__(self, config: LABTransferConfig = None):
        self.config = config or LABTransferConfig()
        self.config.validate()
        self.transfer = LABColorTransfer(self.config)
        self.logger = get_logger()

    def _process_single_image(self, args):
        path, target_lab, output_dir, method = args
        try:
            source_image = Image.open(path).convert('RGB')
            source_lab = self.transfer.rgb_to_lab_optimized(np.array(source_image))
            # Apply transfer method
            if method == 'basic':
                result_lab = self.transfer.basic_lab_transfer(source_lab, target_lab)
            elif method == 'weighted':
                result_lab = self.transfer.weighted_lab_transfer(source_lab, target_lab)
            elif method == 'selective':
                result_lab = self.transfer.selective_lab_transfer(source_lab, target_lab)
            elif method == 'adaptive':
                result_lab = self.transfer.adaptive_lab_transfer(source_lab, target_lab)
            else:
                result_lab = self.transfer.basic_lab_transfer(source_lab, target_lab)
            result_rgb = self.transfer.lab_to_rgb_optimized(result_lab)
            output_path = os.path.join(output_dir, f"lab_transfer_{os.path.basename(path)}")
            Image.fromarray(result_rgb).save(output_path)
            return {'input': path, 'output': output_path, 'success': True}
        except Exception as e:
            return {'input': path, 'output': None, 'success': False, 'error': str(e)}

    def process_image_batch(self, image_paths, target_path, output_dir,
                            method='basic', batch_size=10, max_workers=None):
        """
        Batch process images in parallel using ProcessPoolExecutor.
        """
        # Load and convert target
        target_image = Image.open(target_path).convert('RGB')
        target_lab = self.transfer.rgb_to_lab_optimized(np.array(target_image))
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 8)
        self.logger.info(f"Starting parallel batch processing on {max_workers} workers")
        args_list = [(path, target_lab, output_dir, method) for path in image_paths]
        total = len(image_paths)
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_single_image, args) for args in args_list]
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    res = future.result()
                    results.append(res)
                except Exception:
                    self.logger.exception("Error in worker")
                if i % batch_size == 0 or i == total:
                    self.logger.info(f"Processed {i}/{total} images")
        success = sum(1 for r in results if r.get('success'))
        self.logger.info(f"Batch complete: {success}/{total} succeeded")
        return results

    def process_large_image(self, source_path, target_path, output_path,
                             tile_size=None, overlap=None):
        """
        Process a large image by tiling and smoothing overlaps.
        """
        cfg = self.config
        tile_size = tile_size or cfg.tile_size
        overlap = overlap or cfg.overlap
        src_img = Image.open(source_path).convert('RGB')
        tgt_img = Image.open(target_path).convert('RGB')
        src_arr = np.array(src_img)
        tgt_lab = self.transfer.rgb_to_lab_optimized(np.array(tgt_img))
        h, w, _ = src_arr.shape
        out_arr = np.zeros_like(src_arr)
        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                tile = src_arr[y:y+tile_size, x:x+tile_size]
                lab_tile = self.transfer.rgb_to_lab_optimized(tile)
                result_tile = self.transfer.adaptive_lab_transfer(lab_tile, tgt_lab)
                rgb_tile = self.transfer.lab_to_rgb_optimized(result_tile)
                blended = self.transfer.blend_tile_overlap(rgb_tile, out_arr, x, y, overlap)
                out_arr[y:y+tile_size, x:x+tile_size] = blended
        Image.fromarray(out_arr).save(output_path)
