"""
Advanced LAB Color Transfer implementations.
"""
import numpy as np
from .core import LABColorTransfer
from .metrics import histogram_matching

class LABColorTransferAdvanced(LABColorTransfer):
    """
    Advanced subclass of LABColorTransfer providing hybrid and adaptive methods.
    """
    def __init__(self, config=None):
        super().__init__(config)

    def selective_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """
        Perform selective LAB transfer based on luminance mask.
        """
        return super().selective_lab_transfer(source_lab, target_lab)

    def adaptive_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """
        Perform adaptive LAB transfer using regional statistics.
        """
        return super().adaptive_lab_transfer(source_lab, target_lab)

    def hybrid_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """
        Hybrid transfer: use statistical transfer for L channel, and histogram matching for a and b channels.
        """
        # Statistical on L
        stat_lab = self.basic_lab_transfer(source_lab, target_lab)
        # Histogram matching on a, b channels
        hist_lab = histogram_matching(source_lab, target_lab)
        # Combine
        result = np.copy(source_lab)
        result[:, :, 0] = stat_lab[:, :, 0]
        result[:, :, 1:] = hist_lab[:, :, 1:]
        return result
