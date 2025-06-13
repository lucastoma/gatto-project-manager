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
        self.logger.info("Initialized Advanced LAB Color Transfer.")

    def hybrid_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """
        Hybrid transfer: performs statistical transfer on the L (luminance) channel
        and histogram matching on the a* and b* (color) channels. This approach
        preserves the overall brightness structure while achieving a more precise
        color palette match.

        Args:
            source_lab: Source image in LAB space (H x W x 3).
            target_lab: Target image in LAB space (H x W x 3).

        Returns:
            The transferred image in LAB space.
        """
        self.logger.info("Executing hybrid transfer (L: stats, a/b: histogram).")
        
        # 1. Perform statistical transfer on the L channel only.
        # We use a helper function to avoid calculating for all channels.
        stat_l_channel = self._transfer_channel_stats(source_lab[..., 0], target_lab[..., 0])

        # 2. Perform histogram matching on a* and b* channels.
        # The function now correctly accepts a `channels` argument.
        hist_ab_channels = histogram_matching(source_lab, target_lab, channels=['a', 'b'])

        # 3. Combine the results.
        result_lab = np.copy(source_lab)
        result_lab[..., 0] = stat_l_channel
        result_lab[..., 1] = hist_ab_channels[..., 1]
        result_lab[..., 2] = hist_ab_channels[..., 2]
        
        self.logger.info("Hybrid transfer complete.")
        return result_lab
