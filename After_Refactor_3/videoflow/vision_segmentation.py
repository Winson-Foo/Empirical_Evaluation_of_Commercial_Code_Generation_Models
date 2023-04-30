from typing import List, Tuple
import numpy as np

from ...core.node import ProcessorNode

class Segmenter(ProcessorNode):
    """Abstract class that defines the interface to do image segmentation in images."""

    def _segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment the input image and return a mask of shape (height, width, num_classes).
        """
        raise NotImplementedError('Subclass must implement it')

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment the input image and return the masks, classes, and scores.

        Arguments:
        - image: Input image of shape (height, width, 3)

        Returns:
        - masks: Array of shape (nb_masks, height, width) containing segmentation masks
        - classes: Array of shape (nb_masks,) containing the class labels for each mask
        - scores: Array of shape (nb_masks,) containing the confidence scores for each mask
        """
        masks = self._segment(image)
        nb_masks = masks.shape[-1]
        classes = np.zeros(nb_masks, dtype=np.int32)
        scores = np.zeros(nb_masks, dtype=np.float32)
        return masks, classes, scores