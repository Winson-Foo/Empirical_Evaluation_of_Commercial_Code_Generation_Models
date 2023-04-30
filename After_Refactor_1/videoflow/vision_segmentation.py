from core.node import ProcessorNode
import numpy as np

class ImageSegmenter(ProcessorNode):
    """
    An abstract class that defines the interface to perform image segmentation 
    on images.
    """

    def segment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Perform segmentation on an input RGB image.

        Args:
            image: A numpy array of size (h, w, 3).

        Returns:
            A numpy array of size (h, w, num_classes), representing the binary 
            mask for each class.
        """
        raise NotImplementedError("Subclass must implement this method.")

    def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform segmentation on an input RGB image and return the resulting masks, classes, and scores.

        Args:
            image: A numpy array of size (h, w, 3).

        Returns:
            A tuple containing:
            - masks: A numpy array of size (nb_masks, h, w).
            - classes: A numpy array of size (nb_masks,) representing the class label for each mask.
            - scores: A numpy array of size (nb_masks,) representing the confidence score for each mask.
        """
        masks = self.segment_image(image)
        classes = np.array([0] * len(masks))
        scores = np.array([1.0] * len(masks))
        return masks, classes, scores
