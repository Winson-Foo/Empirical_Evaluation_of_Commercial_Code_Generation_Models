import numpy as np
from ...core.node import ProcessorNode


class ImageSegmenter(ProcessorNode):
    '''
    Abstract class that defines the interface to do image
    segmentation in images
    '''

    def segment(self, image: np.ndarray) -> np.ndarray:
        '''
        Segment the input image into multiple masks.

        Args:
            image: A numpy array of shape (h, w, 3).

        Returns:
            A numpy array of shape (h, w, num_classes) containing the segmented masks.

        Raises:
            ValueError: If the input image array is empty.
        '''
        if not image.size:
            raise ValueError('Input image array is empty')
        return self._segment(image)

    def _segment(self, image: np.ndarray) -> np.ndarray:
        '''
        Abstract method that should be implemented by subclasses.

        Args:
            image: A numpy array of shape (h, w, 3).

        Returns:
            A numpy array of shape (h, w, num_classes) containing the segmented masks.
        '''
        raise NotImplementedError('Subclass must implement this method')

    def process(self, image: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        '''
        Process the input image and return segmented masks, classes and scores.

        Args:
            image: A numpy array of shape (h, w, 3).

        Returns:
            A tuple containing:
            - masks: A numpy array of shape (nb_masks, h, w) containing the segmented masks.
            - classes: A numpy array of shape (nb_masks,) containing the class labels.
            - scores: A numpy array of shape (nb_masks,) containing the scores for each mask.
        '''
        masks = self.segment(image)
        classes = np.array([], dtype=np.int64)
        scores = np.array([], dtype=np.float32)
        return masks, classes, scores