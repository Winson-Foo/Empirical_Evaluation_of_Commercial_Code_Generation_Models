import numpy as np

class BoundingBoxTracker:

    def __init__(self):
        '''
        Initializes the tracker with an empty state.
        '''
        self.state = None

    def track(self, detections: np.array) -> np.array:
        '''
        Tracks bounding boxes from one frame to another.
        It keeps an internal state representation that allows
        it to track across frames.

        Args:
        - detections: np.array of shape (num_boxes, 6)
          Specifically (num_boxes, [ymin, xmin, ymax, xmax, class_index, score])

        Returns:
        - tracks: np.array of shape (num_boxes, 5)
          Specifically (num_boxes, [ymin, xmin, ymax, xmax, track_id])
        '''
        raise NotImplementedError("Subclass must implement track method")

class BoundingBoxTrackerProcessor:

    def __init__(self, tracker: BoundingBoxTracker):
        '''
        Initializes the BoundingBoxTrackerProcessor with the specified tracker.
        '''
        self.tracker = tracker

    def process(self, detections: np.array) -> np.array:
        '''
        Processes detections by tracking them with the registered tracker.

        Args:
        - detections: np.array of shape (num_boxes, 6)
          Specifically (num_boxes, [ymin, xmin, ymax, xmax, class_index, score])

        Returns:
        - tracks: np.array of shape (num_boxes, 5)
          Specifically (num_boxes, [ymin, xmin, ymax, xmax, track_id])
        '''
        return self.tracker.track(detections)
