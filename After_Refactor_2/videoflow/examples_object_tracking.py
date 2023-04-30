'''
Downloads a sample video file of an 
intersection, and runs the detector and 
tracker on it. Outputs annotated video 
to output.avi
'''
import numpy as np
from videoflow import FlowBuilder, VideofileReader, VideofileWriter 
from videoflow.producers import FuncFilter
from videoflow_contrib.detector_tf import TensorflowObjectDetector
from videoflow_contrib.tracker_sort import KalmanFilterBoundingBoxTracker
from videoflow.processors.vision.annotators import TrackerAnnotator
from videoflow.utils.downloader import get_file

BASE_URL_EXAMPLES = "https://github.com/videoflow/videoflow/releases/download/examples/"
VIDEO_NAME = "intersection.mp4"
URL_VIDEO = BASE_URL_EXAMPLES + VIDEO_NAME

class BoundingBoxesFilter(FuncFilter):
    def __init__(self, class_indexes_to_keep):
        self._class_indexes_to_keep = class_indexes_to_keep
        super(BoundingBoxesFilter, self).__init__(self._filter_boxes)
    
    def _filter_boxes(self, dets):
        '''
        Keeps only the boxes with the class indexes
        specified in self._class_indexes_to_keep

        - Arguments:
            - dets: np.array of shape (nb_boxes, 6) \
                Specifically (nb_boxes, [xmin, ymin, xmax, ymax, class_index, score])
        '''
        f = np.array([dets[:, 4] == a for a in self._class_indexes_to_keep])
        f = np.any(f, axis = 0)
        filtered = dets[f]
        return [filtered]

def main():
    input_file = get_file(VIDEO_NAME, URL_VIDEO)
    output_file = "output.avi"
    class_indexes_to_keep = [1] # Class corresponding to automobiles
    
    # Instantiate the flow builder
    f = FlowBuilder("intersection_flow"). \
            add(VideofileReader(input_file)). \
            add(FuncFilter(lambda x: (x.index, x.frame))). \
            add(TensorflowObjectDetector(num_classes = 2, architecture = 'fasterrcnn-resnet101', dataset = 'kitti', nb_tasks = 1)). \
            add(BoundingBoxesFilter(class_indexes_to_keep)). \
            add(KalmanFilterBoundingBoxTracker()). \
            add(TrackerAnnotator()). \
            add(VideofileWriter(output_file, fps = 30))
    
    # Build and run the flow
    flow = f.build()
    flow.run()
    flow.join()

if __name__ == "__main__":
    main()
