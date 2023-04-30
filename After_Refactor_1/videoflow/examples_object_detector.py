import videoflow
import videoflow.core.flow as flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader
from videoflow_contrib.detector_tf import TensorflowObjectDetector
from videoflow.processors.vision.annotators import BoundingBoxAnnotator
from videoflow.utils.downloader import get_file

EXAMPLES_BASE_URL = "https://github.com/videoflow/videoflow/releases/download/examples/"
VIDEO_NAME = 'intersection.mp4'
URL_VIDEO = EXAMPLES_BASE_URL + VIDEO_NAME
OUTPUT_VIDEO_FILE_NAME = "output.avi"

class FrameIndexSplitter(videoflow.core.node.ProcessorNode):
    def __init__(self):
        super(FrameIndexSplitter, self).__init__()
    
    def process(self, data):
        index, frame = data
        return frame

def run_object_detection_on_video(input_file_path, output_file_path):
    reader = VideofileReader(input_file_path)
    frame = FrameIndexSplitter()(reader)
    detector = TensorflowObjectDetector()(frame)
    annotator = BoundingBoxAnnotator()(frame, detector)
    writer = VideofileWriter(output_file_path, fps=30)(annotator)
    video_flow = flow.Flow([reader], [writer], flow_type=BATCH)
    video_flow.run()
    video_flow.join()

if __name__ == "__main__":
    input_file_path = get_file(VIDEO_NAME, URL_VIDEO)
    run_object_detection_on_video(input_file_path, OUTPUT_VIDEO_FILE_NAME)
