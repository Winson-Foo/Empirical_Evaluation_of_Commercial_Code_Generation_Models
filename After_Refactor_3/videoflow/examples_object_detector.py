import videoflow
import videoflow.core.flow as flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader
from videoflow_contrib.detector_tf import TensorflowObjectDetector
from videoflow.processors.vision.annotators import BoundingBoxAnnotator
from videoflow.utils.downloader import get_file

# Constants
BASE_URL_EXAMPLES = "https://github.com/videoflow/videoflow/releases/download/examples/"
VIDEO_NAME = 'intersection.mp4'
URL_VIDEO = BASE_URL_EXAMPLES + VIDEO_NAME

# Functions
def download_video():
    """
    Downloads the sample video file from the provided URL
    """
    input_file = get_file(VIDEO_NAME, URL_VIDEO)
    return input_file

def process_frame(frame):
    """
    Processes each frame of the video and adds bounding boxes around
    objects detected in the frame
    """
    detector = TensorflowObjectDetector()(frame)
    annotator = BoundingBoxAnnotator()(frame, detector)
    return annotator

def main():
    input_file = download_video()
    output_file = "output.avi"

    # Set up videoflow nodes
    reader = VideofileReader(input_file)
    frame = videoflow.core.node.ProcessorNode(process=process_frame)(reader)
    writer = VideofileWriter(output_file, fps=30)(frame)

    # Set up videoflow flow
    fl = flow.Flow([reader], [writer], flow_type=BATCH)

    # Run videoflow flow
    fl.run()
    fl.join()

if __name__ == "__main__":
    main()