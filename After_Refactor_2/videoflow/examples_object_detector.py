import videoflow
import videoflow.core.flow as flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader
from videoflow_contrib.detector_tf import TensorflowObjectDetector
from videoflow.processors.vision.annotators import BoundingBoxAnnotator
from videoflow.utils.downloader import get_file

def download_video(url, file_name):
    """Download a video from a given URL"""
    return get_file(file_name, url)

def split_frames(data):
    """Split a frame index and the corresponding frame"""
    index, frame = data
    return frame

def detect_objects(frame):
    """Detect objects in a given frame"""
    return TensorflowObjectDetector()(frame)

def annotate_frames(frame, detector):
    """Annotate a frame with bounding boxes"""
    return BoundingBoxAnnotator()(frame, detector)

def write_output(file_name, annotator):
    """Write annotated frames to a video file"""
    return VideofileWriter(file_name, fps=30)(annotator)

def run_video_flow(input_file, output_file):
    """Run the entire video processing flow"""
    reader = VideofileReader(input_file)
    frame = split_frames(reader)
    detector = detect_objects(frame)
    annotator = annotate_frames(frame, detector)
    writer = write_output(output_file, annotator)
    fl = flow.Flow([reader], [writer], flow_type=BATCH)
    fl.run()
    fl.join()

def main():
    """Main function to download a video and process it"""
    base_url = "https://github.com/videoflow/videoflow/releases/download/examples/"
    video_name = 'intersection.mp4'
    url_video = base_url + video_name
    output_file = "output.avi"

    input_file = download_video(url_video, video_name)
    run_video_flow(input_file, output_file)

if __name__ == "__main__":
    main()