import sys
import cv2
import videoflow
import videoflow.core.flow as flow
from videoflow.core.constants import REALTIME
from videoflow.producers import VideoUrlReader
from videoflow.consumers import VideofileWriter

class FrameSplitter(videoflow.core.node.ProcessorNode):
    """
    This class processes the data received from the video stream.
    It takes in the data as a tuple containing the frame index and the frame. 
    It then returns the frame only.
    """
    def __init__(self):
        super(FrameSplitter, self).__init__()

    def process(self, data):
        index, frame = data
        return frame

def main():
    # Read in the command line arguments
    stream_address = sys.argv[1]
    username = sys.argv[2]
    password = sys.argv[3]

    # Set up video url reader
    stream_url = f'rtsp://{username}:{password}@{stream_address}'
    reader = VideoUrlReader(stream_url)

    # Set up frame splitter
    split_frame = FrameSplitter()(reader)

    # Set up video file writer
    output_file = "output.avi"
    writer = VideofileWriter(output_file, fps=30)(split_frame)

    # Set up and run the flow pipeline
    pipeline = flow.Flow([reader], [writer], flow_type=REALTIME)
    pipeline.run()
    pipeline.join()

if __name__ == "__main__":
    main()