import sys
import cv2
import videoflow
import videoflow.core.flow as flow
from videoflow.core.constants import REALTIME
from videoflow.producers import VideoUrlReader
from videoflow.consumers import VideofileWriter

class FrameIndexSplitter(videoflow.core.node.ProcessorNode):
    '''
    This class is used to extract frames from the video stream
    '''
    def __init__(self):
        super(FrameIndexSplitter, self).__init__()
    
    def process(self, data):
        index, frame = data
        return frame

def read_rtsp_stream(stream_address, username, password):
    '''
    This function reads the RTSP stream and returns the VideoUrlReader object
    '''
    stream_url = f'rtsp://{username}:{password}@{stream_address}'
    reader = VideoUrlReader(stream_url)
    return reader

def extract_frames(reader):
    '''
    This function extracts frames from the VideoUrlReader and returns a FrameIndexSplitter object
    '''
    frame = FrameIndexSplitter()(reader)
    return frame

def write_frames_to_file(output_file, frame):
    '''
    This function writes the frames to a file using VideofileWriter
    '''
    writer = VideofileWriter(output_file, fps=30)(frame)
    return writer

def main():
    try:
        # Get input parameters from command line arguments
        stream_address = sys.argv[1]
        username = sys.argv[2]
        password = sys.argv[3]
        output_file = "output.avi"

        # Read RTSP stream
        reader = read_rtsp_stream(stream_address, username, password)
        
        # Extract frames from video stream
        frame = extract_frames(reader)
        
        # Write frames to file
        writer = write_frames_to_file(output_file, frame)
        
        # Start the flow of data
        fl = flow.Flow([reader], [writer], flow_type=REALTIME)
        fl.run()
        fl.join()
        
    except Exception as e:
        print("An error occurred: ", e)

if __name__ == "__main__":
    main()