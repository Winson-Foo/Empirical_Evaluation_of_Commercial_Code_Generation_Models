import sys
import videoflow
import videoflow.core.flow as flow
from videoflow.producers import VideoUrlReader
from videoflow.consumers import VideofileWriter
from FrameIndexSplitter import FrameIndexSplitter
from VideoUrlReader import VideoUrlReader
from VideofileWriter import VideofileWriter

def read_from_rtsp_stream(stream_url):
    reader = VideoUrlReader(stream_url)
    frame = FrameIndexSplitter()(reader)
    return frame

def write_to_file(frame, output_file, fps):
    writer = VideofileWriter(output_file, fps)
    writer.write(frame)

def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <stream_address> <username> <password>")
        return
    
    stream_address = sys.argv[1]
    username = sys.argv[2]
    password = sys.argv[3]
    output_file = "output.avi"
    fps = 30

    stream_url = f'rtsp://{username}:{password}@{stream_address}'
    frame = read_from_rtsp_stream(stream_url)
    write_to_file(frame, output_file, fps)
    
if __name__ == "__main__":
    main()    