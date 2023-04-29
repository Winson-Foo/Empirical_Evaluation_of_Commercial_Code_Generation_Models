import cv2
import videoflow.core.node as node

class VideoUrlReader(node.SourceNode):
    def __init__(self, stream_url):
        super(VideoUrlReader, self).__init__()
        self.stream_url = stream_url
        
    def read(self):
        cap = cv2.VideoCapture(self.stream_url)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.emit(frame)
        cap.release()
