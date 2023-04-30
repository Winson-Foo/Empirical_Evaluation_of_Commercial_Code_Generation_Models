import cv2
import videoflow.core.node as node

class VideoFileWriter(node.SinkNode):
    def __init__(self, output_file, fps):
        super(VideoFileWriter, self).__init__()
        self.output_file = output_file
        self.fps = fps
    
    def write(self, item):
        writer = cv2.VideoWriter(
            self.output_file,
            cv2.VideoWriter_fourcc(*'XVID'),
            self.fps,
            (item.shape[1], item.shape[0])
        )
        writer.write(item)