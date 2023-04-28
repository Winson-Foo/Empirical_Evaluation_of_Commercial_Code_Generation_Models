from typing import Tuple
import cv2
import numpy as np

from ..core.node import ConsumerNode

class VideoFileWriter(ConsumerNode):
    '''
    Opens a video file writer object and writes subsequent frames received into the object.
    If video file exists it overwrites it.

    The video writer will open when it receives the first frame.

    Args:
        video_file: Path to video. Folder where video lives must exist. Extension must be .avi
        swap_channels: Whether to swap the color channels of the input frames.
        fps: Frames per second.
    '''

    def __init__(self, video_file: str, swap_channels: bool = True, fps: int = 30) -> None:
        if not video_file.endswith('.avi'):
            raise ValueError('Only .avi format is supported')
        self.video_file = video_file
        self.swap_channels = swap_channels
        self.fps = fps
        self.out = None
        super().__init__()

    def open(self) -> None:
        '''
        We don't open the video writer here. We need to wait for the first frame to
        arrive in order to determine the width and height of the video.
        '''
        pass
    
    def close(self) -> None:
        '''
        Closes the video stream.
        '''
        if self.out is not None and self.out.isOpened():
            self.out.release()

    def consume(self, frame: np.ndarray) -> None:
        '''
        Receives the picture frame to append to the video and appends it to the video.

        If it is the first frame received, it opens the video file and determines the height
        and width of the video from the dimensions of that first frame. Every subsequent frame
        is expected to have the same height and width. If it does not have it, it gets resized to it.

        Args:
            frame: np.ndarray of dimension (height, width, 3)
        '''
        if self.out is None:
            self.height = frame.shape[0]
            self.width = frame.shape[1]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.out = cv2.VideoWriter(self.video_file, fourcc, self.fps, (self.width, self.height))
        
        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.swap_channels:
            resized = resized[..., ::-1]
        self.out.write(resized)