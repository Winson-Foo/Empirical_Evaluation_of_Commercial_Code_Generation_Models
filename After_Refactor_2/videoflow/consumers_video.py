from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import cv2
import numpy as np

from ..core.node import ConsumerNode


class VideofileWriter(ConsumerNode):
    """
    Opens a video file writer object and writes subsequent frames received into the object.
    If the video file exists, it will overwrite it.

    The video writer will open when it receives the first frame.

    Args:
        video_file (str): path to video file.
        swap_channels (bool): determines if RGB channels should be swapped.
        fps (int): frames per second.
    """

    SUPPORTED_VIDEO_EXTENSION = '.avi'
    FOURCC_CODEC = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    def __init__(self, video_file: str, swap_channels: bool = True, fps: int = 30):
        if not video_file.endswith(self.SUPPORTED_VIDEO_EXTENSION):
            raise ValueError('Only .avi format is supported')
        self.video_file = video_file
        self.swap_channels = swap_channels
        self.fps = fps
        self.out = None
        self.height = None
        self.width = None
        super(VideofileWriter, self).__init__()

    def open(self):
        """
        Initialize the video writer.

        We don't open the video writer here.
        We need to wait for the first frame to arrive in order to determine the
        width and height of the video.
        """
        pass
    
    def close(self):
        """
        Close the video writer.
        """
        if self.out is not None and self.out.isOpened():
            self.out.release()

    def consume(self, item: np.array):
        """
        Receive a picture frame to append to the video and append it to the video.

        If it is the first frame received, it opens the video file and determines
        the height and width of the video from the dimensions of that first frame.
        Every subsequent frame is expected to have the same height and width. If it
        does not, it gets resized to it.

        Args:
            item (ndarray): a numpy array of dimension (height, width, 3).
        """
        if self.out is None:
            self.height = item.shape[0]
            self.width = item.shape[1]
            self.out = cv2.VideoWriter(self.video_file, self.FOURCC_CODEC, self.fps, (self.width, self.height))
        
        resized = cv2.resize(item, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.swap_channels:
            resized = resized[..., ::-1]
        self.out.write(resized)
