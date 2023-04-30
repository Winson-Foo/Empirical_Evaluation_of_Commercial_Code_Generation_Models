from typing import Optional, Tuple

import cv2
import numpy as np

from ..core.node import ConsumerNode


class VideofileWriter(ConsumerNode):
    '''
    Opens a video file writer object and writes subsequent frames received into the object.  If video file exists \
    it overwrites it.
    
    The video writer will open when it receives the first frame.
    '''

    def __init__(self, video_file_path: str, swap_channels: bool = True, fps: int = 30):
        '''
        Constructor method for the VideofileWriter class.
        :param video_file_path: str, path of the video file to write.
        :param swap_channels: bool, flag to indicate if swapping channels is needed. Default: True.
        :param fps: int, frames per second. Default: 30.
        '''
        if not video_file_path.endswith('.avi'):
            raise ValueError('Only .avi format is supported')
        self._video_file_path = video_file_path
        self._swap_channels = swap_channels
        self._fps = fps
        self._video_writer = None
        super(VideofileWriter, self).__init__()

    def open(self):
        '''
        Dummy open method. The video writer will be opened when the first frame is received.
        '''
        pass

    def close(self):
        '''
        Closes the video stream.
        '''
        if self._video_writer is not None and self._video_writer.isOpened():
            self._video_writer.release()

    def consume(self, image: np.array):
        '''
        Receives the picture frame to append to the video and appends it to the video.
        If it is the first frame received, it opens the video file and determines \
            the height and width of the video from the dimensions of that first frame. \
            Every subsequent frame is expected to have the same height and width. If it \
            does not has it, it gets resized to it.
        :param image: np.array, (height, width, 3)
        '''
        if self._video_writer is None:
            height, width, _ = image.shape
            self._open_video_writer(width, height)
        
        resized_image = self._resize_image(image)
        if self._swap_channels:
            resized_image = self._swap_color_channels(resized_image)
        
        self._write_to_video(resized_image)

    def _open_video_writer(self, width: int, height: int):
        four_cc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self._video_writer = cv2.VideoWriter(self._video_file_path, four_cc, self._fps, (width, height))

    def _resize_image(self, image: np.array) -> np.array:
        if (image.shape[0], image.shape[1]) != (self._height, self._width):
            image = cv2.resize(image, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return image

    def _swap_color_channels(self, image: np.array) -> np.array:
        return image[..., ::-1]

    def _write_to_video(self, image: np.array):
        self._video_writer.write(image)