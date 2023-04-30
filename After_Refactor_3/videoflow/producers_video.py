from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging 
import cv2
import numpy as np

from ..core.node import ProducerNode

logger = logging.getLogger(__name__)

class ImageProducer(ProducerNode):
    """Reads a single image and produces it."""

    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = image_path
        self.image_returned = False
    
    def open(self):
        pass
    
    def close(self):
        pass
    
    def next(self) -> np.ndarray:
        """Returns image in RGB format."""
        if not self.image_returned:
            image = cv2.imread(self.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_returned = True
            return image
        else:
            raise StopIteration()

class ImageFolderReader(ProducerNode):
    """Reads from a folder of images and returns them one by one. Passes through images in alphabetical order."""

    def __init__(self, folder_path: str):
        super().__init__()
        self.folder_path = folder_path
    
    def open(self):
        raise NotImplementedError()
    
    def close(self):
        raise NotImplementedError()
    
    def next(self) -> np.ndarray:
        raise NotImplementedError()

class VideoStreamError(Exception):
    """Custom error class for handling VideostreamReader exceptions."""

class VideoStreamReader(ProducerNode):
    """Reader of video streams, using `cv2`.
    
    Args:
        url_or_device_id (Union[int, str]): The url, filesystem path or id of the video stream.
        swap_channels (bool): If true, it will change channels from BGR to RGB.
        num_frames (int): The number of frames when to stop. -1 never stops.
        num_retries (int): If there are errors reading the stream, how many times to retry.
    """

    def __init__(self, url_or_device_id: str, swap_channels: bool = True, num_frames: int = -1, num_retries: int = 0):
        super().__init__()
        self.url_or_device_id = url_or_device_id
        self.swap_channels = swap_channels
        self.num_frames = num_frames
        self.num_retries = num_retries
        self.frame_count = 0
        self.retries_count = 0
        self.video = None
    
    def open(self):
        """Opens the video stream."""
        if self.video is None:
            self.video = cv2.VideoCapture(self.url_or_device_id)

    def close(self):
        """Releases the video stream object."""
        if self.video and self.video.isOpened():
            self.video.release()

    def next(self):
        """Returns the next frame in the video stream.

        Returns:
            (int, np.ndarray): Tuple containing the frame number and the frame in np.ndarray format.
        
        Raises:
            StopIteration: if it reaches the specified number of frames to process or if it reaches the number of retries without success.
        """
        if self.frame_count == self.num_frames:
            raise StopIteration()

        while self.retries_count <= self.num_retries:
            if self.video.isOpened():
                success, frame = self.video.read()
                self.frame_count += 1
                if not success:
                    self.retries_count += 1
                    logger.error(f'Error reading video, retrying {self.retries_count}/{self.num_retries}')
                    if self.video.isOpened():
                        self.video.release()
                    self.video = cv2.VideoCapture(self.url_or_device_id)
                else:
                    if self.swap_channels:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return (self.frame_count, frame)
            else:
                self.video = cv2.VideoCapture(self.url_or_device_id)
            self.retries_count += 1
            logger.error(f'Error reading video, retrying {self.retries_count}/{self.num_retries}')
        logger.error(f'Reached maximum retries for video {self.url_or_device_id}')
        raise VideoStreamError(f'Error reading video {self.url_or_device_id}. Reached maximum retries of {self.num_retries}')

class VideoUrlReader(VideoStreamReader):
    """Opens a video capture object and returns subsequent frames from the video url each time ``next`` is called.

    Args:
        url (str): The URL of the video stream.
        num_frames (int): Number of frames to process. -1 means all of them.
        num_retries (int): Number of retries if there are errors reading the stream.
    """

    def __init__(self, url: str, num_frames: int = -1, num_retries: int = 0):
        super().__init__(url, num_frames=num_frames, num_retries=num_retries)

class VideoDeviceReader(VideoStreamReader):
    """Opens a video capture object and returns subsequent frames from the video device each time ``next`` is called.

    Args:
        device_id (int): The ID of the video device connected to the computer.
        num_frames (int): Number of frames to process. -1 means all of them.
        num_retries (int): Number of retries if there are errors reading the stream.
    """

    def __init__(self, device_id: int, num_frames: int = -1, num_retries: int = 0):
        super().__init__(device_id, num_frames=num_frames, num_retries=num_retries)

class VideoFileReader(VideoStreamReader):
    """Opens a video capture object and returns subsequent frames from the video file each time ``next`` is called.

    Args:
        video_file (str): Path to video file.
        swap_channels (bool): If true, swaps from BGR to RGB.
        num_frames (int): Number of frames to process. -1 means all of them.
        num_retries (int): Number of retries if there are errors reading the stream.
    """

    def __init__(self, video_file: str, swap_channels: bool = False, num_frames: int = -1, num_retries: int = 0):
        super().__init__(video_file, swap_channels = swap_channels, num_frames=num_frames, num_retries=num_retries)

# Here for the sake of not breaking old code
VideofileReader = VideoFileReader