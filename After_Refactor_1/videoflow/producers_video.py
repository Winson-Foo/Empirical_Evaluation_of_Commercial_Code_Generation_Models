from typing import Any
import logging 

import cv2
import numpy as np

from ..core.node import ProducerNode

logger = logging.getLogger(__name__)

class SingleImageReader(ProducerNode):
    '''
    Reads a single image and returns it in RGB format
    '''

    def __init__(self, image_path: str):
        self._image_path = image_path
        self._image_returned = False
        super().__init__()
    
    def open(self):
        """
        Opens the image file
        """
        pass
    
    def close(self):
        """
        Closes the image file
        """
        pass
    
    def read(self) -> np.array:
        '''
        Returns the image in RGB format
        '''
        if not self._image_returned:
            im = cv2.imread(self._image_path)
            im = im[...,::-1]
            self._image_returned = True
            return im
        else:
            raise StopIteration()

class ImageFolderReader(ProducerNode):
    '''
    Reads from a folder of images and returns them one by one.
    Passes through images in alphabetical order.
    '''
    def __init__(self):
        super().__init__()
    
    def open(self):
        """
        Opens the image folder
        """
        raise NotImplementedError()
    
    def close(self):
        """
        Closes the image folder
        """
        raise NotImplementedError()
    
    def read(self) -> Any:
        """
        Reads the next image in the folder
        """
        raise NotImplementedError()

class VideoFolderReader(ProducerNode):
    '''
    Reads videos from a folder of videos and returns the frames of 
    the videos one by one.
    Passes through videos in alphabetical order.
    '''
    def __init__(self):
        super().__init__()

    def open(self):
        """
        Opens the video folder
        """
        raise NotImplementedError()
    
    def close(self):
        """
        Closes the video folder
        """
        raise NotImplementedError()
    
    def read(self) -> Any:
        """
        Reads the next frame of the next video in the folder
        """
        raise NotImplementedError()

class VideostreamReader(ProducerNode):
    '''
    Reader of video streams, using ``cv2``
    
    - Arguments:
        - url_or_deviceid: (int or str) The url, filesystem path or id of the \
            video stream.
        - swap_channels: if True, it will change channels from BGR to RGB
        - nb_frames: (int) The number of frames when to stop. -1 never stops
        - nb_retries: (int) If there are errors reading the stream, how \
            many times to retry.
    '''
    def __init__(self, url_or_deviceid: str, swap_channels: bool = True,
                 nb_frames: int = -1, nb_retries: int = 0):
        super().__init__()
        self._url_or_deviceid = url_or_deviceid
        self._video = None
        self._swap_channels = swap_channels
        self._nb_frames = nb_frames
        self._frame_count = 0
        self._nb_retries = nb_retries
        self._retries_count = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        '''
        Opens the video stream
        '''
        if self._video is None:
            self._video = cv2.VideoCapture(self._url_or_deviceid)

    def close(self):
        '''
        Releases the video stream object
        '''
        if self._video and self._video.isOpened():
            self._video.release()

    def read(self) -> np.array:
        '''
        - Returns:
            - frame no / index: integer value of the frame read
            - frame: np.array of shape (h, w, 3)
        
        - Raises:
            - StopIteration: after it finishes reading the videofile \
                or when it reaches the specified number of frames to \
                process, or if it reaches the number of retries wihout \
                success.
        '''
        if self._frame_count == self._nb_frames:
            raise StopIteration()

        for i in range(self._nb_retries + 1):
            if self._video.isOpened():
                success, frame = self._video.read()
                self._frame_count += 1
                if not success:
                    if self._video.isOpened():
                        self._video.release()
                    self._video = cv2.VideoCapture(self._url_or_deviceid)
                else:
                    if self._swap_channels:
                        frame = frame[...,::-1]
                    return (self._frame_count, frame)
            else:
                self._video = cv2.VideoCapture(self._url_or_deviceid)
            logger.error(f'Error reading video, increasing retries count to {i + 1}')
        raise StopIteration()
    
class VideoUrlReader(VideostreamReader):
    '''
    Opens a video capture object and returns subsequent frames
    from the video url each time ``read`` is called.

    - Arguments:
        - device_id: id of the video device connected to the computer
        - nb_frames: number of frames to process. -1 means all of them
    '''
    def __init__(self, url: str, nb_frames: int = -1, nb_retries: int = 0):
        super().__init__(url, nb_frames=nb_frames, nb_retries=nb_retries)

class VideoDeviceReader(VideostreamReader):
    '''
    Opens a video capture object and returns subsequent frames
    from the video device each time ``read`` is called.

    - Arguments:
        - device_id: id of the video device connected to the computer
        - nb_frames: number of frames to process. -1 means all of them
    '''
    def __init__(self, device_id: int, nb_frames: int = -1, nb_retries: int = 0):
        super().__init__(device_id, nb_frames=nb_frames, nb_retries=nb_retries)    


class VideoFileReader(VideostreamReader):
    '''
    Opens a video capture object and returns subsequent frames
    from the video file each time ``read`` is called.
    - Arguments:
        - video_file: path to video file
        - swap_channels: If true, swaps from BGR to RGB
        - nb_frames: number of frames to process. -1 means all of them
    '''
    def __init__(self, video_file: str, swap_channels: bool = False, nb_frames: int = -1):
        super().__init__(video_file, swap_channels=swap_channels, nb_frames=nb_frames, nb_retries=0)

# Here for the sake of not breaking old code
VideofileReader = VideoFileReader
