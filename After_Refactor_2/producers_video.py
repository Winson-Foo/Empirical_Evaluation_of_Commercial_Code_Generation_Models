import cv2
import numpy as np
import logging 

logger = logging.getLogger(__name__)

class ImageProducer:
    '''
    Reads a single image and produces it
    '''

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image_returned = False
    
    def next(self) -> np.ndarray:
        '''
        Returns image in RGB format.
        '''
        if not self.image_returned:
            im = cv2.imread(self.image_path)
            im = im[...,::-1] # Swap channels from BGR to RGB
            self.image_returned = True
            return im
        else:
            raise StopIteration()


class ImageFolderReader:
    '''
    Reads from a folder of images and returns them one by one.
    Passes through images in alphabetical order.
    '''
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.image_paths = sorted([\p for p in Path(self.folder_path).iterdir() if p.is_file()])

    def next(self) -> np.ndarray:
        if len(self.image_paths) < 1:
            raise StopIteration()
        
        image_path = self.image_paths.pop(0)
        im = cv2.imread(str(image_path))
        im = im[...,::-1] # Swap channels from BGR to RGB
        return im


class VideoStreamReader:
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
    def __init__(self, url_or_deviceid, swap_channels=True, nb_frames=-1, nb_retries=0):
        self.url_or_deviceid = url_or_deviceid
        self.video = None
        self.swap_channels = swap_channels
        self.nb_frames = nb_frames
        self.frame_count = 0
        self.nb_retries = nb_retries
        self.retries_count = 0

    def open(self):
        '''
        Opens the video stream
        '''
        if self.video is None:
            self.video = cv2.VideoCapture(self.url_or_deviceid)

    def close(self):
        '''
        Releases the video stream object
        '''
        if self.video and self.video.isOpened():
            self.video.release()

    def next(self):
        '''
        - Returns:
            - frame no / index  : integer value of the frame read
            - frame: np.ndarray of shape (h, w, 3)

        - Raises:
            - StopIteration: after it finishes reading the videofile \
                or when it reaches the specified number of frames to \
                process, or if it reaches the number of retries wihout \
                success.
        '''
        if self.frame_count == self.nb_frames:
            raise StopIteration()

        while self.retries_count <= self.nb_retries:
            if self.video.isOpened():
                success, frame = self.video.read()
                self.frame_count += 1
                if not success:
                    if self.video.isOpened():
                        self.video.release()
                    self.video = cv2.VideoCapture(self.url_or_deviceid)
                else:
                    if self.swap_channels:
                        frame = frame[...,::-1] # Swap channels from BGR to RGB
                    return (self.frame_count, frame)
            else:
                self.video = cv2.VideoCapture(self.url_or_deviceid)
            self.retries_count += 1
            logger.error(f'Error reading video, increasing retries count to {self.retries_count}')
        raise StopIteration()


class VideoFolderReader:
    '''
    Reads videos from a folder of videos and returns the frames of 
    the videos one by one.
    Passes through videos in alphabetical order.
    '''
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.video_paths = sorted([p for p in Path(self.folder_path).iterdir() if p.is_file()])

    def next(self) -> tuple:
        if len(self.video_paths) < 1:
            raise StopIteration()
        
        video_path = self.video_paths.pop(0)
        reader = VideoStreamReader(str(video_path))
        reader.open()
        for i, frame in enumerate(reader):
            if i >= reader.nb_frames and reader.nb_frames != -1:
                break
            yield (i, frame)

        reader.close()


class VideoFileReader(VideoStreamReader):
    '''
    Opens a video capture object and returns subsequent frames
    from the video file each time ``next`` is called.
    - Arguments:
        - video_file: path to video file
        - swap_channels: If true, swaps from BGR to RGB
        - nb_frames: number of frames to process. -1 means all of them
    '''
    def __init__(self, video_file: str, swap_channels: bool = False, nb_frames=-1):
        super().__init__(video_file, swap_channels=swap_channels, nb_frames=nb_frames)


class VideoUrlReader(VideoStreamReader):
    '''
    Opens a video capture object and returns subsequent frames
    from the video url each time ``next`` is called.

    - Arguments:
        - device_id: id of the video device connected to the computer
        - nb_frames: number of frames to process. -1 means all of them
    '''
    def __init__(self, url: str, nb_frames=-1, nb_retries=0):
        super().__init__(url, nb_frames=nb_frames, nb_retries=nb_retries)


class VideoDeviceReader(VideoStreamReader):
    '''
    Opens a video capture object and returns subsequent frames
    from the video device each time ``next`` is called.

    - Arguments:
        - device_id: id of the video device connected to the computer
        - nb_frames: number of frames to process. -1 means all of them
    '''
    def __init__(self, device_id: int, nb_frames=-1, nb_retries=0):
        super().__init__(device_id, nb_frames=nb_frames, nb_retries=nb_retries)   
