import caiman as cm
import cv2
from caiman.utils.utils import download_demo
import queue
from time import time, sleep

def append_frame_to_queue(frame_queue, frame_iterator, init_batch, target_fps):
    t_start = time()
    for t in range(init_batch, len(frame_iterator)):
        # read frame and append to queue
        frame = next(frame_iterator)
        frame_queue.put(frame)
        sleep(max(0, (t+1-init_batch) / target_fps - time() + t_start))

def get_frame_iterator(device=0, fps=None):
    """
    device: device number (int) or filename (string) for reading from camera or file respectively
    fps: frame per second
    """
    if isinstance(device, int):  # capture from camera
        def capture_iter(device=device, fps=fps):
            cap = cv2.VideoCapture(device)
            if fps is not None:  # set frame rate
                cap.set(cv2.CAP_PROP_FPS, fps)
            while True:
                yield cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
        return capture_iter(device, fps)
    else:  # read frame by frame from file
        return cm.base.movies.load_iter(device, var_name_hdf5='Y')

def process_frames(init_batch, frame_iterator):
    frame_queue = queue.Queue()
    append_frame_to_queue(frame_queue, frame_iterator, init_batch, target_fps)

    # Process frames in the queue
    while True:
        try:
            frame = frame_queue.get()
            # Process the frame here
            # ...

        except queue.Empty:
            # Queue is empty, wait for frames to be added
            sleep(1)  # Adjust sleep time as needed

if __name__ == "__main__":
    init_batch_size = 500  # number of frames to use for initialization
    target_fps = 10

    frame_iterator = get_frame_iterator(download_demo('blood_vessel_10Hz.mat'))
    for _ in range(init_batch_size):
        next(frame_iterator)

    process_frames(init_batch_size, frame_iterator)