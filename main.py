import os
import cv2


FRAME_CAPTURE_INTERVAL = 1  # in seconds


def extractImages(inputVideo, frames_dir: str):
    # Create a directory to store the frames
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    else:
        for file in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, file))
    
    count = 0
    vidcap = cv2.VideoCapture(inputVideo)
    success, image = vidcap.read()
    success = True
    while success:
        # added this line
        vidcap.set(cv2.CAP_PROP_POS_MSEC,
                   (count * FRAME_CAPTURE_INTERVAL * 1000))
        success, image = vidcap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(frames_dir, f"frame{count}.jpg"), image)
        count = count + 1
    
    print(f'Saves {count} frames', success)


extractImages("test_package.mp4", 'frames')
