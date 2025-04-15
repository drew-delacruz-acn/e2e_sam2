import cv2
import numpy as np
import random
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

class VideoFrameSampler:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def uniform_sampling(self, interval=30):
        """
        Sample frames uniformly every 'interval' frames.
        """
        sampled_frames = []
        for idx in range(0, self.frame_count, interval):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                sampled_frames.append(frame)
            else:
                break
        return sampled_frames

    def random_sampling(self, num_frames=10):
        """
        Randomly sample 'num_frames' frames from the video.
        """
        sampled_frames = []
        frame_indices = sorted(random.sample(range(self.frame_count), num_frames))
        for idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                sampled_frames.append(frame)
        return sampled_frames

    def sequential_sampling(self, start_frame=0, end_frame=None, step=1):
        """
        Sample frames sequentially from 'start_frame' to 'end_frame' with a given 'step'.
        """
        if end_frame is None:
            end_frame = self.frame_count

        sampled_frames = []
        for idx in range(start_frame, min(end_frame, self.frame_count), step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                sampled_frames.append(frame)
            else:
                break
        return sampled_frames

    def scene_based_sampling(self, threshold=30.0):
        """
        Use PySceneDetect to detect scene boundaries, then sample one frame (the middle frame) from each scene.
        Adjust 'threshold' for sensitivity.
        """
       

        # Create a new SceneManager and add the ContentDetector
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        # Use open_video from PySceneDetect
        video_stream = open_video(self.video_path)

        # Perform scene detection
        scene_manager.detect_scenes(video=video_stream)
        scene_list = scene_manager.get_scene_list()

        sampled_frames = []
        for scene in scene_list:
            start_time, end_time = scene
            # Convert timecode to frame index
            start_frame = start_time.get_frames()
            end_frame = end_time.get_frames()

            # Choose the midpoint frame
            mid_frame = (start_frame + end_frame) // 2

            # Read that midpoint frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = self.cap.read()
            if ret:
                sampled_frames.append(frame)
        return sampled_frames

    def release(self):
        self.cap.release()


# Example Usage
if __name__ == "__main__":
    video_file = "../../data/test.mp4"
    sampler = VideoFrameSampler(video_file)

    # Various sampling methods:
    uniform_frames = sampler.uniform_sampling(interval=60)
    random_frames = sampler.random_sampling(num_frames=5)
    sequential_frames = sampler.sequential_sampling(start_frame=100, end_frame=200, step=10)
    scene_frames = sampler.scene_based_sampling(threshold=30.0)

    # Example: pass the frames to an object detection method
    # detect_objects(scene_frames)

    sampler.release()
