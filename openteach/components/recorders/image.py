import os
import time

import cv2
import h5py
import numpy as np
from openteach.constants import (
    CAM_FPS,
    DEPTH_RECORD_FPS,
    IMAGE_FISHEYE_RECORD_RESOLUTION,
    IMAGE_RECORD_RESOLUTION,
)
from openteach.utils.files import store_pickle_data
from openteach.utils.network import SCRCPY_client, ZMQCameraSubscriber
from openteach.utils.timer import FrequencyTimer

from .recorder import Recorder


# To record realsense streams
class RGBImageRecorder(Recorder):
    def __init__(self, host, image_stream_port, storage_path, filename):
        self.notify_component_start("RGB stream: {}".format(image_stream_port))

        # Subscribing to the image stream port
        self._host, self._image_stream_port = host, image_stream_port
        self.image_subscriber = ZMQCameraSubscriber(
            host=host, port=image_stream_port, topic_type="RGB"
        )

        # Timer
        self.timer = FrequencyTimer(CAM_FPS)

        # Storage path for file
        self._filename = filename
        self._recorder_file_name = os.path.join(storage_path, filename + ".mp4")
        self._metadata_filename = os.path.join(storage_path, filename + ".metadata")

        # Initializing the recorder
        self.recorder = cv2.VideoWriter(
            self._recorder_file_name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            CAM_FPS,
            IMAGE_RECORD_RESOLUTION,
        )
        self.timestamps = []

    def stream(self):
        print(
            "Starting to record RGB frames from port: {}".format(
                self._image_stream_port
            )
        )

        self.num_image_frames = 0
        self.record_start_time = time.time()

        while True:
            try:
                self.timer.start_loop()
                image, timestamp = self.image_subscriber.recv_rgb_image()
                self.recorder.write(image)
                self.timestamps.append(timestamp)

                self.num_image_frames += 1
                self.timer.end_loop()
            except KeyboardInterrupt:
                self.record_end_time = time.time()
                break

        # Closing the socket
        self.image_subscriber.stop()

        # Displaying statistics
        self._display_statistics(self.num_image_frames)

        # Saving the metadata
        self._add_metadata(self.num_image_frames)
        self.metadata["timestamps"] = self.timestamps
        self.metadata["recorder_ip_address"] = self._host
        self.metadata["recorder_image_stream_port"] = self._image_stream_port

        # Storing the data
        print("Storing the final version of the video...")
        self.recorder.release()
        store_pickle_data(self._metadata_filename, self.metadata)
        print("Stored the video in {}.".format(self._recorder_file_name))
        print("Stored the metadata in {}.".format(self._metadata_filename))


class DepthImageRecorder(Recorder):
    def __init__(self, host, image_stream_port, storage_path, filename):
        self.notify_component_start("Depth stream: {}".format(image_stream_port))

        # Subscribing to the image stream port
        self._host, self._image_stream_port = host, image_stream_port
        self.image_subscriber = ZMQCameraSubscriber(
            host=host, port=image_stream_port, topic_type="Depth"
        )

        # Timer
        self.timer = FrequencyTimer(DEPTH_RECORD_FPS)

        # Storage path for file
        self._filename = filename
        self._recorder_file_name = os.path.join(storage_path, filename + ".h5")

        # Intializing the depth data containers
        self.depth_frames = []
        self.timestamps = []

    def stream(self):
        if self.image_subscriber.recv_depth_image() is None:
            raise ValueError("Depth image stream is not active.")

        print(
            "Starting to record depth frames from port: {}".format(
                self._image_stream_port
            )
        )

        self.num_image_frames = 0
        self.record_start_time = time.time()

        while True:
            try:
                self.timer.start_loop()
                depth_data, timestamp = self.image_subscriber.recv_depth_image()
                self.depth_frames.append(depth_data)
                self.timestamps.append(timestamp)

                self.num_image_frames += 1
                self.timer.end_loop()
            except KeyboardInterrupt:
                self.record_end_time = time.time()
                break

        # Closing the socket
        self.image_subscriber.stop()

        # Displaying statistics
        self._display_statistics(self.num_image_frames)

        # Saving the metadata
        self._add_metadata(self.num_image_frames)
        self.metadata["recorder_ip_address"] = self._host
        self.metadata["recorder_image_stream_port"] = self._image_stream_port

        # Writing to dataset - hdf5 is faster and compresses more than blosc zstd with clevel 9
        print("Compressing depth data...")
        with h5py.File(self._recorder_file_name, "w") as file:
            stacked_frames = np.array(self.depth_frames, dtype=np.uint16)
            file.create_dataset(
                "depth_images",
                data=stacked_frames,
                compression="gzip",
                compression_opts=6,
            )

            timestamps = np.array(self.timestamps, np.float64)
            file.create_dataset(
                "timestamps", data=timestamps, compression="gzip", compression_opts=6
            )

            file.update(self.metadata)

        print("Saved compressed depth data in {}.".format(self._recorder_file_name))


class OculusImageRecorder(Recorder):
    def __init__(
        self,
        host,
        image_stream_port,
        storage_path,
        filename,
        SCRCPY_dir,
        FFMPEG_bin,
        ADB_bin,
        SCRCPY_ver,
        oculus_idx,
        oculus_address,
    ):
        self.notify_component_start("Oculus stream: {}".format(image_stream_port))

        # Subscribing to the image stream port
        self._host, self._image_stream_port = host, image_stream_port[oculus_idx]
        self.image_subscriber = SCRCPY_client(
            SCRCPY_dir=SCRCPY_dir,
            FFMPEG_bin=FFMPEG_bin,
            ADB_bin=ADB_bin,
            host=self._host,
            port=self._image_stream_port,
            SCRCPY_ver=SCRCPY_ver,
            oculus_address=oculus_address,
        )
        self.oculus_idx = oculus_idx
        self.oculus_address = oculus_address

        # Timer
        self.timer = FrequencyTimer(30)

        # Storage path for file
        self._filename = filename
        self._recorder_file_name = os.path.join(
            storage_path, filename + "_" + str(self.oculus_idx) + ".avi"
        )
        self._metadata_filename = os.path.join(
            storage_path, filename + "_" + str(self.oculus_idx) + ".metadata"
        )

        # Initializing the recorder
        self.recorder = cv2.VideoWriter(
            self._recorder_file_name,
            cv2.VideoWriter_fourcc(*"XVID"),
            30,
            (self.image_subscriber.WIDTH, self.image_subscriber.HEIGHT),
        )

        # Intializing the depth data containers
        self.oculus_frames = []
        self.timestamps = []

    def stream(self):
        if self.image_subscriber.recv_image() is None:
            raise ValueError("Depth image stream is not active.")

        print(
            "Starting to record Oculus frames from port: {}".format(
                self._image_stream_port
            )
        )

        self.num_image_frames = 0
        self.record_start_time = time.time()

        while True:
            try:
                self.timer.start_loop()
                image, timestamp = self.image_subscriber.recv_image()
                if isinstance(image, (np.ndarray, np.generic)):
                    # print(image.shape)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    self.recorder.write(image)
                    self.timestamps.append(timestamp)

                    self.num_image_frames += 1
                self.timer.end_loop()
            except KeyboardInterrupt:
                self.record_end_time = time.time()
                break

        # Closing the socket
        self.image_subscriber.stop()

        # Displaying statistics
        self._display_statistics(self.num_image_frames)

        # Saving the metadata
        self._add_metadata(self.num_image_frames)
        self.metadata["timestamps"] = self.timestamps
        self.metadata["recorder_ip_address"] = self._host
        self.metadata["recorder_image_stream_port"] = self._image_stream_port

        # Storing the data
        print("Storing the final version of the video...")
        self.recorder.release()
        store_pickle_data(self._metadata_filename, self.metadata)
        print("Stored the video in {}.".format(self._recorder_file_name))
        print("Stored the metadata in {}.".format(self._metadata_filename))


class FishEyeImageRecorder(Recorder):
    def __init__(self, host, image_stream_port, storage_path, filename):
        self.notify_component_start("RGB stream: {}".format(image_stream_port))

        # Subscribing to the image stream port
        print("Image Stream Port", image_stream_port)
        self._host, self._image_stream_port = host, image_stream_port
        self.image_subscriber = ZMQCameraSubscriber(
            host=host, port=image_stream_port, topic_type="RGB"
        )

        # Timer
        self.timer = FrequencyTimer(CAM_FPS)

        # Storage path for file
        self._filename = filename
        self._recorder_file_name = os.path.join(storage_path, filename + ".mp4")
        self._metadata_filename = os.path.join(storage_path, filename + ".metadata")
        self._pickle_filename = os.path.join(storage_path, filename + ".pkl")

        # Initializing the recorder
        self.recorder = cv2.VideoWriter(
            self._recorder_file_name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            CAM_FPS,
            IMAGE_FISHEYE_RECORD_RESOLUTION,
        )
        self.timestamps = []
        self.frames = []

    def stream(self):
        print(
            "Starting to record RGB frames from port: {}".format(
                self._image_stream_port
            )
        )

        self.num_image_frames = 0
        self.record_start_time = time.time()

        while True:
            try:
                self.timer.start_loop()
                image, timestamp = self.image_subscriber.recv_rgb_image()
                self.recorder.write(image)
                self.timestamps.append(timestamp)

                self.frames.append(np.array(image))

                self.num_image_frames += 1
                self.timer.end_loop()
            except KeyboardInterrupt:
                self.record_end_time = time.time()
                break
        self.image_subscriber.stop()

        # Displaying statistics
        self._display_statistics(self.num_image_frames)

        # Saving the metadata
        self._add_metadata(self.num_image_frames)
        self.metadata["timestamps"] = self.timestamps
        self.metadata["recorder_ip_address"] = self._host
        self.metadata["recorder_image_stream_port"] = self._image_stream_port

        # Storing the data
        print("Storing the final version of the video...")
        self.recorder.release()
        store_pickle_data(self._metadata_filename, self.metadata)
        print("Stored the video in {}.".format(self._recorder_file_name))
        print("Stored the metadata in {}.".format(self._metadata_filename))
