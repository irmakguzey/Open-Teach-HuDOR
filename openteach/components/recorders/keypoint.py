import os
import time
from copy import deepcopy as copy

import h5py
import numpy as np
from openteach.constants import *
from openteach.utils.network import ZMQKeypointSubscriber, create_pull_socket
from openteach.utils.timer import FrequencyTimer
from openteach.utils.vectorops import *

from .recorder import Recorder


class KeypointRecorder(Recorder):
    def __init__(
        self,
        data_types,
        host,
        keypoint_port,
        storage_path,
    ):
        self.data_types = data_types
        # Set the keypoint subscriber
        self.host = host
        self.raw_keypoint_socket = create_pull_socket(self.host, 8087)
        self.button_keypoint_socket = create_pull_socket(self.host, 8095)
        self.teleop_reset_socket = create_pull_socket(self.host, 8100)
        self.oculus_camera_socket = create_pull_socket(self.host, 8080)
        self.record_socket = create_pull_socket(self.host, 8097)
        self.realsense_port = 10005

        # Timer
        self.timer = FrequencyTimer(VR_FREQ)  # Will preprocess with time anyways

        # Storage path for file
        self._filename = "keypoints"  # We will save
        self.notify_component_start("{}".format(self._filename))
        self._recorder_file_name = os.path.join(storage_path, self._filename + ".h5")
        # Initializing the data containers
        self.keypoint_information = dict()
        self.storage_path = storage_path

    @property
    def recorder_functions(self):
        return {
            "timestamp": self.get_timestamp,
            "original_keypoints": self.get_original_keypoints,
            "oculus_camera": self.get_oculus_camera,
            "keypoint_status": self.get_keypoint_status,
        }

    def _process_data_token(self, data_token):
        return data_token.decode().strip()

    def _extract_data_from_token(self, token):
        data = self._process_data_token(token)
        # information = dict(hand = 'right' if data.startswith('right') else 'left')
        information = dict()
        keypoint_vals = [0] if data.startswith("absolute") else [1]

        # Data is in the format <hand>:x,y,z|x,y,z|x,y,z
        vector_strings = data.split(":")[1].strip().split("|")
        for vector_str in vector_strings:
            vector_vals = vector_str.split(",")
            for float_str in vector_vals[:3]:
                keypoint_vals.append(float(float_str))

        return keypoint_vals

    def _get_hand_coords(self):
        pause_status = self.teleop_reset_socket.recv()
        button_feedback = self.button_keypoint_socket.recv()
        raw_keypoints = self.raw_keypoint_socket.recv()
        oculus_camera = self.oculus_camera_socket.recv()
        keypoint_status = self.record_socket.recv()

        record = keypoint_status == b"True"

        # Processing the keypoints and publishing them
        raw_keypoints = self._extract_data_from_token(raw_keypoints)
        eye_data = self._process_data_token(oculus_camera)
        eye_data = eye_data.split(",")
        eye_data = [float(x) for x in eye_data]

        return raw_keypoints, eye_data, record

    def get_timestamp(self):
        return time.time()

    def get_original_keypoints(self):
        return self.original_keypoints

    def get_oculus_camera(self):
        print(self.oculus_camera)
        return self.oculus_camera

    def get_keypoint_status(self):
        return self.keypoint_status

    def stream(self):
        print("Starting to record hand keypoints")

        self.num_datapoints = 0
        self.record_start_time = time.time()

        while True:
            try:
                self.timer.start_loop()
                self.original_keypoints, self.oculus_camera, self.keypoint_status = (
                    self._get_hand_coords()
                )  # This returns the data type as well but it's not important if not
                # Save the data
                for attribute_key in self.data_types:
                    if attribute_key == "keypoint_status":
                        if self.recorder_functions[attribute_key]() == True:
                            if attribute_key not in self.keypoint_information.keys():
                                self.keypoint_information[attribute_key] = [time.time()]
                            else:
                                self.keypoint_information[attribute_key].append(
                                    time.time()
                                )
                    else:
                        # attribute_recorder_function = self.recorder_functions[attribute_key]
                        if attribute_key not in self.keypoint_information.keys():
                            self.keypoint_information[attribute_key] = [
                                self.recorder_functions[attribute_key]()
                            ]

                        else:
                            self.keypoint_information[attribute_key].append(
                                self.recorder_functions[attribute_key]()
                            )

                self.num_datapoints += 1
                self.timer.end_loop()

            except KeyboardInterrupt:
                self.record_end_time = time.time()
                break

        # Displaying statistics
        self._display_statistics(self.num_datapoints)

        # Saving the metadata
        self._add_metadata(self.num_datapoints)
        self.keypoint_information["keypoint_status"] = [
            self.keypoint_information["keypoint_status"][0],
            self.keypoint_information["keypoint_status"][-1],
        ]

        # Writing to dataset
        print("Compressing keypoint data...")
        with h5py.File(self._recorder_file_name, "w") as file:
            for key in self.keypoint_information.keys():
                if key == "timestamp":
                    self.keypoint_information["timestamp"] = np.array(
                        self.keypoint_information["timestamp"], dtype=np.float64
                    )
                    file.create_dataset(
                        "timestamps",
                        data=self.keypoint_information["timestamp"],
                        compression="gzip",
                        compression_opts=6,
                    )
                elif key == "keypoint_status":
                    self.keypoint_information["keypoint_status"] = np.array(
                        self.keypoint_information["keypoint_status"], dtype=np.float64
                    )
                    file.create_dataset(
                        "keypoint_status",
                        data=self.keypoint_information["keypoint_status"],
                        compression="gzip",
                        compression_opts=6,
                    )
                else:
                    self.keypoint_information[key] = np.array(
                        self.keypoint_information[key], dtype=np.float32
                    )
                    file.create_dataset(
                        key,
                        data=self.keypoint_information[key],
                        compression="gzip",
                        compression_opts=6,
                    )

            # Other metadata
            file.update(self.metadata)
        print("Saved keypoint data in {}".format(self._recorder_file_name))

        data = self.keypoint_information["oculus_camera"]
        import matplotlib.pyplot as plt

        # Plotting
        plt.plot(data[:, 0] - data[0, 0], color="red")
        plt.plot(data[:, 1] - data[0, 1], color="green")
        plt.plot(data[:, 2] - data[0, 2], color="blue")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Change in translation over time")
        plt.grid(True)

        plot_path = os.path.join(self.storage_path, "keypoints_position.png")
        plt.savefig(plot_path)

        plt.figure()
        plt.plot(data[:, 3] - data[0, 3], color="red")
        plt.plot(data[:, 4] - data[0, 4], color="green")
        plt.plot(data[:, 5] - data[0, 5], color="blue")
        plt.plot(data[:, 6] - data[0, 6], color="black")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Change in orientation over time")
        plt.grid(True)

        plot_path = os.path.join(self.storage_path, "keypoints_orientation.png")
        plt.savefig(plot_path)

        print("The calibration figures have been successfully saved!!")
