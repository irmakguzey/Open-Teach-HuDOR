import os
from abc import ABC
from multiprocessing import Process

import hydra

from openteach.constants import *

from .recorders.image import DepthImageRecorder, FishEyeImageRecorder, RGBImageRecorder
from .sensors import *


class ProcessInstantiator(ABC):
    def __init__(self, configs):
        self.configs = configs
        self.processes = []

    def _start_component(self, configs):
        raise NotImplementedError("Function not implemented!")

    def get_processes(self):
        return self.processes


class RealsenseCameras(ProcessInstantiator):
    """
    Returns all the camera processes. Start the list of processes to start
    the camera stream.
    """

    def __init__(self, configs):
        super().__init__(configs)
        # Creating all the camera processes
        self._init_camera_processes()

    def _start_component(self, cam_idx):
        component = RealsenseCamera(
            stream_configs=dict(
                host=self.configs.host_address,
                port=self.configs.cam_port_offset + cam_idx,
            ),
            cam_serial_num=self.configs.robot_cam_serial_numbers[cam_idx],
            cam_id=cam_idx + 1,
            cam_configs=self.configs.cam_configs,
            stream_oculus=True if self.configs.oculus_cam == cam_idx else False,
        )
        component.stream()

    def _init_camera_processes(self):
        for cam_idx in range(len(self.configs.robot_cam_serial_numbers)):
            self.processes.append(
                Process(target=self._start_component, args=(cam_idx,))
            )


class FishEyeCameras(ProcessInstantiator):
    """
    Returns all the fish eye camera processes. Start the list of processes to start
    the camera stream.
    """

    def __init__(self, configs):
        super().__init__(configs)
        # Creating all the camera processes
        self._init_camera_processes()

    def _start_component(self, cam_idx):
        component = FishEyeCamera(
            cam_index=self.configs.fisheye_cam_numbers[cam_idx],
            stream_configs=dict(
                host=self.configs.host_address,
                port=self.configs.fish_eye_cam_port_offset + cam_idx,
                set_port_offset=self.configs.fish_eye_cam_port_offset,
            ),
            stream_oculus=(
                True
                if self.configs.stream_oculus and self.configs.oculus_cam == cam_idx
                else False
            ),
        )
        component.stream()

    def _init_camera_processes(self):
        for cam_idx in range(len(self.configs.fisheye_cam_numbers)):
            self.processes.append(
                Process(target=self._start_component, args=(cam_idx,))
            )


# NOTE: Teleoperator is not really being used in this setup
class TeleOperator(ProcessInstantiator):
    """
    Returns all the teleoperation processes. Start the list of processes
    to run the teleop.
    """

    def __init__(self, configs):
        super().__init__(configs)

        # For Simulation environment start the environment as well
        if configs.sim_env:
            self._init_sim_environment()
        # Start the Hand Detector
        self._init_detector()
        # Start the keypoint transform
        self._init_keypoint_transform()
        self._init_visualizers()

        if configs.operate:
            self._init_operator()

    # Function to start the components
    def _start_component(self, configs):
        component = hydra.utils.instantiate(configs)
        component.stream()

    # Function to start the detector component
    def _init_detector(self):
        self.processes.append(
            Process(target=self._start_component, args=(self.configs.robot.detector,))
        )

    # Function to start the sim environment
    def _init_sim_environment(self):
        for env_config in self.configs.robot.environment:
            self.processes.append(
                Process(target=self._start_component, args=(env_config,))
            )

    # Function to start the keypoint transform
    def _init_keypoint_transform(self):
        for transform_config in self.configs.robot.transforms:
            self.processes.append(
                Process(target=self._start_component, args=(transform_config,))
            )

    # Function to start the visualizers
    def _init_visualizers(self):

        for visualizer_config in self.configs.robot.visualizers:
            self.processes.append(
                Process(target=self._start_component, args=(visualizer_config,))
            )
        # XELA visualizer
        if self.configs.run_xela:
            for visualizer_config in self.configs.xela_visualizers:
                self.processes.append(
                    Process(target=self._start_component, args=(visualizer_config,))
                )

    # Function to start the operator
    def _init_operator(self):
        for operator_config in self.configs.robot.operators:

            self.processes.append(
                Process(target=self._start_component, args=(operator_config,))
            )


# Data Collector Class
class Collector(ProcessInstantiator):
    """
    Returns all the recorder processes. Start the list of processes
    to run the record data.
    """

    def __init__(self, configs, demo_num):
        super().__init__(configs)
        self.demo_num = demo_num
        self._storage_path = os.path.join(
            self.configs.storage_path, "demonstration_{}".format(self.demo_num)
        )

        self._init_keypoint_recorder()

        # Initializing the recorders
        self._init_oculus_recorder()

        if configs.record_realsense:
            self._init_camera_recorders()

        if configs.record_fisheye:
            self._init_fish_eye_recorders()

    def _create_storage_dir(self):
        if os.path.exists(self._storage_path):
            return
        else:
            os.makedirs(self._storage_path)

    def _start_component(self, component):
        component.stream()

    # Obtaining the rgb and depth components
    def _start_rgb_component(self, cam_idx):
        component = RGBImageRecorder(
            host=self.configs.host_address,
            image_stream_port=self.configs.cam_port_offset + cam_idx,
            storage_path=self._storage_path,
            filename="cam_{}_rgb_video".format(cam_idx),
        )
        component.stream()

    def _start_depth_component(self, cam_idx):
        component = DepthImageRecorder(
            host=self.configs.host_address,
            image_stream_port=self.configs.cam_port_offset
            + cam_idx
            + DEPTH_PORT_OFFSET,
            storage_path=self._storage_path,
            filename="cam_{}_depth".format(cam_idx),
        )
        component.stream()

    def _start_oculus_component(self, configs, oculus_idx, oculus_address):
        component = hydra.utils.instantiate(
            configs,
            storage_path=self._storage_path,
            filename="oculus",
            oculus_idx=oculus_idx,
            oculus_address=oculus_address,
        )
        component.stream()

    def _start_keypoint_component(self, configs):
        component = hydra.utils.instantiate(configs, storage_path=self._storage_path)
        component.stream()

    def _init_keypoint_recorder(self):
        # Start the recorder component
        self.processes.append(
            Process(
                target=self._start_keypoint_component,
                args=(self.configs.keypoint_recorder,),
            )
        )

    def _start_fish_eye_component(self, cam_idx):
        component = FishEyeImageRecorder(
            host=self.configs.host_address,
            image_stream_port=self.configs.fish_eye_cam_port_offset + cam_idx,
            storage_path=self._storage_path,
            filename="cam_{}_fish_eye_video".format(cam_idx),
        )
        component.stream()

    def _init_fish_eye_recorders(self):
        for cam_idx in range(len(self.configs.fisheye_cam_numbers)):
            self.processes.append(
                Process(target=self._start_fish_eye_component, args=(cam_idx,))
            )

    def _init_camera_recorders(self):
        for cam_idx in range(len(self.configs.robot_cam_serial_numbers)):
            self.processes.append(
                Process(target=self._start_rgb_component, args=(cam_idx,))
            )

            self.processes.append(
                Process(target=self._start_depth_component, args=(cam_idx,))
            )

    def _init_oculus_recorder(self):
        for oculus_idx in range(len(self.configs.oculus_addresses)):
            self.processes.append(
                Process(
                    target=self._start_oculus_component,
                    args=(
                        self.configs.oculus_recorder,
                        oculus_idx,
                        self.configs.oculus_addresses[oculus_idx],
                    ),
                )
            )
