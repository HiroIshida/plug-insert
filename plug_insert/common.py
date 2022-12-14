import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from geometry_msgs.msg import PoseStamped
from mohou_ros_utils.config import Config
from mohou_ros_utils.rosbag import bag_to_seqs
from skrobot.models.pr2 import PR2

from rosbag import Bag


def project_path() -> Path:
    return Path("~/.mohou/plug_insert").expanduser()


def set_to_homeposition(robot: PR2):
    path = project_path()
    config = Config.from_project_path(path)
    assert config.home_position is not None
    for name, value in config.home_position.items():
        robot.__dict__[name].joint_angle(value)


@dataclass
class History:
    bag_file_path: Optional[Path]
    pose_list: List[PoseStamped]

    @classmethod
    def from_bag(cls, bag_file: Path, n_point: int) -> "History":
        print("procssing {}".format(bag_file))
        bag = Bag(bag_file, mode="r")
        seq = bag_to_seqs(bag, ["/relative_pose"])[0]
        bag.close()

        # relative pose to reference
        # pose_list = np.random.randn(1000).tolist()
        pose_list: List[PoseStamped] = seq.object_list  # type: ignore
        if not isinstance(pose_list[0], PoseStamped):
            for i, pose in enumerate(pose_list):
                # because rosbag create temporary rosmsg type
                # we will encounter the error like
                # ModuleNotFoundError: No module named 'tmp7szu73zq'
                # when unpickling that

                pose_new = PoseStamped()

                pose_new.header.stamp = pose.header.stamp
                pose_new.header.frame_id = pose.header.frame_id
                pose_new.header.seq = pose.header.seq

                pose_new.pose.position.x = pose.pose.position.x
                pose_new.pose.position.y = pose.pose.position.y
                pose_new.pose.position.z = pose.pose.position.z

                pose_new.pose.orientation.x = pose.pose.orientation.x
                pose_new.pose.orientation.y = pose.pose.orientation.y
                pose_new.pose.orientation.z = pose.pose.orientation.z
                pose_new.pose.orientation.w = pose.pose.orientation.w

                pose_list[i] = pose_new

        indices_list = np.array_split(range(len(pose_list)), n_point)
        heads = [indices[0] for indices in indices_list]
        pose_list_resampled = [pose_list[idx] for idx in heads]
        return cls(bag_file, pose_list_resampled)

    def dump(self) -> None:
        assert self.bag_file_path is not None
        history_path = self.bag_file_path.parent / (self.bag_file_path.stem + ".history")
        with history_path.open(mode="wb") as f:
            pickle.dump(self, f)
        print("dumped to {}".format(history_path))

    @classmethod
    def load_all(cls) -> List["History"]:
        rosbag_base_path = project_path() / "rosbag"
        histories = []
        for p in rosbag_base_path.iterdir():
            if p.name.endswith(".history"):
                with p.open(mode="rb") as f:
                    histories.append(pickle.load(f))
        return histories
