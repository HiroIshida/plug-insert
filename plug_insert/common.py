import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from mohou_ros_utils.rosbag import bag_to_seqs
from mohou_ros_utils.utils import CoordinateTransform, chain_transform
from sensor_msgs.msg import JointState
from skrobot.coordinates import Coordinates, rpy_matrix
from skrobot.coordinates.math import rpy_angle
from skrobot.model.joint import Joint
from skrobot.models.pr2 import PR2

from rosbag import Bag


def project_path() -> Path:
    return Path("~/.mohou/plug_insert").expanduser()


class InvalidSamplePointError(Exception):
    pass


def coords_to_vec(co: Coordinates) -> np.ndarray:
    pos = co.worldpos()
    ypr = rpy_angle(co.worldrot())[1]
    return np.hstack([pos, ypr])


def vec_to_coords(vec: np.ndarray) -> Coordinates:
    pos, ypr = vec[:3], vec[3:]
    mat = rpy_matrix(*ypr)
    return Coordinates(pos=pos, rot=mat)


@dataclass
class History:
    bag_file_path: Path
    angles_vector_table: Dict[str, List[float]]
    tf_rarm2ref_list: List[CoordinateTransform]
    reference_name: str

    @classmethod
    def from_bag(cls, bag_file: Path, n_point: int) -> "History":
        print("procssing {}".format(bag_file))
        topic_name = "/joint_states"

        bag = Bag(bag_file, mode="r")
        seq = bag_to_seqs(bag, [topic_name])[0]
        bag.close()

        msg_list: List[JointState] = seq.object_list  # type: ignore

        rarm_coords_history = []
        larm_coords_history = []
        pr2 = PR2(use_tight_joint_limit=False)
        table = {name: [] for name in pr2.joint_names}  # type: ignore
        for msg in msg_list:
            for name, pos in zip(msg.name, msg.position):
                joint: Joint = pr2.__dict__[name]
                joint.joint_angle(pos)

            for j in pr2.joint_list:
                table[j.name].append(j.joint_angle())

            co_rarm = pr2.rarm_end_coords.copy_worldcoords()
            rarm_coords_history.append(co_rarm)

            co_larm = pr2.larm_end_coords.copy_worldcoords()
            larm_coords_history.append(co_larm)

        larm_init: Coordinates = larm_coords_history[0]
        ref_coords = larm_init

        tf_rarm_to_ref_list = []
        tf_ref_to_base = CoordinateTransform.from_skrobot_coords(ref_coords, "ref", "base")
        for co in rarm_coords_history:
            tf_rarm_to_base = CoordinateTransform.from_skrobot_coords(co, "rarm", "base")
            tf_rarm_to_ref = chain_transform(tf_rarm_to_base, tf_ref_to_base.inverse())
            tf_rarm_to_ref_list.append(tf_rarm_to_ref)

        indices_list = np.array_split(range(len(tf_rarm_to_ref_list)), n_point)
        heads = [indices[0] for indices in indices_list]
        tf_rarm_to_ref_list_resampled = [tf_rarm_to_ref_list[idx] for idx in heads]
        return cls(bag_file, table, tf_rarm_to_ref_list_resampled, "l_gripper_tool_frame")

    def dump(self) -> None:
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
