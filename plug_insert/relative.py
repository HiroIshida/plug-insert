import pickle
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Dict, List

from matplotlib.lines import logging
from mohou_ros_utils.rosbag import bag_to_synced_seqs
from sensor_msgs.msg import JointState
from skrobot.coordinates import Coordinates
from skrobot.model.joint import Joint
from skrobot.models.pr2 import PR2

from plug_insert.common import project_path
from rosbag import Bag

logger = getLogger(__name__)


@dataclass
class History:
    angles_vector_table: Dict[str, List[float]]
    rarm_coords_history: List[Coordinates]
    larm_coords_history: List[Coordinates]

    @classmethod
    def from_bag(cls, bag_file: Path, freq: float) -> "History":
        topic_name = "/joint_states"

        bag = Bag(bag_file, mode="r")
        seq = bag_to_synced_seqs(bag, freq, [topic_name])[0]
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

            rarm_coords_history.append(pr2.rarm_end_coords.copy_worldcoords())
            larm_coords_history.append(pr2.larm_end_coords.copy_worldcoords())

        return cls(table, rarm_coords_history, larm_coords_history)


if __name__ == "__main__":
    rosbag_base_path = project_path() / "rosbag"
    logger.setLevel(logging.INFO)

    for path in rosbag_base_path.iterdir():
        if path.is_symlink():
            continue
        if not path.name.endswith(".bag"):
            continue
        history = History.from_bag(path, 0.5)
        history_path = path.parent / (path.stem + ".history")
        with history_path.open(mode="wb") as f:
            pickle.dump(history, f)
        logger.error("dumped history to {}".format(history_path))
