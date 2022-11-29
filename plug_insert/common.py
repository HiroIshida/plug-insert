import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, overload

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


@dataclass
class Trajectory:
    _points: List[np.ndarray]

    @property
    def length(self) -> float:
        n_point = len(self._points)
        total = 0.0
        for i in range(n_point - 1):
            p0 = self._points[i]
            p1 = self._points[i + 1]
            total += float(np.linalg.norm(p1 - p0))
        return total

    def sample_point(self, dist_from_start: float) -> np.ndarray:

        if dist_from_start > self.length + 1e-6:
            raise InvalidSamplePointError("exceed total length")

        dist_from_start = min(dist_from_start, self.length)
        edge_dist_sum = 0.0
        for i in range(len(self) - 1):
            edge_dist_sum += float(np.linalg.norm(self._points[i + 1] - self._points[i]))
            if dist_from_start <= edge_dist_sum:
                diff = edge_dist_sum - dist_from_start
                vec_to_prev = self._points[i] - self._points[i + 1]
                vec_to_prev_unit = vec_to_prev / np.linalg.norm(vec_to_prev)
                point_new = self._points[i + 1] + vec_to_prev_unit * diff
                return point_new
        raise InvalidSamplePointError()

    def resample(self, n_waypoint: int) -> "Trajectory":
        # yeah, it's inefficient. n^2 instead of n ...
        point_new_list = []
        partial_length = self.length / (n_waypoint - 1)
        for i in range(n_waypoint):
            dist_from_start = partial_length * i
            point_new = self.sample_point(dist_from_start)
            point_new_list.append(point_new)
        return Trajectory(point_new_list)

    def numpy(self):
        return np.array(self._points)

    def visualize(self, fax: Tuple, *args, **kwargs) -> None:
        fig, ax = fax
        arr = self.numpy()
        ax.plot(arr[:, 0], arr[:, 1], *args, **kwargs)

    @classmethod
    def from_two_points(cls, start: np.ndarray, goal: np.ndarray, n_waypoint) -> "Trajectory":
        diff = goal - start
        points = [start + diff / (n_waypoint - 1) * i for i in range(n_waypoint)]
        return cls(points)

    @overload
    def __getitem__(self, indices: List[int]) -> List[np.ndarray]:
        pass

    @overload
    def __getitem__(self, indices: slice) -> List[np.ndarray]:
        pass

    @overload
    def __getitem__(self, index: int) -> np.ndarray:
        pass

    def __getitem__(self, indices_like):
        points = self._points
        return points[indices_like]  # type: ignore

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self):
        return self._points.__iter__()

    def adjusted(self, goal: np.ndarray) -> "Trajectory":
        dists = np.sum((self._points - goal) ** 2, axis=1)
        idx_min = np.argmin(dists).item()
        points_new = self._points[: idx_min + 1] + [goal]
        traj_new = Trajectory(points_new)
        return traj_new.resample(len(traj_new) - 1)

    def append(self, point: np.ndarray) -> None:
        self._points.append(point)

    def __add__(self, other: "Trajectory") -> "Trajectory":
        diff_contact = np.linalg.norm(self._points[-1] - other._points[0])
        assert diff_contact < 1e-6
        points = copy.deepcopy(self._points) + copy.deepcopy(other._points[1:])
        return Trajectory(points)


def coords_to_vec(co: Coordinates) -> np.ndarray:
    pos = co.worldpos()
    ypr = rpy_angle(co.worldrot())[0]
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

        vec_list = []
        tf_ref_to_base = CoordinateTransform.from_skrobot_coords(ref_coords, "ref", "base")
        for co in rarm_coords_history:
            tf_rarm_to_base = CoordinateTransform.from_skrobot_coords(co, "rarm", "base")
            tf_rarm_to_ref = chain_transform(tf_rarm_to_base, tf_ref_to_base.inverse())
            co_transformed = tf_rarm_to_ref.to_skrobot_coords()
            vec_list.append(coords_to_vec(co_transformed))

        # FIXME: somehow, when we resample the trajectory, the first
        # point of the trajectory becomes NaN.
        # This probably because at the initial state of the history
        # the values are almost static.
        # Sampling +1 points and remove the first one is the adhoc
        # workaround for this problem.
        tmp = Trajectory(vec_list).resample(n_point + 1)
        rarm_ef_traj = Trajectory(tmp._points[1:])

        # back to transforms again
        tf_list = []
        for vec in rarm_ef_traj:
            co = vec_to_coords(vec)
            tf = CoordinateTransform.from_skrobot_coords(co, "rarm", "ref")
            tf_list.append(tf)
        return cls(bag_file, table, tf_list, "l_gripper_tool_frame")

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
