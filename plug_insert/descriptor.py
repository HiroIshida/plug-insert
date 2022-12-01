import contextlib
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from mohou_ros_utils.utils import CoordinateTransform
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from skrobot.coordinates.math import (
    quaternion2rpy,
    rpy2quaternion,
    wxyz2xyzw,
    xyzw2wxyz,
)
from skrobot.model import Axis
from skrobot.viewers import TrimeshSceneViewer
from std_msgs.msg import Header

from plug_insert.common import History


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@dataclass
class Descriptor:
    pca: PCA
    cov: np.ndarray
    n_bottleneck: int
    header: Header
    ref_mat: np.ndarray

    @classmethod
    def from_histories(
        cls, histories: List[History], n_bottleneck: Optional[int] = None
    ) -> "Descriptor":
        assert len(histories) > 1
        if n_bottleneck is None:
            n_bottleneck = len(histories)
        header = histories[0].pose_list[0].header
        traj_vec_list = []

        with temp_seed(1):
            ref_mat = Rotation.random().as_matrix()

        for history in histories:
            vec_seq = []
            for pose in history.pose_list:
                header = pose.header
                cotra = CoordinateTransform.from_ros_pose(pose.pose)
                cotra.rot = cotra.rot.dot(ref_mat)
                trans, quat_xyzw = cotra.to_ros_transform()
                rpy = quaternion2rpy(xyzw2wxyz(quat_xyzw))[1]
                vec = np.hstack((trans, rpy))
                vec_seq.append(vec)
            traj_vec = np.array(vec_seq).flatten()
            traj_vec_list.append(traj_vec)

        arr = np.array(traj_vec_list)
        pca = PCA(n_components=n_bottleneck)
        pca.fit(arr)

        Z = pca.transform(np.array(traj_vec_list))
        cov = np.cov(Z.T)
        return cls(pca, cov, n_bottleneck, header, ref_mat)

    def sample(self) -> List[CoordinateTransform]:
        p = np.random.multivariate_normal(np.zeros(self.n_bottleneck), self.cov)
        traj_vec = self.pca.inverse_transform(p)
        traj_arr = traj_vec.reshape((-1, 6))
        transform_list = []
        for vec in traj_arr:
            trans, rpy = vec[:3], vec[3:]
            quat = tuple(wxyz2xyzw(rpy2quaternion(rpy)).tolist())
            transform = CoordinateTransform.from_ros_transform(
                (trans, quat), src="endeffector", dest=self.header.frame_id
            )
            transform.rot = transform.rot.dot(self.ref_mat.T)
            transform_list.append(transform)
        return transform_list


if __name__ == "__main__":
    desc = Descriptor.from_histories(History.load_all())
    transform_list = desc.sample()
    viewer = TrimeshSceneViewer()

    for transform in transform_list:
        co = transform.to_skrobot_coords()
        axis = Axis.from_coords(co)
        viewer.add(axis)
        print("add")
    viewer.show()

    print("==> Press [q] to close window")
    while not viewer.has_exit:
        time.sleep(0.1)
        viewer.redraw()
