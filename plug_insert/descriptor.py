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
class DummyPCA:
    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


@dataclass
class Descriptor:
    pca: PCA
    cov: np.ndarray
    encoded: List[np.ndarray]
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
                rpy = quaternion2rpy(xyzw2wxyz(quat_xyzw))[0]
                vec = np.hstack((trans, rpy))
                vec_seq.append(vec)
            traj_vec = np.array(vec_seq).flatten()
            traj_vec_list.append(traj_vec)

        arr = np.array(traj_vec_list)
        pca = PCA(n_components=n_bottleneck)
        pca.fit(arr)
        # pca = DummyPCA()
        encoded = pca.transform(arr)

        Z = pca.transform(np.array(traj_vec_list))
        cov = np.cov(Z.T)
        return cls(pca, cov, encoded, n_bottleneck, header, ref_mat)

    def sample(self) -> List[CoordinateTransform]:
        p = np.random.multivariate_normal(np.zeros(self.n_bottleneck), self.cov)
        return self._inverse(p)

    def reproduce(self, idx: int) -> List[CoordinateTransform]:
        p = self.encoded[idx]
        return self._inverse(p)

    def _inverse(self, p) -> List[CoordinateTransform]:
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
    hs = History.load_all()
    desc = Descriptor.from_histories(hs)
    # transform_list = desc.reproduce(0)
    transform_list = desc.sample()
    viewer = TrimeshSceneViewer()

    # transform_list = [CoordinateTransform.from_ros_pose(p.pose) for p in hs[0].pose_list]

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
