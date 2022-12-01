from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from geometry_msgs.msg import PoseStamped
from mohou_ros_utils.utils import CoordinateTransform
from sklearn.decomposition import PCA
from std_msgs.msg import Header

from plug_insert.common import History


@dataclass
class Descriptor:
    pca: PCA
    cov: np.ndarray
    n_bottleneck: int
    header: Header

    @classmethod
    def from_histories(
        cls, histories: List[History], n_bottleneck: Optional[int] = None
    ) -> "Descriptor":
        assert len(histories) > 1
        if n_bottleneck is None:
            n_bottleneck = len(histories)
        header = histories[0].pose_list[0].header
        traj_vec_list = []
        for history in histories:
            vec_seq = []
            for pose in history.pose_list:
                header = pose.header
                co = CoordinateTransform.from_ros_pose(pose.pose)
                transform = co.to_ros_transform()
                vec = np.hstack(transform)
                vec_seq.append(vec)
            traj_vec = np.array(vec_seq).flatten()
            traj_vec_list.append(traj_vec)

        arr = np.array(traj_vec_list)
        pca = PCA(n_components=n_bottleneck)
        pca.fit(arr)

        Z = pca.transform(np.array(traj_vec_list))
        cov = np.cov(Z.T)
        return cls(pca, cov, n_bottleneck, header)

    def sample(self) -> List[PoseStamped]:
        p = np.random.multivariate_normal(np.zeros(self.n_bottleneck), self.cov)
        traj_vec = self.pca.inverse_transform(p)
        traj_arr = traj_vec.reshape((-1, 7))
        pose_list = []
        for vec in traj_arr:
            quat = vec[3:]
            norm = np.linalg.norm(quat)
            error = abs(norm - 1.0)
            if error > 1e-6:
                print("fix norm {}".format(norm))
                assert error < 0.1
                quat = quat / norm
            transform = CoordinateTransform.from_ros_transform((vec[:3], quat))
            pose = PoseStamped()
            pose.header = self.header
            pose.pose = transform.to_ros_pose()
            pose_list.append(pose)
        return pose_list


if __name__ == "__main__":
    desc = Descriptor.from_histories(History.load_all())
    while True:
        try:
            transform_list = desc.sample()
            break
        except AssertionError:
            print("failed")
    print("sampled")
