from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.decomposition import PCA

from plug_insert.common import History, Trajectory


@dataclass
class Descriptor:
    pca: PCA
    cov: np.ndarray
    n_bottleneck: int

    @classmethod
    def from_histories(cls, histories: List[History], n_bottleneck: int = 8) -> "Descriptor":
        traj_vec_list = []
        for history in histories:
            vec_seq = []
            for vec in history.tf_rarm2ref_list:
                vec_seq.append(vec)
            traj_vec = np.array(vec_seq).flatten()
            traj_vec_list.append(traj_vec)

        arr = np.array(traj_vec_list)
        pca = PCA(n_components=n_bottleneck)
        pca.fit(arr)

        Z = pca.transform(np.array(traj_vec_list))
        cov = np.cov(Z.T)
        return cls(pca, cov, n_bottleneck)

    def sample(self) -> Trajectory:
        p = np.random.multivariate_normal(np.zeros(self.n_bottleneck), self.cov)
        traj_vec = self.pca.inverse_transform(p)
        traj_arr = traj_vec.reshape((-1, 7))
        return Trajectory(list(traj_arr))


if __name__ == "__main__":
    desc = Descriptor.from_histories(History.load_all(), 8)
    traj = desc.sample()
