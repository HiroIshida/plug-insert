import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rospy
import tf
import tqdm
from geometry_msgs.msg import Point, Pose, Quaternion
from mohou_ros_utils.utils import CoordinateTransform, chain_transform
from skrobot.coordinates import Coordinates
from skrobot.interfaces.ros import PR2ROSRobotInterface  # type: ignore
from skrobot.model import Axis
from skrobot.models.pr2 import PR2
from skrobot.viewers import TrimeshSceneViewer

from plug_insert.common import History
from plug_insert.descriptor import Descriptor


@dataclass
class RolloutExecutor:
    descriptor: Descriptor
    robot: PR2
    ri: PR2ROSRobotInterface
    tf_listener: tf.TransformListener

    @classmethod
    def init(cls) -> "RolloutExecutor":
        robot = PR2()
        ri = PR2ROSRobotInterface(robot)
        robot.angle_vector(ri.angle_vector())
        histories = History.load_all()
        desc = Descriptor.from_histories(histories)
        listener = tf.TransformListener()
        return cls(desc, robot, ri, listener)

    def solve_inverse_kinematics(self, co: Coordinates) -> bool:
        control_joint_names = [
            "r_shoulder_pan_joint",
            "r_shoulder_lift_joint",
            "r_upper_arm_roll_joint",
            "r_elbow_flex_joint",
            "r_forearm_roll_joint",
            "r_wrist_flex_joint",
            "r_wrist_roll_joint",
        ]
        joints = [self.robot.__dict__[jname] for jname in control_joint_names]
        link_list = [joint.child_link for joint in joints]
        end_effector = self.robot.rarm_end_coords
        av_next = self.robot.inverse_kinematics(co, end_effector, link_list, stop=100)
        is_solved = isinstance(av_next, np.ndarray)
        return is_solved

    def send_command(self) -> None:
        self.ri.angle_vector(self.robot.angle_vector(), time=0.5, time_scale=1.0)
        time.sleep(0.5)

    def get_ref2base(self) -> CoordinateTransform:
        ref_to_base: Optional[CoordinateTransform] = None
        while True:
            try:
                pos, quat = self.tf_listener.lookupTransform(
                    "base_footprint", "reference", rospy.Time(0)
                )
                pose = Pose(Point(*pos), Quaternion(*quat))
                ref_to_base = CoordinateTransform.from_ros_pose(pose)
                break
                time.sleep(0.1)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        assert ref_to_base is not None
        return ref_to_base

    def create_rollout_trajectory(
        self,
        tf_ef2ref_list: List[CoordinateTransform],
        tf_efdash2ef: CoordinateTransform,
        tf_ref2base: CoordinateTransform,
    ) -> List[Coordinates]:
        co_list = []
        for transform in tf_ef2ref_list:
            tf_ef2ref = transform
            tf_ef2base = chain_transform(tf_ef2ref, tf_ref2base)
            tf_efdash2base = chain_transform(tf_efdash2ef, tf_ef2base)
            co_list.append(tf_efdash2base.to_skrobot_coords())
        return co_list

    def rollout(self, param: np.ndarray, error: np.ndarray, dryrun: bool = True):
        self.descriptor._inverse(param)
        tf_ef2ref_list = self.descriptor._inverse(param)
        tf_ref2base = self.get_ref2base()

        assert len(error) == 2
        x_trans, y_rot = error
        co = Coordinates()
        co.translate([x_trans, 0, 0])
        co.rotate(y_rot, "y")
        tf_efdash2ref = CoordinateTransform.from_skrobot_coords(co)

        ef_coords_list = self.create_rollout_trajectory(tf_ef2ref_list, tf_efdash2ref, tf_ref2base)

        if dryrun:
            viewer = TrimeshSceneViewer()
            viewer.add(self.robot)
            for co in tqdm.tqdm(ef_coords_list):
                viewer.add(Axis.from_coords(co, axis_radius=0.005, axis_length=0.01))
            viewer.show()
            time.sleep(1)

            for co in tqdm.tqdm(ef_coords_list):
                assert self.solve_inverse_kinematics(co)
                viewer.redraw()
                time.sleep(0.3)
        else:
            for co in tqdm.tqdm(ef_coords_list):
                assert self.solve_inverse_kinematics(co)
                self.send_command()


if __name__ == "__main__":
    # debug run
    sampler = RolloutExecutor.init()
    param = sampler.descriptor.encoded[0]
    sampler.rollout(param, np.zeros(2), dryrun=False)
