#!/usr/bin/env python3
import time
from typing import List, Optional

import numpy as np
import rospy
import tf
import tqdm
from geometry_msgs.msg import Point, Pose, Quaternion
from mohou_ros_utils.utils import CoordinateTransform, chain_transform
from skrobot.interfaces.ros import PR2ROSRobotInterface  # type: ignore
from skrobot.models.pr2 import PR2

from plug_insert.common import History


def solve_inverse_kinematics(
    robot: PR2, control_joint_names: List[str], tf_gripper2base_target: CoordinateTransform
) -> bool:
    joints = [robot.__dict__[jname] for jname in control_joint_names]
    link_list = [joint.child_link for joint in joints]
    end_effector = robot.rarm_end_coords
    av_next = robot.inverse_kinematics(
        tf_gripper2base_target.to_skrobot_coords(), end_effector, link_list, stop=5
    )
    is_solved = isinstance(av_next, np.ndarray)
    return is_solved


def get_ref2base(reference_name: str) -> CoordinateTransform:
    ref_to_base: Optional[CoordinateTransform] = None
    while True:
        try:
            pos, quat = listener.lookupTransform("/base_link", reference_name, rospy.Time(0))
            pose = Pose(Point(*pos), Quaternion(*quat))
            ref_to_base = CoordinateTransform.from_ros_pose(pose)
            break
            time.sleep(0.1)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    assert ref_to_base is not None
    return ref_to_base


rospy.init_node("reproducer")

listener = tf.TransformListener()
histories = History.load_all()
history = histories[0]

tf_list = []
tf_ref2base = get_ref2base(history.reference_name)
for tf_rarm2ref in tqdm.tqdm(history.tf_rarm2ref_list):
    tf_rarm2base = chain_transform(tf_rarm2ref, tf_ref2base)
    tf_list.append(tf_rarm2base)

joint_names = [
    "r_shoulder_pan_joint",
    "r_shoulder_lift_joint",
    "r_upper_arm_roll_joint",
    "r_elbow_flex_joint",
    "r_forearm_roll_joint",
    "r_wrist_flex_joint",
    "r_wrist_roll_joint",
]


robot = PR2()
ri = PR2ROSRobotInterface(robot)
robot.angle_vector(ri.angle_vector())

for transform in tf_list:
    print(transform)
    solve_inverse_kinematics(robot, joint_names, transform)
    ri.angle_vector(robot.angle_vector(), time_scale=1.0, time=0.5)
    time.sleep(0.3)
