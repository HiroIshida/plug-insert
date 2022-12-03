#!/usr/bin/env python3
import argparse
import time
from typing import List, Optional

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Point, Pose, Quaternion
from mohou_ros_utils.utils import CoordinateTransform, chain_transform
from skrobot.coordinates import Coordinates
from skrobot.interfaces.ros import PR2ROSRobotInterface  # type: ignore
from skrobot.model import Axis
from skrobot.models.pr2 import PR2
from skrobot.viewers import TrimeshSceneViewer

from plug_insert.common import History
from plug_insert.descriptor import Descriptor


def solve_inverse_kinematics(robot: PR2, control_joint_names: List[str], co: Coordinates) -> bool:
    joints = [robot.__dict__[jname] for jname in control_joint_names]
    link_list = [joint.child_link for joint in joints]
    end_effector = robot.rarm_end_coords
    av_next = robot.inverse_kinematics(co, end_effector, link_list, stop=100)
    is_solved = isinstance(av_next, np.ndarray)
    return is_solved


def get_ref2base() -> CoordinateTransform:
    ref_to_base: Optional[CoordinateTransform] = None
    while True:
        try:
            pos, quat = listener.lookupTransform("base_footprint", "reference", rospy.Time(0))
            pose = Pose(Point(*pos), Quaternion(*quat))
            ref_to_base = CoordinateTransform.from_ros_pose(pose)
            break
            time.sleep(0.1)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    assert ref_to_base is not None
    return ref_to_base


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryrun", action="store_true", help="dryrun")
    args = parser.parse_args()
    dryrun: bool = args.dryrun

    rospy.init_node("reproducer")

    listener = tf.TransformListener()
    histories = History.load_all()
    # history = histories[3]

    desc = Descriptor.from_histories(histories)
    transform_list = desc.sample()

    co_list: List[Coordinates] = []
    tf_ref2base = get_ref2base()
    for transform in transform_list:
        tf_ef2ref = transform
        tf_ef2base = chain_transform(tf_ef2ref, tf_ref2base)
        co_list.append(tf_ef2base.to_skrobot_coords())

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

    if dryrun:
        viewer = TrimeshSceneViewer()
        viewer.add(robot)
        for co in co_list:
            viewer.add(Axis.from_coords(co, axis_radius=0.005, axis_length=0.01))
        viewer.show()
        time.sleep(2)

        for co in co_list:
            assert solve_inverse_kinematics(robot, joint_names, co)
            viewer.redraw()
            time.sleep(0.3)
    else:
        for co in co_list:
            print(co)
            assert solve_inverse_kinematics(robot, joint_names, co)
            ri.angle_vector(robot.angle_vector(), time=0.5, time_scale=1.0)
            time.sleep(0.5)
