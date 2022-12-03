import time
from typing import Optional

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Point, Pose, Quaternion
from mohou_ros_utils.utils import CoordinateTransform
from skrobot.coordinates import Coordinates
from skrobot.models.pr2 import PR2


def solve_inverse_kinematics(robot: PR2, co: Coordinates) -> bool:
    control_joint_names = [
        "r_shoulder_pan_joint",
        "r_shoulder_lift_joint",
        "r_upper_arm_roll_joint",
        "r_elbow_flex_joint",
        "r_forearm_roll_joint",
        "r_wrist_flex_joint",
        "r_wrist_roll_joint",
    ]
    joints = [robot.__dict__[jname] for jname in control_joint_names]
    link_list = [joint.child_link for joint in joints]
    end_effector = robot.rarm_end_coords
    av_next = robot.inverse_kinematics(co, end_effector, link_list, stop=100)
    is_solved = isinstance(av_next, np.ndarray)
    return is_solved


def get_ref2base(listener: tf.TransformListener) -> CoordinateTransform:
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
