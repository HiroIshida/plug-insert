#!/usr/bin/env python3
import argparse
import time
from typing import List

import rospy
import tf
from mohou_ros_utils.utils import chain_transform
from skrobot.coordinates import Coordinates
from skrobot.interfaces.ros import PR2ROSRobotInterface  # type: ignore
from skrobot.model import Axis
from skrobot.models.pr2 import PR2
from skrobot.viewers import TrimeshSceneViewer

from plug_insert.common import History
from plug_insert.descriptor import Descriptor
from plug_insert.robot import get_ref2base, solve_inverse_kinematics

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
    # transform_list = desc.reproduce(2)

    co_list: List[Coordinates] = []
    tf_ref2base = get_ref2base(listener)
    for transform in transform_list:
        tf_ef2ref = transform
        tf_ef2base = chain_transform(tf_ef2ref, tf_ref2base)
        co_list.append(tf_ef2base.to_skrobot_coords())

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
            assert solve_inverse_kinematics(robot, co)
            viewer.redraw()
            time.sleep(0.3)
    else:
        for co in co_list:
            print(co)
            assert solve_inverse_kinematics(robot, co)
            ri.angle_vector(robot.angle_vector(), time=0.5, time_scale=1.0)
            time.sleep(0.5)
