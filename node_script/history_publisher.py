#!/usr/bin/env python3
import pickle

import rospy
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import Path as PathMsg
from skrobot.coordinates.math import matrix2quaternion, wxyz2xyzw

from plug_insert.common import History, project_path


def create_pathmsg(history: History) -> PathMsg:

    poses = []
    for co in history.rarm_coords_history:
        pos = co.worldpos()
        quat = wxyz2xyzw(matrix2quaternion(co.worldrot()))

        pos_msg = Point(*pos)
        quat_msg = Quaternion(*quat)
        pose = Pose(position=pos_msg, orientation=quat_msg)
        pose_stamped = PoseStamped(pose=pose)
        pose_stamped.header.frame_id = "base_link"
        poses.append(pose_stamped)

    msg = PathMsg(poses=poses)
    msg.header.frame_id = "base_link"
    return msg

histories = History.load_all()
msg_list = [create_pathmsg(h) for h in histories]

pub = rospy.Publisher("history", PathMsg, queue_size=10)
rospy.init_node("history_publisher", anonymous=True)
rate = rospy.Rate(5)
while not rospy.is_shutdown():
    for msg in msg_list:
        pub.publish(msg)
    rospy.loginfo("published")
    rate.sleep()
