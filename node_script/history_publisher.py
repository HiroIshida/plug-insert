#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import Path as PathMsg
from skrobot.coordinates.math import wxyz2xyzw

from plug_insert.common import History


def create_pathmsg(history: History) -> PathMsg:

    poses = []
    for vec in history.endeffector_traj:
        pos, quat_wxyz = vec[:3], vec[3:]
        quat = wxyz2xyzw(quat_wxyz)

        pos_msg = Point(*pos)
        quat_msg = Quaternion(*quat)
        pose = Pose(position=pos_msg, orientation=quat_msg)
        pose_stamped = PoseStamped(pose=pose)
        pose_stamped.header.frame_id = history.reference_name
        poses.append(pose_stamped)

    msg = PathMsg(poses=poses)
    msg.header.frame_id = history.reference_name
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
