#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMsg

from plug_insert.common import History


def create_pathmsg(history: History) -> PathMsg:

    poses = []
    for tf in history.tf_rarm2ref_list:
        pose_stamped = PoseStamped(pose=tf.to_ros_pose())
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
