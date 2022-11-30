#!/usr/bin/env python3
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from mohou_ros_utils.utils import CoordinateTransform

if __name__ == "__main__":
    rospy.init_node("publisher")
    pub = rospy.Publisher("relative_pose", PoseStamped, queue_size=10)

    listener = tf.TransformListener()

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            transform = listener.lookupTransform("reference", "r_gripper_tool_frame", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        co = CoordinateTransform.from_ros_transform(transform)
        pose = PoseStamped(pose=co.to_ros_pose())
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "reference"
        pub.publish(pose)
