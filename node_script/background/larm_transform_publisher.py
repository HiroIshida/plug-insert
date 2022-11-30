#!/usr/bin/env python3
import rospy
import tf
from mohou_ros_utils.utils import CoordinateTransform


def broadcaster(frame_name: str):
    listener = tf.TransformListener()
    broadcaster = tf.TransformBroadcaster()

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            (trans, rot) = listener.lookupTransform(frame_name, "base_link", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        transform_inverted = CoordinateTransform.from_ros_transform((trans, rot)).inverse()
        transform = transform_inverted.to_ros_transform()
        t = rospy.Time.now()
        broadcaster.sendTransform(transform[0], transform[1], t, "reference", "base_link")
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node("broadcaster")
    broadcaster("l_gripper_tool_frame")
