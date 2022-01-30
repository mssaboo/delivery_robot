#!/usr/bin/env python3

"""
marker_pub.py
Edited by Keiko
Description: displays markers at each of the objects we have detected in the scene
"""

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose2D
from group7_project.msg import DetectedObject, DetectedObjectList

class MarkerPublisher:
    def __init__(self):
        self.state = None
        self.detected_objects = {} # place a marker on all objects that can potentially be picked up
        rospy.init_node('marker_node', anonymous=True)
        rospy.Subscriber('/detected_object', DetectedObject, self.update_detection_list)
        rospy.Subscriber('/cmd_nav', Pose2D, self.goal_cb)
        self.rate = rospy.Rate(1)
        self.x_g = None
        self.y_g = None
        self.vis_pub = rospy.Publisher('marker/', Marker, queue_size=10)


    def update_detection_list(self, data):
        # if not data.name in self.detected_objects.keys():
        #     rospy.loginfo("Marker Publisher: Object Detected: {}".format(data.name))
        # else:
        #     rospy.loginfo("Marker Publisher: Object estimate updated: {} x: {}, y: {}".format(data.name, data.x, data.y))
        self.detected_objects[data.name] = (data.name, data.x, data.y, 0.0)

    def goal_cb(self, data):
        self.x_g = data.x
        self.y_g = data.y

    def publisher(self):
        while not rospy.is_shutdown():
            # publish our goal position
            goal_marker = Marker()
            goal_marker.header.frame_id = "map"
            goal_marker.header.stamp = rospy.Time()
            goal_marker.id = 0
            goal_marker.type = 2 # sphere
            goal_marker.pose.position.x = self.x_g
            goal_marker.pose.position.y = self.y_g
            goal_marker.pose.position.z = 0.0

            goal_marker.pose.orientation.x = 0.0
            goal_marker.pose.orientation.y = 0.0
            goal_marker.pose.orientation.z = 0.0
            goal_marker.pose.orientation.w = 1.0

            # not too big
            goal_marker.scale.x = 1./10.
            goal_marker.scale.y = 1./10.
            goal_marker.scale.z = 1./10.

            goal_marker.color.a = 1.0 # Don't forget to set the alpha!
            goal_marker.color.r = 0.0
            goal_marker.color.g = 0.0
            goal_marker.color.b = 1.0

            # use a single topic for all markers
            if self.x_g is not None:
                 #+ obj_name, Marker, queue_size=10)
                self.vis_pub.publish(goal_marker)

            # loop through each object in the list
            obj_num = 1
            for key, detected_object in self.detected_objects.items():
                # unpack
                obj_name, obj_x, obj_y, obj_th = detected_object

                marker = Marker()

                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time()

                # IMPORTANT: If you're creating multiple markers,
                #            each need to have a separate marker ID.

                marker.id = obj_num

                marker.type = 2 # sphere

                marker.pose.position.x = obj_x
                marker.pose.position.y = obj_y
                marker.pose.position.z = 0.0

                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0

                # not too big
                marker.scale.x = 1./20.
                marker.scale.y = 1./20.
                marker.scale.z = 1./20.

                marker.color.a = 1.0 # Don't forget to set the alpha!
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0

                # use a single topic for all markers
                self.vis_pub.publish(marker)

                obj_num += 1

            # print('Published all markers!')
            self.rate.sleep()

if __name__ == '__main__':
    try:
        markerpub_class = MarkerPublisher()
        markerpub_class.publisher()
    except rospy.ROSInterruptException:
        pass
