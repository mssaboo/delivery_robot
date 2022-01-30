#!/usr/bin/env python3

import rospy
import os
# watch out on the order for the next two imports lol
from tf import TransformListener, transformations
try:
    # import tensorflow as tf
    # updated to disable v2 behavior as per https://edstem.org/us/courses/14340/discussion/892146
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    pass
import numpy as np
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from geometry_msgs.msg import Pose2D
from group7_project.msg import DetectedObject, DetectedObjectList
from cv_bridge import CvBridge, CvBridgeError
import cv2
import math
import statistics
from std_msgs.msg import Int8, Bool, Float64

"""
img2global.py
Edited By: Keiko
Description: Converts image frame (x,y,z) normal vector with distance to a global X,Y,Z coordinate
Last Modified: 120521
"""
import numpy as np

def img2global(img_obj_theta, img_obj_dist, global_camera_pose):
    x_norm = np.cos(img_obj_theta)
    y_norm = np.sin(img_obj_theta)
    #z_norm = 1.0 - np.sqrt(x_norm**2 + y_norm**2) # should be close to 0
    #img_obj_normvec = np.array([x_norm, y_norm, z_norm])
    img_obj_position = img_obj_dist * np.array([x_norm, y_norm])
    x_cam, y_cam, th_cam = global_camera_pose

    R = np.array([[np.cos(th_cam), -np.sin(th_cam)],[np.sin(th_cam), np.cos(th_cam)]]) # might be a transpose

    global_obj_position = np.array([x_cam,y_cam]) + np.dot(R, img_obj_position)

    return global_obj_position

#Added by Mahesh
from std_msgs.msg import UInt32MultiArray

def load_object_labels(filename):
    """ loads the coco object readable name """

    fo = open(filename,'r')
    lines = fo.readlines()
    fo.close()
    object_labels = {}
    for l in lines:
        object_id = int(l.split(':')[0])
        label = l.split(':')[1][1:].replace('\n','').replace('-','_').replace(' ','_')
        object_labels[object_id] = label

    return object_labels

class DetectorParams:

    def __init__(self, verbose=False):
    
        # Set to True to use tensorflow and a conv net.
        # False will use a very simple color thresholding to detect stop signs only.
        #self.use_tf = rospy.get_param("use_tf")
        self.use_tf = True # changing to hard coded

        # Path to the trained conv net
        model_path = rospy.get_param("~model_path", "../../tfmodels/ssd_resnet_50_fpn_coco.pb")
        label_path = rospy.get_param("~label_path", "../../tfmodels/coco_labels.txt")
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_path)
        self.label_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), label_path)

        # Minimum score for positive detection
        self.min_score = rospy.get_param("~min_score", 0.5)

        if verbose:
            print("DetectorParams:")
            print("    use_tf = {}".format(self.use_tf))
            print("    model_path = {}".format(model_path))
            print("    label_path = {}".format(label_path))
            print("    min_score = {}".format(self.min_score))

class ObjectEstimate:
    def __init__(self, measurements_to_keep = 10):
        # self.measurements_to_keep = measurements_to_keep
        self.x_values = []
        self.y_values = []

    def add_observation(self, x, y):
        self.x_values.append(x)
        self.y_values.append(y)

        # if len(self.x_values) > self.measurements_to_keep * 2:
        #     extra_objects = len(self.x_values) - self.measurements_to_keep
        #     self.x_values.sort()
        #     self.x_values = self.x_values[extra_objects//2: -extra_objects//2]
        #     self.y_values.sort()
        #     self.y_values = self.y_values[extra_objects//2: -extra_objects//2]
    
    def get_estimate(self):
        x = statistics.mean(self.x_values)
        y = statistics.mean(self.y_values)
        return (x, y)

class Detector:

    def __init__(self):
        rospy.init_node('turtlebot_detector', anonymous=True)
        self.params = DetectorParams()
        self.bridge = CvBridge()

        if self.params.use_tf:
            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.params.model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def,name='')
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
            self.sess = tf.Session(graph=self.detection_graph)

        # camera and laser parameters that get updated
        self.cx = 0.
        self.cy = 0.
        self.fx = 1.
        self.fy = 1.
        self.laser_ranges = []
        self.laser_angle_increment = 0.01 # this gets updated

        self.object_publishers = {}
        self.object_labels = load_object_labels(self.params.label_path)

        self.tf_listener = TransformListener()
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback, queue_size=1)
        rospy.Subscriber('/camera/camera_info', CameraInfo, self.camera_info_callback)
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        rospy.Subscriber("/waiting_at_waypoint", Bool, self.waiting_at_waypoint_callback)

        #Added by Mahesh
        self.objects_toBeDetected = ["fire_hydrant","apple","pizza","traffic_light","cake"]
        self.object_publisher = rospy.Publisher('/detected_object', DetectedObject, queue_size=10)
        self.stop_sign_publisher = rospy.Publisher("/stop_sign", Float64, queue_size=10 )
        self.object_estimates = {key: ObjectEstimate() for key in self.objects_toBeDetected}
        self.msgs = {key: [] for key in self.objects_toBeDetected}

        #Added by Keiko
        self.trans_listener = TransformListener()
        self.x = 0
        self.y = 0
        self.theta = 0
        self.is_publishing = False

    def waiting_at_waypoint_callback(self, data):
        self.is_publishing = data.data

    def run_detection(self, img):
        """ runs a detection method in a given image """

        image_np = self.load_image_into_numpy_array(img)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        if self.params.use_tf:
            # uses MobileNet to detect objects in images
            # this works well in the real world, but requires
            # good computational resources
            with self.detection_graph.as_default():
                (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes,self.d_scores,self.d_classes,self.num_d],
                feed_dict={self.image_tensor: image_np_expanded})

            #return self.filter(boxes[0], scores[0], classes[0], num[0])
            # changing to int to match args to self.filter
            return self.filter(boxes[0], scores[0], classes[0], int(num[0]))

        else:
            # uses a simple color threshold to detect stop signs
            # this will not work in the real world, but works well in Gazebo
            # with only stop signs in the environment
            R = image_np[:,:,0].astype(np.int) > image_np[:,:,1].astype(np.int) + image_np[:,:,2].astype(np.int)
            Ry, Rx, = np.where(R)
            if len(Ry)>0 and len(Rx)>0:
                xmin, xmax = Rx.min(), Rx.max()
                ymin, ymax = Ry.min(), Ry.max()
                boxes = [[float(ymin)/image_np.shape[1], float(xmin)/image_np.shape[0], float(ymax)/image_np.shape[1], float(xmax)/image_np.shape[0]]]
                scores = [.99]
                classes = [13]
                num = 1
            else:
                boxes = []
                scores = 0
                classes = 0
                num = 0

            return boxes, scores, classes, num

    def filter(self, boxes, scores, classes, num):
        """ removes any detected object below MIN_SCORE confidence """

        f_scores, f_boxes, f_classes = [], [], []
        f_num = 0

        for i in range(num):
            if scores[i] >= self.params.min_score:
                f_scores.append(scores[i])
                f_boxes.append(boxes[i])
                f_classes.append(int(classes[i]))
                f_num += 1
            else:
                break

        return f_boxes, f_scores, f_classes, f_num

    def load_image_into_numpy_array(self, img):
        """ converts opencv image into a numpy array """

        (im_height, im_width, im_chan) = img.shape

        return np.array(img.data).reshape((im_height, im_width, 3)).astype(np.uint8)

    def project_pixel_to_ray(self, u, v):
        """ takes in a pixel coordinate (u,v) and returns a tuple (x,y,z)
        that is a unit vector in the direction of the pixel, in the camera frame """

        ########## Code starts here ##########
        # TODO: Compute x, y, z.
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        z = 1.
        rsos = np.sqrt(x**2 + y**2  + z**2)
        x /= rsos
        y /= rsos
        z /= rsos
        ########## Code ends here ##########

        return x, y, z

    def estimate_distance(self, thetaleft, thetaright, ranges):
        """ estimates the distance of an object in between two angles
        using lidar measurements """

        leftray_indx = min(max(0,int(thetaleft/self.laser_angle_increment)),len(ranges))
        rightray_indx = min(max(0,int(thetaright/self.laser_angle_increment)),len(ranges))

        if leftray_indx<rightray_indx:
            meas = ranges[rightray_indx:] + ranges[:leftray_indx]
        else:
            meas = ranges[rightray_indx:leftray_indx]

        num_m, dist = 0, 0
        for m in meas:
            if m>0 and m<float('Inf'):
                dist += m
                num_m += 1
        if num_m>0:
            dist /= num_m

        return dist

    def camera_callback(self, msg):
        """ callback for camera images """

        # save the corresponding laser scan
        img_laser_ranges = list(self.laser_ranges)

        try:
            img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            img_bgr8 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        (img_h,img_w,img_c) = img.shape

        # runs object detection in the image
        (boxes, scores, classes, num) = self.run_detection(img)

        if num > 0:
            # some objects were detected
            for (box,sc,cl) in zip(boxes, scores, classes):
                ymin = int(box[0]*img_h)
                xmin = int(box[1]*img_w)
                ymax = int(box[2]*img_h)
                xmax = int(box[3]*img_w)
                xcen = int(0.5*(xmax-xmin)+xmin)
                ycen = int(0.5*(ymax-ymin)+ymin)

                cv2.rectangle(img_bgr8, (xmin,ymin), (xmax,ymax), (255,0,0), 2)

                # computes the vectors in camera frame corresponding to each sides of the box
                rayleft = self.project_pixel_to_ray(xmin,ycen)
                rayright = self.project_pixel_to_ray(xmax,ycen)

                # convert the rays to angles (with 0 poiting forward for the robot)
                thetaleft = math.atan2(-rayleft[0],rayleft[2])
                thetaright = math.atan2(-rayright[0],rayright[2])
                if thetaleft<0:
                    thetaleft += 2.*math.pi
                if thetaright<0:
                    thetaright += 2.*math.pi

                # estimate the corresponding distance using the lidar
                dist = self.estimate_distance(thetaleft,thetaright,img_laser_ranges)
                object_size = abs( (ymax - ymin) * (xmax - xmin)) / (img_h * img_w)
                close_enough_to_object = (dist < 0.75)
                print("detector: Detected Object {}, dist: {}, size: {}".format(self.object_labels[cl], dist, object_size))
                if self.object_labels[cl] in self.objects_toBeDetected and close_enough_to_object and self.is_publishing:
                    label = self.object_labels[cl]
                    theta_avg = (thetaleft + thetaright)/2.0

                    # print("turtlebot x: {}, y: {}, theta: {}".format(self.x, self.y, self.theta))
                    global_obj_position = img2global(theta_avg, dist, np.array([self.x, self.y, self.theta]))
                    x, y = global_obj_position
                    self.object_estimates[label].add_observation(x, y)
                    # print("{} r: {}, theta: {}".format(label, dist, theta_avg))

                    # print("{} x: {}, y: {}".format(label, global_obj_position[0], global_obj_position[1]))

                    # publishes the detected object and its location
                    object_msg = DetectedObject()
                    object_msg.id = cl
                    object_msg.name = label
                    object_msg.confidence = sc
                    object_msg.distance = dist
                    object_msg.thetaleft = thetaleft
                    object_msg.thetaright = thetaright
                    object_msg.corners = [ymin,xmin,ymax,xmax]
                    object_msg.theta = self.theta
                    object_msg.x, object_msg.y = self.object_estimates[label].get_estimate()
                    # object_msg.x, object_msg.y = x, y
                    object_msg.robot_x = self.x
                    object_msg.robot_y = self.y
                    object_msg.robot_theta = self.theta

                    self.object_publisher.publish(object_msg)
                elif self.object_labels[cl] == "stop_sign":
                    distance = Float64()
                    distance.data = dist
                    self.stop_sign_publisher.publish(distance)

                    

        # displays the camera image
        cv2.imshow("Camera", img_bgr8)
        cv2.waitKey(1)

    def camera_info_callback(self, msg):
        """ extracts relevant camera intrinsic parameters from the camera_info message.
        cx, cy are the center of the image in pixel (the principal point), fx and fy are
        the focal lengths. """

        ########## Code starts here ##########
        # TODO: Extract camera intrinsic parameters.
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        ########## Code ends here ##########

    def laser_callback(self, msg):
        """ callback for thr laser rangefinder """

        self.laser_ranges = msg.ranges
        self.laser_angle_increment = msg.angle_increment

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # try to get state information to update self.x, self.y, self.theta
            try:
                (translation, rotation) = self.trans_listener.lookupTransform(
                    "/map", "/base_footprint", rospy.Time(0)
                )
                self.x = translation[0]
                self.y = translation[1]
                euler = transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
            except Exception as e:
                print(e)
                pass
            rate.sleep()

if __name__=='__main__':
    d = Detector()
    d.run()
