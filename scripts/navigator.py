#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String
import tf
import numpy as np
from numpy import linalg
from utils.utils import wrapToPi
from utils.grids import StochOccupancyGrid2D
from utils.img2global import img2global
from planners import AStar, compute_smoothed_traj
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum

from dynamic_reconfigure.server import Server
from group7_project.cfg import NavigatorConfig

from planners.utils import travellingSalesmanProblem

# imports added by Mahesh
from group7_project.msg import DetectedObject, DetectedObjectList
from std_msgs.msg import Int8, Bool, Float64
from std_msgs.msg import UInt32MultiArray
# state machine modes, not all implemented


class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3
    STOP = 4


# States added by Mahesh for State Machine
class States(Enum):
    REST = 0
    EXPLORE = 1
    INPUT = 2
    PLAN = 3
    SELECT_OBJ = 4
    GOTO_OBJ = 5
    PICKUP_OBJ = 6
    GOTO_START = 7
    MISSION_COMPLETE = 8
    PURE_MAPPING = 9
    EXPLORE_WAYPOINTS = 10

def get_waypoints():
    initial_position = (3.1499805426046885, 1.600005394413161, 0.0003137981697178387)
    fire_hydrant = (3.14, 2.24, 3.12)
    traffic_light = (3.41, 0.75, 3.13)
    left_w1 = (2.44, 0.20, 3.05)
    apple = (0.36, 0.26, 3.01)
    top_w1 = (0.34, 1.68, 1.89)
    pizza = (0.32, 2.03, -3.00)
    top_w2 = (0.45, 1.54, -0.04)
    middle_w1 = (1.57, 1.55, -0.04)
    cake = (2.34, 1.89, 2.69)
    right_w1 = (2.37, 2.79, 3.05)
    right_w2 = (1.08, 2.83, 3.10)
    right_w3 = (3.19, 2.79, -0.65)
    waypoints = [initial_position, fire_hydrant, traffic_light, left_w1, apple, top_w1, middle_w1, cake, right_w1, right_w2, pizza, right_w3, initial_position]
    # # initial_position = (3.1499805426046885, 1.600005394413161, 0.0003137981697178387)
    # # fire_hydrant = (2.78 , 1.9994757121354816, 2.3342335559863283)
    # traffic_light = (3.2, 1.2299933541477714, -1.6613942616449668)
    # checkpoint_bottom_left = (3.2711863767907987, 0.5, -1.2587332091149288)
    # checkpoint_left_middle = (2.6469172700134944, 0.3452726849726044, -3.1048644729556547)
    # apple = (0.3, 0.2727360021975153, 3.056979322594769)
    # checkpoint_top_middle = (0.22346204232971817, 1.0383107089746466, 1.584950108979645)
    # pizza = (0.22780888250101305, 1.9753292448097841, 3.056979322594769)
    # checkpoint_right_middle = (0.21440811836747617, 2.146921920769351, 1.5418321415461642)
    # cake = (2.1887558834226315, 1.8367754830341891, 2.25)

    # waypoints = [initial_position, fire_hydrant, traffic_light, checkpoint_bottom_left, checkpoint_left_middle, apple, checkpoint_top_middle, pizza, checkpoint_right_middle, cake, initial_position]
    # waypoints = [initial_position, checkpoint_bottom_left, apple, checkpoint_top_middle, checkpoint_right_middle, cake, initial_position]
    # # waypoints = [initial_position, apple, pizza, cake, initial_position]
    return waypoints

class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """

    def __init__(self):
        rospy.init_node("turtlebot_navigator", anonymous=True)
        self.mode = Mode.IDLE

        # Added by Mahesh
        # init to REST State
        self.state = States.REST
        # objects_detected is list of objects detected in form of arrays ["Name",x,y,theta]
        # self.objects_detected = []
        self.detected_objects = {}
        self.waypoints = get_waypoints()
        # self.waypoints = []
        # pickup_list is list of objects that TA will select to pickup
        self.pickup_list = []

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None


        # goal state
        self.x_init = None
        self.y_init = None
        self.theta_init = None

        self.th_init = 0.0

        ########### added by Keiko ############
        self.stop_min_dist = 0.8
        self.stop_time = 3.0
        #######################################

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0, 0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False

        # plan parameters
        self.plan_resolution = 0.1
        self.plan_horizon = 15

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.0, 0.0]

        # Robot limits
        self.v_max = 0.2  # maximum velocity
        self.om_max = 0.4  # maximum angular velocity

        self.v_des = 0.12  # desired cruising velocity
        # threshold in theta to start moving forward when path-following
        self.theta_start_thresh = 0.05
        self.start_pos_thresh = (
            0.2  # threshold to be far enough into the plan to recompute it
        )

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.2
        self.at_thresh = 0.05
        self.at_thresh_theta = 0.25 #0.05 increasing because we don't care about matching theta

        # trajectory smoothing
        self.spline_alpha = 0.01
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 5.0
        self.kpy = 5.0
        self.kdx = 1.5
        self.kdy = 1.5

        self.wait_end_time = 0
        self.waiting_at_waypoint = True
        self.last_stop_released_at = 0

        # heading controller parameters
        self.kp_th = 0.5

        self.traj_controller = TrajectoryTracker(
            self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max
        )
        self.pose_controller = PoseController(
            0.1, 0.1, 0.1, self.v_max, self.om_max
        )
        self.heading_controller = HeadingController(self.kp_th, self.om_max)

        self.nav_planned_path_pub = rospy.Publisher(
            "/planned_path", Path, queue_size=10
        )
        self.nav_smoothed_path_pub = rospy.Publisher(
            "/cmd_smoothed_path", Path, queue_size=10
        )
        self.nav_smoothed_path_rej_pub = rospy.Publisher(
            "/cmd_smoothed_path_rejected", Path, queue_size=10
        )
        self.waiting_at_waypoint_pub = rospy.Publisher(
            "/waiting_at_waypoint", Bool, queue_size = 10
        )

        self.nav_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.trans_listener = tf.TransformListener()

        self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/map_metadata", MapMetaData, self.map_md_callback)
        rospy.Subscriber("/cmd_nav", Pose2D, self.cmd_nav_callback)
        rospy.Subscriber("/teleop_cmd_vel", Twist, self.teleop_vel_callback)
        # Added by Mahesh
        rospy.Subscriber('/send_state', Int8, self.state_callback)
        self.cmd_nav_pub = rospy.Publisher("/cmd_nav", Pose2D, queue_size=10)
        rospy.Subscriber("/stop_sign", Float64, self.stop_sign_detected_callback )


        self.subscribers = {}
        #"fire_hydrant","apple","pizza","traffic_light","cake"
        self.objects_map = {
            1:"fire_hydrant",
            2:"apple",
            3:"pizza",
            4:"traffic_light",
            5:"cake",
        }
        rospy.Subscriber('/detected_object', DetectedObject, self.update_detection_list)
        print("finished init")

    # Added by Mahesh

    # def createSubscriber(self,data):
    #     #array of names
    #     #if not subscribed - subscribe
    #     #update that particular id object in map
    #     names_array = data.objects
    #     for i in range(len(names_array)):
    #         if (i in self.subscribers.keys()):
    #             pass
    #         else:
    #             self.subscribers[i] = names_array[i]
    #             rospy.Subscriber('/detector/'+names_array[i],DetectedObject,self.update_detection_list)

    def update_detection_list(self, data):
        if self.state == States.EXPLORE or self.state == States.EXPLORE_WAYPOINTS:
            if not data.name in self.detected_objects.keys():
                rospy.loginfo("Navigator: Object Detected: {}".format(data.name))
            # else:
            #     rospy.loginfo("Navigator: Object estimate updated: {} x: {}, y: {}".format(data.name, data.x, data.y))
            #     rospy.loginfo("Navigator: Robot Position to user: {} x: {}, y: {}, theta: {}".format(data.name, data.robot_x, data.robot_y, data.robot_theta))
            self.detected_objects[data.name] = (data.name, data.robot_x, data.robot_y, data.robot_theta)

    def teleop_vel_callback(self, data):
        if self.state == States.EXPLORE:
            self.teleop_velocity_to_publish = data

    def state_callback(self, data):
        # key 'f'
        if(data.data == 10):
            # self.detected_objects = {}
            self.switch_state(States.EXPLORE_WAYPOINTS)
        # key 'r'
        elif (data.data == 13):
            self.switch_state(States.PURE_MAPPING)
        # key 'g'
        elif(data.data == 11):
            self.switch_state(States.INPUT)
        # key 'h'
        elif(data.data == 12):
            self.switch_state(States.PLAN)
        # key 0-9 is used to select objects
        elif(data.data <= 9):
            # Add object with index data.data to
            # check if size of object_detected is greater or equal to data.data for safety
            object_name = self.objects_map[data.data]
            self.pickup_list.append(object_name)

    def dyn_cfg_callback(self, config, level):
        rospy.loginfo(
            "Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}".format(**config)
        )
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]
        return config

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        if (
            data.x != self.x_g
            or data.y != self.y_g
            or data.theta != self.theta_g
        ):
            self.x_g = data.x
            self.y_g = data.y
            self.theta_g = data.theta
            rospy.loginfo("cmd_nav callback")
            rospy.loginfo("cmd_nav callback: ({}, {}, {})".format(self.x_g, self.y_g, self.theta_g))
            self.replan()

    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x, msg.origin.position.y)

    def map_callback(self, msg):
        """
        receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if (
            self.map_width > 0
            and self.map_height > 0
            and len(self.map_probs) > 0
        ):
            self.occupancy = StochOccupancyGrid2D(
                self.map_resolution,
                self.map_width,
                self.map_height,
                self.map_origin[0],
                self.map_origin[1],
                5,  # CHANGED from 8 to 4
                self.map_probs,
            )
            # self.occupancy.plot()
            if self.x_g is not None and self.mode != Mode.STOP:
                # if we have a goal to plan to, replan
                rospy.loginfo("replanning because of new map")
                self.replan()  # new map, need to replan

    ############ stop sign functions ############
    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """
    
        # distance of the stop sign
        dist = msg.data
        rospy.loginfo("Stop sign detected at distance {}".format(dist))
        # if close enough and in nav mode, stop
        if dist > 0 and dist < self.stop_min_dist and ((rospy.get_rostime().to_sec() - self.last_stop_released_at) > 10) and self.mode == Mode.TRACK:
            rospy.loginfo("Stop entering at : {}".format(rospy.get_rostime().to_sec()))
            self.init_stop_sign()
    
    def init_stop_sign(self):
        """ initiates a stop sign maneuver """
        self.stop_sign_start = rospy.get_rostime()
        self.switch_mode(Mode.STOP)
    
    def has_stopped(self):
        """ checks if stop sign maneuver is over """
    
        return self.mode == Mode.STOP and \
               rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.stop_time)
    ################################################

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        return (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.near_thresh
        )

    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        return (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.at_thresh
            and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta
        )

    def aligned(self):
        """
        returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """
        return (
            abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh
        )

    def close_to_plan_start(self):
        return (
            abs(self.x - self.plan_start[0]) < self.start_pos_thresh
            and abs(self.y - self.plan_start[1]) < self.start_pos_thresh
        )

    def snap_to_grid(self, x):
        return (
            self.plan_resolution * round(x[0] / self.plan_resolution),
            self.plan_resolution * round(x[1] / self.plan_resolution),
        )

    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode

    def switch_state(self, new_state):
        rospy.loginfo("Switching from State %s -> %s", self.state, new_state)
        self.state = new_state

    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i, 0]
            pose_st.pose.position.y = traj[i, 1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        t = self.get_current_plan_time()
        # if self.state == States.EXPLORE:
        #     try:
        #         V = self.teleop_velocity_to_publish.linear.x
        #         om = self.teleop_velocity_to_publish.angular.z
        #     except:
        #         rospy.loginfo("No teleop received yet")
        #         V, om = 0.0, 0.0
        # else:
        if self.mode == Mode.PARK:
            V, om = self.pose_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.TRACK:
            V, om = self.traj_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.ALIGN:
            V, om = self.heading_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        else:
            V = 0.0
            om = 0.0

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):
        t = (rospy.get_rostime() - self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    def get_path_length(self, p1, p2):
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x1, y1 = p1
        x2, y2 = p2
        x_init = self.snap_to_grid((x1, y1))
        x_goal = self.snap_to_grid((x2, y2))
        problem = AStar(
            state_min,
            state_max,
            x_init,
            x_goal,
            self.occupancy,
            self.plan_resolution,
        )
        rospy.loginfo("ASTAR object created, starting solving")
        success = problem.solve()
        rospy.loginfo("ASTAR solved")
        if not success:
            rospy.loginfo(
                "Navigator: solve_tsp: Path length calculation failed")
            # needs to be revisited
            return linalg.norm(np.array([x1 - x2, y1 -y2]))

        planned_path_length = len(problem.path)
        return planned_path_length

    def solve_tsp(self):
        """
        formulates and solves the Travelling Salesman problem
        so as to recover objects in minimum possible time
        """
        rospy.loginfo(
            "Navigator: Starting Solving Travelling Salesman Problem"
        )
        rospy.loginfo(
            "Navigator: solve_tsp: Pickup List: {}".format(self.pickup_list))
        graph_points = [(self.x, self.y)]
        object_numbers = [-1]
        for object_ix in self.pickup_list:
            object_data = self.detected_objects[object_ix]
            x, y = object_data[1], object_data[2]
            graph_points.append((x, y))
            object_numbers.append(object_ix)

        graph = []
        for i, p1 in enumerate(graph_points):
            distances_from_i = []
            for j, p2 in enumerate(graph_points):
                distance = self.get_path_length(p1, p2) if i != j else 0
                rospy.loginfo(
                    "Navigator: solve_tsp: Path Length Object {} to Object {} : {}".format(i, j, distance))
                distances_from_i.append(distance)
            graph.append(distances_from_i)
        rospy.loginfo("Navigator: solve_tsp: TSP Graph: {}".format(graph))
        min_cost_ordering = travellingSalesmanProblem(graph)
        self.pickup_list = [object_numbers[i] for i in min_cost_ordering]
        rospy.loginfo(
            "Navigator: solve_tsp: Pickup List: {}".format(self.pickup_list))

    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """
        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo(
                "Navigator: replanning canceled, waiting for occupancy map."
            )
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        problem = AStar(
            state_min,
            state_max,
            x_init,
            x_goal,
            self.occupancy,
            self.plan_resolution,
        )

        rospy.loginfo("Navigator: computing navigation plan")
        success = problem.solve()
        if not success:
            rospy.loginfo("Planning failed")
            return
        rospy.loginfo("Planning Succeeded")

        planned_path = problem.path

        # Check whether path is too short
        if len(planned_path) < 4:
            rospy.loginfo("Path too short to track")
            self.switch_mode(Mode.PARK)
            return

        # Smooth and generate a trajectory
        traj_new, t_new = compute_smoothed_traj(
            planned_path, self.v_des, self.spline_alpha, self.traj_dt
        )

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
            t_remaining_curr = (
                self.current_plan_duration - self.get_current_plan_time()
            )

            # Estimate duration of new trajectory
            th_init_new = traj_new[0, 2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err / self.om_max)
            t_remaining_new = t_init_align + t_new[-1]

            if t_remaining_new > t_remaining_curr:
                rospy.loginfo(
                    "New plan rejected (longer duration than current plan)"
                )
                self.publish_smoothed_path(
                    traj_new, self.nav_smoothed_path_rej_pub
                )
                return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0, 2]
        self.heading_controller.load_goal(self.th_init)

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return

        rospy.loginfo("Ready to track")
        self.switch_mode(Mode.TRACK)

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
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
                if self.x_init is None:
                    self.x_init = self.x
                    self.y_init = self.y
                    self.theta_init = self.theta

            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ) as e:
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                self.switch_mode(Mode.IDLE)
                print(e)
                pass

            # rospy.loginfo("Turtlebot Current Pose x:{}, y:{}, theta: {}".format(self.x, self.y, self.theta))
            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            if self.mode == Mode.IDLE:
                pass
            elif self.mode == Mode.STOP: # This stuff added by Keiko
                # At a stop sign
                if self.has_stopped():
                    self.last_stop_released_at = rospy.get_rostime().to_sec()
                    rospy.loginfo("Stop released at : {}".format(self.last_stop_released_at))
                    self.replan()
                    # self.switch_mode(Mode.IDLE)
            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.TRACK:
                if self.near_goal():
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    rospy.loginfo("replanning because far from start")
                    self.replan()
                elif (
                    rospy.get_rostime() - self.current_plan_start_time
                ).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
                    self.replan()  # we aren't near the goal but we thought we should have been, so replan
            elif self.mode == Mode.PARK:
                if self.at_goal():
                    # forget about goal:
                    # self.x_g = None
                    # self.y_g = None
                    # self.theta_g = None
                    self.switch_mode(Mode.IDLE)

            # State Machine Code Added by Mahesh
            # Subscribers:
            # 1. Subscribe to keypress for State Change
            # 2. Subscribe to CNN to detect objects
            # 3. Subscribe to keypress for inputs

            """
            Hints: Added by Mahesh
            DetectedObjectsList.objects has names of all detected objects -- all /detector/* topic
            DetectedObjectsList.ob_msgs has all detected objects in form of DetectedObjects -- /detector/objects topic
            """
            if self.state == States.REST:
                # reinit all variables declared for using this SM
                self.pickup_list = []
                # on key press change state to EXPLORE (handled by state_callback)
            elif self.state == States.EXPLORE_WAYPOINTS:
                if (rospy.get_rostime().to_sec() - self.wait_end_time) > 1:
                    if self.x_g is None or self.at_goal():
                        if self.waiting_at_waypoint:
                            self.waiting_at_waypoint = False
                            if (len(self.waypoints) == 0):
                                self.switch_state(States.INPUT)
                            else:
                                x, y, theta = self.waypoints[0]
                                goal = Pose2D()
                                goal.x, goal.y = x, y
                                goal.theta = theta
                                rospy.loginfo("Goal Waypoint : {}, {}, {}".format(goal.x, goal.y, goal.theta))
                                self.cmd_nav_pub.publish(goal)
                                self.waypoints.pop(0)
                        else:
                            self.wait_end_time = rospy.get_rostime().to_sec() + 1
                            self.waiting_at_waypoint = True
                        obj = Bool()
                        obj.data = self.waiting_at_waypoint
                        self.waiting_at_waypoint_pub.publish(obj)

            elif self.state == States.PURE_MAPPING:
                pass
            elif self.state == States.EXPLORE:
                pass
                # use teleop to move the robot (handled by keyboard_teleop)
                # add objects detected while exploring to a list
                # add markers to objects dectected
                # change state to INPUT on key press (handled by state_callback)
            elif self.state == States.INPUT:
                # take input from keyboard and add to pickup_list (handled by state_callback)
                # display the list objects detected to terminal
                rospy.loginfo(self.detected_objects)
            elif self.state == States.PLAN:
                # use tsp to plan the path (pass pickup_list to this function and get back updated sequence)
                # pickup_list will have objects to be picked up in sequence provided by tsp
                # switch state to SELECT_OBJ

                rospy.loginfo(self.pickup_list)
                self.solve_tsp()
                rospy.loginfo(self.pickup_list)
                self.switch_state(States.SELECT_OBJ)
            elif self.state == States.SELECT_OBJ:
                # if pickup_list is not empty:
                if len(self.pickup_list) != 0:
                    # select obj and goal location
                    obj = self.detected_objects[self.pickup_list[0]]

                    ## Keiko ##
                    name, x, y, theta = obj
                    rospy.loginfo("Goal Object : {}, {}, {}, {}".format(name, x, y, theta))

                    # state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
                    # state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
                    # x_init = self.snap_to_grid((self.x, self.y))
                    # x_goal = self.snap_to_grid((x, y))
                    # problem = AStar(
                    #     state_min,
                    #     state_max,
                    #     x_init,
                    #     x_goal,
                    #     self.occupancy,
                    #     self.plan_resolution,
                    # )
                    # success = problem.approx_solve()
                    # set goal position
                    # goal = Pose2D()
                    # goal.x , goal.y = problem.x_goal
                    goal = Pose2D()
                    goal.x , goal.y = x, y
                    goal.theta = theta
                    rospy.loginfo("Goal Object Approximation : {}, {}, {}, {}".format(name, goal.x, goal.y, goal.theta))
                    self.cmd_nav_pub.publish(goal)
                    # change state to GOTO_OBJ
                    self.switch_state(States.GOTO_OBJ)
                # else:
                else:
                    # send goal as start position
                    goal = Pose2D()
                    goal.x = self.x_init
                    goal.y = self.y_init
                    goal.theta = self.theta_init
                    rospy.loginfo("Goto START: x:{}, y:{}, theta:{}".format(goal.x, goal.y, goal.theta))
                    self.cmd_nav_pub.publish(goal)
                    # change state to GOTO_START
                    self.switch_state(States.GOTO_START)
            elif self.state == States.GOTO_OBJ:
                # if distance to goal >=epsilon:
                rospy.loginfo("Distance to Goal: {}".format(linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))))
                if self.at_goal():
                    # change state to pickup_obj
                    rospy.loginfo("Reached Location")
                    self.switch_mode(Mode.STOP)
                    self.switch_state(States.PICKUP_OBJ)
                    t1=rospy.get_rostime().to_sec()
            elif self.state == States.PICKUP_OBJ:
                # pop from pickup_list
                # set velocities to 0
                #use MODE.stop to do this
                rospy.loginfo("Picking up object")
                # cmd_vel=Twist()
                # cmd_vel.linear.x=0.0
                # cmd_vel.angular.z=0.0
                # self.nav_vel_pub.publish(cmd_vel)
                # wait for 3 sec
                t2=rospy.get_rostime().to_sec()
                # rospy.loginfo("t1: {}".format(t1))
                # rospy.loginfo("t2: {}".format(t2))
                if(t2-t1 >= 3):
                # change state to SELECT_OBJ
                    self.pickup_list.pop(0)
                    self.switch_state(States.SELECT_OBJ)

            elif self.state == States.GOTO_START:
                # if distance to goal >=epsilon:
                if not self.at_goal():
                    pass
                    # ask navigator to navigate to the goal location
                    # detect stop signs and stop for 2 sec during navigation
                # else:
                else:
                    # change state to MISSION_COMPLETE
                    self.switch_state(States.MISSION_COMPLETE)
                    t3=rospy.get_rostime().to_sec()
            elif self.state == States.MISSION_COMPLETE:
                # wait for 1 sec
                t4=rospy.get_rostime().to_sec()
                if(t4-t3 >= 1):
                    self.switch_state(States.REST)
                # change state to REST

            self.publish_control()
            rate.sleep()


if __name__ == "__main__":
    nav=Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    nav.run()
