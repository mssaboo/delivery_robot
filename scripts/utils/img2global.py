#!/usr/bin/env python3

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
