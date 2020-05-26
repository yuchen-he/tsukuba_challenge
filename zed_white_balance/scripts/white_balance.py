#!/usr/bin/env python
# coding: UTF-8

import rospy
import yaml
import numpy as np
import cv2
import os
import rosparam
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import dynamic_reconfigure.client

class white_balance_adjust:

    def __init__(self):

        # Read yaml file directly
        # current_path = os.path.dirname(__file__)
        # yaml_file    = os.path.join(current_path, "target_parameter.yaml")
        # f = open(yaml_file, "r+")
        # self.target_param         =  yaml.load(f)
        # self.start_row            =  self.target_param['start_row']
        # self.target_brightness    =  self.target_param['target_brightness'] 
        # self.roi_mode             =  self.target_param['roi_mode']
        # self.left_top             =  self.target_param['left_top']
        # self.right_bottom         =  self.target_param['right_bottom']

        # Read from launch file
        self.roi_mode             =  rospy.get_param("/roi_mode")         
        self.start_row            =  rospy.get_param("/start_row") 
        self.target_brightness    =  rospy.get_param("/target_brightness") 
        self.left_top             =  rospy.get_param("/left_top") 
        self.right_bottom         =  rospy.get_param("/right_bottom") 

        #self.auto_exposure        =  rospy.get_param("/nodelet_manager/auto_exposure")

        self.gain       = 50
        self.exposure   = 50

        # Subscribe the images
        rospy.Subscriber('/sensors/zed/left/image_rect_color', Image, self.callback)
        # rospy.Subscriber('/sensors/zed/right/image_rect_color', Image, self.callback)  


    def callback(self, image):

        # Use cv_bridge to change the format of input image
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
        #cv_image.shape -> (h, w, channels) = (720, 1280, 3)


        # Get the average brightness based on roi_mode
        if (self.roi_mode == True):
            cv_image_sliced    = cv2.cvtColor(cv_image[self.left_top[1]:self.right_bottom[1], self.left_top[0]:self.right_bottom[0], :], cv2.COLOR_BGR2GRAY)
        else:
            cv_image_sliced    = cv2.cvtColor(cv_image[self.start_row:, :, :], cv2.COLOR_BGR2GRAY)

        average_brightness = np.average(cv_image_sliced[:, :])
        #print('The auto_exposure mode is : ', self.auto_exposure)
        #print('The average_brightness is : ', average_brightness)
        #print('The target_brightness  is : ', self.target_brightness)
 

        # Get parameters: gain and exposure
        self.gain       = rosparam.get_param("/nodelet_manager/gain")
        self.exposure   = rosparam.get_param("/nodelet_manager/exposure")
        #print('Present gain is           : ', self.gain)
        #print('Present exposure is       : ', self.exposure)


        # Dynamic reconfigure
        client = dynamic_reconfigure.client.Client("nodelet_manager", timeout=30)
        brightness_diff = average_brightness - self.target_brightness

        if (brightness_diff < -5):
            if (self.gain < 100):
                self.gain     += (-brightness_diff) *0.1
            elif (self.exposure < 100):
                self.exposure += (-brightness_diff) *0.1
            else:
                #print("Both gain and exposure has reached 100!")
            client.update_configuration({"gain":self.gain, "exposure":self.exposure})

        elif (brightness_diff > 5):
            if (self.gain > 0):
                self.gain     -= (brightness_diff) *0.1
            elif (self.exposure > 0):
                self.exposure -= (brightness_diff) *0.1
            else:
                #print("Both gain and exposure has reached 0!")
            client.update_configuration({"gain":self.gain, "exposure":self.exposure})

    
def main():

    rospy.init_node('white_balance', anonymous=True)
    wb = white_balance_adjust()

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException: pass

