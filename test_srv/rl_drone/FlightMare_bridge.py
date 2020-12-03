#!/usr/bin/env python3

import numpy as np
from test_srv.srv import QuadState
import rospy
from cv_bridge import CvBridge

class NormalizedActions():
    def action(self, action):
        low_vel  = arg_params['low_vel']
        high_vel = arg_params['high_vel']

        low_heading  = arg_params['low_heading']
        high_heading = arg_params['high_heading']
        
        action[0] = low_vel + (action[0] + 1.0) * 0.5 * (high_vel - low_vel)
        action[0] = np.clip(action[0], low_vel, high_vel)

        action[1] = low_heading + (action[1] + 1.0) * 0.5 * (high_heading - low_heading)
        action[1] = np.clip(action[1], low_heading, high_heading)
        
        return action

    def _reverse_action(self, action):
        low_vel  = arg_params['low_vel']
        high_vel = arg_params['high_vel']

        low_heading  = arg_params['low_heading']
        high_heading = arg_params['high_heading']
        
        action[0] = 2 * (action[0] - low_vel) / (high_vel - low_vel) - 1
        action[0] = np.clip(action[0], low_vel, high_vel)

        action[1] = 2 * (action[1] - low_heading) / (high_heading - low_heading) - 1
        action[1] = np.clip(action[1], low_heading, high_heading)
        
        return action

class FlightMare():

  def __init__(self, arg_params):

    self.max_x = arg_params['simulation_max_x']
    self.max_y = arg_params['simulation_max_y']
    self.min_x = arg_params['simulation_min_x']
    self.min_y = arg_params['simulation_min_y']

    self.bridge = CvBridge()
    self.arg_params = arg_params

    self.state_ = [np.zeros((arg_params['img_height'],arg_params['img_width'],arg_params['n_channels'])),np.zeros((2,)),np.zeros((2,))] 

    self.reward_ = 0

    self.is_done_ = False

    self.prev_state_ = [np.zeros((2,)),np.zeros((2,))] #store only pos and goal of the prev_state

    self.crash_ = False

    self.curr_action_ = np.zeros((2,1)) #heading and velocity

    rospy.wait_for_service('navigator/get_state')

    self.updated_state = rospy.ServiceProxy('navigator/get_state', QuadState)

    # FOR TESTING
    '''
    rospy.wait_for_service('get_state')
    self.updated_state = rospy.ServiceProxy('get_state', QuadState)
    '''

  def step(self, action):
    # Execute one time step within the environment

    self.curr_action_ = action
    self._take_action()
    self.get_reward_()

    return self.state_, self.reward_, self.is_done_

  def get_reward_(self):

    if (self.is_done_ and not self.crash_):
      self.reward_ = self.arg_params['completion_reward']
    
    else:

      if not (self.crash_):
        self.reward_ = self.arg_params['dist_reward_weight']*(np.linalg.norm(self.prev_state_[1]-self.prev_state_[0]) - np.linalg.norm(self.state_[2]-self.state_[1]))
      else:
        self.reward_ = self.arg_params['crash_reward']
  
  def _take_action(self):

    self.prev_state_[0] = self.state_[1].copy()
    self.prev_state_[1] = self.state_[2].copy()
    
    msg = "1 " + str(self.curr_action_[0,0]) + " " + str(self.curr_action_[0,1])
    
    rsp = self.updated_state(msg)

    self.crash_ = rsp.crash.data
    self.is_done_ = rsp.done.data or self.crash_
    if self.crash_:
        print("Crashed!")

    # self.state_[0] = self.bridge.imgmsg_to_cv2(rsp.image, desired_encoding='passthrough') #(H,W,n)
    tmp_img = self.bridge.imgmsg_to_cv2(rsp.image, desired_encoding='passthrough')
    img_cvt = tmp_img.astype(np.float32)/255.0
    self.state_[0] = img_cvt
    

    self.state_[1][0] = rsp.current_position.x
    self.state_[1][1] = rsp.current_position.y
    
    self.state_[2][0] = rsp.goal_position.x
    self.state_[2][1] = rsp.goal_position.y

  def reset(self):
    msg = "0 0 0"
    #ros action service to get new starting_position, goal and image
    
    rsp = self.updated_state(msg)

    self.is_done_ = False
    self.crash_ = False

    '''
    print("rsp image dimensions:")
    print(len(rsp.image.data))
    print(rsp.image.height)
    print(rsp.image.width)
    print(rsp.image.encoding)
    '''

    # self.state_[0] = self.bridge.imgmsg_to_cv2(rsp.image, desired_encoding='passthrough') #(H,W,n)
    tmp_img = self.bridge.imgmsg_to_cv2(rsp.image, desired_encoding='passthrough')
    img_cvt = tmp_img.astype(np.float32)/255.0
    self.state_[0] = img_cvt

    self.state_[1][0] = rsp.current_position.x
    self.state_[1][1] = rsp.current_position.y
    
    self.state_[2][0] = rsp.goal_position.x
    self.state_[2][1] = rsp.goal_position.y

    return self.state_
