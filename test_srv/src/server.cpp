#include "ros/ros.h"
#include "geometry_msgs/Point.h"
#include "sensor_msgs/Image.h"
#include "std_msgs/Header.h"
#include "std_msgs/Bool.h"
#include "test_srv/QuadState.h"
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

bool random_message_generator(test_srv::QuadState::Request  &req,
         test_srv::QuadState::Response &res)
{
  ROS_INFO("Got request %s\n", req.in.c_str());
  cv::Mat image = cv::Mat::zeros(cv::Size(128,128),CV_32FC4);
  geometry_msgs::Point a;
  a.x = 0;
  a.y = 0;
  a.z = 0;
  std_msgs::Bool c;
  c.data = 0;  
  res.done = c;
  res.crash = c;
  res.current_position = a;
  res.goal_position = a;

  //sensor_msgs::Image img_msg;
  //cv_bridge::CvImage img_bridge;

  std_msgs::Header header;
  header.stamp = ros::Time::now();
  //sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(header, "32FC4", image).toImageMsg();
  //img_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::32FC4,image);
  //img_bridge.toImageMsg(img_msg);
  res.image = *(cv_bridge::CvImage(header, "32FC4", image).toImageMsg());
  //res.image = img_msg;

  res.header = header;
  return true;
}
  
int main(int argc, char **argv)
{
  ros::init(argc, argv, "get_state_server");
  ros::NodeHandle n;
 
  ros::ServiceServer service = n.advertiseService("get_state", random_message_generator);
  ROS_INFO("Ready to genearete random messages");
  ros::spin();
 
  return 0;
}
