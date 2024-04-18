#include <ros/ros.h>
#include "offboard_takeoff/offboard_control.h"

namespace OffboardControl{
    position_hover::position_hover(ros::NodeHandle& nh) : nh_(nh){
        local_pose_pub = nh_.advertise<geometry_msgs::PoseStamped>("mavros/setpoint_position/local", 10);
    }
    void position_hover::moveToPosition(float x, float y, float z){
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = x;
        pose.pose.position.y = y;
        pose.pose.position.z = z;
        pose.pose.orientation.w = 0.707;
        pose.pose.orientation.z = 0.707;
        ros::Rate rate(30.0);
        while(ros::ok()){
            local_pose_pub.publish(pose);
            ros::spinOnce();
            rate.sleep();
        }
    }
}