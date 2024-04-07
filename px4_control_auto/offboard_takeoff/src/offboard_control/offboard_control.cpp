#include <ros/ros.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include "offboard_takeoff/offboard_control.h"

namespace OffboardControl {
    offboard_takeoff::offboard_takeoff(ros::NodeHandle& nh) : nh_(nh) {
        pose_init_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("mavros/setpoint_position/local", 10);
        // 订阅当前状态信息
        state_sub_ = nh_.subscribe<mavros_msgs::State>("mavros/state", 10, boost::bind(&offboard_takeoff::stateCallback, this, _1));
        // 发布解锁和设置模式的服务
        arming_client = nh_.serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");
        set_mode_client = nh_.serviceClient<mavros_msgs::SetMode>("mavros/set_mode");
    }

    // 进入Offboard模式并解锁
    bool offboard_takeoff:: takeoff() {
        // 等待当前状态可用
        ros::Rate rate(20.0);
        while (ros::ok() && !current_state_.connected) {
            ros::spinOnce();
            rate.sleep();
        }
        // init the target position, before set_mode_'OFFBOARD'& arm UAV  maybe the value(0,0,1.5)
        initial_pose_.pose.position.x = 0;
        initial_pose_.pose.position.y = 0;
        initial_pose_.pose.position.z = 1;
        initial_pose_.pose.orientation.w = 0.707;
        initial_pose_.pose.orientation.z = 0.707;
        for(int i = 50; ros::ok() && i > 0; --i){
            pose_init_pub_.publish(initial_pose_);
            ros::spinOnce();
            rate.sleep();
        }
        // 设置Offboard模式
        if (!setMode(set_mode_client)) {
            ROS_ERROR("Failed to set Offboard mode!");
            return false;
        }     
        // 解锁
        if (!sendArmCommand(arming_client)) {
            ROS_ERROR("Failed to arm the vehicle!");
            return false;
        }       
        ROS_INFO("Vehicle armed and in Offboard mode");
        return true;
    }

    // 当前状态回调函数
    void offboard_takeoff::stateCallback(const mavros_msgs::State::ConstPtr& msg) {
        current_state_ = *msg;
    }

    // 发送解锁指令
    bool sendArmCommand(ros::ServiceClient arming_client) {
        mavros_msgs::CommandBool arm_cmd;
        arm_cmd.request.value = true;        
        if (!arming_client.exists()) {
            ROS_ERROR("Arming service is not available!");
            return false;
        }

        if (!arming_client.call(arm_cmd) || !arm_cmd.response.success) {
            ROS_ERROR("Failed to call arming service!");
            return false;
        }
        return true;
    }

    // // 发送设置模式指令
    bool setMode(ros::ServiceClient set_mode_client) {
        mavros_msgs::SetMode offb_set_mode;
        offb_set_mode.request.custom_mode = "OFFBOARD";        
        if (!set_mode_client.exists()) {
            ROS_ERROR("Set mode service is not available!");
            return false;
        }

        if (!set_mode_client.call(offb_set_mode) || !offb_set_mode.response.mode_sent) {
            ROS_ERROR("Failed to call set mode service!");
            return false;
        }

        return true;
    }
};