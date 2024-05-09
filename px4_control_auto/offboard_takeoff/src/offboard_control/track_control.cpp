#include "offboard_takeoff/track_control.h"
#include <Eigen/Dense>
namespace OffboardControl{
    track_control::track_control(ros::NodeHandle& nh) : nh_(nh){
        // object_position_sub_ = 
        //nh_.subscribe<geometry_msgs::PoseStamped>("/camera_frame/center_point", 10, boost::bind(&track_control::updatePositionBufferWithSmooth, this, _1));
        object_position_sub_ = 
        nh_.subscribe<geometry_msgs::PoseStamped>("/camera_frame/center_point", 10, boost::bind(&track_control::updatePosition, this, _1));
        distance_sub_ = nh_.subscribe<std_msgs::Float64>("/object_detect/object_distance", 10, boost::bind(&track_control::updateDistance, this, _1));
        yaw_vel_pub = nh_.advertise<geometry_msgs::TwistStamped>("mavros/setpoint_velocity/cmd_vel", 10);
        target_pose_pub = nh_.advertise<geometry_msgs::PoseStamped>("mavros/setpoint_position/local", 10);
        distance_ = -1;
        delta_x_ = -1;
    }

    void track_control::updatePositionBufferWithSmooth(const Eigen::Vector3d new_point){
        position_buffer_.push_back(new_point);
        if (position_buffer_.size() > 5) {
            position_buffer_.erase(position_buffer_.begin());
        }
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        for (const auto& point : position_buffer_) {
            sum += point;
        }
        target_position_ = sum / position_buffer_.size();
    }

    void track_control::updatePosition(const geometry_msgs::PoseStamped::ConstPtr& msg){
        if(nullptr == msg){
            return;
        }
        delta_x_ = msg->pose.position.x;
        return;
    }
    void track_control::updateDistance(const std_msgs::Float64::ConstPtr& msg){
        distance_ = msg->data;
        ROS_INFO("the distance is %f", distance_);
    }

    void track_control::observe_mode(geometry_msgs::Twist vel){
        vel.angular.z = 0.5;
        yaw_vel_pub.publish(vel);
    }
    void track_control::track_mode(geometry_msgs::Twist vel){
        vel.angular.z = 0.0;
    }
    void track_control::track_process(geometry_msgs::Twist vel, geometry_msgs::PoseStamped target_pose){
        if(distance_ > 0 && -0.1 <= delta_x_ && delta_x_ <= 0.1){
            track_control::track_mode(vel);
            target_pose.pose.position.x = distance_ - 2.0;
        }
        else{
            track_control::observe_mode(vel);
        }
        target_pose_pub.publish(target_pose);    
    }
}