#include "offboard_takeoff/track_control.h"
#include <Eigen/Dense>
namespace OffboardControl{
    track_control::track_control(ros::NodeHandle& nh) : nh_(nh){
        // object_position_sub_ = 
        //nh_.subscribe<geometry_msgs::PoseStamped>("/camera_frame/center_point", 10, boost::bind(&track_control::updatePositionBufferWithSmooth, this, _1));
        object_position_sub_ = 
        nh_.subscribe<geometry_msgs::PoseStamped>("/camera_frame/center_point", 10, boost::bind(&track_control::updatePosition, this, _1));
        distance_sub_ = nh_.subscribe<std_msgs::Float64>("/object_detect/object_distance", 10, boost::bind(&track_control::updateDistance, this, _1));
        height_sub = nh.subscribe<mavros_msgs::Altitude>("/mavros/altitude", 10, boost::bind(&track_control::getCurrentHeight, this, _1));
        distance_ = -1;
        delta_x_ = -1;
        target_pose.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
        target_pose.type_mask = 
            // mavros_msgs::PositionTarget::IGNORE_PX |
            // mavros_msgs::PositionTarget::IGNORE_PY |
            // mavros_msgs::PositionTarget::IGNORE_PZ |
            mavros_msgs::PositionTarget::IGNORE_VX |
            mavros_msgs::PositionTarget::IGNORE_VY |
            mavros_msgs::PositionTarget::IGNORE_VZ |
            mavros_msgs::PositionTarget::IGNORE_AFX |
            mavros_msgs::PositionTarget::IGNORE_AFY |
            mavros_msgs::PositionTarget::IGNORE_AFZ |
            mavros_msgs::PositionTarget::FORCE |
            mavros_msgs::PositionTarget::IGNORE_YAW |
            mavros_msgs::PositionTarget::IGNORE_YAW_RATE;    
        target_pose.position.x = 0;
        target_pose.position.y = 0;   
        target_pose.position.z = 1;        
    }
    void track_control::getCurrentHeight(const mavros_msgs::Altitude::ConstPtr &msg){
        current_height = msg->local;
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

    void track_control::observe_mode(){
        ROS_INFO("Observe Mode...");
        target_pose.type_mask = 
            // mavros_msgs::PositionTarget::IGNORE_PX |
            // mavros_msgs::PositionTarget::IGNORE_PY |
            // mavros_msgs::PositionTarget::IGNORE_PZ |
            mavros_msgs::PositionTarget::IGNORE_VX |
            mavros_msgs::PositionTarget::IGNORE_VY |
            mavros_msgs::PositionTarget::IGNORE_VZ |
            mavros_msgs::PositionTarget::IGNORE_AFX |
            mavros_msgs::PositionTarget::IGNORE_AFY |
            mavros_msgs::PositionTarget::IGNORE_AFZ |
            mavros_msgs::PositionTarget::FORCE |
            mavros_msgs::PositionTarget::IGNORE_YAW;
            // mavros_msgs::PositionTarget::IGNORE_YAW_RATE; 
        target_pose.yaw_rate = 0.5;       
    }
    void track_control::track_mode(){
        ROS_INFO("Track Mode...");
        target_pose.type_mask = 
            // mavros_msgs::PositionTarget::IGNORE_PX |
            // mavros_msgs::PositionTarget::IGNORE_PY |
            // mavros_msgs::PositionTarget::IGNORE_PZ |
            mavros_msgs::PositionTarget::IGNORE_VX |
            mavros_msgs::PositionTarget::IGNORE_VY |
            mavros_msgs::PositionTarget::IGNORE_VZ |
            mavros_msgs::PositionTarget::IGNORE_AFX |
            mavros_msgs::PositionTarget::IGNORE_AFY |
            mavros_msgs::PositionTarget::IGNORE_AFZ |
            mavros_msgs::PositionTarget::FORCE |
            mavros_msgs::PositionTarget::IGNORE_YAW |
            mavros_msgs::PositionTarget::IGNORE_YAW_RATE;
        target_pose.position.y = distance_ - 2;

    }
    void track_control::track_process(){
        if(distance_ > 0 && -0.1 <= delta_x_ && delta_x_ <= 0.1){
            track_control::track_mode();
        }
        else{
            track_control::observe_mode();
        }    
    }
}