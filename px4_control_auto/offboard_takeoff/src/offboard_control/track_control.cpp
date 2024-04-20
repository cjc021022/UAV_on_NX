#include "offboard_takeoff/track_control.h"
#include <Eigen/Dense>
namespace OffboardControl{
    track_control::track_control(ros::NodeHandle& nh) : nh_(nh){
        // object_position_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("/camera_frame/center_point", 10, boost::bind(&track_control::updatePositionBufferWithSmooth, this, _1));
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
}