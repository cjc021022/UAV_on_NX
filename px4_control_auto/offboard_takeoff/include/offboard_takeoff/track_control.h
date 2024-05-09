#include <ros/ros.h>
#include <vector>
#include "std_msgs/Float64.h"
#include "realsense_depth/depth_helper.h"
#include"geometry_msgs/TwistStamped.h"
#include <geometry_msgs/PoseStamped.h>
namespace OffboardControl{
    class track_control{
        public:
            ros::Publisher target_pose_pub;
            ros::Publisher yaw_vel_pub;
            track_control(ros::NodeHandle& nh);
            void observe_mode(geometry_msgs::Twist vel);
            void track_mode(geometry_msgs::Twist vel);
            void track_process(geometry_msgs::Twist vel, geometry_msgs::PoseStamped target_pose);
        private:
            ros::NodeHandle nh_;
            ros::Subscriber object_position_sub_;
            ros::Subscriber distance_sub_;
            std::vector<Eigen::Vector3d> position_buffer_;
            Eigen::Vector3d target_position_;
            
            double delta_x_;
            double distance_;
            void updatePosition(const geometry_msgs::PoseStamped::ConstPtr& msg);
            void updatePositionBufferWithSmooth(const Eigen::Vector3d new_point);
            void updateDistance(const std_msgs::Float64::ConstPtr& msg);

    };
}