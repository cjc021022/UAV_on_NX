#include <ros/ros.h>
#include <vector>
#include "std_msgs/Float64.h"
#include "realsense_depth/depth_helper.h"
#include <mavros_msgs/Altitude.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/PositionTarget.h>
namespace OffboardControl{
    class track_control{
        public:
            ros::Subscriber height_sub;
            double current_height;
            mavros_msgs::PositionTarget target_pose;
            track_control(ros::NodeHandle& nh);
            void getCurrentHeight(const mavros_msgs::Altitude::ConstPtr &msg);
            void observe_mode();
            void track_mode();
            void track_process();
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