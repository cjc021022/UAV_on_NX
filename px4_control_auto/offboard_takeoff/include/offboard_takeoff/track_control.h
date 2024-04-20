#include <ros/ros.h>
#include <vector>
#include "realsense_depth/depth_helper.h"
namespace OffboardControl{
    class track_control{
        public:
            track_control(ros::NodeHandle& nh);
        private:
            ros::NodeHandle nh_;
            ros::Subscriber object_position_sub_;
            std::vector<Eigen::Vector3d> position_buffer_;
            Eigen::Vector3d target_position_;
            void updatePositionBufferWithSmooth(const Eigen::Vector3d new_point);

    };
}