#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include<vector>
#include"yolov8_ros_msgs/BoundingBoxes.h"
namespace realsenseHelper{
    class depth_helper{
        public:
            depth_helper(ros::NodeHandle& nh, std::string boudingboxes_topic); 
            int64_t x_center;
            int64_t y_center;
        private:
            ros::NodeHandle nh_;
            ros::Subscriber align_depth_image_sub_;
            ros::Subscriber boudingboxes_sub_;
            std::vector<yolov8_ros_msgs::BoundingBox> bounding_boxes_;
            std::string class_name;
            double confidence;
            double center_distance_;
            int64_t track_id;
            void readDepthImage(const sensor_msgs::Image::ConstPtr &msg);
            void updateBoudingBoxes(const yolov8_ros_msgs::BoundingBoxes::ConstPtr &msg);
    };
}