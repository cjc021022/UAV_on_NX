#pragma once
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include<vector>
#include"yolov8_ros_msgs/BoundingBoxes.h"
#include<Eigen/Dense>
namespace realsenseHelper{
    struct depth_intri{
        double fx = 607.0803833007812;
        double fy = 607.2030639648438;
        double ppx = 330.08758544921875;
        double ppy = 241.1981658935547;
        int height = 480;
        int width = 640;
        double depth_scale = 0.0010000000474974513;
    };
    Eigen::Matrix3d intrinToMatrix(const depth_intri& depth_intrin);
    class depth_helper{
        public:
            depth_helper(ros::NodeHandle& nh, std::string boudingboxes_topic); 
            int64_t x_center;
            int64_t y_center;
            bool getDepthDistanceFromPoint();
            Eigen::Vector3d imageToBodyCoords();
            void publisher_point();
        private:
            ros::NodeHandle nh_;
            ros::Subscriber align_depth_image_sub_;
            ros::Subscriber boudingboxes_sub_;
            ros::Publisher camera_frame_pub_; 
            std::vector<yolov8_ros_msgs::BoundingBox> bounding_boxes_;
            std::string class_name;
            cv::Mat depth_frame_;
            Eigen::Matrix3d depth_intrin_;
            double confidence;
            double center_distance_;
            int64_t track_id;
            void readDepthImage(const sensor_msgs::Image::ConstPtr &msg);
            void updateBoudingBoxes(const yolov8_ros_msgs::BoundingBoxes::ConstPtr &msg);
            
    };
}