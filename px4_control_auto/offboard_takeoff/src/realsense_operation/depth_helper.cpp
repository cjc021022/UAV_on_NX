#include"realsense_depth/depth_helper.h"
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <geometry_msgs/PoseStamped.h>
#include"yolov8_ros_msgs/BoundingBox.h"
#include<Eigen/Dense>
namespace realsenseHelper{
    Eigen::Matrix3d intrinToMatrix(const depth_intri& depth_intrin) {
        Eigen::Matrix3d K;
        K << 1.0 / depth_intrin.fx, 0, -depth_intrin.ppx / depth_intrin.fx,
             0, 1.0 / depth_intrin.fy, -depth_intrin.ppy / depth_intrin.fy,
             0, 0, 1.0;
        return K;
    }

    depth_helper::depth_helper(ros::NodeHandle & nh, std::string boudingboxes_topic) : nh_(nh), x_center(-1), y_center(-1){
        align_depth_image_sub_ = nh_.subscribe<sensor_msgs::Image>("/camera/aligned_depth_to_color/image_raw", 1, boost::bind(&depth_helper::readDepthImage, this, _1));
        boudingboxes_sub_ = nh_.subscribe<yolov8_ros_msgs::BoundingBoxes>(boudingboxes_topic, 10, boost::bind(&depth_helper::updateBoudingBoxes, this, _1));
        camera_frame_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/camera_frame/center_point", 10);
        depth_intri default_intrin;
        depth_intrin_ = intrinToMatrix(default_intrin);
        // std::cout << "Depth intrinsic matrix:\n" << depth_intrin_ << std::endl;
    }

    void depth_helper::readDepthImage(const sensor_msgs::Image::ConstPtr &msg){
        if (nullptr == msg){
            return;
        }
        cv_bridge::CvImagePtr 
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        depth_frame_ = cv_ptr->image;
    }

    void depth_helper::updateBoudingBoxes(const yolov8_ros_msgs::BoundingBoxes::ConstPtr &msg){
        bounding_boxes_.clear();
        for (const auto& bounding_box : msg->bounding_boxes){
            x_center = (bounding_box.xmin + bounding_box.xmax) / 2;
            y_center = (bounding_box.ymin + bounding_box.ymax) / 2;
            bounding_boxes_.push_back(bounding_box);
        }
    }
    bool depth_helper::getDepthDistanceFromPoint(){
        if (depth_frame_.empty() || x_center < 0 || y_center < 0) {
            ROS_DEBUG("Depth image not available.");
            return false;
        }        
        ushort dis = depth_frame_.at<ushort>(y_center, x_center); 
        // ushort dis = depth_image.at<ushort>(320, 240);
        center_distance_ = double (dis)/1000 ;
        return true;
        // ROS_INFO("Value of depth_pic's pixel= %.2f", center_distance_);
    }
    Eigen::Vector3d depth_helper::imageToBodyCoords(){
        if(!getDepthDistanceFromPoint()){
            return Eigen::Vector3d::Zero();
        }
        Eigen::Vector3d pixel_coords(x_center, y_center, 1.0);
        Eigen::Vector3d normalized_coords = depth_intrin_ * pixel_coords;  // [x_norm, y_norm, 1]^T
        Eigen::Vector3d camera_coords(center_distance_, center_distance_, center_distance_); 
        Eigen::Vector3d camera_point_vector = normalized_coords.cwiseProduct(camera_coords); //camera_point[0--2] xyz
        double theta = 30 * M_PI / 180; // 30°对应的弧度值
        Eigen::Matrix3d Rx;
        Rx << 1, 0, 0,
            0, cos(theta), -sin(theta),
            0, sin(theta), cos(theta);
        Eigen::Vector3d translation_vector(0, 0, 0); // 平移
        Eigen::Vector3d transformed_point = Rx * camera_point_vector;  
        return transformed_point;           
    }
    void depth_helper::publisher_point(){
        ros::Rate rate(30);
        geometry_msgs::PoseStamped target_point;
        target_point.header.frame_id = "UAV_body_frame"; 
        while (ros::ok()){
            Eigen::Vector3d UAV_body_point = imageToBodyCoords();
            if (!UAV_body_point.isZero()){
                target_point.pose.position.x = UAV_body_point[0];
                target_point.pose.position.y = UAV_body_point[2];
                target_point.pose.position.z = UAV_body_point[1];                  
            }
            target_point.header.stamp = ros::Time::now(); // 设置当前时间为时间戳
            camera_frame_pub_.publish(target_point);
            ros::spinOnce();
            rate.sleep();
        }
    }
}