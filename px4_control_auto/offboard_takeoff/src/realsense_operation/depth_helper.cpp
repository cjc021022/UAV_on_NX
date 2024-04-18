#include"realsense_depth/depth_helper.h"
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include"yolov8_ros_msgs/BoundingBox.h"

namespace realsenseHelper{
    depth_helper::depth_helper(ros::NodeHandle & nh, std::string boudingboxes_topic) : nh_(nh){
        align_depth_image_sub_ = nh_.subscribe<sensor_msgs::Image>("/camera/aligned_depth_to_color/image_raw", 1, boost::bind(&depth_helper::readDepthImage, this, _1));
        boudingboxes_sub_ = nh_.subscribe<yolov8_ros_msgs::BoundingBoxes>(boudingboxes_topic, 10, boost::bind(&depth_helper::updateBoudingBoxes, this, _1));
    }

    void depth_helper::readDepthImage(const sensor_msgs::Image::ConstPtr &msg){
        if (nullptr == msg){
            return;
        }
        cv_bridge::CvImagePtr 
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        cv::Mat depth_image = cv_ptr->image;    
        ushort dis = depth_image.at<ushort>(x_center, y_center); 
        // ushort dis = depth_image.at<ushort>(320, 240);
        center_distance_ = double (dis)/1000 ;
        ROS_INFO("Value of depth_pic's pixel= %.2f", center_distance_);  
    }
    void depth_helper::updateBoudingBoxes(const yolov8_ros_msgs::BoundingBoxes::ConstPtr &msg){
        bounding_boxes_.clear();
        for (const auto& bounding_box : msg->bounding_boxes) {
            x_center = (bounding_box.xmin + bounding_box.xmax) / 2;
            y_center = (bounding_box.ymin + bounding_box.ymax) / 2;            
            bounding_boxes_.push_back(bounding_box);
        }
    }
}