#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <librealsense2/rs.hpp>
#include"yolov8_ros_msgs/BoundingBoxes.h"
#include"yolov8_ros_msgs/BoundingBox.h"
using namespace std;

using namespace cv;
 
cv::Mat depth_pic;        //定义全局变量，图像矩阵Ｍat形式
float d_value;
// realsense_dev::depth_value command_to_pub;   //待发布数据

void boudingboxCallBack(const yolov8_ros_msgs::BoundingBoxes::ConstPtr &msg){
    vector<yolov8_ros_msgs::BoundingBox> boxes_vector;
    for (const auto& bounding_box : msg->bounding_boxes) {
        int x_center = (bounding_box.xmin + bounding_box.xmax) / 2;
        int y_center = (bounding_box.ymin + bounding_box.ymax) / 2;         
        boxes_vector.push_back(bounding_box);
    } 
    ROS_INFO("first bouding box is %d", boxes_vector[0].xmin);
}
void depthCallback(const sensor_msgs::Image::ConstPtr&msg)
{
    cv_bridge::CvImagePtr Dest ;
    Dest = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
    depth_pic = Dest->image;
    //cout<<"Output some info about the depth image in cv format"<<endl;
    //cout<< "Rows of the depth iamge = "<<depth_pic.rows<<endl;                       //获取深度图的行数height
    //cout<< "Cols of the depth iamge = "<<depth_pic.cols<<endl;                           //获取深度图的列数width
    //cout<< "Type of depth_pic's element = "<<depth_pic.type()<<endl;             //深度图的类型
    ushort dis = depth_pic.at<ushort>(depth_pic.rows/2,depth_pic.cols/2);           //读取深度值，数据类型为ushort单位为ｍｍ
    //将深度图的像素点根据内参转换到深度摄像头坐标系下的三维点
    d_value = float(dis)/1000 ;      //强制转换
    ROS_INFO("Value of depth_pic's pixel= %.2f", d_value);
    // cout<< "Value of depth_pic's pixel= "<<d_value<<endl;    //读取深度值
}

float getDepthAtPixel(const cv::Mat& depth_image, int x, int y)
{
    if (!depth_image.empty() && x >= 0 && x < depth_image.cols && y >= 0 && y < depth_image.rows)
    {
        return depth_image.at<uint16_t>(y, x) * 0.001; // Convert to meters
    }
    else
    {
        return std::numeric_limits<float>::quiet_NaN(); // Return NaN if pixel is out of bounds or no depth data available
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "stream_pub");               // ros节点初始化
    ros::NodeHandle nh;                                           //创建节点句柄
    ros::AsyncSpinner spinner(1);
    spinner.start();

    ros::Subscriber image_sub = nh.subscribe<sensor_msgs::Image>("/camera/aligned_depth_to_color/image_raw", 1, depthCallback);   //订阅深度图像
    ros::Subscriber boudingboxes_sub = nh.subscribe<yolov8_ros_msgs::BoundingBoxes>("/yolov8/BoundingBoxes", 10, boudingboxCallBack);
    //ros::Subscriber element_sub = nh.subscribe<sensor_msgs::Image>("/camera/aligned_depth_to_color/image_raw",100,pixelCallback);     //订阅像素点坐标
    // ros::Publisher mode_pub = nh.advertise<realsense_dev::depth_value>("/depth_info", 10);

    // command_to_pub.Value = 0;    //初始化深度值
    // ros::Rate rate(20.0);    //设定自循环的频率
    // while(ros::ok)
    // {
    //     command_to_pub.header.stamp      = ros::Time::now();
    //     command_to_pub.Value = d_value;     //depth_pic.rows/2,depth_pic.cols/2  为像素点
    //     mode_pub.publish(command_to_pub);
    // }
    
    ros::Duration(10).sleep();    //设定自循环的频率
    return 0 ;
}