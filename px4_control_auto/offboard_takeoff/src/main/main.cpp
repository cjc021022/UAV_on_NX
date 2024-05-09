#include <ros/ros.h>
#include "offboard_takeoff/offboard_control.h"
#include "realsense_depth/depth_helper.h"
#include"offboard_takeoff/track_control.h"
int main(int argc, char** argv) {
    ros::init(argc, argv, "offboard_hover_node");
    ros::NodeHandle nh;
    OffboardControl::offboard_takeoff offboard_takeoff(nh);
    bool success = offboard_takeoff.takeoff();
    // OffboardControl::position_hover position_hover(nh);
    // if(success){
    //     position_hover.moveToPosition(0,0,1);
    // }
    std::string boudingbox_topic = "/yolov8/BoundingBoxes";
    realsenseHelper::depth_helper depth_helper(nh, boudingbox_topic);
    geometry_msgs::PoseStamped target_point;
    target_point.header.frame_id = "UAV_body_frame";
    std_msgs::Float64 distance_msg;
    OffboardControl::track_control track_control(nh);
    geometry_msgs::Twist vel;
    geometry_msgs::PoseStamped target_pose;
    target_pose.pose.orientation.w = 0.707;
    target_pose.pose.orientation.z = 0.707; 
    target_pose.pose.position.x = 0;
    target_pose.pose.position.y = 0;   
    target_pose.pose.position.z = 1;  
    ros::Rate rate(30);
    while(ros::ok){
        if(success){
            depth_helper.publisher_point(target_point, distance_msg);
            track_control.track_process(vel, target_pose);
        }
        ros::spinOnce();
        rate.sleep();        
    }
    // ros::spin();
    return 0;
}