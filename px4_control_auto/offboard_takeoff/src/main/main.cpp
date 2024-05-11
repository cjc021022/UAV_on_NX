#include <ros/ros.h>
#include "offboard_takeoff/offboard_control.h"
#include "realsense_depth/depth_helper.h"
#include "offboard_takeoff/track_control.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "offboard_hover_node");
    ros::NodeHandle nh;
    ros::Publisher target_pose_pub;
    target_pose_pub = nh.advertise<mavros_msgs::PositionTarget>("mavros/setpoint_raw/local", 10);
    OffboardControl::offboard_takeoff offboard_takeoff(nh);
    bool success = offboard_takeoff.takeoff();
    ''' hover at the position (x, y, z) '''
    // OffboardControl::position_hover position_hover(nh);  
    // if(success){
    //     position_hover.moveToPosition(0,0,1);
    // }
    std::string boudingbox_topic = "/yolov8/BoundingBoxes";
    realsenseHelper::depth_helper depth_helper(nh, boudingbox_topic);
    OffboardControl::track_control track_control(nh);
    ros::Rate rate(30);
    while(ros::ok){
        if(success){
            track_control.target_pose.header.stamp = ros::Time::now();
            if(0.9 <= track_control.current_height && track_control.current_height <= 1.1){
                depth_helper.publisher_point();
                track_control.track_process();
            }
            target_pose_pub.publish(track_control.target_pose);
        }
        ros::spinOnce();
        rate.sleep();        
    }
    return 0;
}