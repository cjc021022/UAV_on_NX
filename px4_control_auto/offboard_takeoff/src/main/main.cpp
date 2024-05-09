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
    depth_helper.publisher_point();
    OffboardControl::track_control track_control(nh);
    if(success){
        track_control.track_process();
    }
    // ros::spin();
    return 0;
}