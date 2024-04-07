#include <ros/ros.h>
#include "offboard_takeoff/offboard_control.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "offboard_hover_node");
    ros::NodeHandle nh;

    OffboardControl::offboard_takeoff offboard_takeoff(nh);
    bool success = offboard_takeoff.takeoff();
    OffboardControl::position_hover position_hover(nh);
    if(success){
        position_hover.moveToPosition(0,0,1.5);
    }
    ros::spin();
    return 0;
}