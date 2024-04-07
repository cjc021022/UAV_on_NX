#include <ros/ros.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/State.h>
#include <geometry_msgs/PoseStamped.h>

namespace OffboardControl{
    bool setMode(ros::ServiceClient set_mode_client);
    bool sendArmCommand(ros::ServiceClient arming_client);
    class offboard_takeoff{
        public:
            ros::ServiceClient arming_client;
            ros::ServiceClient set_mode_client;        
            offboard_takeoff(ros::NodeHandle& nh);
            bool pose_init();
            bool takeoff();

        private:
            ros::NodeHandle nh_;
            ros::Publisher pose_init_pub_;
            ros::Subscriber state_sub_;
            mavros_msgs::State current_state_;        
            geometry_msgs::PoseStamped initial_pose_;
            void stateCallback(const mavros_msgs::State::ConstPtr& msg);
            // bool sendArmCommand(const mavros_msgs::CommandBool& arm_cmd);
            // bool setMode(const mavros_msgs::SetMode& offb_set_mode);
    };
    class position_hover{
        public:
            ros::Publisher local_pose_pub;        
            position_hover(ros::NodeHandle & nh);
            void moveToPosition(float x, float y, float z);
        private:
            ros::NodeHandle nh_;

    };
}