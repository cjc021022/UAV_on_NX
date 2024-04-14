#! /home/nx/miniconda3/envs/yolo_depth_detect/bin/python
import os
import sys
sys.path.append('/home/nx/catkin_ws/src/yolo_depth_detect/include/yolo_depth_detect')
print("PYTHONPATH:", os.getenv("PYTHONPATH"))
from one_object_element import one_Object_Element
import rospy

one_test = one_Object_Element(87, 0.11)
rospy.init_node('test_import', anonymous=True)
rospy.loginfo('class id is %d', one_test.class_id)