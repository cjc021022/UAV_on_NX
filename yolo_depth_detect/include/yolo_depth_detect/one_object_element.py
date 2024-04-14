#! /home/nx/miniconda3/envs/yolo_depth_detect/bin/python
import pyrealsense2 as rs
import numpy as np
class one_Object_Element:
    def __init__(self, class_id, confidence) -> None:
        self.class_id = class_id
        self.class_name = ''
        self.id = -1
        self.confidence = confidence
        self.corner_points = []
        self.body_position = []
    
    def uv_trans_to_body(self, aligned_depth_frame, depth_intrin):
        ux = int((self.corner_points[0][0] + self.corner_points[1][0])/2)
        uy = int((self.corner_points[0][1] + self.corner_points[1][1])/2)
        dis = aligned_depth_frame.get_distance(ux, uy)
        camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)  # 计算相机坐标系的xyz
        camera_xyz = np.round(np.array(camera_xyz), 3)
        camera_xyz = camera_xyz.tolist()
        self.body_position = camera_xyz
    