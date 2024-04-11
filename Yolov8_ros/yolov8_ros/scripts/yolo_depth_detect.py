#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import torch
import rospy
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.ops import scale_boxes
import time
import pyrealsense2 as rs
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes

pipeline = rs.pipeline()  # 定义流程pipeline
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)  # 流程开始
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)
interest_class_list = [0, 64, 66, 67, 41, 73]
interest_class_dict = {
    0  : 'person',
    64 : 'mouse',
    67 : 'cell phone',
    66 : 'keyboard',
    41 : 'cup',
    73 : ' book'
}
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
        camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
        camera_xyz = camera_xyz.tolist()
        self.body_position = camera_xyz

class Yolo_Dect:
    def __init__(self):
        # load parameters
        weight_path = rospy.get_param('~weight_path', '')
        # pub_topic = rospy.get_param('~pub_topic', '/yolov8/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.5')
        self.visualize = rospy.get_param('~visualize', 'True')
        # which device will be used
        if (rospy.get_param('/use_cpu', 'false')):
            self.device = 'cpu'
            self.is_half = False
        else:
            self.device = 'cuda'
            self.is_half = True
        self.model = YOLO(weight_path)
        self.model.fuse()
        self.model.conf = conf
        self.color_image = Image()
    
    def preprocess(self, color_image):
        img_resize = cv2.resize(color_image, (640, 640))
        img_resize = img_resize[:, :, ::-1]
        img_resize = np.expand_dims(img_resize, axis=0)
        img_resize = img_resize.transpose(0, 3, 1, 2)
        img_resize = np.ascontiguousarray(img_resize)
        self.img_resize = img_resize
        img_torch = torch.from_numpy(img_resize.astype(np.float32) / 255.0).to(self.device)
        img_torch = img_torch.half() if self.is_half else img_torch.float()  # 格式转换 uint8-> 浮点数
        if img_torch.ndimension() == 3:
            img_torch = img_torch.unsqueeze(0)
        self.img_torch = img_torch
        return img_resize

    @torch.no_grad()
    def detect(self, color_image):
        img_resize = self.preprocess(color_image)
        results = self.model.predict(self.img_torch, half=self.is_half, classes=interest_class_list, show=False, agnostic_nms=True, conf=0.7)
        another_result = results[0].boxes.xyxy.clone()
        another_result = scale_boxes(img_resize.shape[2:], another_result, color_image.shape, ratio_pad=None)
        object_list = []
        index = 0
        if results is not None:
            for result in results[0].boxes:
                class_id = int(result.cls.item())
                one_object = one_Object_Element(result.cls.item, result.conf.item())
                one_object.class_name = interest_class_dict[result.cls.item()]
                one_corner_position = []
                another_corner_position = []
                one_corner_position.extend((another_result[index][0], another_result[index][1]))
                another_corner_position.extend((another_result[index][2], another_result[index][3]))
                one_object.corner_points.extend((one_corner_position, another_corner_position))
                object_list.append(one_object)   
                index += 1        
        return object_list  

    def publish_image(self, imgdata, height, width):
        image_temp = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = self.camera_frame
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(imgdata).tobytes()
        image_temp.header = header
        image_temp.step = width * 3
        self.image_pub.publish(image_temp)

def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

    ############### 相机参数的获取 #######################
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile(
    ).intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }'''

    # 保存内参到本地
    # with open('./intrinsics.json', 'w') as fp:
    #json.dump(camera_parameters, fp)
    #######################################################
    color_image = np.asanyarray(color_frame.get_data())  # RGB图

    # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
    return depth_intrin, color_image, aligned_depth_frame

def main():
    rospy.init_node('yolov8_ros', anonymous=True)
    rospy.loginfo("initial the Model of YOLOv8....")
    yolo_detect = Yolo_Dect()  
    rospy.loginfo("YOLO model load success")
    try:
        while not rospy.is_shutdown():
            depth_intrin, color_image, aligned_depth_frame = get_aligned_images()
            if not color_image.any():
                continue
            t_start = time.time()
            object_list = yolo_detect.detect(color_image) 
            if object_list is None or len(object_list) == 0:
                continue
            print(f"This frame is detected, and result is below:")
            for one_object in object_list:
                one_object.uv_trans_to_body(aligned_depth_frame, depth_intrin)
                class_name = one_object.class_name
                conf = one_object.confidence
                body_position = one_object.body_position
                print(f"Class : {class_name}")
                print(f"confidence : {conf}")
                print(f"position : {body_position}")
                print(f"-----")
            t_end = time.time()
            fps = int(1.0 / (t_end - t_start))
            print(f"FPS : {fps}")
            # rospy.spin() 
    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    main()
