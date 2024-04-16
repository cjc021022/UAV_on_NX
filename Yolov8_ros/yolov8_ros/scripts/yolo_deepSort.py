#! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import torch
import rospy
import numpy as np
from ultralytics import YOLO
import datetime
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from deep_sort_realtime.deepsort_tracker import DeepSort

interest_class_list = [0, 64, 66, 67, 41, 73]
class Yolo_Dect:
    def __init__(self):

        # load parameters
        weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param(
            '~image_topic', '/camera/color/image_raw')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.5')
        self.visualize = rospy.get_param('~visualize', 'True')
        self.depth_image_width=rospy.get_param('depth_image_width','')
        self.depth_image_height=rospy.get_param('depth_image_height','')

        # which device will be used
        if (rospy.get_param('/use_cpu', 'false')):
            self.device = 'cpu'
            self.is_half = False
        else:
            self.device = 'cuda'
            self.is_half = True
        rospy.loginfo("initial the Model of YOLOv8....")
        self.model = YOLO(weight_path)
        self.model.fuse()
        rospy.loginfo("YOLO model load success")
        self.model.conf = conf
        self.color_image = Image()
        self.getImageStatus = False
        self.tracker = DeepSort(max_age=20)
        # image subscribe
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback,
                                          queue_size=1, buff_size=52428800)
        while (not self.getImageStatus):
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)

    def image_callback(self, image):
        self.getImageStatus = True
        color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1)
        self.color_image = color_image
        color_image = color_image[:, :, ::-1]
        color_image = np.expand_dims(color_image, axis=0)
        color_image = color_image.transpose(0, 3, 1, 2)
        color_image = np.ascontiguousarray(color_image)
        img_torch = torch.from_numpy(color_image.astype(np.float32) / 255.0).to(self.device)
        img_torch = img_torch.half() if self.is_half else img_torch.float()  # 格式转换 uint8-> 浮点数
        if img_torch.ndimension() == 3:
            img_torch = img_torch.unsqueeze(0)        
        results = self.model.predict(img_torch, half=self.is_half, classes=interest_class_list, show=False, conf=0.7)

        self.dectshow(results, image.height, image.width)

        # cv2.waitKey(3)

    def dectshow(self, results, height, width):
        t_start = datetime.datetime.now()
        detections = []
        for data in results[0].boxes.data.tolist():
            class_id = int(data[-1])
            confidence = data[4]
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            detections.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        tracks = self.tracker.update_tracks(detections, frame=self.color_image)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = int(ltrb[0]), int(
                ltrb[1]), int(ltrb[2]), int(ltrb[3])
            rospy.loginfo(f"box position is {[xmin, ymin, xmax, ymax]}")
            rospy.loginfo(f"track id is {track_id}")                 
        t_end = datetime.datetime.now()
        rospy.loginfo(f"FPS : {1 / (t_end - t_start).total_seconds():.2f}")     
        if self.visualize :
            cv2.imshow('YOLOv8', self.frame)


def main():
    rospy.init_node('yolov8_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()


if __name__ == "__main__":

    main()
