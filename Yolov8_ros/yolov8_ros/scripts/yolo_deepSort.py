#! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import torch
import rospy
import numpy as np
from ultralytics import YOLO
import time
import random
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes
from deep_sort_realtime.deepsort_tracker import DeepSort

interest_class_list = [0, 64, 66, 67, 41, 73]
class Yolo_Dect:
    def __init__(self):

        # load parameters
        weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param(
            '~image_topic', '/camera/color/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov8/BoundingBoxes')
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

        self.model = YOLO(weight_path)
        self.model.fuse()

        self.model.conf = conf
        self.color_image = Image()
        self.getImageStatus = False
        self.tracker = DeepSort(max_age=15)
        # Load class color
        self.classes_colors = {}

        # image subscribe
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback,
                                          queue_size=1, buff_size=52428800)

        # output publishers
        # self.position_pub = rospy.Publisher(
        #     pub_topic,  BoundingBoxes, queue_size=1)

        # self.image_pub = rospy.Publisher(
        #     '/yolov8/detection_image',  Image, queue_size=1)

        # if no image messages
        while (not self.getImageStatus):
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)

    def image_callback(self, image):

        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        self.getImageStatus = True
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1)

        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        results = self.model.predict(self.color_image, half=self.is_half, classes=interest_class_list, show=False, conf=0.7)

        self.dectshow(results, image.height, image.width)

        cv2.waitKey(3)

    def dectshow(self, results, height, width):
        t_start = time.time()
        self.frame = results[0].plot()
        # print(type(self.frame))
        # print(str(results[0].speed['inference']))
        fps = 1000.0/ results[0].speed['inference']
        cv2.putText(self.frame, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        detections = []
        for data in results[0].boxes.data.tolist():
            class_id = int(data[-1])
            confidence = data[4]
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            detections.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        tracks = self.tracker.update_tracks(detections, frame=self.frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(
                ltrb[1]), int(ltrb[2]), int(ltrb[3])
            rospy.loginfo(f"box position is {[xmin, ymin, xmax, ymax]}")
            # draw the bounding box and the track id
            cv2.rectangle(self.frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.rectangle(self.frame, (xmin, ymin - 20), (xmin + 20, ymin), (0, 255, 0), -1)
            cv2.putText(self.frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)                      
        t_end = time.time()
        fps = int(1.0 / (t_end - t_start))
        rospy.loginfo("FPS : %d", fps)
        # print(f"FPS : {fps}")        
        if self.visualize :
            cv2.imshow('YOLOv8', self.frame)


def main():
    rospy.loginfo("code start here!")
    rospy.init_node('yolov8_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()


if __name__ == "__main__":

    main()
