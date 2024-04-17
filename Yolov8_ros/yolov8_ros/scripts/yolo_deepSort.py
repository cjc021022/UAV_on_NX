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
from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes
from deep_sort_realtime.deepsort_tracker import DeepSort

interest_class_list = [0, 64, 66, 67, 41, 73]
interest_class_dict = {
    0  : 'person',
    64 : 'mouse',
    67 : 'cell phone',
    66 : 'keyboard',
    41 : 'cup',
    73 : ' book'
}

def compute_center(box):
    x, y, w, h = box
    return int((x + w / 2)), int((y + h / 2))

def compute_distance(box1, box2):
    x1, y1 = compute_center(box1)
    x2, y2 = compute_center(box2)
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
class Yolo_Dect:
    def __init__(self):

        # load parameters
        weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param(
            '~image_topic', '/camera/color/image_raw')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.5')
        self.depth_image_width=rospy.get_param('depth_image_width','')
        self.depth_image_height=rospy.get_param('depth_image_height','')
        pub_topic = rospy.get_param('~pub_topic', '')
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
        self.tracker = DeepSort(max_age=50)
        # image subscribe
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback,
                                          queue_size=1, buff_size=52428800)
        self.position_pub = rospy.Publisher(pub_topic, BoundingBoxes, queue_size=1)        
        while (not self.getImageStatus):
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)

    def image_callback(self, image):
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
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

    def dectshow(self, results, height, width):
        t_start = datetime.datetime.now()
        detections = []
        for data in results[0].boxes.data.tolist():
            class_id = int(data[-1])
            confidence = data[4]
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            detections.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        tracks = self.tracker.update_tracks(detections, frame=self.color_image)
        if tracks is not None:
            self.match_detection_and_tracks(detections, tracks)
        rospy.loginfo(f"box position is {[xmin, ymin, xmax, ymax]}"))                 
        t_end = datetime.datetime.now()
        rospy.loginfo(f"FPS : {1 / (t_end - t_start).total_seconds():.2f}")     

    def match_detection_and_tracks(self, detections, tracks):
        matches = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            one_box = BoundingBox()
            track_box = track.to_tlwh()
            one_box.track_id = track.track_id
            one_box.x_center, one_box.y_center = compute_center(track_box)
            one_box.xmin, one_box.ymin, one_box.xmax, one_box.ymax = int(track_box[0]), int(track_box[1]), int(track_box[0]+track_box[2]), int(track_box[1]+track_box[3])
            if detections is None:
                one_box.cls = 'Unknow'
                one_box.confidence = -1.0
                self.boundingBoxes.bounding_boxe.append(one_box)
                return 
            for detection in detections:
                det_box = detection[0]  # xywh
                min_distance = float('inf')
                distance = compute_distance(det_box, track_box)
                if distance < min_distance:
                    min_distance = distance
                    matched_detect = detection
            one_box.confidence = matched_detect[1]
            one_box.cls = interest_class_dict[matched_detect[2]]
            self.boundingBoxes.bounding_boxe.append(one_box)
        return matches

def main():
    rospy.init_node('yolov8_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()


if __name__ == "__main__":

    main()
