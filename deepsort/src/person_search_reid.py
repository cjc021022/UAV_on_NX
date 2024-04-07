#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
import warnings
import argparse
import rospy
#import onnxruntime as ort

import sys
sys.path.insert(0,"/home/lenovo/some_test/yolov8_deepsort_ros_ws/src/deepsort/utils/*")
#from datasets import LoadStreams, LoadImages

from utils.datasets import LoadStreams, LoadImages
from utils.draw import draw_boxes, draw_person
from utils.general import check_img_size
from utils.torch_utils import time_synchronized
from person_detect_yolov5 import Person_detect
from deep_sort import build_tracker
from utils.parser import get_config
from utils.log import get_logger
from utils.torch_utils import select_device, load_classifier, time_synchronized
from sklearn.metrics.pairwise import cosine_similarity
#from deep_sort.deep_sort import DeepSort

from sensor_msgs.msg import Image
from deepsort.msg import  BoundingBoxes , BoundingBox

from std_msgs.msg import Int16,Bool
    
class yolo_reid():
    def __init__(self, cfg):
        # self.args = args
        # self.video_path = path
                
        self.device= rospy.get_param('~device')
        self.sort=rospy.get_param('~sort')
        self.display= rospy.get_param('~display')
        self.frame_interval = rospy.get_param('~cpu_or_gpu')
        self.query = rospy.get_param('~query_features')
        self.names = rospy.get_param('~name')
        self.img_size = rospy.get_param('~img_size')
        self.image_topic = rospy.get_param('~image_topic')
        self.detect_result_topic = rospy.get_param('~detect_result_topic')
        self.track_id=0

        # self.device= 'cuda:0'
        # self.sort=True
        # self.config_deepsort = True
        # self.frame_interval = True
        # self.query = "/home/lenovo/some_test/yolov8_deepsort_ros_ws/src/deepsort/src/fast_reid/query/query_features.npy"
        # self.names = "/home/lenovo/some_test/yolov8_deepsort_ros_ws/src/deepsort/src/fast_reid/query/names.npy"
        # self.img_size = 1080
        # self.image_topic = "/camera/color/image_raw"
        # self.detect_result_topic = "/yolov8/BoundingBoxes"

        self.bbox_xywh=[]
        self.cls_conf=[]
        self.xy=[]
        self.stat_track=Bool()
        self.stat_track.data=False
        use_cuda = self.device and torch.cuda.is_available()

        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        # Person_detect行人检测类
        #self.person_detect = Person_detect(self.args, self.video_path)
        # deepsort 类

        self.deepsort = build_tracker(cfg, self.sort, use_cuda=use_cuda)

        imgsz = check_img_size(self.img_size, s=32)  # self.model.stride.max())  # check img_size
        #self.dataset = LoadImages(self.video_path, img_size=imgsz) #加载文件夹里面的图片或者视频
        self.query_feat = np.load(self.query)
        self.names = np.load(self.names)




    def init_parse(self):
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.orginal_image_callback, queue_size=10)
        self.image_sub = rospy.Subscriber(self.detect_result_topic,BoundingBoxes,self.detect_results_callback,queue_size=10)
        self.track_person_id=rospy.Subscriber("/track_person_id",Int16,self.get_track_id,queue_size=10)
        self.pub_track_box= rospy.Publisher("/track_box", BoundingBoxes, queue_size=10)
        self.pub_start_track=rospy.Publisher("/start_track",Bool,queue_size=10)
        self.send_track=rospy.Timer(rospy.Duration(0.2), callback=self.send_start_track)
        
        
    def send_start_track(self,event):
        self.pub_start_track.publish(self.stat_track)

    def get_track_id(self,id):
        self.track_id=id.data
    
    def orginal_image_callback(self,image):
        self.ori_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1)
        self.ori_image = cv2.cvtColor(self.ori_image, cv2.COLOR_BGR2RGB)


    
    def detect_results_callback(self,results):
        self.bbox_xywh=[]
        self.cls_conf=[]
        self.xy=[]
        for i,boundingbox in enumerate(results.bounding_boxes):
            self.bbox_xywh+=[[boundingbox.xywh[0],boundingbox.xywh[1],boundingbox.xywh[2],boundingbox.xywh[3]]]
            self.cls_conf+=[boundingbox.probability]
            # self.xy+=[[torch.Tensor(boundingbox.xmin),torch.Tensor(boundingbox.ymin),torch.Tensor(boundingbox.xmax),torch.Tensor(boundingbox.ymax)]] 
            self.xy+=[[boundingbox.xmin,boundingbox.ymin,boundingbox.xmax,boundingbox.ymax]] 
        # if(len(results.bounding_boxes)!=0):       
        # #     print("bbox_xywh:")
        # #     print(self.bbox_xywh)
        # #     print("cls_conf:")
        # #     print(self.cls_conf)
        #      print(len(self.xy))
        if(len(results.bounding_boxes)>0):
            self.bbox_xywh = np.array(self.bbox_xywh)
            # print(len(self.cls_conf))

            outputs, features = self.deepsort.update(self.bbox_xywh,self.cls_conf,self.ori_image)
            person_cossim = cosine_similarity(features, self.query_feat)
            max_idx = np.argmax(person_cossim, axis=1)
            maximum = np.max(person_cossim, axis=1)
            max_idx[maximum < 0.6] = -1
            score = maximum
            reid_results = max_idx
            draw_person(self.ori_image, self.xy, reid_results, self.names)


            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im ,track_bounding_box= draw_boxes(self.ori_image, bbox_xyxy, identities,self.track_id)
                track_boxes=BoundingBoxes()
                track_box=BoundingBox()   
                if(self.track_id!=0):
            
                    track_box.xmin=np.int64(track_bounding_box[0])
                    track_box.ymin=np.int64(track_bounding_box[1])
                    track_box.xmax=np.int64(track_bounding_box[2])
                    track_box.ymax=np.int64(track_bounding_box[3])
                    track_box.xywh.append(np.int64((track_bounding_box[0]+track_bounding_box[2])/2))
                    track_box.xywh.append(np.int64((track_bounding_box[1]+track_bounding_box[3])/2))
                    track_box.xywh.append(np.int64(track_bounding_box[2]-track_bounding_box[0]))
                    track_box.xywh.append(np.int64(track_bounding_box[3]-track_bounding_box[1]))
                                                                                            
                    track_boxes.bounding_boxes.append(track_box)
                    self.stat_track.data=True
                    # print(track_boxes)
                    
                self.pub_track_box.publish(track_boxes)
            
                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

            
                if self.display:
                    cv2.imshow("test",self.ori_image)
                    cv2.waitKey(2)

            if(self.track_id==0):
                self.stat_track.data=False


            # results.append((idx_frame - 1, bbox_tlwh, identities))
        # print("yolo+deepsort:", time_synchronized() - t1)

        # if self.display:
        #     cv2.imshow("test", self.ori_image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break


    # def deep_sort(self):
    #     idx_frame = 0
    #     results = []
    #     for video_path, img, ori_img, vid_cap in self.dataset:
    #         #
    #         idx_frame += 1
    #         # print('aaaaaaaa', video_path, img.shape, im0s.shape, vid_cap)
    #         t1 = time_synchronized()

    #         # yolo detection
    #         bbox_xywh, cls_conf, cls_ids, xy = self.person_detect.detect(video_path, img, ori_img, vid_cap)

    #         # do tracking  # features:reid模型输出512dim特征
    #         outputs, features = self.deepsort.update(bbox_xywh, cls_conf, ori_img)
    #         print(len(outputs), len(bbox_xywh), features.shape)

    #         person_cossim = cosine_similarity(features, self.query_feat)
    #         max_idx = np.argmax(person_cossim, axis=1)
    #         maximum = np.max(person_cossim, axis=1)
    #         max_idx[maximum < 0.6] = -1
    #         score = maximum
    #         reid_results = max_idx
    #         draw_person(ori_img, xy, reid_results, self.names)  # draw_person name



    #         # print(features.shape, self.query_feat.shape, person_cossim.shape, features[1].shape)

    #         if len(outputs) > 0:
    #             bbox_tlwh = []
    #             bbox_xyxy = outputs[:, :4]
    #             identities = outputs[:, -1]
    #             ori_im = draw_boxes(ori_img, bbox_xyxy, identities)

    #             for bb_xyxy in bbox_xyxy:
    #                 bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

    #             # results.append((idx_frame - 1, bbox_tlwh, identities))
    #         # print("yolo+deepsort:", time_synchronized() - t1)

    #         if self.args.display:
    #             cv2.imshow("test", ori_img)
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 break



if __name__ == '__main__':
    rospy.init_node("deepsort", anonymous=True)
    cfg = get_config()

    config_deepsort = '/home/lenovo/some_test/yolov8_deepsort_ros_ws/src/deepsort/configs/deep_sort.yaml'
    cfg.merge_from_file(config_deepsort)
    yolo_reid = yolo_reid(cfg)
    # print("aaa")
    yolo_reid.init_parse()   
    rospy.spin()
    # with torch.no_grad():
    #     yolo_reid.deep_sort()
