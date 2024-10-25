import cv2
import numpy as np
import os
#加载训练数据集文件
recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read('trainer/trainer.yml')
#准备识别的图片
img = cv2.imread('D:/Develop/code/imageEngineering/att_faces/s2/7.pgm')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#调用opencv的分类器
face_detect = cv2.CascadeClassifier('D:/Develop/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
#检测人脸图像
face = face_detect.detectMultiScale(gray)
#绘制矩形框
for x,y,w,h in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
    #人脸识别
    #获取标签id和置信度
    id,confidence = recognizer.predict(gray[y:y+h,x:x+w])
    print('标签id',id,'置信评分',confidence)
cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()