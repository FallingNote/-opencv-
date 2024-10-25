import os
import cv2
import sys
from PIL import Image
import numpy as np


def getImageAndLabels(path):
    facesSamples = []
    ids = []
    #读取每个图片的名称并拼接为图片路径
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    #导入人脸检测模块
    face_detector = cv2.CascadeClassifier('D:/Develop/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    #遍历列表中的图片
    for imagePath in imagePaths:
        #以灰度图的形式打开图片
        PIL_image = Image.open(imagePath).convert('L')
        #将图像转化为灰度数组
        np_img = np.array(PIL_image,'uint8')
        #获取图片中的人脸部分
        faces = face_detector.detectMultiScale(np_img)
        #获取每张图片的id
        id = int(os.path.split(imagePath)[1].split('.')[0])
        for x,y,w,h in faces:
            #将每张图片的人脸区域保存到facesSamples
            facesSamples.append(np_img[y:y+h,x:x+w])
            ids.append(id)
    return facesSamples,ids


if __name__ == '__main__':
    #图片路径
    path = './data/jm/'
    #获取图像数组和id标签数组
    faces,ids = getImageAndLabels(path)
    #获取循环对象
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    recognizer.train(faces,np.array(ids))
    #保存训练数据
    recognizer.write('trainer/trainer.yml')