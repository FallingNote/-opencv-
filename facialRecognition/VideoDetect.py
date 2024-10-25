#导入opencv模块
import cv2

#人脸检测方法
def face_detect(img):
    #将图片尺寸进行转换
    img = cv2.resize(img, dsize=(400, 700))
    # 将图片进行灰度转换
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #调用opencv的分类器
    face_detect = cv2.CascadeClassifier('D:/Develop/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    #检测人脸图像
    face = face_detect.detectMultiScale(gray)
    #绘制矩形框
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
    cv2.imshow('result',img)

#读取摄像头
cap = cv2.VideoCapture('')


#等待键盘输入，单位ms
#设定只有输入q的时候退出
while True:
    flag,frame = cap.read()
    if not flag:
        break
    face_detect(frame)
    if ord('q') == cv2.waitKey(100):
        break
#释放内存
cv2.destroyAllWindows()
cap.release()