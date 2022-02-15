'''
   @author: HeQingsong
   @date: 2022-02-15 14:54
   @filename: Video2Sumiao.py
   @project: opencv_demo
   @python version: 3.7 by Anaconda
   @description: 
'''
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from cv2 import VideoWriter_fourcc

video_file = '/Users/mac/Desktop/1644907451467651.mp4'
tmp_file = '/Users/mac/Desktop/tmp'
res_video_file = '/Users/mac/Desktop/res.avi'
ori_dir = tmp_file + '/ori_pics'
sumiao_dir = tmp_file + '/sumiao_pics'
fps = 0
width = 0
height = 0


def init():
    if not os.path.exists(video_file):
        print("源视频文件不存在")
    if os.path.exists(tmp_file):
        shutil.rmtree(tmp_file)
    os.mkdir(tmp_file)
    if not os.path.exists(ori_dir):
        os.mkdir(ori_dir)
    if not os.path.exists(sumiao_dir):
        os.mkdir(sumiao_dir)


def video_to_pics():
    cap = cv2.VideoCapture(video_file)
    global fps, width, height
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('fps:{fps}, width:{width}, height:{height}'.format(fps=fps, width=width, height=height))

    index = 0
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                index += 1
                newPath = ori_dir + '/' + str(index) + ".jpeg"
                cv2.imencode('.jpg', frame)[1].tofile(newPath)
        else:
            break


def to_sumiao():
    file_list = os.listdir(ori_dir)
    for file in file_list:
        img = cv2.imread(ori_dir + '/' + file, cv2.IMREAD_GRAYSCALE)
        # 方法1 start
        img_edge = cv2.adaptiveThreshold(img, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY,
                                         blockSize=5,
                                         C=7)
        cv2.imwrite(sumiao_dir + '/' + file, img_edge)
        # 方法1 end

        # 方法2 start
        # GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # r, b = cv2.threshold(GrayImage, 100, 255, 8)
        # im = Image.fromarray(b)
        # im.save(sumiao_dir + '/' + file)
        # 方法2 end

        # 方法3 start
        # out1 = np.asarray(Image.open(ori_dir + '/' + file).convert('L')).astype('float')
        # depth = 10.
        # grad = np.gradient(out1)
        # grad_x, grad_y = grad
        # grad_x = grad_x * depth / 100.
        # grad_y = grad_y * depth / 100.
        # A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)
        # uni_x = grad_x / A
        # uni_y = grad_y / A
        # uni_z = 1. / A
        # vec_el = np.pi / 2.2
        # vec_az = np.pi / 4.
        # dx = np.cos(vec_el) * np.cos(vec_az)
        # dy = np.cos(vec_el) * np.sin(vec_az)
        # dz = np.sin(vec_el)
        # b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)
        # b = b.clip(0, 255)
        # im = Image.fromarray(b.astype('uint8'))
        # im.save(sumiao_dir + '/' + file)
        # 方法3 end


def to_video():
    fourcc = VideoWriter_fourcc(*'MJPG')  # 支持jpg
    videoWriter = cv2.VideoWriter(res_video_file, fourcc, fps, (width, height))
    im_names = os.listdir(sumiao_dir)
    print(len(im_names))
    for im_name in range(len(im_names) - 2):
        string = sumiao_dir + '/' + str(im_name + 1) + '.jpeg'
        frame = cv2.imread(string)
        frame = cv2.resize(frame, (width, height))
        videoWriter.write(frame)
    videoWriter.release()


if __name__ == '__main__':
    init()
    video_to_pics()
    to_sumiao()
    to_video()
