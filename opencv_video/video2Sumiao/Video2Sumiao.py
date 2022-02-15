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


class Video2Sumiao:
    def __init__(self, file):
        self.video_file = file
        last_index = self.video_file.rfind('/')
        self.root_path = self.video_file[0:last_index]
        self.tmp_file = self.root_path + '/tmp'
        self.res_video_file = self.root_path + '/res.avi'
        self.ori_dir = self.tmp_file + '/ori_pics'
        self.sumiao_dir = self.tmp_file + '/sumiao_pics'
        self.fps = 0
        self.width = 0
        self.height = 0
        if not os.path.exists(self.video_file):
            print("源视频文件不存在")
        if os.path.exists(self.tmp_file):
            shutil.rmtree(self.tmp_file)
        os.mkdir(self.tmp_file)
        if not os.path.exists(self.ori_dir):
            os.mkdir(self.ori_dir)
        if not os.path.exists(self.sumiao_dir):
            os.mkdir(self.sumiao_dir)

    def video_to_pics(self):
        cap = cv2.VideoCapture(self.video_file)
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
                    newPath = self.ori_dir + '/' + str(index) + ".jpeg"
                    cv2.imencode('.jpg', frame)[1].tofile(newPath)
            else:
                break

    def to_sumiao(self, calc_opt: int):
        file_list = os.listdir(self.ori_dir)
        for file in file_list:
            infile = self.ori_dir + '/' + file
            outfile = self.sumiao_dir + '/' + file
            if calc_opt == 1:
                self.__calc1__(infile, outfile)
            elif calc_opt == 2:
                self.__calc2__(infile, outfile)
            elif calc_opt == 3:
                self.__calc3__(infile, outfile)
            elif calc_opt == 4:
                self.__calc4__(infile, outfile)

    def __calc1__(self, infile, outfile):
        img = cv2.imread(infile, cv2.IMREAD_GRAYSCALE)
        img_edge = cv2.adaptiveThreshold(img, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY,
                                         blockSize=5,
                                         C=7)
        cv2.imwrite(outfile, img_edge)

    def __calc2__(self, infile, outfile):
        img = cv2.imread(infile, cv2.IMREAD_GRAYSCALE)
        GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r, b = cv2.threshold(GrayImage, 100, 255, 8)
        im = Image.fromarray(b)
        im.save(outfile)

    def __calc3__(self, infile, outfile):
        out1 = np.asarray(Image.open(infile).convert('L')).astype('float')
        depth = 10.
        grad = np.gradient(out1)
        grad_x, grad_y = grad
        grad_x = grad_x * depth / 100.
        grad_y = grad_y * depth / 100.
        A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)
        uni_x = grad_x / A
        uni_y = grad_y / A
        uni_z = 1. / A
        vec_el = np.pi / 2.2
        vec_az = np.pi / 4.
        dx = np.cos(vec_el) * np.cos(vec_az)
        dy = np.cos(vec_el) * np.sin(vec_az)
        dz = np.sin(vec_el)
        b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)
        b = b.clip(0, 255)
        im = Image.fromarray(b.astype('uint8'))
        im.save(outfile)

    '''
    来源于：https://www.yzlfxy.com/jiaocheng/python/366629.html
    转换为卡通图
    '''

    def __calc4__(self, infile, outfile):
        num_down = 2
        num_bilateral = 7
        img_rgb = cv2.imread(infile)
        img_color = img_rgb
        for _ in range(num_down):
            img_color = cv2.pyrDown(img_color)
        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
        for _ in range(num_down):
            img_color = cv2.pyrUp(img_color)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY,
                                         blockSize=9,
                                         C=2)
        height = img_rgb.shape[0]
        width = img_rgb.shape[1]
        img_color = cv2.resize(img_color, (width, height))
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        img_cartoon = cv2.bitwise_and(img_color, img_edge)
        cv2.imwrite(outfile, img_cartoon)

    def to_video(self):
        fourcc = VideoWriter_fourcc(*'MJPG')  # 支持jpg
        videoWriter = cv2.VideoWriter(self.res_video_file, fourcc, fps, (width, height))
        im_names = os.listdir(self.sumiao_dir)
        print(len(im_names))
        for im_name in range(len(im_names) - 2):
            string = self.sumiao_dir + '/' + str(im_name + 1) + '.jpeg'
            frame = cv2.imread(string)
            frame = cv2.resize(frame, (width, height))
            videoWriter.write(frame)
        videoWriter.release()

    def img_to_char(self, image_path, height):
        img = Image.open(image_path)
        img_width, img_height = img.size
        width = 3 * height * img_width // img_height
        img = img.resize((width, height), Image.ANTIALIAS)
        data = np.array(img.convert('L'))
        chars = "#RMNHQODBWGPZ*@$C&98?32I1>!:-;. "
        N = len(chars)
        # 计算每个字符的区间,//取整
        n = 256 // N
        # result是字符结果
        result = ''
        for i in range(height):
            for j in range(width):
                result += chars[data[i][j] // n]
            result += '\n'
        with open('img.txt', mode='w') as f:
            f.write(result)

    def transVideo(self):
        self.video_to_pics()
        self.to_sumiao(1)
        self.to_video()
        shutil.rmtree(self.tmp_file)


if __name__ == '__main__':
    Video2Sumiao('/Users/mac/Desktop/1644907451467651.mp4').transVideo()
