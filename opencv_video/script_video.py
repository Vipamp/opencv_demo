#script_draw
from PIL import Image
import numpy as np
import cv2
import os
video_path=input("video_path?")
save_path=input("save_path?")
save_video=input("save_video?")
def imgs2video(path,save_video,lenght):
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0') 
    post_video=cv2.VideoWriter(save_path, fourcc, 15,(1280,720))
    for i in range(lenght):
        print(path+str(i)+".jpg")
        img=cv2.imread(path+str(i)+".jpg")
        post_video.write(img)
        
        del img
 
    post_video.release()
    return "done"
def save_imgs(imgs,save_path):
    for img_i in range(len(imgs)):   
        img = Image.fromarray(imgs[img_i])
        img.save(save_path+str(img_i)+".jpg")
        print("saving imgs:",img_i+1,"/",len(imgs))
        
def video2imgs(video_name):
    img_list=[]
    cap = cv2.VideoCapture(video_name)
    while cap.isOpened():
        ret,frame=cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img=cv2.resize(gray,(1280,720),interpolation=cv2.INTER_AREA)
            img_list.append(img)
        else:
            break
    cap.release()
    return img_list
def array2script(array):
    depth = 10.                     
    grad = np.gradient(array)            
    grad_x, grad_y =grad               
    grad_x = grad_x*depth/100. 
    grad_y = grad_y*depth/100. 
    A = np.sqrt(grad_x**2 + grad_y**2 + 1.) 
    uni_x = grad_x/A 
    uni_y = grad_y/A
    uni_z = 1./A  
    vec_el = np.pi/2.2             
    vec_az = np.pi/4.                   
    dx = np.cos(vec_el)*np.cos(vec_az)  
    dy = np.cos(vec_el)*np.sin(vec_az)  
    dz = np.sin(vec_el)             
    b = 255*(dx*uni_x + dy*uni_y + dz*uni_z)   
    b = b.clip(0,255)  
    img = Image.fromarray(b.astype('uint8'))
    img=np.asarray(img)
    return img
def imgs2scripts(img_list):
    script_list=[]
    for i in range(len(img_list)):
        script_list.append(array2script(img_list[i]))
        print("have loaded:",i)

            
    return script_list
imgs=video2imgs(video_path)
print(imgs[0])
scripts=imgs2scripts(imgs)
print(scripts[0].shape)
lenght=len(scripts)
save_imgs(scripts,save_path)
imgs2video(save_path,save_video,lenght)
