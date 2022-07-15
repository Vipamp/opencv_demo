'''
   @author: HeQingsong
   @date: 2022-02-15 23:51
   @filename: videotest.py
   @project: opencv_demo
   @python version: 3.7 by Anaconda
   @description: 
'''
from moviepy.editor import *


def from_video_get_mp3():
    video = VideoFileClip('/Users/mac/Desktop/1644907451467651.mp4')
    audio = video.audio
    audio.write_audiofile('/Users/mac/Desktop/test.mp3')


def add_mp3(avifile,audiofile):
    video_clip = VideoFileClip(avifile)
    audio_clip = AudioFileClip(audiofile)
    video_clip2 = video_clip.set_audio(audio_clip)
    video_clip2.write_videofile('/Users/mac/Desktop/bws_audio.mp4')


if __name__ == '__main__':
    add_mp3('/Users/mac/Desktop/res.mp4','/Users/mac/Desktop/test.mp3')
