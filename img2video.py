import cv2
import os
import glob

fps = 10
size = (512, 512)
videowriter = cv2.VideoWriter("test.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
path = '/Users/xiao/Desktop/output/mountain_2_style_350000/frame*'
imgfiles = sorted(glob.glob(path))
print(imgfiles)
for i in imgfiles:
    print(i)
    img = cv2.imread(i)
    img = cv2.resize(img, (512, 512))
    videowriter.write(img)