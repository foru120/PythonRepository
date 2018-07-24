import os
import PIL
import cv2
import numpy as np
from PIL import Image

def data_augmentation():
    for root, dirs, files in os.walk("./png"):
        for file in files:
            imagePath=os.path.join(root, file)
            c_png = Image.open(imagePath)
            c_png_fl = c_png.transpose(Image.FLIP_LEFT_RIGHT)
            c_png_ft = c_png.transpose(Image.FLIP_TOP_BOTTOM)
            c_png_r90 = c_png.transpose(Image.ROTATE_90)
            c_png_r180 = c_png.transpose(Image.ROTATE_180)
            c_png_r270 = c_png.transpose(Image.ROTATE_270)
            pre=imagePath[:-4]
            c_png_fl.save(pre+ "_fl.png")
            c_png_ft.save(pre + "_ft.png")
            c_png_r90.save(pre + "_r90.png")
            c_png_r180.save(pre + "_r180.png")
            c_png_r270.save(pre + "_r270.png")