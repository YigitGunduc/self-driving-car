import numpy as np
from PIL import ImageGrab
import cv2
from matplotlib import pyplot as plt

def grabscreen():
    # Reading image
    img = np.array(ImageGrab.grab(bbox=(0,40,800,680)))
    #converting image to gray scale
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #resizeing image
    img = cv2.resize(img, (128, 128))

    return np.array(img) / 255


