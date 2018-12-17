import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

path = 'Detection/data/Image/Ai_Sugiyama_0001.jpg'
im = Image.open(path)
im = np.array(im)

cv2.imshow("dsa", im)