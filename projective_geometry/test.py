import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.pyplot import Line2D
import sys
import os
path = "tilef.jpg"
img = plt.imread(path)
plt.imshow(img, cmap = "gray")
plt.axis('image')
p = plt.ginput(2)
point_x = [(x[0]) for x in p]
point_y = [(y[1]) for y in p]
plt.plot(point_x,point_y,'r')
# plt.show()


