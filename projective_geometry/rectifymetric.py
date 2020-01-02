import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils import *
import sys
import os
if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    print  ('please input the your image path!!!')
    exit(0)
img = plt.imread(filename)
H,imgrect = metricAffine(img,nLinePairs=2)
plt.close('all')
fileparts = os.path.splitext(filename)
plt.imshow(imgrect, cmap = 'gray')
plt.axis('off')
savepath = fileparts[0]+ "_rect2"+fileparts[1]
plt.savefig(savepath,pad_inches=0.0,bbox_inches='tight')
plt.show()
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()