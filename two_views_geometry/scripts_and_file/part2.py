import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def show_disparity():
    imgL = cv2.imread('KITTI_Left.png',cv2.IMREAD_GRAYSCALE)     
    imgR = cv2.imread('KITTI_Right.png',cv2.IMREAD_GRAYSCALE) 
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
    disparity = stereo.compute(imgL,imgR)
#     disparity.convertTo(disparity, CV_32F, 1.0 / 16.0)
    disparity = disparity / 16.0
#     print(disparity[5][25])#-16
    plt.imshow(disparity,'gray')
    plt.show()
    key = cv2.waitKey(0)
    if key == 27: #press Q 
        cv2.destroyAllWindows()
def  reconstruction(imgL, imgR):
    #compute diaparity map
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL,imgR)
    pointscloud = []
    f = 7.215377e+02
    cx = 6.095593e+02
    cy = 1.728540e+02
    Tx = 0.3875744
    h,w = imgL.shape[:2]
    disparity = np.array(disparity)
    for v in range(h):
        for u in range(w):
            if disparity[v][u] <= 0.0 or disparity[v][u]>=96.0:
                continue
            x = (u - cx) / f
            y  = (v - cy) / f 
            depth = f * Tx / disparity[v][u] 
            pointscloud.append(x * depth)
            pointscloud.append(y * depth)
            pointscloud.append(depth)
    pointscloud = np.array(pointscloud).reshape(-1,3)
    print(pointscloud)
    x1 = pointscloud[:,0]
    y1 = pointscloud[:,1]
    z1 = pointscloud[:,2]
    fig = plt.figure()
    ax =Axes3D(fig)
    ax.scatter(x1,y1,z1,c='b',marker='.',s=2,linewidth=0,alpha=1,cmap='spectral')
    ax.axis('scaled')          
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    key = cv2.waitKey(0)
    if key == 27: #press Q 
        cv2.destroyAllWindows()
# def main():
#     img1 = cv2.imread('KITTI_Left.png',cv2.IMREAD_GRAYSCALE)     
#     img2 = cv2.imread('KITTI_Right.png',cv2.IMREAD_GRAYSCALE) 
#     reconstruction(img1,img2)
    #show_disparity()
# img1 = cv2.imread('KITTI_Left.png',cv2.IMREAD_GRAYSCALE)     
# img2 = cv2.imread('KITTI_Right.png',cv2.IMREAD_GRAYSCALE) 
if __name__ =='__main__':  
        img1 = cv2.imread('KITTI_Left.png',cv2.IMREAD_GRAYSCALE)     
        img2 = cv2.imread('KITTI_Right.png',cv2.IMREAD_GRAYSCALE)
        #show_disparity()
        reconstruction(img1,img2)
