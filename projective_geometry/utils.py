
from math import pi
import matplotlib.pyplot as plt
import scipy as sp
from numpy import linalg as la
import numpy as np
import cv2
import scipy.linalg 
#use the warpPerspective funtion to get the Affine image
def applyH(img,H):
    return cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
#using the svd to get the solutions of s
def nullspace(A, eps=1e-15):
    u, s, vh = sp.linalg.svd(A,full_matrices=1,compute_uv=1)
    # Pad so that we get the nullspace of a wide matrix. 
    N = A.shape[1]
    K = s.shape[0]
    if K < N:
        s[K+1:N] = 0
        s2 = np.zeros((N))
        s2[0:K] = s
        s = s2
    null_mask = (s <= eps)
    null_space = sp.compress(null_mask, vh, axis=0)
    return sp.transpose(null_space)
# get two pairs of lines by ploting on the image
def getlines_affine(nLinePairs):
    lines=[]
    for i in range(nLinePairs):
        #here 2 means we need two pairs of lines
        for j in range(2):
            points_plot = plt.ginput(2)
            point_homo = [(x[0],x[1],1) for x in points_plot]
            point_x = [(x[0]) for x in points_plot]
            point_y = [(y[1]) for y in points_plot]
            plt.plot(point_x,point_y,'r')
            lines.append(np.cross(point_homo[0],point_homo[1]))
    return lines
# get two pairs of lines by ploting on the image, what we should pay atteneion to is 
#the second pairs lines that not parallel to the first set
def getlines_metric(nLinePairs):
    lines=[]
    for i in range(nLinePairs):
        if i ==1:
            plt.suptitle('Please select the first pairs lines')
        if i ==1:
            plt.suptitle('Please select the second pairs lines that not parallel to the first set')
        for j in range(2):
            points_plot = plt.ginput(2)
            point_homo = [(x[0],x[1],1) for x in points_plot]
            point_x = [(x[0]) for x in points_plot]
            point_y = [(y[1]) for y in points_plot]
            if j ==0:
                plt.plot(point_x,point_y,'r')
            else:
                plt.plot(point_x,point_y,'g')
            lines.append(np.cross(point_homo[0],point_homo[1]))
    return lines
# return smallest singular vector of A (or the nullspace if A is 2x3)
def smallestSingularVector(A):
    if A.shape[0] == 2 and A.shape[1] == 3:
        return nullspace(A)
    elif(A.shape[0] > 2):
        u,s,vh = np.linalg.svd(A,full_matrices=1,compute_uv=1)
        vN = vh[vh.shape[0]-1,:]
        vN = vN.conj().T
        return vN
    else:
        raise Exception("bad shape of A: %d %d" % (A.shape[0], A.shape[1]))
#to get the H(Affine) matrix
def rectifyAffine(im, nLinePairs):
    plt.imshow(im,cmap='gray') 
    plt.axis('image')
    lines = getlines_affine(nLinePairs)
    #normalized the points
    x1 = np.cross(lines[0],lines[1])
    x1 = x1/x1[2]
    x2 = np.cross(lines[2],lines[3])
    x2 = x2/x2[2]
    #calculate the vanishing points
    l_vanish = np.cross(x1,x2)
    l_vanish = l_vanish/l_vanish[2]
    #create the H matrix using the method in the slides
    H = [[1,0,0],[0,1,0],[l_vanish[0],l_vanish[1],l_vanish[2]]]
    H = np.array(H)
    img_rect = applyH(im,H)
    return H ,img_rect
#to get the H(metric) matrix
def metricAffine(img, nLinePairs):
    plt.imshow(img,cmap='gray') 
    plt.axis('image')
    lines = getlines_metric(nLinePairs)
    A = np.zeros((2, 3))
    lines = np.array(lines).reshape(2,6)# [[l1,l2,l3,m1,m2,m3],[l_1,l_2,l_3,m_1,m_2,m_3]]
    for i, val in enumerate(lines):
        A[i][0] = val[0] * val[3] # l1 * m1
        A[i][1] = val[0] * val[4] + val[1] * val[3] #l1 * m2 + m1 * l2
        A[i][2] = val[1] * val[4] # l2*m2
        #svd
    s = smallestSingularVector(A) 
    S = np.array([[s[0], s[1]], [s[1], s[2]]]).reshape((2,2))
    #normalized
    S = S / max(s[0],s[2])
    U2, sigma2, V2 = la.svd(S)
    #here we used the cholesky method, because we can not get the good performance
    #using the method written in the slides!
    A = np.linalg.cholesky(S)
    H = np.eye(3)
    H[0:2,0:2] = A
    Hinv = np.linalg.inv(H)
    #check it so we print it
    print("H:")
    print(H)
    print('A:')
    print(A)
    print('S:')
    print(S)
    img_rect = applyH(img,Hinv)
    return Hinv, img_rect

        
        
    
    
    
