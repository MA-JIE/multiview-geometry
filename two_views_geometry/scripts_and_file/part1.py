import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
def draw_epiline(img1,img2,F):
    plt.imshow(img1,cmap='gray') 
    plt.axis('image')
    point_plot = plt.ginput(1)
    point_homo = np.array([point_plot[0][0],point_plot[0][1],1]).T
    epiline = F.dot(point_homo)
    x = np.linspace(0,640,50)
    y = -epiline[0] * x / epiline[1] - epiline[2]/ epiline[1]
    plt.imshow(img2,cmap='gray') 
    plt.plot(x,y,'r')
    plt.axis('off')
    plt.savefig('epipolar_img2.png',pad_inches=0.0,bbox_inches='tight')
    plt.show()
    key = cv2.waitKey(0)
    if key == 27: #press Q 
        cv2.destroyAllWindows()
    return epiline
def smallestsingularvector(F):#3x3
        u,s,vh = np.linalg.svd(F,full_matrices=1,compute_uv=1)
        vN = vh[vh.shape[0]-1,:]
        vN = vN.conj().T
        return vN
def compute_epipole(F):
    F_T = np.array(F).T
    e = smallestsingularvector(F_T)
    return e/e[2]
def get_matchs(img1,img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches , kp1, kp2
def draw_matches(img1,img2):
    matches, kp1, kp2 = get_matchs(img1,img2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()
def get_pixel_points(img1,img2):
    matches, kp1, kp2 = get_matchs(img1,img2)
    x1 = []
    x2 = []
    for i, val in enumerate(matches):
        if matches[i].distance < max(2*matches[0].distance,30.0):
            a = [kp1[val.queryIdx].pt[0],kp1[val.queryIdx].pt[1],1]
            x1.append(a)
            b = [kp2[val.trainIdx].pt[0],kp2[val.trainIdx].pt[1],1]
            x2.append(b)
    x1 = np.int32(x1)
    x2 = np.int32(x2)
    return x1,x2
def compute_F_RANSAC(img1,img2):
    x1,x2=get_pixel_points(img1,img2)
    if len(x1) < 8:
        raise RuntimeError('Fundamental matrix requires N >= 8 pts')
    else:
        F, mask = cv2.findFundamentalMat(x1, x2,cv2.FM_RANSAC)
    # U,S,V = np.linalg.svd(F)
    # S[2] = 0
    # F = np.dot(U,np.dot(np.diag(S),V))
    return F, mask 
def compute_fundamental(x1,x2):
    #get the first 8 points that have best matching
    # x1 = x1[:, 0:8]# array 3 x 8
    # x2 = x2[:, 0:8]
    A = np.zeros((8,9))
    for i in range(8):
        A[i] = [x2[0,i] * x1[0,i], x1[1,i] * x2[0,i], x2[0,i], x1[0,i] * x2[1,i], x1[1,i] * x2[1,i], x2[1,i], x1[0,i], x1[1,i], 1]
    #get the solution of at least square
    U, S, V = np.linalg.svd(A)
    F = np.array(V[-1]).reshape(3,3)
    # rank(F) = 2
    U,S,V = np.linalg.svd(F)# in python V =V'
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S),V))
    return F
def compute_F_8(img1,img2):
    x1,x2=get_pixel_points(img1,img2)
    x1 = np.array(x1).T#array 3 x n
    x2 = np.array(x2).T
    x1 = x1[:, 0:8]# array 3 x 8
    x2 = x2[:, 0:8]
    mean_x1 = np.mean(x1[:2], axis = 1)
    mean_x2 = np.mean(x2[:2], axis = 1)
    # x1_hat = np.zeros((2,8))
    # x2_hat = np.zeros((2,8))
    # x1_hat[0] = x1[0] - mean_x1[0]
    # x1_hat[1] = x1[1] - mean_x1[1]
    # x2_hat[0] = x2[0] - mean_x2[0]
    # x2_hat[1] = x2[1] - mean_x2[1]
    # d1_hat = np.mean([np.sqrt(np.square(x[0]+np.square([x[1]]))) for x in x1_hat.T]) 
    # d2_hat = np.mean([np.sqrt(np.square(x[0]+np.square([x[1]]))) for x in x2_hat.T])
    # S1 = np.sqrt(2) / d1_hat
    # S2 = np.sqrt(2) / d2_hat
    S1 = np.sqrt(2) / np.std(x1[:2])
    S2 = np.sqrt(2) / np.std(x2[:2])
    T1 = np.array([S1, 0, -S1 * mean_x1[0], 0, S1, -S1 * mean_x1[1],0,0,1 ]).reshape(3,3)
    T2 = np.array([S2, 0, -S2 * mean_x2[0], 0, S2, -S2 * mean_x2[1],0,0,1 ]).reshape(3,3)
    x1_ = np.dot(T1,x1)
    x2_ = np.dot(T2, x2)
    #compute fundamental matrix using normalized coordinates
    F = compute_fundamental(x1_,x2_) 
    # anti- normalized
    # F = np.dot(np.linalg.inv(T2), np.dot(F, T1))
    F = np.dot(T2.T, np.dot(F.T, T1))
    return F / F[2,2]
def question_1():
    # Load the two images
    img1 = cv2.imread('chapel00.png',cv2.IMREAD_GRAYSCALE)     # queryImage
    img2 = cv2.imread('chapel01.png',cv2.IMREAD_GRAYSCALE) # trainImage
    F=np.loadtxt('chapel.00.01.F')
    l = draw_epiline(img1,img2,F)
    e = compute_epipole(F)
    print("F_T .  e' = {}".format(e.dot(l)))
    print("fundamental matrix in the txt is \n {}".format(F / F[2,2]))
def question_2():
    img1 = cv2.imread('chapel00.png',cv2.IMREAD_GRAYSCALE)     # queryImage
    img2 = cv2.imread('chapel01.png',cv2.IMREAD_GRAYSCALE) # trainImage
    matches, kp1, kp2 = get_matchs(img1,img2)
    F_ransac , mask = compute_F_RANSAC(img1,img2)
    print("fundamental matrix using RANSAC is \n {}".format(F_ransac))
def question_3():
    img1 = cv2.imread('chapel00.png',cv2.IMREAD_GRAYSCALE)     # queryImage
    img2 = cv2.imread('chapel01.png',cv2.IMREAD_GRAYSCALE) # trainImage
    F8 =  compute_F_8(img1,img2)
    print("and fundamental matrix using 8 point is \n {}".format(F8))
if __name__ == '__main__':
    #----------question 1.1-------------
    question_1()
    #----------question 1.2.1 The RANSAC Algorithm------------
    question_2()
    #-----------8-Points Algorithm----------------------
    question_3()

