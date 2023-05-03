import numpy as np
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt

import cv2

# 1. Load the two temple images and the points from data/some_corresp.npz
# img1 = cv2.cvtColor(cv2.imread('../data/im1.png'), cv2.COLOR_BGR2GRAY).astype(np.float32)
# img2 = cv2.cvtColor(cv2.imread('../data/im2.png'), cv2.COLOR_BGR2GRAY).astype(np.float32)

# 2. Run eight_point to compute F
import numpy as np
from helper import displayEpipolarF, epipolarMatchGUI, camera2
from submission import eight_point, epipolar_correspondences, essential_matrix, hom2cart, cart2hom

def triangulate(P1, pts1, P2, pts2):
    N = pts1.shape[0]
    pts3d = np.zeros((N, 3))
    for i in range(N):
        A = np.zeros((4, 4))
        A[0] = pts1[i,0]*P1[2] - P1[0]
        A[1] = pts1[i,1]*P1[2] - P1[1]
        A[2] = pts2[i,0]*P2[2] - P2[0]
        A[3] = pts2[i,1]*P2[2] - P2[1]
        _, _, vt = np.linalg.svd(A)
        pts3d[i] = vt[-1,:3] / vt[-1,3]
    return pts3d
def make_P(K, R, t):
    # K: 3x3 camera intrinsic matrix
    # R: 3x3 rotation matrix
    # t: 3x1 translation vector

    # create 3x4 projection matrix
    P = np.zeros((3, 4))
    P[:, :3] = np.dot(K, R)
    P[:, 3] = np.dot(K, t).T

    return P

# Load point correspondences
data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']

# Load images
img1 = cv2.imread('../data/im1.png')
img2 = cv2.imread('../data/im2.png')

# Estimate fundamental matrix
M = max(img1.shape)
F = eight_point(pts1, pts2, M)


pts2 = epipolar_correspondences(img1, img2, F, pts1)

# Load the intrinsic camera matrices
intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']


# Compute the Essential matrix using the function
E = essential_matrix(F, K1, K2)
print(E)

M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P1 = K1 @ M1
Ps = camera2(E)
# print(Ps)

# # 遍历每个P矩阵
# for i, P in enumerate(Ps):
#     # 将4x4的P矩阵转换为3x4的相机矩阵
#     P = P[:3, :4]
#     # 判断矩阵P是否为合法解
#     if np.linalg.det(P[:, :3]) > 0:
#         P2 = P
#         break

max_in_front = 0
P2_best = None
for i in range(4):
    in_front = 0
    P2 = Ps[:,:,i]
    pts3d = triangulate(P1, pts1, P2, pts2)
    for j in range(len(pts3d)):
        if (pts3d[j][2] > 0):
            in_front += 1
    if in_front > max_in_front:
       max_in_front = in_front
       posi = i
       
P2_best = Ps[:,:,posi]
pts4d = triangulate(P1, pts1, P2_best, pts2)


# calculate P2
# R, t = P2[:, :3], P2[:, 3]
# P2 = make_P(K2, R, t)

# print(P1)

# perform triangulation
# pts4d = triangulate(P1, pts1, P2, pts2)
# print(pts4d)


# calculate re-projection error
pts3d = hom2cart(pts4d)
pts1_proj_hom = np.dot(pts4d, P1)
eps = 1e-12
mask = np.abs(pts1_proj_hom[-1]) >= eps
pts1_proj_hom[:, mask] /= pts1_proj_hom[-1, mask]
pts1_proj_hom[:, ~mask] /= eps
pts1_proj = hom2cart(pts1_proj_hom)
reproj_error = np.mean(np.linalg.norm(pts1[:, :2] - pts1_proj[:, :2], axis=1))
print('Reprojection error:', reproj_error)






# F = eight_point(pts1, pts2, M)
# print(F)

# Display epipolar lines
# displayEpipolarF(img1, img2, F)
# epipolarMatchGUI(img1, img2, F)



# 3. Load points in image 1 from data/temple_coords.npz

# 4. Run epipolar_correspondences to get points in image 2

# 5. Compute the camera projection matrix P1

# 6. Use camera2 to get 4 camera projection matrices P2

# 7. Run triangulate using the projection matrices

# 8. Figure out the correct P2

# 9. Scatter plot the correct 3D points

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
