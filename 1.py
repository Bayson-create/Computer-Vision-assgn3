import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd
from scipy.optimize import minimize
import submission as sub
from helper import displayEpipolarF, epipolarMatchGUI, camera2
from submission import eight_point, epipolar_correspondences, essential_matrix, hom2cart, cart2hom
from mpl_toolkits.mplot3d import Axes3D
import cv2

def triangulate(P1, pts1, P2, pts2):
    N = pts1.shape[0]
    pts3d = []
    for i in range(N):
        A = np.zeros((4, 4))
        A[0] = pts1[i,1]*P1[2] - P1[1]
        A[1] = P1[0] - pts1[i,0]*P1[2]
        A[2] = pts2[i,1]*P2[2] - P2[1]
        A[3] = P2[0] - pts2[i,0]*P2[2]
        u, s, vt = np.linalg.svd(A)
        pt3d = np.array(vt[len(s)-1])
        pt3d = pt3d / pt3d[3]
        pts3d.append(pt3d)
    pts3d = np.array(pts3d)
    return pts3d

# Load data
corresp = np.load('../data/some_corresp.npz')
pts1 = corresp['pts1']
pts2 = corresp['pts2']
# temple_coords = np.load('../data/temple_coords.npz')
# pts1 = temple_coords['pts1']
# y1 = temple_coords['y1']
intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']

# Load images
img1 = cv2.imread('../data/im1.png')
img2 = cv2.imread('../data/im2.png')
M = max(img1.shape)

# 1. Compute fundamental matrix
F = eight_point(pts1, pts2, M) #有误，已修改

#load the points in image 1 contained in data/temple_coords.npz
temple_coords = np.load('../data/temple_coords.npz')
pts1 = temple_coords['pts1']
# y1 = temple_coords['y1']

# 2. Get epipolar correspondences
pts2 = epipolar_correspondences(img1, img2, F, pts1)


# 3. Compute essential matrix
E = essential_matrix(F, K1, K2)


# 4. Compute camera projection matrices
P1 = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
P2s = camera2(E)
print(P2s)

# 5. Triangulate points with all four P2 candidates
I = np.array([[1,0,0,0], [0,1,0,0],[0,0,1,0]])
P1 = np.dot(K1, I)

max_in_front = 0
P2_best = None
# for P2 in P2s:
for i in range(4):
    in_front = 0
    # P2 = P2[:3, :4]
    # pts3d_best = triangulate(K1.dot(P1), pts1, K2.dot(P2), pts2)
    P2 = P2s[:,:,i]
    pts3d_best = triangulate(P1, pts1, P2, pts2)
    # in_front = np.sum(pts3d_best[:, 2] > 0)
    for j in range(len(pts3d_best)):
        if (pts3d_best[j][2] > 0):
            in_front += 1
    if in_front > max_in_front:
       max_in_front = in_front
       posi = i
       
P2_best = P2s[:,:,posi]
pts3d_best = triangulate(P1, pts1, P2_best, pts2)

print()
print(pts3d_best)

# pts3d_best = None
# max_in_front = 0
# for i in range(4):
#     P2 = P2s[:, :, i]
#     pts4d = triangulate(P1, pts1, P2, pts2)
#     in_front = np.sum(pts4d[:, -1] > 0)
#     if in_front > max_in_front:
#         max_in_front = in_front
#         pts3d_best = pts4d
#         P2_best = P2

# 6. Plot 3D points
fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')
# ax.scatter(pts3d_best[:, 0], pts3d_best[:, 1], pts3d_best[:, 2])
for i in range(len(pts3d_best)):
    ax.scatter(pts3d_best[i][2], pts3d_best[i][0], pts3d_best[i][1], color='blue', s=10)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
plt.show()

print(P2_best)
# 7. Save extrinsic parameters
R1, t1 = np.eye(3), np.zeros((3, 1))
R2, t2 = P2_best[:, :3], P2_best[:, -1:]
np.savez('../data/extrinsics.npz', R1=R1, R2=R2, t1=t1, t2=t2)
