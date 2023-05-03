"""
Homework 5
Submission Functions
"""

# import packages here

import numpy as np
from scipy.signal import convolve2d
from numpy.linalg import svd, inv, qr
from helper import refineF, displayEpipolarF
from scipy.spatial.distance import cdist
import cv2
from scipy import ndimage
from scipy.optimize import least_squares
from helper import refineF

def cart2hom(pts):
    """
    Transforms from cartesian to homogeneous coordinates.

    Parameters:
        pts: numpy array of shape (N, 2) with cartesian coordinates.

    Returns:
        pts_hom: numpy array of shape (N, 3) with homogeneous coordinates.
    """
    N = pts.shape[0]
    pts_hom = np.hstack([pts, np.ones((N, 1))])
    return pts_hom


def hom2cart(pts_hom):
    """
    Transforms from homogeneous to cartesian coordinates.

    Parameters:
        pts_hom: numpy array of shape (N, 3) with homogeneous coordinates.

    Returns:
        pts: numpy array of shape (N, 2) with cartesian coordinates.
    """
    N = pts_hom.shape[0]
    pts = pts_hom[:, :2] / pts_hom[:, 2].reshape(N, 1)
    return pts


def normalize(pts):
    # Shift points to the origin
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid

    # Scale points to have mean distance sqrt(2) from the origin
    mean_dist = np.mean(np.sqrt(np.sum(pts_centered**2, axis=1)))
    scale = np.sqrt(2) / mean_dist
    pts_scaled = pts_centered * scale

    # Construct the transformation matrix
    T = np.array([
        [scale, 0, -scale*centroid[0]],
        [0, scale, -scale*centroid[1]],
        [0, 0, 1]
    ])

    return pts_scaled, T


"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    # replace pass by your implementation

    # Normalize points
    # pts1_norm, T1 = normalize(pts1)
    # pts2_norm, T2 = normalize(pts2)

    # # Construct the A matrix
    # A = np.zeros((pts1.shape[0], 9))
    # for i in range(pts1.shape[0]):
    #     x, y = pts1_norm[i]
    #     u, v = pts2_norm[i]
    #     A[i] = [x*u, y*u, u, x*v, y*v, v, x, y, 1]

    # # Solve the linear system using SVD
    # U, D, V = np.linalg.svd(A)
    # F = V[-1].reshape(3, 3)

    # # Enforce rank 2 constraint
    # U, D, V = np.linalg.svd(F)
    # D[2] = 0
    # F = np.dot(U, np.dot(np.diag(D), V))

    # # Unnormalize F
    # F = np.dot(T2.T, np.dot(F, T1))

    # # Refine F using local minimization
    # F = refineF(F, pts1, pts2)

    # return F / F[2, 2]

    if len(pts1) >= 4 :
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        pts1 = pts1/M
        pts2 = pts2/M
        T = np.array([[1/M,0,0],[0,1/M,0],[0,0,1]])
        A = []
        for i in range(len(pts1)):
            x1 = pts1[i][0]
            y1 = pts1[i][1]
            x2 = pts2[i][0]
            y2 = pts2[i][1]
            A.append([x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
        A = np.array(A)
        u, s, v = np.linalg.svd(A)

        F = v[len(s)-1]

        F = refineF(F,pts1,pts2)

        S,V,D = np.linalg.svd(F)
        V = np.array([[V[0], 0, 0], [0, V[1], 0], [0, 0, 0]])

        F_con = np.dot(S, V)
        F_con = np.dot(F_con, D)
        F_r = np.dot(T.T,F_con)
        F_r = np.dot(F_r,T)
        return F_r




"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
# def epipolar_correspondences(im1, im2, F, pts1):
    # replace pass by your implementation


def epipolar_correspondences(im1, im2, F, pts1, w=5):
    # h,w,_ = im2.shape
    # dis = 33333
    # pts2 = []
    # for i in range(len(pts1)):
    #     pts1_x = pts1[i][0]
    #     pts1_y = pts1[i][1]
    #     p1 = np.array([pts1_x,pts1_y,1])
    #     l = np.dot(F, p1)
    #     s = np.sqrt(l[0] ** 2 + l[1] ** 2)
    #     l = l/s
    #     for i in range(2,w - 2):
    #         p2_y = int(-(l[0] * i + l[2])/ l[1])
    #         a1 = np.array([[im1[pts1_y-2, pts1_x-2, :], im1[pts1_y-2, pts1_x-1, :], im1[pts1_y-2,pts1_x, :], im1[pts1_y-2, pts1_x+1, :],im1[pts1_y-2, pts1_x+2, :]],
    #                        [im1[pts1_y-1, pts1_x-2, :], im1[pts1_y-1, pts1_x-1, :], im1[pts1_y-1, pts1_x, :], im1[pts1_y-1, pts1_x+1, :], im1[pts1_y-2, pts1_x+2, :]],
    #                        [im1[pts1_y, pts1_x-2, :], im1[pts1_y, pts1_x-1, :],im1[pts1_y, pts1_x, :],im1[pts1_y, pts1_x+1, :], im1[pts1_y, pts1_x+2, :]],
    #                        [im1[pts1_y+1, pts1_x-2, :], im1[pts1_y+1, pts1_x-1, :],im1[pts1_y+1, pts1_x, :],im1[pts1_y+1, pts1_x+1, :], im1[pts1_y+1, pts1_x+2, :]],
    #                        [im1[pts1_y+2, pts1_x-2 , :],im1[pts1_y+2, pts1_x-1, :],im1[pts1_y+2, pts1_x, :],im1[pts1_y+2, pts1_x+1, :], im1[pts1_y+2, pts1_x+2, :]]])
    #         a2 = np.array([[im2[p2_y-2, i-2, :],im2[p2_y-2, i-1, :], im2[p2_y-2, i, :], im2[p2_y-2, i+1, :], im2[p2_y-2, i+2, :]],
    #                        [im2[p2_y-1, i-2, :], im2[p2_y-1, i-1, :], im2[p2_y-1, i, :], im2[p2_y-1, i+1, :], im2[p2_y-1, i+2, :]],
    #                        [im2[p2_y , i - 2, :], im2[p2_y , i - 1, :], im2[p2_y , i, :],im2[p2_y , i + 1, :], im2[p2_y , i + 2, :]],
    #                        [im2[p2_y +1, i - 2, :], im2[p2_y + 1, i - 1, :], im2[p2_y + 1, i, :],im2[p2_y + 1, i + 1, :], im2[p2_y + 1, i + 2, :]],
    #                        [im2[p2_y +2, i - 2, :], im2[p2_y +2, i - 1, :], im2[p2_y +2, i, :],im2[p2_y +2, i + 1, :], im2[p2_y +2, i + 2, :]],
    #                        ])
    #         d = a1-a2
    #         d = np.abs(d)
    #         d_Manhattan = np.sum(d)
    #         if d_Manhattan <= dis:
    #             dis = d_Manhattan
    #             p2x = i
    #             p2y = p2_y
    #     pts2.append([p2x,p2y])
    # pts2 = np.array(pts2)
    # return pts2
    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape
    pts2 = np.zeros_like(pts1)
    
    pts1 = pts1.astype(np.float32)

    # calculate epipolar lines for all points in pts1
    epilines = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    for i, (pt1, epl) in enumerate(zip(pts1, epilines)):
        # calculate the two endpoints of the epipolar line
        x0, y0 = map(int, [0, -epl[2]/epl[1]])
        x1, y1 = map(int, [w2, -(epl[2] + epl[0]*w2)/epl[1]])

        # crop a window around pt1 in image1
        win1 = im1[max(0, int(pt1[1]-w)):min(h1, int(pt1[1]+w+1)),
          max(0, int(pt1[0]-w)):min(w1, int(pt1[0]+w+1))]


        # search for the best matching point along the epipolar line in image2
        line2 = im2[y0:y1, x0:x1]
        if line2.size == 0:
            continue

        # compute similarity scores between pt1 window and all windows along epipolar line in image2
        scores = np.zeros(line2.shape[:2])
        for y in range(line2.shape[0]-w):
            for x in range(line2.shape[1]-w):
                win2 = line2[y:y+w*2+1, x:x+w*2+1]
                if win2.shape != win1.shape:
                    continue
                score = np.sum(np.abs(win1 - win2))
                scores[y, x] = score

        # get the index of the best matching point
        y, x = np.unravel_index(np.argmax(scores), scores.shape)

        # add the offset to get the final location in image2
        pt2 = np.array([x0 + x, y0 + y])

        pts2[i] = pt2

    return pts2





"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # replace pass by your implementation
    E = K2.T @ F @ K1
    # U, S, Vt = np.linalg.svd(E)
    # S[2] = 0
    # E = U @ np.diag(S) @ Vt
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    # replace pass by your implementation

    N = pts1.shape[0]
    pts3d_hom = np.zeros((N, 4))
    
    for i in range(N):
        A = np.vstack([pts1[i, 0] * P1[2, :] - P1[0, :],
                       pts1[i, 1] * P1[2, :] - P1[1, :],
                       pts2[i, 0] * P2[2, :] - P2[0, :],
                       pts2[i, 1] * P2[2, :] - P2[1, :]])
        _, _, V = np.linalg.svd(A)
        pts3d_hom[i, :] = V[-1, :]
        
    pts3d_hom = pts3d_hom / pts3d_hom[:, 3][:, np.newaxis]
    pts3d = hom2cart(pts3d_hom)
    
    # re-projection error
    # P1_proj = P1 @ cart2hom(pts3d).T
    # P2_proj = P2 @ cart2hom(pts3d).T
    
    # pts1_reproj = hom2cart(P1_proj.T)
    # pts2_reproj = hom2cart(P2_proj.T)
    
    # err1 = np.sqrt(np.sum((pts1_reproj - pts1)**2, axis=1))
    # err2 = np.sqrt(np.sum((pts2_reproj - pts2)**2, axis=1))
    
    # return pts3d, np.mean(err1), np.mean(err2)
    return pts3d



"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def skew(v): #function new added
    return np.array([[0, -v[2][0], v[1][0]],
                 [v[2][0], 0, -v[0][0]],
                 [-v[1][0], v[0][0], 0]])


def rectify_pair(K1, K2, R1, R2, t1, t2):
    # Step 1: Compute the optical centers of each camera
    c1 = np.linalg.inv(K1 @ R1) @ (-t1)
    c2 = np.linalg.inv(K2 @ R2) @ (-t2)
    
    # Step 2: Compute the new rotation matrix Re
    r1 = (c1 - c2) / np.linalg.norm(c1 - c2)
    print(r1)
    r2 = np.cross(R1[2,:], r1)
    # r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r2, r1)
    R1p = R2p = np.column_stack((r1, r2, r3))
    
    # Step 3: Compute the new intrinsic parameters
    K1p = K2p = K2
    
    # Step 4: Compute the new translation vectors
    t1p = -R1p @ c1
    t2p = -R2p @ c2
    
    # Step 5: Compute the rectification matrices
    M1 = K1p @ R1p @ np.linalg.inv(K1 @ R1)
    M2 = K2p @ R2p @ np.linalg.inv(K2 @ R2)
    
    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    high, wide = im1.shape
    dispM = np.ones([high, wide])
    w = int((win_size - 1)/2)
    for i in range(w, high - w - 1):
        for j in range(w, wide - w - 1):
            im1_test = im1[i - w: i + w + 1, j - w: j + w + 1]
            dis = [100000000 for i in range(max_disp+1)]
            for d in range(max_disp+1):
                if j  >= d + w:
                    im2_test = im2[i - w: i + w + 1,  j - w - d: j + w + 1 - d]
                    dis_m = (im1_test - im2_test)**2
                    dis[d] = np.sum(dis_m)
            dispM[i][j] = np.argmin(dis)

    return dispM
    # # Create empty disparity map
    # dispM = np.zeros(im1.shape)

    # # Compute half window size
    # w = (win_size - 1) // 2

    # # Iterate over each pixel in the left image
    # for y in range(w, im1.shape[0] - w):
    #     for x in range(w, im1.shape[1] - w):
    #         # Compute the search range for the corresponding pixel in the right image
    #         search_range = min(max_disp, im1.shape[1] - x - w)

    #         # Create a window around the current pixel in the left image
    #         window = im1[y - w : y + w + 1, x - w : x + w + 1]

    #         # Initialize the minimum distance and corresponding disparity
    #         min_dist = float('inf')
    #         min_disp = 0

    #         # Iterate over each pixel in the search range in the right image
    #         for d in range(search_range):
    #             # Create a window around the corresponding pixel in the right image
    #             window2 = im2[y - w : y + w + 1, x - w - d : x + w + 1 - d]

    #             # Compute the distance between the two windows
    #             dist = np.sum(np.square(window - window2))

    #             # Update the minimum distance and corresponding disparity
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 min_disp = d

    #         # Assign the minimum disparity to the current pixel in the disparity map
    #         dispM[y, x] = min_disp

    # return dispM



"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation

    # Compute baseline
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)
    b = np.linalg.norm(c1 - c2)

    # Compute focal length
    f = K1[0, 0]

    # Compute depth
    depthM = np.zeros_like(dispM, dtype=np.float32)
    for y in range(dispM.shape[0]):
        for x in range(dispM.shape[1]):
            disp = dispM[y, x]
            if disp != 0:
                depthM[y, x] = b * f / disp
            else: depthM[y, x] = 0

    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation

    """
    Estimate the camera matrix P given 2D and 3D points using Direct Linear Transform (DLT)
    
    Arguments:
    x -- 2 x N matrix denoting the (x, y) coordinates of the N points on the image plane
    X -- 3 x N matrix denoting the (x, y, z) coordinates of the corresponding points in the 3D world
    
    Returns:
    P -- 3 x 4 camera matrix
    """

    # Construct matrix A
    A = []
    for i in range(len(x)):
        A.append([X[i][0], X[i][1], X[i][2] , 1, 0, 0, 0, 0, -X[i][0] * x[i][0], -X[i][1] * x[i][0], -X[i][2] * x[i][0], -x[i][0]])
        A.append([0, 0, 0, 0, X[i][0], X[i][1], X[i][2], 1, -X[i][0] * x[i][1], -X[i][1] * x[i][1], -X[i][2] * x[i][1], -x[i][1]])
    A = np.array(A)
    
    # Compute SVD of A
    U, D, Vt = np.linalg.svd(A)
    
    # Camera matrix
    P = Vt[len(D) - 1]
    P_norm = P.reshape((3, 4))

    return P_norm


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation

    u , s, v = np.linalg.svd(P)

    c = v[-1, :]
    c = c[0:3]/c[3]
    M = P[:3, :3]

    R, K = np.linalg.qr(np.linalg.inv(M))
    K = np.linalg.inv(K)
    K = K/K[2,2]
    R = R.T
    t = (-R.dot(c)).reshape(3,1)

    return K, R, t