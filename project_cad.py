import cv2
import numpy as np
from matplotlib import pyplot as plt
import submission as sub

# Load data
data = np.load('../data/pnp.npz', allow_pickle = True)
img = data['image']
cad = data['cad']
x = data['x']
X = data['X']
# K = data['K']
P = sub.estimate_pose(x,X)
K, R, t = sub.estimate_params(P)

# Estimate camera matrix and intrinsic matrix
_, rvec, tvec = cv2.solvePnP(X, x, K, distCoeffs=None)
R, _ = cv2.Rodrigues(rvec)
P = np.hstack((R, tvec))

# Project 3D points onto the image
img_points, _ = cv2.projectPoints(X, rvec, tvec, K, distCoeffs=None)

# Plot 2D and projected 3D points
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.scatter(x[:, 0], x[:, 1], color='red')
plt.scatter(img_points[:, 0, 0], img_points[:, 0, 1], color='blue')
plt.show()

pts = cad[0][0][0]
pts_dou = cad[0][0][1]


ax = plt.axes(projection='3d')
pt = []
for i in range(len(pts)):
    pts[i] = np.array(pts[i])
    pt.append(np.dot(R, pts[i]))
pt = np.array(pt)
ax.plot3D(xs = pt[:,0], ys = pt[:, 1] , zs = pt[:, 2], color= 'blue', ms = 0.01)
plt.show()


f, ax2= plt.subplots(1, 1, figsize=(12, 9))

# Rotate CAD model
cad_rotated = np.dot(R, cad)

# Project CAD vertices onto the image
cad_points, _ = cv2.projectPoints(cad_rotated, rvec, tvec, K, distCoeffs=None)

# Plot projected CAD model
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.plot(cad_points[0:2, 0, 0], cad_points[0:2, 0, 1], color='green')
plt.plot(cad_points[1:3, 0, 0], cad_points[1:3, 0, 1], color='green')
plt.plot(cad_points[2:4, 0, 0], cad_points[2:4, 0, 1], color='green')
plt.plot([cad_points[0, 0, 0], cad_points[3, 0, 0]], [cad_points[0, 0, 1], cad_points[3, 0, 1]], color='green')
plt.scatter(x[:, 0], x[:, 1], color='red')
plt.show()
