import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy
# --- 1. Inputs ---

# 3D World Coordinates of GCPs (X, Y, Z) - example data
# Ensure these are in a consistent coordinate system (e.g., UTM, local meters)
# At least 4 non-coplanar points are needed for solvePnP. More is better.
# For solvePnPRansac, it can handle outliers if you have more points.
scaling = False # Set to False if you want to use the original coordinates without scaling
scaling_factor = 1000.0 # Scale factor to convert from meters to millimeters (or other units)

object_points = np.array([
[112989.91409784222196322,552154.64224467158783227,0.69480002],
[113050.60488277713011485,552092.87901645083911717,0.74900001],
[113050.71205896152241621,551689.52053703810088336,0.53350002],
[113490.0272401758847991,552144.36423021694645286,0.59299999],
[113746.99576699010503944,551909.75011235871352255,0.44319999],
[113570.84898267970129382,551796.37950754794292152,0.48969999],
[113713.57496283081127331,551649.0721996248466894,0.64929998],
[113662.3701781995332567,551852.85045757621992379,0.4294],
[113235.42748870696232188,552191.37642782612238079,0.39610001],
[113381.63397003299905919,551881.35023998084943742,0.54180002],
[113258.9771846032817848,551768.33204498805571347,0.61519998],
[113220.07767920961487107,552009.80907174595631659,0.45019999],
# [113567.70224852213868871,552072.74866445199586451,5.9110999],
# [113304.82361678876623046,552099.70178270724136382,6.2086]# Point 5 in world (add some Z variation)
], dtype=np.float64)
shift = np.array([np.mean(object_points,axis=0)[:2][0],np.mean(object_points,axis=0)[:2][1],0])
object_points = object_points - shift
 # Scale down to meters or millimeters as needed
# 2D Image Pixel Coordinates of GCPs (px, py) - corresponding to object_points
# (column, row) or (x, y) from top-left
image_points = np.array([
[12545.07368950936506735,2035.07012215582199133],
[11049.46411504126990621,3492.31905832603297313],
[1339.21411504126854197,3460.56054768773174146],
[12242.3761051535821025,14079.03493826990961679],
[6568.1400628622932345,20234.66355500830104575],
[3858.84098310769877571,15985.53682954818032158],
[303.34750151262142026,19400.10047985495111789],
[5207.95179599114180746,18193.1503264806997322],
[13402.92917329174815677,7953.45956727824341215],
[5921.42085208606204105,11442.91793657300513587],
[3214.90344411061551,8480.24055927238987351],
[9030.81295331306318985,7566.84293657300713676],
# [10530.60990390158804075,15975.24809446153631143],
# [11214.26447309210925596,9607.12772892706016137]# Point 5 in image
], dtype=np.float64)
#image_points = np.vstack((image_points[:, 1], 14016 - image_points[:,0])).T # Ensure shape is (N, 2)
# Image dimensions (needed for guessing intrinsics if not calibrated)
image_width = 14016
image_height = 20544

# --- 2. Camera Intrinsic Parameters ---
# A. If you have calibrated your camera:
# fx = K[0,0], fy = K[1,1], cx = K[0,2], cy = K[1,2]
# camera_matrix = np.array(K, dtype=np.float64)
# dist_coeffs = np.array(D, dtype=np.float64) # [k1, k2, p1, p2, k3]

# B. Educated Guess (replace with your actual values if known or calibrated)
#    This is a common weak point if not properly calibrated.
fx = image_width  # A common rough guess if FoV is unknown
fy = image_width  # Assuming square pixels and fx approx fy
cx = image_width / 2.0
cy = image_height / 2.0

camera_matrix = np.array([
    [21170.2127, 0, cx],
    [0, 21170.2127, cy],
    [0, 0, 1]
], dtype=np.float64)

if scaling:
    object_points = object_points/scaling_factor 
    image_points = image_points/scaling_factor
    camera_matrix = camera_matrix/scaling_factor
    
# Assume no lens distortion if not calibrated.
# For high accuracy, calibrate your camera to get these.
dist_coeffs = np.zeros((5, 1), dtype=np.float64) # k1, k2, p1, p2, k3

print("Camera Matrix (Intrinsic Parameters):\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# --- 3. Solve PnP ---
# Ensure you have at least 4 points for iterative PnP.
# Use cv2.SOLVEPNP_P3P if you only have 3 (can have ambiguities).
# Use cv2.SOLVEPNP_AP3P for 3 points, more robust.
# cv2.SOLVEPNP_ITERATIVE is a common default for >= 4 points.
# cv2.SOLVEPNP_EPNP is also good.

if len(object_points) < 4:
    print("Need at least 4 points for SOLVEPNP_ITERATIVE or similar methods.")
    # Consider SOLVEPNP_P3P or AP3P if you only have 3, but handle multiple solutions.
    exit()

# `success`: boolean, true if solution was found.
# `rvec`: Output rotation vector (Rodrigues)
# `tvec`: Output translation vector
# `inliers`: Output vector of inliers indices (if using RANSAC)

# Using solvePnPRansac for robustness to potential outlier GCPs
(success, rvec, tvec, inliers) = cv2.solvePnPRansac(
    object_points, image_points, camera_matrix, dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE # Or cv2.SOLVEPNP_EPNP etc.
    # reprojectionError=8.0, # Default is 8.0, max reprojection error to be an inlier
    # iterationsCount=100,   # Default is 100
    # confidence=0.99        # Default is 0.99
) # 113332.35654 551927.80771 880.17058  | -0.09765364 -0.02852858 90.23929679


# tvect_n = np.array([tvec[1], tvec[0],-1*tvec[2]])
# location = shift - tvect_n.T
R_world_to_camera, _ = cv2.Rodrigues(rvec)
R_camera_to_world = R_world_to_camera.T # Transpose is inverse for rotation matrices
camera_position_world = -R_camera_to_world @ tvec # Matrix multiplication
location = shift + camera_position_world.T

rotation = Rscipy.from_matrix(R_world_to_camera)
yaw_pitch_roll = rotation.as_euler('ZYX', degrees=True)

if scaling:
    tvec_n = tvec*scaling_factor
    tvec_n = np.array([tvec_n[1],tvec_n[0],tvec_n[2]])
    
    
    print("Scaled Translation Vector (tvec):\n", tvec_n)
if not success:
    print("solvePnP failed to find a solution.")
    exit()

print("\n--- solvePnP Results (World relative to Camera) ---")
print("Rotation vector (rvec):\n", rvec)
print("Translation vector (tvec):\n", tvec)
if inliers is not None:
    print(f"Number of inliers: {len(inliers)} out of {len(object_points)}")
else:
    print("Note: solvePnP (non-RANSAC) does not return inliers.")

# --- 4. Derive Camera Pose in World Coordinates ---

# Convert rotation vector to rotation matrix (world to camera)
R_world_to_camera, _ = cv2.Rodrigues(rvec)
print("\nRotation Matrix (R_world_to_camera):\n", R_world_to_camera)

# The tvec from solvePnP is the position of the world origin in camera coordinates.
# We want the camera's position in world coordinates.
# Camera Position (C_world) = -R_transpose * tvec
R_camera_to_world = R_world_to_camera.T # Transpose is inverse for rotation matrices
camera_position_world = -R_camera_to_world @ tvec # Matrix multiplication

print("\n--- Camera Pose in World Coordinates ---")
print("Camera Position (X, Y, Z) in World Coords:\n", camera_position_world.flatten())

# Camera orientation in World Coordinates
# The camera's Z-axis in its own coordinate system is (0,0,1).
# To find its direction in world coordinates, we transform it by R_camera_to_world.
# camera_z_axis_camera_coords = np.array([[0],[0],[1]], dtype=np.float64)
# camera_direction_world = R_camera_to_world @ camera_z_axis_camera_coords
# This is equivalent to the 3rd column of R_camera_to_world

camera_direction_vector_world = R_camera_to_world[:, 2] # Z-axis of camera in world
# This vector points from the camera outwards, along its optical axis.

print("Camera Direction Vector (pointing along optical Z-axis) in World Coords:\n", camera_direction_vector_world.flatten())

# Optional: Calculate Euler angles for a more human-readable orientation
# (e.g., yaw, pitch, roll). Be mindful of angle conventions and gimbal lock.
# Note: Decomposing to Euler angles can be tricky and ambiguous.
# For many applications, the rotation matrix or quaternion is preferred.

# A common convention: ZYX Euler angles (Yaw, Pitch, Roll)
# sy = math.sqrt(R_camera_to_world[0,0] * R_camera_to_world[0,0] +  R_camera_to_world[1,0] * R_camera_to_world[1,0])
# singular = sy < 1e-6
# if not singular:
#     x_angle = math.atan2(R_camera_to_world[2,1] , R_camera_to_world[2,2]) # Roll
#     y_angle = math.atan2(-R_camera_to_world[2,0], sy)                    # Pitch
#     z_angle = math.atan2(R_camera_to_world[1,0], R_camera_to_world[0,0]) # Yaw
# else:
#     x_angle = math.atan2(-R_camera_to_world[1,2], R_camera_to_world[1,1])
#     y_angle = math.atan2(-R_camera_to_world[2,0], sy)
#     z_angle = 0
# print(f"Camera Orientation (Euler XYZ in radians - Roll, Pitch, Yaw approx.):\n [{x_angle}, {y_angle}, {z_angle}]")
# print(f"Camera Orientation (Euler XYZ in degrees - Roll, Pitch, Yaw approx.):\n [{np.degrees(x_angle)}, {np.degrees(y_angle)}, {np.degrees(z_angle)}]")


# --- 5. (Optional) Verification: Project world points back to image ---
if success:
    projected_image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)

    # Calculate reprojection error
    total_error = 0
    for i in range(len(object_points)):
        # Use only inliers if RANSAC was used
        if inliers is not None and i not in inliers.flatten():
            continue
        error = cv2.norm(image_points[i], projected_image_points[i][0], cv2.NORM_L2)
        total_error += error*error
        # print(f"Point {i}: Original {image_points[i]}, Projected {projected_image_points[i][0]}, Error {error}")

    num_points_for_error = len(inliers) if inliers is not None else len(object_points)
    if num_points_for_error > 0:
        mean_error = np.sqrt(total_error / num_points_for_error)
        print(f"\nMean Reprojection Error (for inliers if RANSAC): {mean_error:.4f} pixels")
    else:
        print("\nNo inliers found to calculate reprojection error.")