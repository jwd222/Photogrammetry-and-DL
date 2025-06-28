import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QDoubleSpinBox, QGroupBox, QGridLayout, 
                             QSpinBox, QCheckBox, QDialog, QDialogButtonBox, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtOpenGL import QGLWidget, QGLFormat, QGL
from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial.transform import Rotation


class GLWidget(QGLWidget):
    def __init__(self, parent=None):
        # Request an OpenGL 2.1 context for compatibility with GLU functions
        fmt = QGLFormat()
        fmt.setVersion(2, 1)
        fmt.setProfile(QGLFormat.CompatibilityProfile) # Use compatibility profile
        super(GLWidget, self).__init__(fmt, parent)
        self.main_window = parent
        
        # Data for solvePnP (original EPSG coordinates)
        self.original_world_points = np.array([], dtype=np.float64).reshape(0,3)
        self.image_points = np.array([], dtype=np.float32).reshape(0,2)
        
        # Data for OpenGL drawing (scaled and centered)
        self.display_world_points = np.array([], dtype=np.float32).reshape(0,3)
        
        # Default intrinsic parameters (will be updated by dialog)
        self.fx, self.fy = 800.0, 800.0
        self.cx, self.cy = 512.0, 384.0
        self.image_width, self.image_height = 1024.0, 768.0

        # Camera pose in display coordinates for OpenGL
        self.camera_position = np.array([0, 0, 0], dtype=np.float32) # Scaled for display
        self.camera_rotation = np.eye(3, dtype=np.float32) # Rotation matrix (world to camera) - remains unscaled
        self.camera_found = False
        
        # Viewpoint rotation
        self.x_rot = 30
        self.y_rot = -30
        self.z_rot = 0
        
        # Viewpoint translation (for panning)
        self.x_trans = 0.0
        self.y_trans = 0.0

        self.zoom = 10.0
        self.show_rays = True
        self.show_camera = True
        self.show_points = True
        self.show_reprojection = True
        self.show_image_plane = True
        self.image_plane_points_3d = [] # Stores 3D points on the image plane (scaled for display)
        self.image_plane_corners_3d = [] # Stores 3D corners of the image plane (scaled for display)
        
        # Estimated Camera Pose for Disambiguation (in original EPSG coordinates)
        self.use_estimated_pose_for_disambiguation = False
        self.estimated_camera_position_epsg = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.estimated_camera_rotation_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32) # Roll, Pitch, Yaw in degrees

        # Scaling parameters received from MainWindow for internal scaling/unscaling
        self.local_origin_3d_vis = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.scale_factor_vis = 1.0
        self.z_exaggeration_factor_vis = 1.0
        self.exaggerate_z_vis = False

        self.setMinimumSize(600, 500)

        self.quadric = None # To store the GLU quadric object

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialfv(GL_FRONT, GL_SHININESS, 100.0)

        # Initialize GLU quadric object for drawing spheres and cones
        self.quadric = gluNewQuadric()
        gluQuadricNormals(self.quadric, GLU_SMOOTH)
        gluQuadricTexture(self.quadric, GL_TRUE)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w/h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Apply panning translation first
        glTranslatef(self.x_trans, self.y_trans, -self.zoom)
        
        # Then apply rotations
        glRotatef(self.x_rot, 1, 0, 0)
        glRotatef(self.y_rot, 0, 1, 0)
        glRotatef(self.z_rot, 0, 0, 1)
        
        self.draw_axes()
        
        # Draw world points (using display_world_points)
        if self.show_points:
            glColor3f(1, 0.5, 0) # Orange
            for point in self.display_world_points:
                glPushMatrix()
                glTranslatef(point[0], point[1], point[2])
                gluSphere(self.quadric, 0.1, 16, 16) 
                glPopMatrix()
        
        # Draw camera (using camera_position which is already scaled)
        if self.camera_found and self.show_camera:
            glPushMatrix()
            # Apply the inverse of the camera's world pose to position the camera in the scene
            # The camera_position is the world coordinates of the camera center (scaled for display).
            # The camera_rotation is the rotation matrix from world to camera coordinates.
            
            # First, translate to the camera's world position (scaled)
            glTranslatef(*self.camera_position)
            
            # Then, apply the camera's world rotation (inverse of the rotation matrix from solvePnP)
            # solvePnP gives R_camera_world, so we need R_world_camera = R_camera_world.T
            rot_matrix_world_camera = self.camera_rotation.T
            
            # Convert rotation matrix to a 4x4 matrix for glMultMatrixf
            m = np.identity(4, dtype=np.float32)
            m[:3, :3] = rot_matrix_world_camera
            glMultMatrixf(m.T) # OpenGL expects column-major order, so transpose again
            
            # --- Draw fixed-size wireframe frustum ---
            aspect_ratio = self.image_width / self.image_height
            base_height = 2.0
            base_width = aspect_ratio * base_height
            depth = 1.0  # Fixed depth of image plane

            # Corners of the image plane in camera-local coordinates
            half_w = base_width / 2
            half_h = base_height / 2
            image_plane_corners_local = np.array([
                [-half_w,  half_h, depth],  # top-left
                [ half_w,  half_h, depth],  # top-right
                [ half_w, -half_h, depth],  # bottom-right
                [-half_w, -half_h, depth],  # bottom-left
            ], dtype=np.float32)

            # Draw frustum as lines
            glColor3f(0.2, 0.6, 1.0)  # Light blue
            glLineWidth(1.5)
            glBegin(GL_LINES)

            # Draw pyramid sides (camera origin to image plane corners)
            for corner in image_plane_corners_local:
                glVertex3f(0, 0, 0)
                glVertex3fv(corner)

            # Draw image plane rectangle edges
            for i in range(4):
                glVertex3fv(image_plane_corners_local[i])
                glVertex3fv(image_plane_corners_local[(i + 1) % 4])

            glEnd()

            glPopMatrix() # Pop the camera's world transformation matrix
            
            # Draw camera axes (relative to the camera's world position and orientation)
            glDisable(GL_LIGHTING)
            glBegin(GL_LINES)
            # X-axis (red)
            glColor3f(1, 0, 0)
            glVertex3fv(self.camera_position)
            glVertex3fv(self.camera_position + rot_matrix_world_camera[:, 0] * 0.8)
            
            # Y-axis (green)
            glColor3f(0, 1, 0)
            glVertex3fv(self.camera_position)
            glVertex3fv(self.camera_position + rot_matrix_world_camera[:, 1] * 0.8)
            
            # Z-axis (blue) - this is the direction the camera is looking
            glColor3f(0, 0.5, 1)
            glVertex3fv(self.camera_position)
            glVertex3fv(self.camera_position + rot_matrix_world_camera[:, 2] * 0.8)
            glEnd()
            glEnable(GL_LIGHTING)
        
        # Draw rays from camera to points (using display_world_points)
        if self.show_rays and self.camera_found:
            glDisable(GL_LIGHTING)
            glBegin(GL_LINES)
            glColor3f(0.8, 0.8, 0.8) # Grey
            for i in range(len(self.display_world_points)):
                glVertex3fv(self.camera_position)
                glVertex3fv(self.display_world_points[i])
            glEnd()
            glEnable(GL_LIGHTING)
        
        # Draw reprojection (showing the world points again, but in a different color)
        if self.show_reprojection and self.camera_found:
            glDisable(GL_LIGHTING)
            glColor3f(0, 1, 0) # Green
            for i in range(len(self.display_world_points)):
                glPushMatrix()
                glTranslatef(self.display_world_points[i][0], self.display_world_points[i][1], self.display_world_points[i][2])
                gluSphere(self.quadric, 0.05, 10, 10) 
                glPopMatrix()
            glEnable(GL_LIGHTING)

        # Draw Image Plane and Projected Points (using scaled image_plane_corners_3d and image_plane_points_3d)
        if self.show_image_plane and self.camera_found and self.image_plane_corners_3d:
            glDisable(GL_LIGHTING)
            glLineWidth(1.0)
            
            # Draw the image plane rectangle
            glColor3f(0.5, 0.5, 0.5) # Light grey for the plane
            glBegin(GL_QUADS)
            for corner in self.image_plane_corners_3d:
                glVertex3fv(corner)
            glEnd()

            # Draw the image plane border
            glColor3f(0.7, 0.7, 0.7) # Darker grey for border
            glBegin(GL_LINE_LOOP)
            for corner in self.image_plane_corners_3d:
                glVertex3fv(corner)
            glEnd()

            # Draw the 2D image points on the 3D plane
            glColor3f(1, 0, 1) # Magenta for image points
            for point_3d in self.image_plane_points_3d:
                glPushMatrix()
                glTranslatef(point_3d[0], point_3d[1], point_3d[2])
                gluSphere(self.quadric, 0.03, 8, 8) # Small sphere for image points
                glPopMatrix()
            glEnable(GL_LIGHTING)
    
    def draw_axes(self):
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(2, 0, 0)
        # Y axis (green)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 2, 0)
        # Z axis (blue)
        glColor3f(1, 1, 0) # Changed to yellow for better visibility against blue camera axis
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 2)
        glEnd()
        
        # Draw axis labels
        self.renderText(2.1, 0, 0, "X")
        self.renderText(0, 2.1, 0, "Y")
        self.renderText(0, 0, 2.1, "Z")
        glEnable(GL_LIGHTING)
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.zoom = max(0.1, min(self.zoom - delta, 100.0)) 
        self.update()
    
    def mousePressEvent(self, event):
        self.last_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        
        if event.buttons() & Qt.LeftButton:
            self.x_rot = (self.x_rot + dy) % 360
            self.y_rot = (self.y_rot + dx) % 360
        elif event.buttons() & Qt.MiddleButton: # Middle mouse button for panning
            # Scale translation by zoom level to make it feel natural
            self.x_trans += dx * 0.005 * (self.zoom / 10.0) 
            self.y_trans -= dy * 0.005 * (self.zoom / 10.0) # Invert Y for intuitive panning
        elif event.buttons() & Qt.RightButton:
            self.z_rot = (self.z_rot + dx) % 360
        
        self.last_pos = event.pos()
        self.update()
        
    def set_estimated_pose(self, x, y, z, roll, pitch, yaw):
        # Store estimated pose in original EPSG coordinates
        self.estimated_camera_position_epsg = np.array([x, y, z], dtype=np.float64)
        self.estimated_camera_rotation_euler = np.array([roll, pitch, yaw], dtype=np.float32)
        self.camera_found = False # Invalidate current pose, force recompute
        self.update()

    def set_use_estimated_pose(self, checked):
        self.use_estimated_pose_for_disambiguation = True if checked else False
        self.camera_found = False # Invalidate current pose, force recompute
        self.update()

    def set_intrinsic_parameters(self, fx, fy, cx, cy, img_w, img_h):
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.image_width, self.image_height = img_w, img_h
        self.camera_found = False # Invalidate pose if intrinsics change
        self.update()

    def set_original_world_points(self, points_epsg):
        self.original_world_points = points_epsg
        self.camera_found = False
        # No update() here, as set_world_points_for_display will trigger it.

    def set_world_points_for_display(self, points_scaled):
        self.display_world_points = points_scaled
        self.camera_found = False
        self.update()

    def set_image_points(self, points_2d):
        self.image_points = points_2d
        self.camera_found = False
        self.update()

    def set_scaling_params(self, origin, scale_factor, z_factor, exaggerate_z):
        self.local_origin_3d_vis = origin
        self.scale_factor_vis = scale_factor
        self.z_exaggeration_factor_vis = z_factor
        self.exaggerate_z_vis = exaggerate_z
        self.camera_found = False # Invalidate pose if scaling changes
        self.update()

    def scale_point_for_display(self, point_epsg):
        """Internal helper to scale a single point (or array of points) from EPSG to display coordinates."""
        if point_epsg.ndim == 1:
            point_epsg = point_epsg.reshape(1, 3)
        
        centered = point_epsg - self.local_origin_3d_vis
        # if self.exaggerate_z_vis:
        #     centered[:, 2] *= 2
        scaled = centered * self.scale_factor_vis
        return scaled.flatten() if point_epsg.shape[0] == 1 else scaled

    def unscale_point_from_display(self, point_scaled):
        """Internal helper to unscale a single point (or array of points) from display to EPSG coordinates."""
        if point_scaled.ndim == 1:
            point_scaled = point_scaled.reshape(1, 3)

        unscaled = point_scaled / self.scale_factor_vis
        # if self.exaggerate_z_vis:
        #     unscaled[:, 2] /= self.z_exaggeration_factor_vis
        original = unscaled + self.local_origin_3d_vis
        return original.flatten() if point_scaled.shape[0] == 1 else original

    def compute_p3p(self):
        """
        Computes the camera pose from 3D-2D point correspondences using PnP-RANSAC.
        This version correctly implements pre-processing and post-processing for numerical stability.
        """
        if len(self.original_world_points) < 4: # RANSAC needs at least 4 points
            print("Need at least 4 world and 4 image points for PnP-RANSAC.")
            self.camera_found = False
            # ... (rest of your early exit logic)
            return None

        camera_matrix = np.array([[self.fx, 0, self.cx],
                                  [0, self.fy, self.cy],
                                  [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # --- 1. PRE-PROCESSING AS PER YOUR METHOD ---
        # WARNING: This method of scaling K and image points is not standard and can
        # lead to less stable results than only scaling world points.
        # The post-processing has been adapted to reverse this specific transformation.
        world_points_mean = np.mean(self.original_world_points, axis=0)
        shifted_world_points = self.original_world_points - world_points_mean
        
        scale_factor = 100.0
        
        scaled_shifted_world_points = shifted_world_points / scale_factor
        scaled_image_points = self.image_points / scale_factor
        
        # Create the scaled camera matrix. Note that K[2,2] must remain 1.
        scaled_camera_matrix = camera_matrix.copy()
        scaled_camera_matrix[0, 0] /= scale_factor # fx' = fx / s
        scaled_camera_matrix[1, 1] /= scale_factor # fy' = fy / s
        scaled_camera_matrix[0, 2] /= scale_factor # cx' = cx / s
        scaled_camera_matrix[1, 2] /= scale_factor # cy' = cy / s
        
        
        # # --- 1. Your Initial Guess Data ---
        # # The initial guess of the CAMERA's pose in the ORIGINAL WORLD coordinate system
        # t_wc_guess_original = np.array([113332.35654, 551927.80771, 880.17058])
        # euler_degrees_guess = [-0.09765364, -0.02852858, 90.23929679] # [yaw, pitch, roll]

        # # --- THIS IS THE KEY CORRECTION ---
        # # Apply the 180-degree yaw adjustment to the initial guess to align it
        # # with the expected output coordinate system.
        # # Assuming 'ZYX' order, yaw is the first angle (Z-axis).
        # adjusted_euler_guess = euler_degrees_guess.copy()
        # adjusted_euler_guess[0] =  abs(adjusted_euler_guess[0]) - 180 # Add 180 to the yaw component
        # # adjusted_euler_guess[0] += 180 # Add 180 to the yaw component

        # # A. Transform the translation guess to match the pre-processed world points
        # t_wc_guess_scaled = (t_wc_guess_original - world_points_mean) / scale_factor

        # # B. Convert the *adjusted* Euler angles to a rotation matrix (R_wc)
        # # Use the 'ZYX' convention as before.
        # r = Rotation.from_euler('ZYX', adjusted_euler_guess, degrees=True)
        # R_wc_guess = r.as_matrix()

        # # C. Invert the pose to get the OBJECT's pose in the CAMERA frame
        # R_cw_guess = R_wc_guess.T
        # t_cw_guess_scaled = -R_cw_guess @ t_wc_guess_scaled

        # # D. Convert to OpenCV's rvec/tvec format
        # rvec_guess, _ = cv2.Rodrigues(R_cw_guess)
        # tvec_guess = t_cw_guess_scaled.reshape(3, 1)
        
        # print("\nFinal rvec for solvePnP:\n", rvec_guess)
        # print("\nFinal tvec for solvePnP:\n", tvec_guess)

        # # print("\n--- Converted for OpenCV ---")
        # # print("Refined Rotation Matrix (Object in Camera):\n", R_cw_guess)
        # # print("\nRefined Translation Vector (Object in Camera):\n", t_cw_guess)

        # # # --- 4. Convert the Final Rotation Matrix to a Rotation Vector (rvec) ---
        # # # OpenCV's Rodrigues function does this conversion.
        # # rvec_guess, _ = cv2.Rodrigues(R_cw_guess)
        # # tvec_guess = t_cw_guess.reshape(3, 1) # Ensure tvec is a column vector


        # # --- 5. Now, use these guesses with solvePnPRansac ---
        

        # --- 2. SOLVE PNP WITH PRE-PROCESSED POINTS (OpenCV) ---
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            scaled_shifted_world_points,
            scaled_image_points,
            scaled_camera_matrix,
            dist_coeffs,
            reprojectionError=5.0,
            iterationsCount=100, # Number of RANSAC iterations
            confidence=0.99,     # Desired confidence
            flags=cv2.SOLVEPNP_ITERATIVE # Using a more robust flag
        )
        
        self.last_inliers = inliers

        if not success:
            print("PnP solution not found.")
            return None
        
        # Convert rotation vector to rotation matrix
        self.camera_rotation, _ = cv2.Rodrigues(rvec) # This is R_camera_world (world to camera)
        R_world_to_cam = self.camera_rotation        

        R_cam_to_world = R_world_to_cam.T
        camera_position= (-R_cam_to_world @ tvec).flatten()
        camera_position_epsg = world_points_mean + camera_position * scale_factor
        
        self.camera_position = self.scale_point_for_display(camera_position_epsg)
        self.camera_found = True
        
        euler_angles_deg = Rotation.from_matrix(R_world_to_cam).as_euler('ZYX', degrees=True)
        yaw_z = euler_angles_deg[2]
        pitch_y = euler_angles_deg[1]
        roll_x = euler_angles_deg[0]


        print("\n#############################################################################")
        print("Extrinsic Input Type: Euler")
        print("  Euler Convention: Extrinsic ZYX (Order: Yaw(Z), Pitch(Y), Roll(X))")
        print(f"  Euler Angles (Input): [{180 + yaw_z:.5f}, {pitch_y:.5f}, {roll_x:.5f}]")
        print("Rotation Matrix (R_world_to_cam):")
        for row in R_world_to_cam:
            print(f"  [{row[0]: .6f}, {row[1]: .6f}, {row[2]: .6f}]")
        print(f"Camera Center C (Cx, Cy, Cz): [{camera_position_epsg[0]:.3f}, {camera_position_epsg[1]:.3f}, {camera_position_epsg[2]:.3f}]")
        print("#############################################################################")
        
        # Project world points to image plane
        scaled_projected_image_points, _ = cv2.projectPoints(
            scaled_shifted_world_points, # Use original 3D points
            rvec,                       # The rotation vector from solvePnP
            tvec,                       # The translation vector from solvePnP
            scaled_camera_matrix,
            dist_coeffs
        )
        
        projected_image_points = scaled_projected_image_points.reshape(-1, 2) * scale_factor

        # Calculate reprojection error for each point
        reprojection_errors = np.linalg.norm(projected_image_points - self.image_points, axis=1)

        # Calculate mean reprojection error
        mean_reprojection_error = np.mean(reprojection_errors)

        print(f"--- Reprojection Error ---")
        for i in range(len(self.original_world_points)):
            # Mark inliers if RANSAC was used
            inlier_status = ""
            if self.last_inliers is not None and i in self.last_inliers:
                inlier_status = " (INLIER)"
            elif self.last_inliers is not None:
                inlier_status = " (OUTLIER)"

            print(f"Point {i+1}:{inlier_status}")
            print(f"  3D World Point (Xw, Yw, Zw): ({self.original_world_points[i][0]:.2f}, {self.original_world_points[i][1]:.2f}, {self.original_world_points[i][2]:.2f})")
            print(f"  Provided 2D Image Point (u_img, v_img): {self.image_points[i]}")
            print(f"  Projected (u, v): ({projected_image_points[i, 0]:.2f}, {projected_image_points[i, 1]:.2f})")
            print(f"  Reprojection Error: {reprojection_errors[i]:.2f} pixels")
        print(f"Mean Reprojection Error (All Points): {mean_reprojection_error:.2f} pixels")

        if self.last_inliers is not None and len(self.last_inliers) > 0:
            inlier_reprojection_errors = reprojection_errors[self.last_inliers.flatten()]
            mean_inlier_reprojection_error = np.mean(inlier_reprojection_errors)
            print(f"Mean Reprojection Error (Inliers Only): {mean_inlier_reprojection_error:.2f} pixels")
        print("#############################################################################")

        # --- Calculate 3D points for image plane visualization ---
        self.image_plane_points_3d = []
        self.image_plane_corners_3d = []

        if self.camera_found:
            # Focal length for the image plane (can be scaled for visualization)
            # Using a small scale factor to make the plane visible and not too large
            # relative to the camera model.
            plane_distance = 3 # Distance from camera center to image plane for visualization (in display units)
            
            # Calculate image plane corners in camera coordinates (normalized, then scaled)
            # These are in camera's local coordinate system, so they are not affected by EPSG scaling
            tl_cam = np.array([(0 - self.cx) / self.fx * plane_distance, (0 - self.cy) / self.fy * plane_distance, plane_distance])            
            tr_cam = np.array([(self.image_width - self.cx) / self.fx * plane_distance, (0 - self.cy) / self.fy * plane_distance, plane_distance])
            br_cam = np.array([(self.image_width - self.cx) / self.fx * plane_distance, (self.image_height - self.cy) / self.fy * plane_distance, plane_distance])
            bl_cam = np.array([(0 - self.cx) / self.fx * plane_distance, (self.image_height - self.cy) / self.fy * plane_distance, plane_distance])

            # Transform corners from camera coordinates to *world EPSG coordinates* first
            R_world_camera = self.camera_rotation.T # Rotation from camera to world
            T_world_camera_epsg = camera_position_epsg.reshape(3,1) # Camera center in world EPSG

            corners_epsg = [
                (R_world_camera @ tl_cam.reshape(3,1) + T_world_camera_epsg).flatten(),
                (R_world_camera @ tr_cam.reshape(3,1) + T_world_camera_epsg).flatten(),
                (R_world_camera @ br_cam.reshape(3,1) + T_world_camera_epsg).flatten(),
                (R_world_camera @ bl_cam.reshape(3,1) + T_world_camera_epsg).flatten()
            ]
            
            # Then scale these EPSG corners for display
            self.image_plane_corners_3d = [self.scale_point_for_display(corner) for corner in corners_epsg]

            # Project image points onto this 3D plane
            for img_pt in self.image_points:
                u, v = img_pt[0], img_pt[1]
                # Convert 2D pixel to 3D point on the plane in camera coordinates
                pt_on_plane_cam = np.array([(u - self.cx) / self.fx * plane_distance, (v - self.cy) / self.fy * plane_distance, plane_distance])
                
                # Transform to world EPSG coordinates
                pt_on_plane_epsg = (R_world_camera @ pt_on_plane_cam.reshape(3,1) + T_world_camera_epsg).flatten()
                
                # Then scale for display
                self.image_plane_points_3d.append(self.scale_point_for_display(pt_on_plane_epsg))

        self.update()

    def set_point_count(self, n):
        # This method is called by MainWindow.update_num_points_in_dialogs
        # It's primarily for resizing the internal arrays.
        # The actual values are managed by MainWindow and pushed via set_original_world_points etc.
        old_n_original = len(self.original_world_points)
        old_n_display = len(self.display_world_points)
        old_n_image = len(self.image_points)

        # Resize original_world_points (will be filled by MainWindow)
        if n > old_n_original:
            self.original_world_points = np.append(self.original_world_points, np.zeros((n - old_n_original, 3), dtype=np.float64), axis=0)
        else:
            self.original_world_points = self.original_world_points[:n]

        # Resize display_world_points (will be filled by MainWindow)
        if n > old_n_display:
            self.display_world_points = np.append(self.display_world_points, np.zeros((n - old_n_display, 3), dtype=np.float32), axis=0)
        else:
            self.display_world_points = self.display_world_points[:n]

        # Resize image_points (will be filled by MainWindow)
        if n > old_n_image:
            self.image_points = np.append(self.image_points, np.zeros((n - old_n_image, 2), dtype=np.float32), axis=0)
        else:
            self.image_points = self.image_points[:n]

        self.camera_found = False # Invalidate camera pose on point count change
        self.image_plane_points_3d = [] # Clear image plane data
        self.image_plane_corners_3d = [] # Clear image plane data
        self.update()

    def closeEvent(self, event):
        # Clean up GLU quadric object when widget is closed
        if self.quadric:
            gluDeleteQuadric(self.quadric)
        super().closeEvent(event)

# --- Dialog Classes (No changes needed here, they deal with raw input) ---

class IntrinsicDialog(QDialog):
    def __init__(self, fx, fy, cx, cy, img_w, img_h, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Intrinsic Parameters")
        self.setModal(True)
        self.layout = QGridLayout(self)

        self.fx_spin = QDoubleSpinBox(); self.fx_spin.setRange(100.0, 999999.0); self.fx_spin.setSingleStep(10.0); self.fx_spin.setValue(fx)
        self.fy_spin = QDoubleSpinBox(); self.fy_spin.setRange(100.0, 999999.0); self.fy_spin.setSingleStep(10.0); self.fy_spin.setValue(fy)
        self.cx_spin = QDoubleSpinBox(); self.cx_spin.setRange(0.0, 999999.0); self.cx_spin.setSingleStep(1.0); self.cx_spin.setValue(cx)
        self.cy_spin = QDoubleSpinBox(); self.cy_spin.setRange(0.0, 999999.0); self.cy_spin.setSingleStep(1.0); self.cy_spin.setValue(cy)
        self.img_w_spin = QSpinBox(); self.img_w_spin.setRange(100, 999999); self.img_w_spin.setSingleStep(1); self.img_w_spin.setValue(img_w)
        self.img_h_spin = QSpinBox(); self.img_h_spin.setRange(100, 999999); self.img_h_spin.setSingleStep(1); self.img_h_spin.setValue(img_h)

        self.layout.addWidget(QLabel("Focal Length (fx):"), 0, 0); self.layout.addWidget(self.fx_spin, 0, 1)
        self.layout.addWidget(QLabel("Focal Length (fy):"), 1, 0); self.layout.addWidget(self.fy_spin, 1, 1)
        self.layout.addWidget(QLabel("Principal Point (cx):"), 2, 0); self.layout.addWidget(self.cx_spin, 2, 1)
        self.layout.addWidget(QLabel("Principal Point (cy):"), 3, 0); self.layout.addWidget(self.cy_spin, 3, 1)
        self.layout.addWidget(QLabel("Image Width (px):"), 4, 0); self.layout.addWidget(self.img_w_spin, 4, 1)
        self.layout.addWidget(QLabel("Image Height (px):"), 5, 0); self.layout.addWidget(self.img_h_spin, 5, 1)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons, 6, 0, 1, 2)

    def get_parameters(self):
        return (self.fx_spin.value(), self.fy_spin.value(), 
                self.cx_spin.value(), self.cy_spin.value(),
                self.img_w_spin.value(), self.img_h_spin.value())

class WorldPointsDialog(QDialog):
    def __init__(self, num_points, current_world_points_epsg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("World Points (EPSG:28992)")
        self.setModal(True)
        self.layout = QGridLayout(self)
        self.num_points = num_points
        self.spinboxes = []

        # Use the first point as a reference for sensible defaults if available
        self.default_origin = np.array([132000.0, 457000.0, 10.0], dtype=np.float64)
        if len(current_world_points_epsg) > 0:
            self.default_origin = current_world_points_epsg[0]

        for i in range(self.num_points):
            x_spin = QDoubleSpinBox(); x_spin.setDecimals(2); x_spin.setRange(0.0, 300000.0); x_spin.setSingleStep(1.0)
            y_spin = QDoubleSpinBox(); y_spin.setDecimals(2); y_spin.setRange(300000.0, 650000.0); y_spin.setSingleStep(1.0)
            z_spin = QDoubleSpinBox(); z_spin.setDecimals(2); z_spin.setRange(-50.0, 100.0); z_spin.setSingleStep(0.1)

            if i < len(current_world_points_epsg):
                x_spin.setValue(current_world_points_epsg[i][0])
                y_spin.setValue(current_world_points_epsg[i][1])
                z_spin.setValue(current_world_points_epsg[i][2])
            else:
                # Provide sensible defaults for new points in EPSG:28992 range
                x_spin.setValue(self.default_origin[0] + (i * 0.5))
                y_spin.setValue(self.default_origin[1] + (i * 0.5))
                z_spin.setValue(self.default_origin[2] + (i * 0.1))


            self.spinboxes.append((x_spin, y_spin, z_spin))
            self.layout.addWidget(QLabel(f"P{i+1}:"), i, 0)
            self.layout.addWidget(QLabel("X:"), i, 1); self.layout.addWidget(x_spin, i, 2)
            self.layout.addWidget(QLabel("Y:"), i, 3); self.layout.addWidget(y_spin, i, 4)
            self.layout.addWidget(QLabel("Z:"), i, 5); self.layout.addWidget(z_spin, i, 6)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons, self.num_points, 0, 1, 7)

    def get_points(self):
        points_epsg = []
        for x_spin, y_spin, z_spin in self.spinboxes:
            points_epsg.append([x_spin.value(), y_spin.value(), z_spin.value()])
        return np.array(points_epsg, dtype=np.float64) # Return as float64 for precision

class ImagePointsDialog(QDialog):
    def __init__(self, num_points, current_image_points, image_width, image_height, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Points (Pixel Coordinates)")
        self.setModal(True)
        self.layout = QGridLayout(self)
        self.num_points = num_points
        self.spinboxes = []
        self.image_width = image_width
        self.image_height = image_height

        for i in range(self.num_points):
            u_spin = QDoubleSpinBox(); u_spin.setDecimals(2); u_spin.setRange(0.0, self.image_width); u_spin.setSingleStep(1.0)
            v_spin = QDoubleSpinBox(); v_spin.setDecimals(2); v_spin.setRange(0.0, self.image_height); v_spin.setSingleStep(1.0)

            if i < len(current_image_points):
                u_spin.setValue(current_image_points[i][0])
                v_spin.setValue(current_image_points[i][1])
            else:
                # Sensible defaults for new points
                u_spin.setValue(self.image_width / 2 + (i * 5))
                v_spin.setValue(self.image_height / 2 + (i * 5))

            self.spinboxes.append((u_spin, v_spin))
            self.layout.addWidget(QLabel(f"P{i+1}:"), i, 0)
            self.layout.addWidget(QLabel("U:"), i, 1); self.layout.addWidget(u_spin, i, 2)
            self.layout.addWidget(QLabel("V:"), i, 3); self.layout.addWidget(v_spin, i, 4)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons, self.num_points, 0, 1, 5)

    def get_points(self):
        points_2d = []
        for u_spin, v_spin in self.spinboxes:
            points_2d.append([u_spin.value(), v_spin.value()])
        return np.array(points_2d, dtype=np.float32)

# --- Main Window Class ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projective n-Point Algorithm Visualizer")
        self.setGeometry(100, 100, 1200, 700)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # GL Widget
        self.gl_widget = GLWidget(parent=self)
        main_layout.addWidget(self.gl_widget, 3)

        # Control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)

        # Data Storage for EPSG:28992, Intrinsics and Scale factors in visualizers
        self.current_fx, self.current_fy = 21170.2127, 21170.2127
        self.current_img_w, self.current_img_h = 14016, 20544
        self.current_cx, self.current_cy = self.current_img_w/2, self.current_img_h/2
        self.current_world_points_epsg = np.array([[112989.91409784, 552154.64224467, 0.69480002],
                                                   [113050.60488278, 552092.87901645, 0.74900001],
                                                   [113050.71205896, 551689.52053704, 0.53350002],
                                                   [113490.02724018, 552144.36423022, 0.59299999],
                                                   [113746.99576700, 551909.75011236, 0.44319999],
                                                   [113570.84898268, 551796.37950755, 0.48969999],
                                                   [113713.57496283, 551649.07219962, 0.64929998],
                                                   [113662.37017820, 551852.85045758, 0.42940000],
                                                   [113235.42748871, 552191.37642783, 0.39610001],
                                                   [113381.63397003, 551881.35023998, 0.54180002],
                                                   [113258.97718460, 551768.33204499, 0.61519998],
                                                   [113220.07767921, 552009.80907175, 0.45019999],
                                                #    [113567.70224852, 552072.74866445, 5.91109990],
                                                #    [113304.82361679, 552099.70178271, 6.20860000]
                                                   ], dtype=np.float64)

        self.current_image_points = np.array([[12545.07368951,  2035.07012216],
                                              [11049.46411504,  3492.31905833],
                                              [ 1339.21411504,  3460.56054769],
                                              [12242.37610515, 14079.03493827],
                                              [ 6568.14006286, 20234.66355501],
                                              [ 3858.84098311, 15985.53682955],
                                              [  303.34750151, 19400.10047985],
                                              [ 5207.95179599, 18193.15032648],
                                              [13402.92917329,  7953.45956728],
                                              [ 5921.42085209, 11442.91793657],
                                              [ 3214.90344411,  8480.24055927],
                                              [ 9030.81295331,  7566.84293657],
                                            #   [10530.60990390, 15975.24809446],
                                            #   [11214.26447309,  9607.12772893]
                                              ], dtype=np.float32)
        
        # Scaling parameters for visualization
        self.local_origin_3d = np.array([0.0, 0.0, 0.0], dtype=np.float64) # Will be set to first world point
        self.scale_factor = 1.0 # Calculated dynamically
        self.z_exaggeration_factor = 1000.0 # Example factor, can be adjusted
        self.exaggerate_z = True # Toggle for Z exaggeration

        # --- Buttons to open dialogs ---
        data_input_group = QGroupBox("Data Input")
        data_input_layout = QVBoxLayout()
        
        self.num_points_spinbox = QSpinBox()
        self.num_points_spinbox.setRange(3, 20)
        self.num_points_spinbox.setValue(len(self.current_world_points_epsg))
        self.num_points_spinbox.valueChanged.connect(self.update_num_points_in_dialogs)
        data_input_layout.addWidget(QLabel("Number of Correspondences:"))
        data_input_layout.addWidget(self.num_points_spinbox)

        btn_intrinsics = QPushButton("Set Camera Intrinsics")
        btn_intrinsics.clicked.connect(self.open_intrinsics_dialog)
        data_input_layout.addWidget(btn_intrinsics)

        btn_world_points = QPushButton("Set World Points (EPSG:28992)")
        btn_world_points.clicked.connect(self.open_world_points_dialog)
        data_input_layout.addWidget(btn_world_points)

        btn_image_points = QPushButton("Set Image Points (Pixels)")
        btn_image_points.clicked.connect(self.open_image_points_dialog)
        data_input_layout.addWidget(btn_image_points)

        data_input_group.setLayout(data_input_layout)
        control_layout.addWidget(data_input_group)

        # Estimated Pose Group (remains in main window)
        self.estimated_pose_group = QGroupBox("Estimated Camera Pose (for Disambiguation)")
        self.estimated_pose_layout = QGridLayout()

        self.use_estimate_checkbox = QCheckBox("Use Estimated Pose")
        self.use_estimate_checkbox.setChecked(self.gl_widget.use_estimated_pose_for_disambiguation)
        self.use_estimate_checkbox.stateChanged.connect(self.gl_widget.set_use_estimated_pose)
        self.estimated_pose_layout.addWidget(self.use_estimate_checkbox, 0, 0, 1, 6)

        # Estimated Position (in EPSG:28992 coordinates)
        self.est_x_spin = QDoubleSpinBox(); self.est_x_spin.setDecimals(2); self.est_x_spin.setRange(0.0, 300000.0); self.est_x_spin.setSingleStep(1.0); self.est_x_spin.setValue(132252.06)
        self.est_y_spin = QDoubleSpinBox(); self.est_y_spin.setDecimals(2); self.est_y_spin.setRange(300000.0, 650000.0); self.est_y_spin.setSingleStep(1.0); self.est_y_spin.setValue(457578.38)
        self.est_z_spin = QDoubleSpinBox(); self.est_z_spin.setDecimals(2); self.est_z_spin.setRange(-500.0, 10000.0); self.est_z_spin.setSingleStep(0.1); self.est_z_spin.setValue(4388.13)
        
        self.estimated_pose_layout.addWidget(QLabel("Pos X:"), 1, 0); self.estimated_pose_layout.addWidget(self.est_x_spin, 1, 1)
        self.estimated_pose_layout.addWidget(QLabel("Y:"), 1, 2); self.estimated_pose_layout.addWidget(self.est_y_spin, 1, 3)
        self.estimated_pose_layout.addWidget(QLabel("Z:"), 1, 4); self.estimated_pose_layout.addWidget(self.est_z_spin, 1, 5)

        # Estimated Rotation (Euler angles: Roll, Pitch, Yaw in degrees)
        self.est_roll_spin = QDoubleSpinBox(); self.est_roll_spin.setDecimals(1); self.est_roll_spin.setRange(-180.0, 180.0); self.est_roll_spin.setSingleStep(1.0); self.est_roll_spin.setValue(0.00712)
        self.est_pitch_spin = QDoubleSpinBox(); self.est_pitch_spin.setDecimals(1); self.est_pitch_spin.setRange(-180.0, 180.0); self.est_pitch_spin.setSingleStep(1.0); self.est_pitch_spin.setValue(-0.04339)
        self.est_yaw_spin = QDoubleSpinBox(); self.est_yaw_spin.setDecimals(1); self.est_yaw_spin.setRange(-180.0, 180.0); self.est_yaw_spin.setSingleStep(1.0); self.est_yaw_spin.setValue(-89.99288)

        self.estimated_pose_layout.addWidget(QLabel("Rot Roll:"), 2, 0); self.estimated_pose_layout.addWidget(self.est_roll_spin, 2, 1)
        self.estimated_pose_layout.addWidget(QLabel("Pitch:"), 2, 2); self.estimated_pose_layout.addWidget(self.est_pitch_spin, 2, 3)
        self.estimated_pose_layout.addWidget(QLabel("Yaw:"), 2, 4); self.estimated_pose_layout.addWidget(self.est_yaw_spin, 2, 5)

        # Connect signals to update estimated pose in GLWidget
        self.est_x_spin.valueChanged.connect(self.update_estimated_pose)
        self.est_y_spin.valueChanged.connect(self.update_estimated_pose)
        self.est_z_spin.valueChanged.connect(self.update_estimated_pose)
        self.est_roll_spin.valueChanged.connect(self.update_estimated_pose)
        self.est_pitch_spin.valueChanged.connect(self.update_estimated_pose)
        self.est_yaw_spin.valueChanged.connect(self.update_estimated_pose)

        self.estimated_pose_group.setLayout(self.estimated_pose_layout)
        control_layout.addWidget(self.estimated_pose_group)

        # Add control panel to layout
        main_layout.addWidget(control_panel, 2)

        # Initialize GLWidget with current data
        self.update_gl_widget_data()
        
        # # Create dropdown for PnP method selection
        # self.pnp_selector = QComboBox()
        # self.pnp_selector.addItems([
        #     "SQPnP (≥3 pts)",
        #     "Iterative (≥4 pts)",
        #     "EPnP (≥4 pts)",
        #     "UPnP (≥4 pts)",
        #     "P3P (4 pts, multiple)",
        #     "AP3P (4 pts, multiple)",
        #     "DLS (≥6 pts)"
        # ])
        # control_layout.addWidget(self.pnp_selector)
        
        # Add compute button
        compute_btn = QPushButton("Compute Camera Pose (PnP)") 
        compute_btn.clicked.connect(self.compute_camera_pose)
        control_layout.addWidget(compute_btn)
        
        # Add visualization options
        vis_group = QGroupBox("Visualization Options")
        vis_layout = QVBoxLayout()
        
        rays_cb = QPushButton("Toggle Rays")
        rays_cb.clicked.connect(lambda: self.toggle_option('rays'))
        vis_layout.addWidget(rays_cb)
        
        camera_cb = QPushButton("Toggle Camera")
        camera_cb.clicked.connect(lambda: self.toggle_option('camera'))
        vis_layout.addWidget(camera_cb)
        
        points_cb = QPushButton("Toggle World Points") 
        points_cb.clicked.connect(lambda: self.toggle_option('points'))
        vis_layout.addWidget(points_cb)
        
        reproj_cb = QPushButton("Toggle Reprojection Markers") 
        reproj_cb.clicked.connect(lambda: self.toggle_option('reprojection'))
        vis_layout.addWidget(reproj_cb)

        image_plane_cb = QPushButton("Toggle Image Plane") # New button for image plane
        image_plane_cb.clicked.connect(lambda: self.toggle_option('image_plane'))
        vis_layout.addWidget(image_plane_cb)
        
        vis_group.setLayout(vis_layout)
        control_layout.addWidget(vis_group)
        
        # Add reset button
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        control_layout.addWidget(reset_btn)
        
        # Add status label
        self.status_label = QLabel("Ready. Set world and image points, then compute.")
        control_layout.addWidget(self.status_label)
        
        main_layout.addWidget(control_panel, 1)
        
    def compute_camera_pose(self):
        # selected = self.pnp_selector.currentText()
        # if "SQPnP" in selected:
        #     self.gl_widget.pnp_flag = cv2.SOLVEPNP_SQPNP
        # elif "Iterative" in selected:
        #     self.gl_widget.pnp_flag = cv2.SOLVEPNP_ITERATIVE
        # elif "EPnP" in selected:
        #     self.gl_widget.pnp_flag = cv2.SOLVEPNP_EPNP
        # elif "UPnP" in selected:
        #     self.gl_widget.pnp_flag = cv2.SOLVEPNP_UPNP
        # elif "P3P" in selected:
        #     self.gl_widget.pnp_flag = cv2.SOLVEPNP_P3P
        # elif "AP3P" in selected:
        #     self.gl_widget.pnp_flag = cv2.SOLVEPNP_AP3P
        # elif "DLS" in selected:
        #     self.gl_widget.pnp_flag = cv2.SOLVEPNP_DLS
        # else:
        #     print("Unknown method selected.")
        #     return

        self.gl_widget.compute_p3p()

    def calculate_and_set_scaling_params(self):
        """Calculates the local origin and scale factor for visualization."""
        if len(self.current_world_points_epsg) > 0:
            self.local_origin_3d = self.current_world_points_epsg[0].copy()
        else:
            self.local_origin_3d = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            self.scale_factor = 1.0
            return # No points, no scaling needed

        centered_points = self.current_world_points_epsg - self.local_origin_3d
        
        # Apply Z exaggeration for range calculation if enabled
        temp_points_for_range = centered_points.copy()
        if self.exaggerate_z:
            temp_points_for_range[:, 2] *= self.z_exaggeration_factor

        max_range = np.max(np.linalg.norm(temp_points_for_range, axis=1))
        
        target_display_range = 10.0 # Max extent in OpenGL view
        if max_range < 1e-6:
            self.scale_factor = 1.0
        else:
            self.scale_factor = target_display_range / max_range
        
        print(f"Calculated local origin: {self.local_origin_3d}")
        print(f"Calculated scale factor: {self.scale_factor:.6f}")
        print(f"Z Exaggeration: {self.exaggerate_z} (Factor: {self.z_exaggeration_factor})")

    def scale_point_for_display(self, point_epsg):
        """Scales a single point (or array of points) from EPSG to display coordinates."""
        if point_epsg.ndim == 1:
            point_epsg = point_epsg.reshape(1, 3)
        
        centered = point_epsg - self.local_origin_3d
        if self.exaggerate_z:
            centered[:, 2] *= self.z_exaggeration_factor
        scaled = centered * self.scale_factor
        return scaled.flatten() if point_epsg.shape[0] == 1 else scaled

    def unscale_point_from_display(self, point_scaled):
        """Unscales a single point (or array of points) from display to EPSG coordinates."""
        if point_scaled.ndim == 1:
            point_scaled = point_scaled.reshape(1, 3)

        unscaled = point_scaled / self.scale_factor
        if self.exaggerate_z:
            unscaled[:, 2] /= self.z_exaggeration_factor
        original = unscaled + self.local_origin_3d
        return original.flatten() if point_scaled.shape[0] == 1 else original

    def update_num_points_in_dialogs(self, count):
        old_count = len(self.current_world_points_epsg)
        if count > old_count:
            # Append dummy points if increasing size
            if old_count > 0:
                last_point = self.current_world_points_epsg[-1]
                # Add small offsets to new points relative to the last one
                new_epsg_points = np.array([last_point + [i*0.1, i*0.1, i*0.01] for i in range(1, count - old_count + 1)], dtype=np.float64)
            else: # If no points initially, use a default EPSG origin
                new_epsg_points = np.array([[132000.0, 457000.0, 10.0]] * (count - old_count), dtype=np.float64)
                if count - old_count > 1: # Offset subsequent points
                    for i in range(1, count - old_count):
                        new_epsg_points[i] += [i*0.1, i*0.1, i*0.01]

            self.current_world_points_epsg = np.append(self.current_world_points_epsg, new_epsg_points, axis=0)
            
            # For image points, use image center as a base
            new_image_points = np.array([[self.current_img_w / 2 + (i * 5), self.current_img_h / 2 + (i * 5)] for i in range(count - old_count)], dtype=np.float32)
            self.current_image_points = np.append(self.current_image_points, new_image_points, axis=0)
        else:
            # Truncate if decreasing size
            self.current_world_points_epsg = self.current_world_points_epsg[:count]
            self.current_image_points = self.current_image_points[:count]
        
        # Update GLWidget's internal arrays to match the new size and values
        self.gl_widget.set_point_count(count) # This resizes GLWidget's internal arrays
        self.update_gl_widget_data() # This pushes the actual data and recalculates scaling
        self.status_label.setText(f"Number of points set to {count}. Open dialogs to fine-tune values.")
            
    def open_intrinsics_dialog(self):
        dialog = IntrinsicDialog(self.current_fx, self.current_fy, self.current_cx, self.current_cy,
                                 self.current_img_w, self.current_img_h, self)
        if dialog.exec_():
            self.current_fx, self.current_fy, self.current_cx, self.current_cy, self.current_img_w, self.current_img_h = dialog.get_parameters()
            self.gl_widget.set_intrinsic_parameters(self.current_fx, self.current_fy, self.current_cx, self.current_cy, self.current_img_w, self.current_img_h)
            self.status_label.setText("Intrinsic parameters updated.")

    def open_world_points_dialog(self):
        dialog = WorldPointsDialog(self.num_points_spinbox.value(), self.current_world_points_epsg, self)
        if dialog.exec_():
            self.current_world_points_epsg = dialog.get_points()
            self.update_gl_widget_data() # Recalculate scaling and push to GLWidget
            self.status_label.setText("World points updated.")

    def open_image_points_dialog(self):
        dialog = ImagePointsDialog(self.num_points_spinbox.value(), self.current_image_points, 
                                   self.current_img_w, self.current_img_h, self)
        if dialog.exec_():
            self.current_image_points = dialog.get_points()
            self.gl_widget.set_image_points(self.current_image_points)
            self.status_label.setText("Image points updated.")
            
    def update_gl_widget_data(self):
        """Pushes all relevant data to the GLWidget."""
        self.calculate_and_set_scaling_params() # Recalculate scaling params first

        # Pass original world points for solvePnP
        self.gl_widget.set_original_world_points(self.current_world_points_epsg)
        
        # Pass scaled world points for display
        scaled_world_points = self.scale_point_for_display(self.current_world_points_epsg)
        self.gl_widget.set_world_points_for_display(scaled_world_points)

        self.gl_widget.set_image_points(self.current_image_points)
        self.gl_widget.set_intrinsic_parameters(self.current_fx, self.current_fy, self.current_cx, self.current_cy, self.current_img_w, self.current_img_h)
        
        # Pass scaling parameters to GLWidget for internal use (e.g., unscaling estimated pose)
        self.gl_widget.set_scaling_params(self.local_origin_3d, self.scale_factor, self.z_exaggeration_factor, self.exaggerate_z)

        self.update_estimated_pose() # Push initial estimated pose (in EPSG)
        self.gl_widget.update() # Trigger redraw

    def update_estimated_pose(self):
        x = self.est_x_spin.value()
        y = self.est_y_spin.value()
        z = self.est_z_spin.value()
        roll = self.est_roll_spin.value()
        pitch = self.est_pitch_spin.value()
        yaw = self.est_yaw_spin.value()
        # Pass estimated pose in original EPSG coordinates to GLWidget
        self.gl_widget.set_estimated_pose(x, y, z, roll, pitch, yaw)
    
    def toggle_option(self, option):
        if option == 'rays':
            self.gl_widget.show_rays = not self.gl_widget.show_rays
        elif option == 'camera':
            self.gl_widget.show_camera = not self.gl_widget.show_camera
        elif option == 'points':
            self.gl_widget.show_points = not self.gl_widget.show_points
        elif option == 'reprojection':
            self.gl_widget.show_reprojection = not self.gl_widget.show_reprojection
        elif option == 'image_plane': # New toggle
            self.gl_widget.show_image_plane = not self.gl_widget.show_image_plane
        self.gl_widget.update()
    
    def reset_view(self):
        self.gl_widget.x_rot = 30
        self.gl_widget.y_rot = -30
        self.gl_widget.z_rot = 0
        self.gl_widget.x_trans = 0.0
        self.gl_widget.y_trans = 0.0
        self.gl_widget.zoom = 10.0
        self.gl_widget.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())