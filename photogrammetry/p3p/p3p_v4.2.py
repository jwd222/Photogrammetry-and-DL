import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QDoubleSpinBox, QGroupBox, QGridLayout, 
                             QSpinBox, QCheckBox, QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtOpenGL import QGLWidget, QGLFormat, QGL
from OpenGL.GL import *
from OpenGL.GLU import *

class GLWidget(QGLWidget):
    def __init__(self, parent=None):
        # Request an OpenGL 2.1 context for compatibility with GLU functions
        fmt = QGLFormat()
        fmt.setVersion(2, 1)
        fmt.setProfile(QGLFormat.CompatibilityProfile) # Use compatibility profile
        super(GLWidget, self).__init__(fmt, parent)

        # # Initial world points (example: a triangle in 3D space, small scale for visualization)
        # self.world_points = np.array([[1.0, 0.0, 3.0], [-1.0, 0.0, 3.0], [0.0, 1.0, 3.0]], dtype=np.float32)
        # # Initial image points (pixel coordinates, consistent with camera_matrix)
        # # Assuming a 1024x768 image, cx=512, cy=384
        # self.image_points = np.array([[600.0, 300.0], [400.0, 300.0], [512.0, 450.0]], dtype=np.float32)
        
        # Data will be set externally by MainWindow
        self.world_points = np.array([], dtype=np.float32).reshape(0,3)
        self.image_points = np.array([], dtype=np.float32).reshape(0,2)
        
        # Default intrinsic parameters (will be updated by dialog)
        self.fx, self.fy = 800.0, 800.0
        self.cx, self.cy = 512.0, 384.0
        self.image_width, self.image_height = 1024.0, 768.0

        self.camera_position = np.array([0, 0, 0], dtype=np.float32)
        self.camera_rotation = np.eye(3, dtype=np.float32)
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
        self.show_image_plane = True # New toggle for image plane
        self.image_plane_points_3d = [] # Stores 3D points on the image plane
        self.image_plane_corners_3d = [] # Stores 3D corners of the image plane
        
        # Estimated Camera Pose for Disambiguation
        self.use_estimated_pose_for_disambiguation = False
        self.estimated_camera_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.estimated_camera_rotation_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32) # Roll, Pitch, Yaw in degrees

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
        
        # Draw world points
        if self.show_points:
            glColor3f(1, 0.5, 0) # Orange
            for point in self.world_points:
                glPushMatrix()
                glTranslatef(point[0], point[1], point[2])
                gluSphere(self.quadric, 0.1, 16, 16) 
                glPopMatrix()
        
        # Draw camera
        if self.camera_found and self.show_camera:
            glPushMatrix()
            # Apply the inverse of the camera's world pose to position the camera in the scene
            # The camera_position is the world coordinates of the camera center.
            # The camera_rotation is the rotation matrix from world to camera coordinates.
            # To draw the camera in the world, we need to apply its world pose.
            # The camera's world rotation is R_world_camera = R_camera_world.T
            # The camera's world translation is T_world_camera = -R_world_camera @ T_camera_world
            
            # First, translate to the camera's world position
            glTranslatef(self.camera_position[0], self.camera_position[1], self.camera_position[2])
            
            # Then, apply the camera's world rotation (inverse of the rotation matrix from solvePnP)
            # solvePnP gives R_camera_world, so we need R_world_camera = R_camera_world.T
            rot_matrix_world_camera = self.camera_rotation.T
            
            # Convert rotation matrix to a 4x4 matrix for glMultMatrixf
            m = np.identity(4, dtype=np.float32)
            m[:3, :3] = rot_matrix_world_camera
            glMultMatrixf(m.T) # OpenGL expects column-major order, so transpose again
            
            # Draw camera body (a sphere) - Made smaller
            glColor3f(0.2, 0.4, 0.8) # Blueish
            gluSphere(self.quadric, 0.15, 16, 16) # Reduced radius from 0.3 to 0.15
            
            # Draw camera lens/frustum (a cone) - Made smaller
            glPushMatrix()
            glTranslatef(0, 0, 0.15) # Move slightly forward from sphere center (adjusted for new sphere size)
            glRotatef(90, 1, 0, 0) # Rotate to make cone point along Z-axis (initially points along Y for gluCylinder)
            gluCylinder(self.quadric, 0.2, 0.0, 0.4, 16, 16) # Reduced base radius from 0.4 to 0.2, height from 0.8 to 0.4
            gluDisk(self.quadric, 0, 0.2, 16, 1) # Base of the cone (adjusted for new base radius)
            glPopMatrix()

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
        
        # Draw rays from camera to points
        if self.show_rays and self.camera_found:
            glDisable(GL_LIGHTING)
            glBegin(GL_LINES)
            glColor3f(0.8, 0.8, 0.8) # Grey
            for i in range(len(self.world_points)):
                glVertex3fv(self.camera_position)
                glVertex3fv(self.world_points[i])
            glEnd()
            glEnable(GL_LIGHTING)
        
        # Draw reprojection (showing the world points again, but in a different color)
        # This visually confirms that the camera pose found aligns with these points.
        if self.show_reprojection and self.camera_found:
            glDisable(GL_LIGHTING)
            glColor3f(0, 1, 0) # Green
            for i in range(len(self.world_points)):
                glPushMatrix()
                glTranslatef(self.world_points[i][0], self.world_points[i][1], self.world_points[i][2])
                gluSphere(self.quadric, 0.05, 10, 10) 
                glPopMatrix()
            glEnable(GL_LIGHTING)

        # Draw Image Plane and Projected Points
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
        glColor3f(1, 1, 0)
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
            # Adjust sensitivity based on your preference
            self.x_trans += dx * 0.005 * (self.zoom / 10.0) 
            self.y_trans -= dy * 0.005 * (self.zoom / 10.0) # Invert Y for intuitive panning
        elif event.buttons() & Qt.RightButton:
            self.z_rot = (self.z_rot + dx) % 360
        
        self.last_pos = event.pos()
        self.update()
        
    def set_estimated_pose(self, x, y, z, roll, pitch, yaw):
        self.estimated_camera_position = np.array([x, y, z], dtype=np.float32)
        self.estimated_camera_rotation_euler = np.array([roll, pitch, yaw], dtype=np.float32)
        self.camera_found = False # Invalidate current pose, force recompute
        self.update()

    def set_use_estimated_pose(self, checked):
        self.use_estimated_pose_for_disambiguation = checked
        self.camera_found = False # Invalidate current pose, force recompute
        self.update()

    def set_intrinsic_parameters(self, fx, fy, cx, cy, img_w, img_h):
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.image_width, self.image_height = img_w, img_h
        self.camera_found = False # Invalidate pose if intrinsics change
        self.update()

    def set_world_points(self, points_3d_local):
        self.world_points = np.array(points_3d_local, dtype=np.float32)
        self.camera_found = False
        self.update()

    def set_image_points(self, points_2d):
        self.image_points = np.array(points_2d, dtype=np.float32)
        self.camera_found = False
        self.update()
    
    def compute_p3p_old(self):
        # Define camera intrinsic parameters
        fx, fy = 800.0, 800.0 # Focal lengths in pixels
        cx, cy = 512.0, 384.0 # Principal point (center of image) in pixels for 1024x768 image
        image_width, image_height = 1024.0, 768.0 # Image dimensions
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        
        # No distortion coefficients for simplicity, assume ideal camera
        dist_coeffs = np.zeros((4, 1), dtype=np.float64) 
        
        # Use SOLVEPNP_SQPNP for 3 or more points. It can return multiple solutions.
        success, rvecs, tvecs = cv2.solveP3P(
            self.world_points.reshape(-1, 1, 3),
            self.image_points.reshape(-1, 1, 2),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_P3P
        )

        if not success or len(rvecs) == 0:
            print("PnP solution not found. Check point data or intrinsics.")
            self.camera_found = False
            self.image_plane_points_3d = []
            self.image_plane_corners_3d = []
            self.update()
            return

        print(f"Found {len(rvecs)} solutions from solveP3P (using SOLVEPNP_P3P)")

        best_idx = 0 # Default to first solution
        if self.use_estimated_pose_for_disambiguation and len(rvecs) > 1:
            print("Using estimated pose for disambiguation...")
            min_diff = float('inf')
            
            # Convert estimated Euler angles to rotation matrix
            # OpenCV uses ZYX (Yaw-Pitch-Roll) convention for Euler angles
            # Roll (X), Pitch (Y), Yaw (Z)
            # R_x = np.array([[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]])
            # R_y = np.array([[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]])
            # R_z = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])
            # R_est = R_z @ R_y @ R_x (for ZYX)
            
            # For simplicity and consistency with typical camera systems, let's assume
            # the estimated_camera_rotation_euler represents rotations around X, Y, Z axes
            # in radians. We'll convert degrees to radians.
            roll_rad, pitch_rad, yaw_rad = np.deg2rad(self.estimated_camera_rotation_euler)

            # Create rotation matrices for each axis
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(roll_rad), -np.sin(roll_rad)],
                           [0, np.sin(roll_rad), np.cos(roll_rad)]], dtype=np.float64)
            Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                           [0, 1, 0],
                           [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]], dtype=np.float64)
            Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                           [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                           [0, 0, 1]], dtype=np.float64)
            
            # Combined rotation matrix (e.g., ZYX order, common for camera orientation)
            # This R_est is R_camera_world_estimate
            R_est_camera_world = Rz @ Ry @ Rx 
            T_est_camera_world = self.estimated_camera_position.reshape(3,1) # This is the world origin in camera coords

            # Convert estimated camera position (world coords) to tvec (world origin in camera coords)
            # T_camera_world = -R_camera_world @ C_world
            # So, if self.estimated_camera_position is C_world, then T_est_tvec = -R_est_camera_world @ self.estimated_camera_position
            T_est_tvec = -R_est_camera_world @ self.estimated_camera_position.reshape(3,1)


            for i in range(len(rvecs)):
                rvec_candidate = rvecs[i]
                tvec_candidate = tvecs[i]

                R_candidate, _ = cv2.Rodrigues(rvec_candidate)
                
                # Calculate translation difference
                trans_diff = np.linalg.norm(tvec_candidate - T_est_tvec)
                
                # Calculate rotation difference (angular distance)
                R_diff = R_candidate @ R_est_camera_world.T # R_candidate * R_est_inv
                rvec_diff, _ = cv2.Rodrigues(R_diff)
                rot_diff = np.linalg.norm(rvec_diff) # Magnitude of rotation vector is angle in radians

                # Combine differences (can be weighted)
                # A simple sum, you might want to weight rotation more or less
                total_diff = trans_diff + rot_diff * 10 # Weight rotation more as it's in radians

                if total_diff < min_diff:
                    min_diff = total_diff
                    best_idx = i
            print(f"Selected solution {best_idx+1} based on minimum difference ({min_diff:.4f}) to estimate.")
            
        rvec = rvecs[best_idx]
        tvec = tvecs[best_idx]

        # Convert rotation vector to rotation matrix
        self.camera_rotation, _ = cv2.Rodrigues(rvec)
        
        # tvec is the translation vector of the world origin in camera coordinates.
        # To get the camera's position in world coordinates (C_w), we use:
        # C_w = -R_world_camera * T_camera_world
        # where R_world_camera = R_camera_world.T (self.camera_rotation.T)
        # and T_camera_world is tvec        
        self.camera_position = -self.camera_rotation.T @ tvec
        self.camera_position = self.camera_position.flatten() 
        self.camera_found = True
        print("Chosen camera position (world coords):", self.camera_position)
        print("Rotation matrix (world to camera):\n", self.camera_rotation)

        # --- Calculate 3D points for image plane visualization ---
        self.image_plane_points_3d = []
        self.image_plane_corners_3d = []

        if self.camera_found:
            # Focal length for the image plane (can be scaled for visualization)
            # Using a small scale factor to make the plane visible and not too large
            # relative to the camera model.
            # A common approach is to use a "normalized" image plane at Z=1,
            # then scale its X/Y dimensions by (image_width/fx) and (image_height/fy)
            # Or, use a fixed distance like 0.5 units in front of the camera.
            plane_distance = 0.5 # Distance from camera center to image plane for visualization
            
            # Calculate image plane corners in camera coordinates (normalized, then scaled)
            # (u,v) -> (x_c, y_c, z_c)
            # x_c = (u - cx) / fx * z_c
            # y_c = (v - cy) / fy * z_c
            # We choose z_c = plane_distance
            
            # Top-Left corner (0,0 pixel)
            tl_cam = np.array([(0 - cx) / fx * plane_distance, (0 - cy) / fy * plane_distance, plane_distance])
            # Top-Right corner (width,0 pixel)
            tr_cam = np.array([(image_width - cx) / fx * plane_distance, (0 - cy) / fy * plane_distance, plane_distance])
            # Bottom-Right corner (width,height pixel)
            br_cam = np.array([(image_width - cx) / fx * plane_distance, (image_height - cy) / fy * plane_distance, plane_distance])
            # Bottom-Left corner (0,height pixel)
            bl_cam = np.array([(0 - cx) / fx * plane_distance, (image_height - cy) / fy * plane_distance, plane_distance])

            # Transform corners from camera to world coordinates
            R_world_camera = self.camera_rotation.T
            T_world_camera = self.camera_position.reshape(3,1)

            self.image_plane_corners_3d = [
                (R_world_camera @ tl_cam.reshape(3,1) + T_world_camera).flatten(),
                (R_world_camera @ tr_cam.reshape(3,1) + T_world_camera).flatten(),
                (R_world_camera @ br_cam.reshape(3,1) + T_world_camera).flatten(),
                (R_world_camera @ bl_cam.reshape(3,1) + T_world_camera).flatten()
            ]

            # Project image points onto this 3D plane
            for img_pt in self.image_points:
                u, v = img_pt[0], img_pt[1]
                # Convert 2D pixel to 3D point on the plane in camera coordinates
                pt_on_plane_cam = np.array([(u - cx) / fx * plane_distance, (v - cy) / fy * plane_distance, plane_distance])
                # Transform to world coordinates
                self.image_plane_points_3d.append((R_world_camera @ pt_on_plane_cam.reshape(3,1) + T_world_camera).flatten())

        self.update()

    def compute_p3p(self):
        if len(self.world_points) < 3 or len(self.image_points) < 3:
            print("Need at least 3 world and 3 image points to compute PnP.")
            self.camera_found = False
            self.image_plane_points_3d = []
            self.image_plane_corners_3d = []
            self.update()
            return

        camera_matrix = np.array([[self.fx, 0, self.cx], 
                                  [0, self.fy, self.cy], 
                                  [0, 0, 1]], dtype=np.float64)        
        # No distortion coefficients for simplicity, assume ideal camera
        dist_coeffs = np.zeros((4, 1), dtype=np.float64) 
        
        # Use SOLVEPNP_SQPNP for 3 or more points. It can return multiple solutions.
        success, rvecs, tvecs = cv2.solveP3P(
            self.world_points.reshape(-1, 1, 3),
            self.image_points.reshape(-1, 1, 2),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_P3P
        )

        if not success or len(rvecs) == 0:
            print("PnP solution not found. Check point data or intrinsics.")
            self.camera_found = False
            self.image_plane_points_3d = []
            self.image_plane_corners_3d = []
            self.update()
            return

        print(f"Found {len(rvecs)} solutions from solveP3P (using SOLVEPNP_P3P)")

        best_idx = 0 # Default to first solution
        if self.use_estimated_pose_for_disambiguation and len(rvecs) > 1:
            print("Using estimated pose for disambiguation...")
            min_diff = float('inf')
            
            # Convert estimated Euler angles to rotation matrix
            # OpenCV uses ZYX (Yaw-Pitch-Roll) convention for Euler angles
            # Roll (X), Pitch (Y), Yaw (Z)
            # R_x = np.array([[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]])
            # R_y = np.array([[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]])
            # R_z = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])
            # R_est = R_z @ R_y @ R_x (for ZYX)
            
            # For simplicity and consistency with typical camera systems, let's assume
            # the estimated_camera_rotation_euler represents rotations around X, Y, Z axes
            # in radians. We'll convert degrees to radians.
            roll_rad, pitch_rad, yaw_rad = np.deg2rad(self.estimated_camera_rotation_euler)

            # Create rotation matrices for each axis
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(roll_rad), -np.sin(roll_rad)],
                           [0, np.sin(roll_rad), np.cos(roll_rad)]], dtype=np.float64)
            Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                           [0, 1, 0],
                           [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]], dtype=np.float64)
            Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                           [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                           [0, 0, 1]], dtype=np.float64)
            
            # Combined rotation matrix (e.g., ZYX order, common for camera orientation)
            # This R_est is R_camera_world_estimate
            R_est_camera_world = Rz @ Ry @ Rx 
            T_est_camera_world = self.estimated_camera_position.reshape(3,1) # This is the world origin in camera coords

            # Convert estimated camera position (world coords) to tvec (world origin in camera coords)
            # T_camera_world = -R_camera_world @ C_world
            # So, if self.estimated_camera_position is C_world, then T_est_tvec = -R_est_camera_world @ self.estimated_camera_position
            T_est_tvec = -R_est_camera_world @ self.estimated_camera_position.reshape(3,1)


            for i in range(len(rvecs)):
                rvec_candidate = rvecs[i]
                tvec_candidate = tvecs[i]

                R_candidate, _ = cv2.Rodrigues(rvec_candidate)
                
                # Calculate translation difference
                trans_diff = np.linalg.norm(tvec_candidate - T_est_tvec)
                
                # Calculate rotation difference (angular distance)
                R_diff = R_candidate @ R_est_camera_world.T # R_candidate * R_est_inv
                rvec_diff, _ = cv2.Rodrigues(R_diff)
                rot_diff = np.linalg.norm(rvec_diff) # Magnitude of rotation vector is angle in radians

                # Combine differences (can be weighted)
                # A simple sum, you might want to weight rotation more or less
                total_diff = trans_diff + rot_diff * 10 # Weight rotation more as it's in radians

                if total_diff < min_diff:
                    min_diff = total_diff
                    best_idx = i
            print(f"Selected solution {best_idx+1} based on minimum difference ({min_diff:.4f}) to estimate.")
            
        rvec = rvecs[best_idx]
        tvec = tvecs[best_idx]

        # Convert rotation vector to rotation matrix
        self.camera_rotation, _ = cv2.Rodrigues(rvec)
        
        # tvec is the translation vector of the world origin in camera coordinates.
        # To get the camera's position in world coordinates (C_w), we use:
        # C_w = -R_world_camera * T_camera_world
        # where R_world_camera = R_camera_world.T (self.camera_rotation.T)
        # and T_camera_world is tvec        
        self.camera_position = -self.camera_rotation.T @ tvec
        self.camera_position = self.camera_position.flatten() 
        self.camera_found = True
        print("Chosen camera position (world coords):", self.camera_position)
        print("Rotation matrix (world to camera):\n", self.camera_rotation)

        # --- Calculate 3D points for image plane visualization ---
        self.image_plane_points_3d = []
        self.image_plane_corners_3d = []

        if self.camera_found:
            # Focal length for the image plane (can be scaled for visualization)
            # Using a small scale factor to make the plane visible and not too large
            # relative to the camera model.
            # A common approach is to use a "normalized" image plane at Z=1,
            # then scale its X/Y dimensions by (image_width/fx) and (image_height/fy)
            # Or, use a fixed distance like 0.5 units in front of the camera.
            plane_distance = 0.5 # Distance from camera center to image plane for visualization
            
            # Calculate image plane corners in camera coordinates (normalized, then scaled)
            # (u,v) -> (x_c, y_c, z_c)
            # x_c = (u - cx) / fx * z_c
            # y_c = (v - cy) / fy * z_c
            # We choose z_c = plane_distance
            
            # Top-Left corner (0,0 pixel)
            tl_cam = np.array([(0 - self.cx) / self.fx * plane_distance, (0 - self.cy) / self.fy * plane_distance, plane_distance])            
            # Top-Right corner (width,0 pixel)
            tr_cam = np.array([(self.image_width - self.cx) / self.fx * plane_distance, (0 - self.cy) / self.fy * plane_distance, plane_distance])
            # Bottom-Right corner (width,height pixel)
            br_cam = np.array([(self.image_width - self.cx) / self.fx * plane_distance, (self.image_height - self.cy) / self.fy * plane_distance, plane_distance])
            # Bottom-Left corner (0,height pixel)
            bl_cam = np.array([(0 - self.cx) / self.fx * plane_distance, (self.image_height - self.cy) / self.fy * plane_distance, plane_distance])

            # Transform corners from camera to world coordinates
            R_world_camera = self.camera_rotation.T
            T_world_camera = self.camera_position.reshape(3,1)

            self.image_plane_corners_3d = [
                (R_world_camera @ tl_cam.reshape(3,1) + T_world_camera).flatten(),
                (R_world_camera @ tr_cam.reshape(3,1) + T_world_camera).flatten(),
                (R_world_camera @ br_cam.reshape(3,1) + T_world_camera).flatten(),
                (R_world_camera @ bl_cam.reshape(3,1) + T_world_camera).flatten()
            ]

            # Project image points onto this 3D plane
            for img_pt in self.image_points:
                u, v = img_pt[0], img_pt[1]
                # Convert 2D pixel to 3D point on the plane in camera coordinates
                pt_on_plane_cam = np.array([(u - self.cx) / self.fx * plane_distance, (v - self.cy) / self.fy * plane_distance, plane_distance])
                # Transform to world coordinates
                self.image_plane_points_3d.append((R_world_camera @ pt_on_plane_cam.reshape(3,1) + T_world_camera).flatten())

        self.update()


    def set_point_count(self, n):
        old_n = len(self.world_points)
        
        # Define default new point values (small scale for visualization)
        new_3d_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        new_2d_point = np.array([0.0, 0.0], dtype=np.float32) # Default pixel coords

        if n > old_n:
            # Append dummy points if increasing size
            self.world_points = np.append(self.world_points, np.zeros((n - old_n, 3), dtype=np.float32), axis=0)
            self.image_points = np.append(self.image_points, np.zeros((n - old_n, 2), dtype=np.float32), axis=0)
        else:
            # Truncate if decreasing size
            self.world_points = self.world_points[:n]
            self.image_points = self.image_points[:n]

        self.camera_found = False # Invalidate camera pose on point count change
        self.image_plane_points_3d = [] # Clear image plane data
        self.image_plane_corners_3d = [] # Clear image plane data
        self.update()  # trigger redraw if needed

    def closeEvent(self, event):
        # Clean up GLU quadric object when widget is closed
        if self.quadric:
            gluDeleteQuadric(self.quadric)
        super().closeEvent(event)

# --- Dialog Classes ---

class IntrinsicDialog(QDialog):
    def __init__(self, fx, fy, cx, cy, img_w, img_h, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Intrinsic Parameters")
        self.setModal(True)
        self.layout = QGridLayout(self)

        self.fx_spin = QDoubleSpinBox(); self.fx_spin.setRange(100.0, 2000.0); self.fx_spin.setSingleStep(10.0); self.fx_spin.setValue(fx)
        self.fy_spin = QDoubleSpinBox(); self.fy_spin.setRange(100.0, 2000.0); self.fy_spin.setSingleStep(10.0); self.fy_spin.setValue(fy)
        self.cx_spin = QDoubleSpinBox(); self.cx_spin.setRange(0.0, 2000.0); self.cx_spin.setSingleStep(1.0); self.cx_spin.setValue(cx)
        self.cy_spin = QDoubleSpinBox(); self.cy_spin.setRange(0.0, 2000.0); self.cy_spin.setSingleStep(1.0); self.cy_spin.setValue(cy)
        self.img_w_spin = QSpinBox(); self.img_w_spin.setRange(100, 4000); self.img_w_spin.setSingleStep(1); self.img_w_spin.setValue(img_w)
        self.img_h_spin = QSpinBox(); self.img_h_spin.setRange(100, 4000); self.img_h_spin.setSingleStep(1); self.img_h_spin.setValue(img_h)

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

        # Store the local origin for conversion
        self.local_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if len(current_world_points_epsg) > 0:
            self.local_origin = current_world_points_epsg[0] # Use first point as origin for display

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
                x_spin.setValue(self.local_origin[0] + (i * 0.5))
                y_spin.setValue(self.local_origin[1] + (i * 0.5))
                z_spin.setValue(self.local_origin[2] + (i * 0.1))


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
        self.gl_widget = GLWidget()
        main_layout.addWidget(self.gl_widget, 3)

        # Control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)

        # Data Storage for EPSG:28992 and Intrinsics
        self.current_fx, self.current_fy = 800.0, 800.0
        self.current_cx, self.current_cy = 512.0, 384.0
        self.current_img_w, self.current_img_h = 1024, 768
        self.current_world_points_epsg = np.array([[155001.0, 465000.0, 13.0], 
                                                   [154999.0, 465000.0, 13.0], 
                                                   [155000.0, 465001.0, 13.0]], dtype=np.float64)
        self.current_image_points = np.array([[600.0, 300.0], [400.0, 300.0], [512.0, 450.0]], dtype=np.float32)
        
        # Local origin for visualization (will be the first world point by default)
        self.local_origin_3d = self.current_world_points_epsg[0] if len(self.current_world_points_epsg) > 0 else np.array([0.0, 0.0, 0.0])

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

        # Estimated Position
        self.est_x_spin = QDoubleSpinBox(); self.est_x_spin.setDecimals(2); self.est_x_spin.setRange(-10.0, 10.0); self.est_x_spin.setSingleStep(0.1); self.est_x_spin.setValue(0.0)
        self.est_y_spin = QDoubleSpinBox(); self.est_y_spin.setDecimals(2); self.est_y_spin.setRange(-10.0, 10.0); self.est_y_spin.setSingleStep(0.1); self.est_y_spin.setValue(0.0)
        self.est_z_spin = QDoubleSpinBox(); self.est_z_spin.setDecimals(2); self.est_z_spin.setRange(-10.0, 10.0); self.est_z_spin.setSingleStep(0.1); self.est_z_spin.setValue(0.0)
        
        self.estimated_pose_layout.addWidget(QLabel("Pos X:"), 1, 0); self.estimated_pose_layout.addWidget(self.est_x_spin, 1, 1)
        self.estimated_pose_layout.addWidget(QLabel("Y:"), 1, 2); self.estimated_pose_layout.addWidget(self.est_y_spin, 1, 3)
        self.estimated_pose_layout.addWidget(QLabel("Z:"), 1, 4); self.estimated_pose_layout.addWidget(self.est_z_spin, 1, 5)

        # Estimated Rotation (Euler angles: Roll, Pitch, Yaw in degrees)
        self.est_roll_spin = QDoubleSpinBox(); self.est_roll_spin.setDecimals(1); self.est_roll_spin.setRange(-180.0, 180.0); self.est_roll_spin.setSingleStep(1.0); self.est_roll_spin.setValue(0.0)
        self.est_pitch_spin = QDoubleSpinBox(); self.est_pitch_spin.setDecimals(1); self.est_pitch_spin.setRange(-180.0, 180.0); self.est_pitch_spin.setSingleStep(1.0); self.est_pitch_spin.setValue(0.0)
        self.est_yaw_spin = QDoubleSpinBox(); self.est_yaw_spin.setDecimals(1); self.est_yaw_spin.setRange(-180.0, 180.0); self.est_yaw_spin.setSingleStep(1.0); self.est_yaw_spin.setValue(0.0)

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
        
        # Add compute button
        compute_btn = QPushButton("Compute Camera Pose (PnP)") 
        compute_btn.clicked.connect(self.gl_widget.compute_p3p)
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

    def update_num_points_in_dialogs(self, count):
        # This updates the internal data arrays to the new size
        # The actual values will be populated when dialogs are opened
        old_count = len(self.current_world_points_epsg)
        if count > old_count:
            self.current_world_points_epsg = np.append(self.current_world_points_epsg, np.zeros((count - old_count, 3), dtype=np.float64), axis=0)
            self.current_image_points = np.append(self.current_image_points, np.zeros((count - old_count, 2), dtype=np.float32), axis=0)
        else:
            self.current_world_points_epsg = self.current_world_points_epsg[:count]
            self.current_image_points = self.current_image_points[:count]
        
        # Update GLWidget's internal arrays to match the new size (values will be dummy for now)
        self.gl_widget.set_point_count(count)
        self.status_label.setText(f"Number of points set to {count}. Open dialogs to set values.")\
            
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
            # Update local origin for visualization
            if len(self.current_world_points_epsg) > 0:
                self.local_origin_3d = self.current_world_points_epsg[0]
            else:
                self.local_origin_3d = np.array([0.0, 0.0, 0.0])
            
            # Convert EPSG:28992 to local coordinates for GLWidget
            points_3d_local = self.current_world_points_epsg - self.local_origin_3d
            self.gl_widget.set_world_points(points_3d_local)
            self.status_label.setText("World points updated (local origin set to first point).")

    def open_image_points_dialog(self):
        dialog = ImagePointsDialog(self.num_points_spinbox.value(), self.current_image_points, 
                                   self.current_img_w, self.current_img_h, self)
        if dialog.exec_():
            self.current_image_points = dialog.get_points()
            self.gl_widget.set_image_points(self.current_image_points)
            self.status_label.setText("Image points updated.")

    def update_gl_widget_data(self):
        # Call this once at startup to push initial data to GLWidget
        self.gl_widget.set_intrinsic_parameters(self.current_fx, self.current_fy, self.current_cx, self.current_cy, self.current_img_w, self.current_img_h)
        
        # Convert initial EPSG:28992 points to local for GLWidget
        if len(self.current_world_points_epsg) > 0:
            self.local_origin_3d = self.current_world_points_epsg[0]
        else:
            self.local_origin_3d = np.array([0.0, 0.0, 0.0])
        points_3d_local = self.current_world_points_epsg - self.local_origin_3d
        self.gl_widget.set_world_points(points_3d_local)
        
        self.gl_widget.set_image_points(self.current_image_points)
        self.update_estimated_pose() # Also push initial estimated pose

    def update_world_point(self, idx, axis, value):
        if axis == 'x':
            self.gl_widget.world_points[idx][0] = value
        elif axis == 'y':
            self.gl_widget.world_points[idx][1] = value
        elif axis == 'z':
            self.gl_widget.world_points[idx][2] = value
        self.gl_widget.camera_found = False # Invalidate pose
        self.gl_widget.image_plane_points_3d = [] # Clear image plane data
        self.gl_widget.image_plane_corners_3d = [] # Clear image plane data
        self.gl_widget.update()

    def update_image_point(self, idx, axis, value):
        if axis == 'u':
            self.gl_widget.image_points[idx][0] = value
        elif axis == 'v':
            self.gl_widget.image_points[idx][1] = value
        self.gl_widget.camera_found = False # Invalidate pose
        self.gl_widget.image_plane_points_3d = [] # Clear image plane data
        self.gl_widget.image_plane_corners_3d = [] # Clear image plane data
        self.gl_widget.update()
        
    def update_estimated_pose(self):
        x = self.est_x_spin.value()
        y = self.est_y_spin.value()
        z = self.est_z_spin.value()
        roll = self.est_roll_spin.value()
        pitch = self.est_pitch_spin.value()
        yaw = self.est_yaw_spin.value()
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