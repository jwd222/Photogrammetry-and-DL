import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QDoubleSpinBox, QGroupBox, QGridLayout, QSpinBox)
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

        # Initial world points (example: a triangle in 3D space, small scale for visualization)
        self.world_points = np.array([[1.0, 0.0, 3.0], [-1.0, 0.0, 3.0], [0.0, 1.0, 3.0]], dtype=np.float32)
        # Initial image points (pixel coordinates, consistent with camera_matrix)
        # Assuming a 1024x768 image, cx=512, cy=384
        self.image_points = np.array([[600.0, 300.0], [400.0, 300.0], [512.0, 450.0]], dtype=np.float32)

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
            self.x_trans += dx * 0.01 * (self.zoom / 10.0) 
            self.y_trans -= dy * 0.01 * (self.zoom / 10.0) # Invert Y for intuitive panning
        elif event.buttons() & Qt.RightButton:
            self.z_rot = (self.z_rot + dx) % 360
        
        self.last_pos = event.pos()
        self.update()
    
    def compute_p3p(self):
        # Define camera intrinsic parameters
        fx, fy = 800.0, 800.0 # Focal lengths in pixels
        cx, cy = 512.0, 384.0 # Principal point (center of image) in pixels for 1024x768 image
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        
        # No distortion coefficients for simplicity, assume ideal camera
        dist_coeffs = np.zeros((4, 1), dtype=np.float64) 
        
        # Use SOLVEPNP_SQPNP for 3 or more points. It can return multiple solutions.
        success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            self.world_points.reshape(-1, 1, 3),
            self.image_points.reshape(-1, 1, 2),
            camera_matrix,
            dist_coeffs, # Pass distortion coefficients
            flags=cv2.SOLVEPNP_SQPNP
        )

        if not success or len(rvecs) == 0:
            print("PnP solution not found. Check point data or intrinsics.")
            self.camera_found = False
            self.update()
            return

        print(f"Found {len(rvecs)} solutions from PnP (using SQPNP)")

        # For visualization, we'll just pick the first solution.
        # In a real application, you'd use a 4th point or other criteria to disambiguate.
        best_idx = 0 
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
        self.camera_position = self.camera_position.flatten() # Convert to 1D array
        self.camera_found = True
        print("Chosen camera position (world coords):", self.camera_position)
        print("Rotation matrix (world to camera):\n", self.camera_rotation)
        
        self.update()

    def set_point_count(self, n):
        old_n = len(self.world_points)
        
        # Define default new point values (small scale for visualization)
        new_3d_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        new_2d_point = np.array([0.0, 0.0], dtype=np.float32) # Default pixel coords

        if n > old_n:
            # Add new points
            for i in range(old_n, n):
                # Simple default values for new points, slightly offset
                self.world_points = np.append(self.world_points, [new_3d_point + np.array([i*0.1, i*0.05, i*0.02])], axis=0)
                self.image_points = np.append(self.image_points, [new_2d_point + np.array([i*10, i*5])], axis=0)
        else:
            # Truncate
            self.world_points = self.world_points[:n]
            self.image_points = self.image_points[:n]

        self.camera_found = False # Invalidate camera pose on point count change
        self.update()  # trigger redraw if needed

    def closeEvent(self, event):
        # Clean up GLU quadric object when widget is closed
        if self.quadric:
            gluDeleteQuadric(self.quadric)
        super().closeEvent(event)


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

        # Number of points selector
        self.num_points_spinbox = QSpinBox()
        self.num_points_spinbox.setRange(3, 20)
        self.num_points_spinbox.setValue(3)
        self.num_points_spinbox.valueChanged.connect(self.update_num_points)
        control_layout.addWidget(QLabel("Number of Correspondences:"))
        control_layout.addWidget(self.num_points_spinbox)

        # World points group
        self.world_layout = QGridLayout()
        self.world_group = QGroupBox("World Points (3D Coordinates)") 
        self.world_group.setLayout(self.world_layout)
        control_layout.addWidget(self.world_group)

        # Image points group
        self.image_layout = QGridLayout()
        self.image_group = QGroupBox("Image Points (2D Image Coordinates)")
        self.image_group.setLayout(self.image_layout)
        control_layout.addWidget(self.image_group)

        # Add control panel to layout
        main_layout.addWidget(control_panel, 2)

        # Initialize points (this will also populate spinboxes)
        self.update_num_points(self.num_points_spinbox.value())
        
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

    def update_num_points(self, count):
        # Clear existing widgets
        for layout in [self.world_layout, self.image_layout]:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)

        # Resize data arrays in GLWidget (this also invalidates camera_found)
        self.gl_widget.set_point_count(count)

        # Re-populate spinboxes based on the (potentially new) data in gl_widget
        for i in range(count):
            # World point spinboxes
            x_spin = QDoubleSpinBox(); x_spin.setDecimals(2); x_spin.setRange(-10.0, 10.0); x_spin.setSingleStep(0.1)
            y_spin = QDoubleSpinBox(); y_spin.setDecimals(2); y_spin.setRange(-10.0, 10.0); y_spin.setSingleStep(0.1)
            z_spin = QDoubleSpinBox(); z_spin.setDecimals(2); z_spin.setRange(0.1, 20.0); z_spin.setSingleStep(0.1) # Z > 0 for points in front of camera

            x_spin.setValue(self.gl_widget.world_points[i][0])
            y_spin.setValue(self.gl_widget.world_points[i][1])
            z_spin.setValue(self.gl_widget.world_points[i][2])

            x_spin.valueChanged.connect(lambda v, idx=i: self.update_world_point(idx, 'x', v))
            y_spin.valueChanged.connect(lambda v, idx=i: self.update_world_point(idx, 'y', v))
            z_spin.valueChanged.connect(lambda v, idx=i: self.update_world_point(idx, 'z', v))

            self.world_layout.addWidget(QLabel(f"P{i+1}:"), i, 0)
            self.world_layout.addWidget(QLabel("X:"), i, 1)
            self.world_layout.addWidget(x_spin, i, 2)
            self.world_layout.addWidget(QLabel("Y:"), i, 3)
            self.world_layout.addWidget(y_spin, i, 4)
            self.world_layout.addWidget(QLabel("Z:"), i, 5)
            self.world_layout.addWidget(z_spin, i, 6)
            self.world_layout.setColumnStretch(6, 1) # Add stretch to last column

            # Image point spinboxes (assuming 1024x768 image)
            u_spin = QDoubleSpinBox(); u_spin.setDecimals(2); u_spin.setRange(0.0, 1024.0); u_spin.setSingleStep(1.0)
            v_spin = QDoubleSpinBox(); v_spin.setDecimals(2); v_spin.setRange(0.0, 768.0); v_spin.setSingleStep(1.0)

            u_spin.setValue(self.gl_widget.image_points[i][0])
            v_spin.setValue(self.gl_widget.image_points[i][1])

            u_spin.valueChanged.connect(lambda v, idx=i: self.update_image_point(idx, 'u', v))
            v_spin.valueChanged.connect(lambda v, idx=i: self.update_image_point(idx, 'v', v))

            self.image_layout.addWidget(QLabel(f"P{i+1}:"), i, 0)
            self.image_layout.addWidget(QLabel("U:"), i, 1)
            self.image_layout.addWidget(u_spin, i, 2)
            self.image_layout.addWidget(QLabel("V:"), i, 3)
            self.image_layout.addWidget(v_spin, i, 4)
            self.image_layout.setColumnStretch(5, 1) # Add stretch to last column

    def update_world_point(self, idx, axis, value):
        if axis == 'x':
            self.gl_widget.world_points[idx][0] = value
        elif axis == 'y':
            self.gl_widget.world_points[idx][1] = value
        elif axis == 'z':
            self.gl_widget.world_points[idx][2] = value
        self.gl_widget.camera_found = False # Invalidate pose
        self.gl_widget.update()

    def update_image_point(self, idx, axis, value):
        if axis == 'u':
            self.gl_widget.image_points[idx][0] = value
        elif axis == 'v':
            self.gl_widget.image_points[idx][1] = value
        self.gl_widget.camera_found = False # Invalidate pose
        self.gl_widget.update()
    
    def toggle_option(self, option):
        if option == 'rays':
            self.gl_widget.show_rays = not self.gl_widget.show_rays
        elif option == 'camera':
            self.gl_widget.show_camera = not self.gl_widget.show_camera
        elif option == 'points':
            self.gl_widget.show_points = not self.gl_widget.show_points
        elif option == 'reprojection':
            self.gl_widget.show_reprojection = not self.gl_widget.show_reprojection
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