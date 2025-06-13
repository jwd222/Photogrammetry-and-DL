import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QDoubleSpinBox, QGroupBox, QGridLayout, QSpinBox, QSlider)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtOpenGL import QGLWidget, QGLFormat, QGL
from OpenGL.GL import *
from OpenGL.GLU import *

class GLWidget(QGLWidget):
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)
        self.world_points = np.array([[1, 0, 3], [-1, 0, 3], [0, 1, 3]], dtype=np.float32)
        self.image_points = np.array([[0.3, 0.3], [-0.3, 0.3], [0, -0.3]], dtype=np.float32)
        self.camera_position = np.array([0, 0, 0], dtype=np.float32)
        self.camera_rotation = np.eye(3, dtype=np.float32)
        self.camera_found = False
        self.x_rot = 30
        self.y_rot = -30
        self.z_rot = 0
        self.zoom = 10.0
        self.show_rays = True
        self.show_camera = True
        self.show_points = True
        self.show_reprojection = True
        self.setMinimumSize(600, 500)

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

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w/h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
    
    @staticmethod    
    def draw_sphere(radius=0.1, slices=16, stacks=16):
        quadric = gluNewQuadric()
        gluSphere(quadric, radius, slices, stacks)
        gluDeleteQuadric(quadric)

    @staticmethod
    def draw_wire_cone(base=0.4, height=0.8, slices=16, stacks=1):
        glBegin(GL_LINE_LOOP)
        for i in range(slices):
            angle = 2 * np.pi * i / slices
            x = base * np.cos(angle)
            y = base * np.sin(angle)
            glVertex3f(x, y, 0)
        glEnd()

        glBegin(GL_LINES)
        for i in range(slices):
            angle = 2 * np.pi * i / slices
            x = base * np.cos(angle)
            y = base * np.sin(angle)
            glVertex3f(x, y, 0)
            glVertex3f(0, 0, height)
        glEnd()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Camera positioning
        glTranslatef(0, 0, -self.zoom)
        glRotatef(self.x_rot, 1, 0, 0)
        glRotatef(self.y_rot, 0, 1, 0)
        glRotatef(self.z_rot, 0, 0, 1)
        
        self.draw_axes()
        
        # Draw world points
        if self.show_points:
            glColor3f(1, 0.5, 0)
            for point in self.world_points:
                glPushMatrix()
                glTranslatef(point[0], point[1], point[2])
                self.draw_sphere(0.1, 16, 16)
                glPopMatrix()
        
        # Draw camera
        if self.camera_found and self.show_camera:
            glPushMatrix()
            glTranslatef(self.camera_position[0], self.camera_position[1], self.camera_position[2])
            
            # Draw camera body
            glColor3f(0.2, 0.4, 0.8)
            glScalef(0.5, 0.5, 0.5)
            self.draw_sphere(0.3, 16, 16)
            glTranslatef(0, 0, -0.3)
            self.draw_wire_cone(0.4, 0.8, 16, 16)
            glPopMatrix()
            
            # Draw camera axes
            glBegin(GL_LINES)
            glColor3f(1, 0, 0)
            glVertex3fv(self.camera_position)
            glVertex3fv(self.camera_position + self.camera_rotation[:, 0] * 0.8)
            
            glColor3f(0, 1, 0)
            glVertex3fv(self.camera_position)
            glVertex3fv(self.camera_position + self.camera_rotation[:, 1] * 0.8)
            
            glColor3f(0, 0.5, 1)
            glVertex3fv(self.camera_position)
            glVertex3fv(self.camera_position + self.camera_rotation[:, 2] * 0.8)
            glEnd()
        
        # Draw rays from camera to points
        if self.show_rays and self.camera_found:
            glBegin(GL_LINES)
            glColor3f(0.8, 0.8, 0.8)
            for i in range(3):
                glVertex3fv(self.camera_position)
                glVertex3fv(self.world_points[i])
            glEnd()
        
        # Draw reprojection if available
        if self.show_reprojection and self.camera_found:
            glDisable(GL_LIGHTING)
            glColor3f(0, 1, 0)
            for i in range(3):
                # Draw a small sphere at the reprojected point
                glPushMatrix()
                glTranslatef(self.world_points[i][0], self.world_points[i][1], self.world_points[i][2])
                self.draw_sphere(0.05, 10, 10)
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
        glColor3f(0, 0, 1)
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
        self.zoom = max(5, min(self.zoom - delta, 30))
        self.update()
    
    def mousePressEvent(self, event):
        self.last_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        
        if event.buttons() & Qt.LeftButton:
            self.x_rot = (self.x_rot + dy) % 360
            self.y_rot = (self.y_rot + dx) % 360
        elif event.buttons() & Qt.RightButton:
            self.z_rot = (self.z_rot + dx) % 360
        
        self.last_pos = event.pos()
        self.update()
    
    def compute_p3p_v1(self):
        # Create a camera matrix (intrinsic parameters)
        fx, fy = 800, 800
        cx, cy = 400, 300
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        
        # Solve PnP using 3 points
        success, rvec, tvec = cv2.solvePnP(
            self.world_points,
            self.image_points,
            camera_matrix,
            None,
            flags=cv2.SOLVEPNP_P3P
        )
        
        if success:
            # Convert rotation vector to rotation matrix
            self.camera_rotation, _ = cv2.Rodrigues(rvec)
            self.camera_position = -self.camera_rotation.T @ tvec
            self.camera_position = self.camera_position.flatten()
            self.camera_found = True
            print("Camera position:", self.camera_position)
            print("Rotation matrix:\n", self.camera_rotation)
        else:
            self.camera_found = False
            print("P3P solution not found")
        
        self.update()
      
    def compute_p3p(self):
        fx, fy = 800, 800
        cx, cy = 512, 384
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        ########################################################
        # | Method               | # of points   | Notes                                   |
        # | -------------------- | ------------- | --------------------------------------- |
        # | `SOLVEPNP_P3P`       | **Exactly 4** | Use for minimal solution                |
        # | `SOLVEPNP_ITERATIVE` | **≥ 4**       | Can use `useExtrinsicGuess=True` with 3 |
        # | `SOLVEPNP_SQPNP`     | **≥ 3**       | Use this if you have only 3 points      |
        ########################################################

        success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            self.world_points.reshape(-1, 1, 3),
            self.image_points.reshape(-1, 1, 2),
            camera_matrix,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_SQPNP
        )


        if not success or len(rvecs) == 0:
            print("P3P solution not found")
            self.camera_found = False
            return

        print(f"Found {len(rvecs)} solutions from P3P")

        # Optional: You can choose based on reprojection error, or later use a 4th point
        best_idx = 0  # stub: currently just use first
        rvec = rvecs[best_idx]
        tvec = tvecs[best_idx]

        self.camera_rotation, _ = cv2.Rodrigues(rvec)
        self.camera_position = -self.camera_rotation.T @ tvec
        self.camera_position = self.camera_position.flatten()
        self.camera_found = True

        print("Chosen camera position:", self.camera_position)
        print("Rotation matrix:\n", self.camera_rotation)

        self.update()

    def update_world_point(self, index, x, y, z):
        if 0 <= index < 3:
            self.world_points[index] = [x, y, z]
            self.camera_found = False
            self.update()
    
    def update_image_point(self, index, u, v):
        if 0 <= index < 3:
            self.image_points[index] = [u, v]
            self.camera_found = False
            self.update()

    def set_point_count(self, n):
        old_n = len(self.world_points)
        new_3d_point = np.array([[0.0, 0.0, 0.0]])
        new_2d_point = np.array([[0.0, 0.0]])
        
        if n > old_n:
            # Add new points
            for _ in range(n - old_n):
                self.world_points = np.append(self.world_points, new_3d_point, axis=0)
                self.image_points = np.append(self.image_points, new_2d_point, axis=0)
        else:
            # Truncate
            self.world_points = self.world_points[:n]
            self.image_points = self.image_points[:n]

        self.update()  # trigger redraw if needed

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
        control_layout.addWidget(QLabel("Number of Correspondences"))
        control_layout.addWidget(self.num_points_spinbox)

        # World points group
        self.world_layout = QGridLayout()
        self.world_group = QGroupBox("World Points (3D Coordinates, EPSG:28992)")
        self.world_group.setLayout(self.world_layout)
        control_layout.addWidget(self.world_group)

        # Image points group
        self.image_layout = QGridLayout()
        self.image_group = QGroupBox("Image Points (2D Image Coordinates)")
        self.image_group.setLayout(self.image_layout)
        control_layout.addWidget(self.image_group)

        # Add control panel to layout
        main_layout.addWidget(control_panel, 2)

        # Initialize points
        self.update_num_points(self.num_points_spinbox.value())
        
        # Add compute button
        compute_btn = QPushButton("Compute Camera Pose (P3P)")
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
        
        points_cb = QPushButton("Toggle Points")
        points_cb.clicked.connect(lambda: self.toggle_option('points'))
        vis_layout.addWidget(points_cb)
        
        reproj_cb = QPushButton("Toggle Reprojection")
        reproj_cb.clicked.connect(lambda: self.toggle_option('reprojection'))
        vis_layout.addWidget(reproj_cb)
        
        vis_group.setLayout(vis_layout)
        control_layout.addWidget(vis_group)
        
        # Add reset button
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        control_layout.addWidget(reset_btn)
        
        # Add status label
        self.status_label = QLabel("Ready. Set 3 world points and 3 image points, then compute.")
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

        self.world_spinboxes = []
        self.image_spinboxes = []

        # Resize data arrays in GLWidget
        self.gl_widget.set_point_count(count)

        for i in range(count):
            # World point spinboxes
            x_spin = QDoubleSpinBox(); x_spin.setDecimals(2)
            y_spin = QDoubleSpinBox(); y_spin.setDecimals(2)
            z_spin = QDoubleSpinBox(); z_spin.setDecimals(2)

            # EPSG:28992 ranges (example based on RD New meters)
            x_spin.setRange(0, 300000)
            y_spin.setRange(300000, 650000)
            z_spin.setRange(-50, 100)  # elevation range

            x_spin.setValue(self.gl_widget.world_points[i][0])
            y_spin.setValue(self.gl_widget.world_points[i][1])
            z_spin.setValue(self.gl_widget.world_points[i][2])

            x_spin.valueChanged.connect(lambda v, idx=i: self.update_world_point(idx, 'x', v))
            y_spin.valueChanged.connect(lambda v, idx=i: self.update_world_point(idx, 'y', v))
            z_spin.valueChanged.connect(lambda v, idx=i: self.update_world_point(idx, 'z', v))

            self.world_spinboxes.append((x_spin, y_spin, z_spin))
            self.world_layout.addWidget(QLabel(f"Point {i+1}:"), i, 0)
            self.world_layout.addWidget(x_spin, i, 1)
            self.world_layout.addWidget(y_spin, i, 2)
            self.world_layout.addWidget(z_spin, i, 3)

            # Image point spinboxes
            u_spin = QDoubleSpinBox(); u_spin.setDecimals(2)
            v_spin = QDoubleSpinBox(); v_spin.setDecimals(2)

            # Image coordinates in normalized or pixel range
            u_spin.setRange(-10000, 10000)
            v_spin.setRange(-10000, 10000)

            u_spin.setValue(self.gl_widget.image_points[i][0])
            v_spin.setValue(self.gl_widget.image_points[i][1])

            u_spin.valueChanged.connect(lambda v, idx=i: self.update_image_point(idx, 'u', v))
            v_spin.valueChanged.connect(lambda v, idx=i: self.update_image_point(idx, 'v', v))

            self.image_spinboxes.append((u_spin, v_spin))
            self.image_layout.addWidget(QLabel(f"Point {i+1}:"), i, 0)
            self.image_layout.addWidget(u_spin, i, 1)
            self.image_layout.addWidget(v_spin, i, 2)

    def update_world_point(self, idx, axis, value):
        if axis == 'x':
            self.gl_widget.world_points[idx][0] = value
        elif axis == 'y':
            self.gl_widget.world_points[idx][1] = value
        elif axis == 'z':
            self.gl_widget.world_points[idx][2] = value
        self.gl_widget.update()

    def update_image_point(self, idx, axis, value):
        if axis == 'u':
            self.gl_widget.image_points[idx][0] = value
        elif axis == 'v':
            self.gl_widget.image_points[idx][1] = value
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
        self.gl_widget.zoom = 10.0
        self.gl_widget.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())