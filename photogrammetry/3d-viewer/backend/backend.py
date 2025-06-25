from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# --- Flask App Initialization ---
app = Flask(__name__)
# Enable CORS to allow requests from the frontend (running on a different origin)
CORS(app)

# --- API Endpoint for PnP ---
@app.route('/solve_pnp', methods=['POST'])
def solve_pnp_endpoint():
    """
    Receives 3D-2D point correspondences and camera intrinsics,
    solves for camera pose using OpenCV's PnP+RANSAC, and returns the pose.
    """
    try:
        # 1. Get data from the POST request
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        world_points = np.array(data['worldPoints'], dtype=np.float64)
        image_points = np.array(data['imagePoints'], dtype=np.float64)
        intrinsics = data['intrinsics']
        solver_options = data['solverOptions']

        # 2. Construct the camera matrix (K) from intrinsics
        camera_matrix = np.array([
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ], dtype=np.float64)

        # Assume no lens distortion
        dist_coeffs = np.zeros((4, 1))

        # 3. Solve for pose using solvePnPRansac
        # This function finds an object pose from 3D-2D point correspondences.
        # It returns the rotation (rvec) and translation (tvec) vectors
        # that transform a 3D point from the world coordinate space to the camera coordinate space.
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            world_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            iterationsCount=solver_options.get('iterations', 100),
            reprojectionError=solver_options.get('reprojErr', 8.0),
            flags=cv2.SOLVEPNP_ITERATIVE # A robust and common choice
        )

        if not success:
            return jsonify({"error": "PnP solution failed. Check input points and intrinsics."}), 500

        # 4. Convert the pose to the format needed by the frontend
        # The rvec and tvec from OpenCV represent the transformation from World to Camera (t_c_w).
        # We need the camera's pose *in* the world (t_w_c). This is the inverse transformation.
        
        # Convert rotation vector to a rotation matrix
        R_camera_from_world, _ = cv2.Rodrigues(rvec)

        # To get the camera's pose in the world, we invert the transformation.
        # Rotation: R_world_from_camera = R_camera_from_world.T
        # Translation: C = -R_world_from_camera @ tvec
        R_world_from_camera = R_camera_from_world.T
        camera_position_in_world = -R_world_from_camera @ tvec

        # Convert the world-space rotation matrix to Euler angles (ZYX extrinsic) for the frontend
        # Scipy's 'zyx' is equivalent to sequential rotations around Z, then new Y, then new X.
        scipy_rotation = Rotation.from_matrix(R_world_from_camera)
        euler_angles_deg = scipy_rotation.as_euler('zyx', degrees=True)

        # 5. Calculate reprojection errors for all points (not just inliers)
        reprojected_points, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, dist_coeffs)
        reprojected_points = reprojected_points.reshape(-1, 2)
        errors = np.linalg.norm(image_points - reprojected_points, axis=1)
        mean_error = np.mean(errors)

        # 6. Prepare the JSON response
        response_data = {
            "success": True,
            "camera_position": camera_position_in_world.flatten().tolist(),
            "rotation_euler_degrees": {
                "yaw_z": euler_angles_deg[0],
                "pitch_y": euler_angles_deg[1],
                "roll_x": euler_angles_deg[2]
            },
            "reprojection_errors": errors.tolist(),
            "mean_reprojection_error": float(mean_error),
            "inliers": inliers.flatten().tolist() if inliers is not None else list(range(len(world_points)))
        }

        return jsonify(response_data)

    except Exception as e:
        # Catch any other errors and return a helpful message
        return jsonify({"error": str(e)}), 500

# --- Main execution ---
if __name__ == '__main__':
    # Run the app on localhost, port 5000
    app.run(debug=True, port=5000)