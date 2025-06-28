import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from scipy.spatial.transform import Rotation
import mimetypes # <--- IMPORT THE LIBRARY

# This line ensures that the server serves .js files with the correct MIME type.
mimetypes.add_type('application/javascript', '.js')

# --- Backend Calculation Logic (Refactored from GLWidget) ---

def solve_pnp_pose(world_points_np, image_points_np, camera_intrinsics):
    """
    Computes camera pose from 3D-2D correspondences.
    
    Args:
        world_points_np (np.ndarray): Nx3 array of 3D world points.
        image_points_np (np.ndarray): Nx2 array of 2D image points.
        camera_intrinsics (dict): Dictionary with 'fx', 'fy', 'cx', 'cy'.

    Returns:
        dict: A dictionary containing the results or an error message.
    """
    if len(world_points_np) < 4:
        return {"success": False, "error": "PnP-RANSAC requires at least 4 point correspondences."}

    # Build camera matrix from intrinsics
    K = np.array([
        [camera_intrinsics['fx'], 0, camera_intrinsics['cx']],
        [0, camera_intrinsics['fy'], camera_intrinsics['cy']],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    try:
        # Use RANSAC for robustness against outliers
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            world_points_np,
            image_points_np,
            K,
            dist_coeffs,
            iterationsCount=200,
            reprojectionError=8.0, # Standard default
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP # A good general-purpose algorithm
        )

        if not success:
            return {"success": False, "error": "solvePnPRansac failed to find a solution."}

        # --- Process Results ---
        # Convert rotation vector to matrix (R_world_to_cam)
        R_world_to_cam, _ = cv2.Rodrigues(rvec)
        
        # The camera's pose in the world is the inverse of the view matrix.
        # R_cam_to_world = R_world_to_cam.T
        # t_cam_in_world = -R_cam_to_world @ tvec
        R_cam_to_world = R_world_to_cam.transpose()
        t_cam_in_world = -np.dot(R_cam_to_world, tvec)


        # For Three.js, a quaternion is most convenient.
        # Scipy's 'from_matrix' creates the quaternion for the rotation.
        # This quaternion represents the orientation of the camera *in the world frame*.
        quat_cam_in_world = Rotation.from_matrix(R_cam_to_world).as_quat()

        # --- Calculate Reprojection Error ---
        projected_points, _ = cv2.projectPoints(world_points_np, rvec, tvec, K, dist_coeffs)
        projected_points = projected_points.reshape(-1, 2)
        
        reprojection_errors = np.linalg.norm(image_points_np - projected_points, axis=1)
        mean_error = np.mean(reprojection_errors)
        
        inlier_indices = inliers.flatten().tolist() if inliers is not None else []
        
        reprojection_details = []
        for i in range(len(world_points_np)):
            reprojection_details.append({
                'world_point': world_points_np[i].tolist(),
                'image_point': image_points_np[i].tolist(),
                'projected_point': projected_points[i].tolist(),
                'error_pixels': reprojection_errors[i],
                'is_inlier': i in inlier_indices
            })

        result = {
            "success": True,
            "camera_position": t_cam_in_world.flatten().tolist(),
            "camera_quaternion": quat_cam_in_world.tolist(), # [x, y, z, w]
            "reprojection": {
                "mean_error_pixels": mean_error,
                "details": reprojection_details,
            }
        }
        return result

    except cv2.error as e:
        return {"success": False, "error": f"OpenCV Error: {e}"}
    except Exception as e:
        return {"success": False, "error": f"An unexpected error occurred: {e}"}


# --- Flask App Setup ---

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app) # Enable Cross-Origin Resource Sharing for the API

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/api/solve-pnp', methods=['POST'])
def api_solve_pnp():
    """API endpoint to receive data and compute the pose."""
    data = request.get_json()
    
    try:
        # Extract and convert data to numpy arrays
        world_points = np.array(data['worldPoints'], dtype=np.float64)
        image_points = np.array(data['imagePoints'], dtype=np.float32)
        intrinsics = data['cameraIntrinsics']
        
        # Perform the PnP calculation
        result = solve_pnp_pose(world_points, image_points, intrinsics)
        
        return jsonify(result)

    except (KeyError, TypeError) as e:
        return jsonify({"success": False, "error": f"Invalid or missing data in request: {e}"}), 400

# In app.py, add this new endpoint function after your existing api_solve_pnp

@app.route('/api/solve-multiple-pnp', methods=['POST'])
def api_solve_multiple_pnp():
    """
    API endpoint to solve for multiple cameras in a single request.
    Expects a JSON payload like:
    {
        "worldPoints": [[x,y,z], ...],
        "tasks": [
            { "imagePoints": [[u,v], ...], "cameraIntrinsics": {...} },
            { "imagePoints": [[u,v], ...], "cameraIntrinsics": {...} }
        ]
    }
    """
    data = request.get_json()

    try:
        world_points_np = np.array(data['worldPoints'], dtype=np.float64)
        tasks = data['tasks']

        results = []
        for task in tasks:
            image_points_np = np.array(task['imagePoints'], dtype=np.float32)
            intrinsics = task['cameraIntrinsics']
            
            # REUSE the existing single-pose solver function!
            pose_result = solve_pnp_pose(world_points_np, image_points_np, intrinsics)
            results.append(pose_result)
        
        return jsonify({"success": True, "results": results})

    except (KeyError, TypeError) as e:
        return jsonify({"success": False, "error": f"Invalid or missing data in request: {e}"}), 400


if __name__ == '__main__':
    # Running in debug mode is helpful for development
    app.run(debug=True, port=5000)