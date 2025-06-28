import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.spatial.transform import Rotation

app = Flask(__name__)
CORS(app)

@app.route('/solve_pnp', methods=['POST'])
def solve_pnp_endpoint():
    try:
        data = request.get_json()
        
        world_points = np.array(data['worldPoints'], dtype=np.float64)
        image_points = np.array(data['imagePoints'], dtype=np.float32)
        intrinsics = data['intrinsics']
        solver_options = data['solverOptions']

        if len(world_points) < 4:
            return jsonify({"error": "RANSAC requires at least 4 point correspondences."}), 400

        camera_matrix = np.array([
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        ################################### MODIFIED FROM P3P_V5 ###################################
        # --- 1. PRE-PROCESSING AS PER YOUR METHOD ---
        # WARNING: This method of scaling K and image points is not standard and can
        # lead to less stable results than only scaling world points.
        # The post-processing has been adapted to reverse this specific transformation.
        world_points_mean = np.mean(world_points, axis=0)
        shifted_world_points = world_points - world_points_mean
        
        scale_factor = 100.0
        
        scaled_shifted_world_points = shifted_world_points / scale_factor
        scaled_image_points = image_points / scale_factor
        
        # Create the scaled camera matrix. Note that K[2,2] must remain 1.
        scaled_camera_matrix = camera_matrix.copy()
        scaled_camera_matrix[0, 0] /= scale_factor # fx' = fx / s
        scaled_camera_matrix[1, 1] /= scale_factor # fy' = fy / s
        scaled_camera_matrix[0, 2] /= scale_factor # cx' = cx / s
        scaled_camera_matrix[1, 2] /= scale_factor # cy' = cy / s
        
        
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
        
        # self.last_inliers = inliers
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec) # This is R_camera_world (world to camera)
        R_world_to_cam = R        

        R_cam_to_world = R_world_to_cam.T
        camera_position= (-R_cam_to_world @ tvec).flatten()
        camera_position_epsg = world_points_mean + camera_position * scale_factor
        
        # self.camera_position = self.scale_point_for_display(camera_position_epsg)
        # self.camera_found = True
        
        euler_angles_deg = Rotation.from_matrix(R_world_to_cam).as_euler('ZYX', degrees=True)
        yaw_z = euler_angles_deg[2]
        pitch_y = euler_angles_deg[1]
        roll_x = euler_angles_deg[0]
        
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
        reprojection_errors = np.linalg.norm(projected_image_points - image_points, axis=1)

        # Calculate mean reprojection error
        mean_reprojection_error = np.mean(reprojection_errors)
        ################################### MODIFIED FROM P3P_V5 ###################################
        # success, rvec, tvec, inliers = cv2.solvePnPRansac(
        #     world_points,
        #     image_points,
        #     camera_matrix,
        #     dist_coeffs,
        #     iterationsCount=solver_options['iterations'],
        #     reprojectionError=solver_options['reprojErr'],
        #     confidence=0.99,
        #     flags=cv2.SOLVEPNP_EPNP
        # )
        
        # if not success:
        #     return jsonify({"error": "solvePnPRansac failed to find a solution."}), 500

        # R, _ = cv2.Rodrigues(rvec)
        # camera_position = -np.matrix(R).T @ np.matrix(tvec)
        
        # # Note: cv2.RQDecomp3x3 returns angles in degrees for the ZYX extrinsic convention
        # # The order is [roll_x, pitch_y, yaw_z]
        # euler_angles_deg = cv2.RQDecomp3x3(R)[0]
        
        # projected_points, _ = cv2.projectPoints(
        #     world_points, rvec, tvec, camera_matrix, dist_coeffs
        # )
        # projected_points = projected_points.reshape(-1, 2)
        # errors = np.linalg.norm(image_points - projected_points, axis=1)
        # mean_error = np.mean(errors)

        response_data = {
            "success": True,
            "camera_position": camera_position.tolist(),
            "rotation_euler_degrees": {
                "roll_x": roll_x,
                "pitch_y": pitch_y,
                "yaw_z": yaw_z
            },
            "reprojection_errors": reprojection_errors.tolist(),
            "mean_reprojection_error": mean_reprojection_error,
            "inliers": inliers.flatten().tolist() if inliers is not None else list(range(len(world_points)))
        }
        
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)