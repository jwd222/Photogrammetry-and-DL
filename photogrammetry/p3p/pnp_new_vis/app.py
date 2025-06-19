import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

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

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            world_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            iterationsCount=solver_options['iterations'],
            reprojectionError=solver_options['reprojErr'],
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success:
            return jsonify({"error": "solvePnPRansac failed to find a solution."}), 500

        R, _ = cv2.Rodrigues(rvec)
        camera_position = -np.matrix(R).T @ np.matrix(tvec)
        
        # Note: cv2.RQDecomp3x3 returns angles in degrees for the ZYX extrinsic convention
        # The order is [roll_x, pitch_y, yaw_z]
        euler_angles_deg = cv2.RQDecomp3x3(R)[0]
        
        projected_points, _ = cv2.projectPoints(
            world_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        errors = np.linalg.norm(image_points - projected_points, axis=1)
        mean_error = np.mean(errors)

        response_data = {
            "success": True,
            "camera_position": camera_position.A1.tolist(),
            "rotation_euler_degrees": {
                "roll_x": euler_angles_deg[0],
                "pitch_y": euler_angles_deg[1],
                "yaw_z": euler_angles_deg[2]
            },
            "reprojection_errors": errors.tolist(),
            "mean_reprojection_error": mean_error,
            "inliers": inliers.flatten().tolist() if inliers is not None else list(range(len(world_points)))
        }
        
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)